import dataclasses
import functools
import logging
import os
import random
import warnings
from typing import Iterable, Iterator, List, Optional, Tuple

import jax
import ml_collections
import numpy as np
import tensorflow.compat.v1 as tf

from manyfold.data.parsers import parse_fasta
from manyfold.model import features
from manyfold.model.tf import input_pipeline, input_pipeline_plmfold, proteins_dataset
from manyfold.train.gcp_utils import GCPBucketFilepath


def _check_tfrecords_pattern(pattern, pattern_name="patter"):
    if os.path.splitext(pattern)[-1] != ".tfrecord":
        logging.warn(f"{pattern_name} does not end with '.tfrecord' - is this correct?")


def get_train_val_filepaths(pattern, test_pattern=None, test_frac=0.1, as_lists=True):
    _check_tfrecords_pattern(pattern)
    filepaths = tf.data.Dataset.list_files(pattern, shuffle=True)

    if test_pattern:
        _check_tfrecords_pattern(test_pattern)
        train_filepaths = filepaths
        val_filepaths = tf.data.Dataset.list_files(test_pattern, shuffle=True)
    else:
        data_size = tf.cast(tf.data.Dataset.cardinality(filepaths), tf.float32)
        test_size = tf.cast(data_size * test_frac, tf.int32)
        train_filepaths = filepaths.skip(test_size)
        val_filepaths = filepaths.take(test_size)

    _train_size = tf.data.Dataset.cardinality(train_filepaths)
    _test_size = tf.data.Dataset.cardinality(val_filepaths)
    logging.info(f"{_train_size}/{_test_size} train/test data files found.")

    if as_lists:
        train_filepaths = [name.numpy() for name in train_filepaths]
        val_filepaths = [name.numpy() for name in val_filepaths]

    return train_filepaths, val_filepaths


def get_train_val_filepaths_gcp(
    gcp_bucket_name: str,
    fasta_filepath: str = "sequences_ready_for_training_pre_casp14.fasta",
    tf_record_filename: str = "features_dict.tfrecord",
    test_frac: float = 0.1,
    shuffle_proteins: bool = True,
    shuffle_seed: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    gcp_fasta_filepath = GCPBucketFilepath(
        gcp_bucket_name=gcp_bucket_name,
        filepath=fasta_filepath,
    )

    fasta_str = gcp_fasta_filepath.download_to_string()[0]
    if type(fasta_str) is bytes:
        fasta_str = fasta_str.decode("utf-8")
    _, all_protein_ids = parse_fasta(fasta_str)

    filepaths = [
        GCPBucketFilepath(
            gcp_bucket_name=gcp_bucket_name,
            filepath=os.path.join(protein_id, tf_record_filename),
        ).get_gcp_filepath()
        for protein_id in all_protein_ids
    ]

    # Note filepaths are determinisitcally ordered at this point by order of fasta.
    if shuffle_proteins:
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(filepaths)
        else:
            random.shuffle(filepaths)

    _test_size = int(len(filepaths) * test_frac)
    _train_size = len(filepaths) - _test_size

    train_filepaths = filepaths[:_train_size]
    test_filepaths = filepaths[-_test_size:] if _test_size > 0 else []

    return train_filepaths, test_filepaths


def _accept_chain(features):
    """Filter chains according to SM 1.2.5"""
    # Filter according to chain length.
    num_residues = tf.cast(
        proteins_dataset._first(features["seq_length"]), dtype=tf.float32
    )
    accept_prob = tf.maximum(
        tf.minimum(tf.constant(512, dtype=tf.float32), num_residues),
        tf.constant(256, dtype=tf.float32),
    ) / tf.constant(512, dtype=tf.float32)

    # Filter according to the size of the PDB cluster this chain falls into
    cluster_size = tf.cast(
        proteins_dataset._first(features["pdb_cluster_size"]), dtype=tf.float32
    )
    accept_prob = accept_prob / cluster_size

    accept = tf.cast(accept_prob, dtype=tf.float32) > tf.random.uniform(
        shape=(), dtype=tf.float32
    )

    return accept


@dataclasses.dataclass
class TFDataloaderParams:
    filepaths: List[str]  # List of filepaths to tensorflow records, each corresponding
    # to the features for a protein chain. The filepaths can be GCP bucket filepaths
    # (in which case they should start with gs://)
    config: ml_collections.ConfigDict
    batch_dims: Tuple[int]  # Batches of training samples will be reshaped to this
    # shape before being outputed (the effective batch size is np.prod(batch_dims)). This
    # is useful for preparing the batches in a shape that matches what jax.pmap or jax.vmap
    # expects.
    shuffle: bool = True  # If True, shuffle the filepaths at the beginning of all
    # epochs
    apply_filters: bool = True  # If True, apply filters described in SM 1.2.5
    is_training: bool = True  # Apply data augmentation filters specific to training
    # samples
    is_validation_pipeline: bool = False  # If True, generate features for loss
    # computation during validation
    compression_type: str = "GZIP"
    num_parallel_reads: int = tf.data.AUTOTUNE  # Number of files to read in parallel
    num_parallel_calls: int = tf.data.AUTOTUNE  # Number of threads dedicated to
    # applying transformations to the raw features recovered from the files
    # (data augmentation, etc...)
    prefetch_factor: int = 2  # Number of batches to prefetch in the background
    max_num_epochs: Optional[int] = None  # The dataset iterator will stop yielding
    # samples after this number of epochs over the filepaths. If None, the iterator
    # never stops yielding examples.
    deterministic: Optional[bool] = False  # Control the level of determinism in
    # the dataloader when reading samples in parallel.
    drop_remainder: Optional[bool] = True  # Drop the last batch if smaller than
    # the effective batch size.
    process_msa_features: Optional[bool] = True
    split_data_across_pod_slice: Optional[bool] = False
    ignore_errors: Optional[bool] = False

    def __post_init__(self):
        assert self.compression_type in ["GZIP", "NONE", "GZIP", "ZLIB"]
        if self.split_data_across_pod_slice:
            # This assumes that all devices are self.filepaths in the same order!
            # If loaded from get_train_val_filepaths_gcp(...) this means either running
            # with get_train_val_filepaths_gcp(..., shuffle=False) or a fixed shuffle
            # seed on all devices!
            global_device_count = jax.device_count()
            local_device_count = jax.local_device_count()
            total_processes = global_device_count // local_device_count
            step = len(self.filepaths) // total_processes
            loc_id = jax.process_index()
            self.filepaths = self.filepaths[step * loc_id : step * (loc_id + 1)]

    def make_dataset(self) -> tf.data.Dataset:
        batch_size = np.prod(self.batch_dims)
        # When keeping remainder, repeat the last filepath to fill last batch up
        # to the effective batch size.
        # Note that this is meant for evaluation, when we know the sample ids and
        # their order in a fasta file (shuffle=False).
        extra_samples = len(self.filepaths) % batch_size
        if not self.drop_remainder and extra_samples:
            if self.shuffle:
                warnings.warn(
                    "In TFDataloaderParams, drop_remainder=False and shuffle=True. "
                    "This is meant for evaluation only."
                )
            num_reps = batch_size - extra_samples
            self.filepaths += [self.filepaths[-1]] * num_reps

        filepaths_dataset = tf.data.Dataset.from_tensor_slices(self.filepaths)
        if self.shuffle:
            filepaths_dataset = filepaths_dataset.shuffle(
                len(self.filepaths), reshuffle_each_iteration=True
            )
        filepaths_dataset = filepaths_dataset.repeat(self.max_num_epochs)

        dataset = tf.data.TFRecordDataset(
            filepaths_dataset,
            compression_type=self.compression_type,
            num_parallel_reads=self.num_parallel_reads,
        )
        if self.ignore_errors:
            dataset = dataset.apply(tf.data.experimental.ignore_errors())

        # Faster dataloading by relaxing determinism on order of parallel read samples.
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = self.deterministic
        dataset = dataset.with_options(ignore_order)

        data_cfg, feature_names = features.make_data_config(
            self.config, is_training=self.is_training
        )

        # Transforms the serialized tf.Example stored in the files
        # into dicts of tf tensors, potentially keeping only a subset of the features
        # stored in the tf.Example.
        create_tensor_dict = functools.partial(
            proteins_dataset.create_tensor_dict, features=feature_names
        )
        dataset = dataset.map(
            create_tensor_dict,
            num_parallel_calls=self.num_parallel_calls,
            deterministic=self.deterministic,
        )

        # Filter chains according to SM 1.2.5
        if self.apply_filters:
            dataset = dataset.filter(_accept_chain)

        if self.process_msa_features:
            process_tensors_from_config = input_pipeline.process_tensors_from_config
        else:
            process_tensors_from_config = (
                input_pipeline_plmfold.process_tensors_from_config
            )
        # Apply data augmentation
        process_tensors_from_config = functools.partial(
            process_tensors_from_config,
            data_config=data_cfg,
            is_training=self.is_training,
            is_validation_pipeline=self.is_validation_pipeline,
        )
        dataset = dataset.map(
            process_tensors_from_config,
            num_parallel_calls=self.num_parallel_calls,
            deterministic=self.deterministic,
        )

        # Fetch multiple batches of samples
        dataset = dataset.batch(
            batch_size,
            num_parallel_calls=self.num_parallel_calls,
            drop_remainder=self.drop_remainder,
        )

        # Reshape the batches to the expected self.batch_dims shape
        def _reshape(batch):
            for k in batch.keys():
                out_shape = self.batch_dims + batch[k][0].shape
                batch[k] = tf.reshape(batch[k], out_shape)
            return batch

        dataset = dataset.map(
            _reshape,
            num_parallel_calls=self.num_parallel_calls,
            deterministic=self.deterministic,
        )
        dataset = dataset.prefetch(self.prefetch_factor)
        return dataset


class TFDataloader(Iterable[features.FeatureDict]):
    """
    Iterable over samples to feed to the alphafold models (stored in the form
    of features.FeatureDict) using TFDataloaderParams.
    """

    def __init__(
        self,
        params: TFDataloaderParams,
        random_seed: Optional[int] = None,
    ):
        self.params = params
        self._session = None
        self._logger = logging.getLogger(f"{__name__}")

        self._logger.debug("Creating tensorflow graph")
        self._tf_graph = tf.Graph()
        with self._tf_graph.as_default(), tf.device("/device:CPU:0"):
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            dataset = self.params.make_dataset()
            self._iterator = dataset.make_initializable_iterator()
            self._next_element = self._iterator.get_next()
            self._tf_graph.finalize()
        self._logger.debug("Tensorflow graph has just been finalized")

    def __enter__(self):
        self._session = tf.Session(graph=self._tf_graph)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._session.close()

    def __iter__(self) -> Iterator[features.FeatureDict]:
        if self._session is None:
            raise RuntimeError("This class must be used within a context manager")

        self._logger.debug("Initializing the dataset")
        self._session.run(
            self._iterator.initializer,
        )
        self._logger.debug("Done initializing the dataset")

        while True:
            try:
                yield self._session.run(self._next_element)
            except tf.errors.OutOfRangeError:
                return
