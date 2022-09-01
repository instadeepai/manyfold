import logging
import os

import hydra
import jax
import numpy as np
from omegaconf import DictConfig

from manyfold.data.parsers import parse_fasta
from manyfold.model import config
from manyfold.model.model import get_confidence_metrics
from manyfold.train.dataloader_tf import TFDataloader
from manyfold.validation.inference_model import ValidModel, reshape_feat_dict
from manyfold.validation.pdb import run_relaxation_pdb, save_predicted_pdb
from manyfold.validation.utils import (
    filter_fasta,
    load_file,
    select_unpad_features,
    unpad_prediction,
)


@hydra.main(
    config_path="../manyfold/model/config", version_base=None, config_name="config_val"
)
def validate_model(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    cfg = config.convert_to_ml_dict(cfg)
    model_config = cfg.model_config
    args = cfg.args

    # Fetch command line arguments and overwrite config defaults accordingly.
    data_dir = args.data_dir
    results_dir = args.results_dir
    fasta_path = args.fasta_path
    filter_min_length = args.filter_min_length
    filter_max_length = args.filter_max_length
    model_name = args.model_name
    params_dir = args.params_dir
    num_devices = args.num_devices
    batch_size = args.batch_size
    max_batches = args.max_batches
    seed_value = args.seed_value
    use_relaxed_predictions = args.use_relaxed_predictions

    # Initial checkings on variables and directories.
    if not data_dir.startswith("gs://"):
        assert os.path.isdir(data_dir)
    assert filter_min_length >= 1
    assert filter_max_length >= 1
    assert filter_min_length <= filter_max_length
    assert batch_size >= 1
    assert num_devices >= 1
    effective_batch_size = num_devices * batch_size

    # Create local directory to store the output results.
    results_dir = os.path.join(results_dir, model_name)
    os.makedirs(results_dir, exist_ok=True)

    # ----- FASTA SEQUENCES, SEQIDS AND MAX(SEQ LENGTH) -----
    logging.info("Loading fasta file...")
    # Load fasta file (from bucket or locally).
    fasta_lines = load_file(fasta_path, from_numpy=False)
    # Get sequences and ids from fasta.
    sequences, seq_ids = parse_fasta(fasta_lines)
    logging.info(f"Found {len(seq_ids)} sequences in fasta.")

    if (filter_min_length > 1) or (filter_max_length < 999999):
        # Filter fasta to get sequences with lengths between min and max.
        sequences, seq_ids = filter_fasta(
            sequences, seq_ids, filter_min_length, filter_max_length
        )
        logging.info(f"Fasta filtered to {len(seq_ids)} sequences.")

    # Obtain maximum sequence length over sequences in fasta.
    max_seq_length = 0
    for seq in sequences:
        max_seq_length = max(len(seq), max_seq_length)
    logging.info(f"Maximum sequence length is {max_seq_length}.")

    # ----- MODEL CONFIG, PARAMETERS, DATALOADER AND DEVICES -----
    valid_model = ValidModel(model_name, model_config)
    # Set model configuration.
    logging.info("Preparing model config...")
    valid_model.prepare_config(max_seq_length, batch_size, num_devices)
    # Prepare validation dataloader.
    logging.info("Preparing validation dataloader...")
    val_filenames = [
        os.path.join(os.path.join(data_dir, seqid), "features_dict.tfrecord")
        for seqid in seq_ids
    ]
    val_dataloader_params = valid_model.prepare_dataloader(val_filenames)

    if model_config.model.global_config.plmfold_mode:
        # Load PLM parameters
        logging.info("Loading PLM parameters...")
        valid_model.load_params_plm()
    # Load model parameters.
    logging.info("Loading model parameters...")
    valid_model.load_params(params_dir)
    # Set up devices for evaluation.
    logging.info("Setting up devices...")
    valid_model.set_up_devices()

    # ----- EVALUATION -----
    logging.info("Starting evaluation...")
    with TFDataloader(val_dataloader_params, seed_value) as val_dataloader:
        # Iterate over the validation dataset and get the results.
        for batch_id, batch in enumerate(val_dataloader):
            if (max_batches > 0) and batch_id >= max_batches:
                break
            logging.info(f"Process batch number {batch_id}.")

            # Batched forward pass.
            logging.info("Inference pass.")
            results = valid_model.inference(batch, seed_value)

            logging.info("Save confidence metrics and PDB files.")
            # Get remainder if last batch is smaller than effective-batch-size.
            num_samples_batch = effective_batch_size
            if effective_batch_size * (batch_id + 1) > len(seq_ids):
                num_samples_batch = len(seq_ids) % effective_batch_size

            init_pos = effective_batch_size * batch_id
            seq_ids_batch = seq_ids[init_pos : init_pos + num_samples_batch]

            # Flatten first two dimensions (num-devices, batch-size).
            batch = reshape_feat_dict(batch)
            results = reshape_feat_dict(results)

            # Compute metrics for each individual sample in the batch.
            for i, seqid in enumerate(seq_ids_batch):
                # Get sequence id and create directory for sample.
                sample_save_dir = os.path.join(results_dir, seqid)
                os.makedirs(sample_save_dir, exist_ok=True)

                # Unbatch, get sequence length, and unpad features for sample.
                feats = jax.tree_util.tree_map(lambda x: x[i], batch)
                nres = feats["seq_length"][0]
                feats = select_unpad_features(feats, nres)
                # Unbatch and unpad predictions for sample.
                preds = jax.tree_util.tree_map(lambda x: x[i], results)
                preds = unpad_prediction(preds, nres)

                # Get confidence metrics for sample.
                preds.update(get_confidence_metrics(preds))
                sample_metrics = {
                    "plddt_mean": preds["ranking_confidence"],
                    "plddt_local": preds["plddt"],
                }
                # Add predicted aligned error metrics if available.
                if "ptm" in preds:
                    sample_metrics.update(
                        {
                            "ptm": preds["ptm"].item(),
                            "max_pae_value": preds[
                                "max_predicted_aligned_error"
                            ].item(),
                            "pae": preds["predicted_aligned_error"],
                        }
                    )
                # Save confidence metrics to .npy file.
                np.save(
                    os.path.join(sample_save_dir, "metrics.npy"),
                    sample_metrics,
                    allow_pickle=True,
                )

                # Save model prediction to .pdb files.
                save_predicted_pdb(preds, feats, output_dir=sample_save_dir)

                if use_relaxed_predictions:
                    # Run relaxation, and save relaxed structure.
                    run_relaxation_pdb(output_dir=sample_save_dir)


if __name__ == "__main__":
    validate_model()
