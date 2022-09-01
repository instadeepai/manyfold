import copy
import logging
import os
import sys

import hydra
import jax
from omegaconf import DictConfig

import manyfold.train.gcp_utils as gcp_utils
from manyfold.data.parsers import parse_fasta
from manyfold.model import config
from manyfold.train.dataloader_tf import TFDataloader, TFDataloaderParams
from manyfold.train.trainer import InvalidParametersAction, Trainer
from manyfold.train.utils import get_model_haiku_params_maybe_gcp
from manyfold.validation.utils import load_file


@hydra.main(
    config_path="../manyfold/model/config",
    version_base=None,
    config_name="config_train",
)
def train_model(cfg: DictConfig) -> None:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["JAX_ENABLE_X64"] = "False"

    ###
    # Read in config and make appropriate changes.
    ###

    cfg = config.convert_to_ml_dict(cfg)
    model_config = cfg.model_config
    args = cfg.args
    plmfold_mode = model_config.model.global_config.plmfold_mode

    # Fetch command line arguments and overwrite config defaults accordingly.
    resume_training = args.continue_from_last_checkpoint
    data_dir_train = args.data_dir_train
    data_dir_val = args.data_dir_val
    fasta_path_train = args.fasta_path_train
    fasta_path_val = args.fasta_path_val
    pretrained_models_dir = args.pretrained_models_dir
    checkpoint_dir = args.checkpoint_dir
    batch_size = args.batch_size
    crop_size = args.crop_size
    if plmfold_mode:
        force_uniform_crop_plm = args.force_uniform_crop_plm
        crop_size_plm = args.crop_size_plm

    if not (gcp_utils.is_gcp_path(checkpoint_dir) or os.path.exists(checkpoint_dir)):
        os.makedirs(checkpoint_dir)

    for directory in [data_dir_train, data_dir_val, checkpoint_dir]:
        if not gcp_utils.is_gcp_path(directory):
            assert os.path.isdir(directory), f"Directory not found {directory}"

    if not resume_training:
        (success, error_msg, checkpoint_paths) = Trainer.checkpoint_paths(
            checkpoint_dir
        )
        if not success:
            raise RuntimeError(error_msg)
        if checkpoint_paths:
            logging.warning(
                f"Found {len(checkpoint_paths)} existing checkpoints in "
                f"{checkpoint_dir} but --cont was not selected, deleting them."
            )
            for filename in checkpoint_paths:
                filepath = os.path.join(checkpoint_dir, filename)
                (success, error_msg) = gcp_utils.from_filepath(filepath).delete()
                if not success:
                    logging.warning(f"Failed to delete {filepath}, error: {error_msg}")

    model_config.train.continue_from_last_checkpoint = resume_training
    model_config.train.checkpointing.checkpoint_dir = checkpoint_dir

    model_config.data.common.crop_size = crop_size  # training crop_size
    if plmfold_mode:
        model_config.data.eval.force_uniform_crop_plm = force_uniform_crop_plm
        model_config.data.common.crop_size_plm = crop_size_plm  # training crop_size_plm

    model_config.data.common.batch_size = args.max_batch_size_per_device

    if model_config.train.invalid_paramaters_action is None:
        model_config.train.invalid_paramaters_action = InvalidParametersAction.ERROR
    else:
        model_config.train.invalid_paramaters_action = InvalidParametersAction.RESTORE

    ###
    # Check number of global/local devices and configure accordingly.
    ###

    num_global_devices = jax.device_count()
    num_local_devices = jax.local_device_count()
    # local_devices = jax.local_devices()
    all_devices = jax.devices()

    if jax.process_index() == 0:
        print(
            "---Devices---\n"
            + f"\tglobal device count: {num_global_devices}\n"
            + f"\tlocal device count: {num_local_devices}"
        )

    # Each tpu process has num_local_devices cores
    # --> samples_per_device: Number of samples in the batch on each tpu process.
    # --> batch_size_per_step: Number of samples across all global devicies.
    num_tpu_processes = num_global_devices // num_local_devices
    samples_per_device = jax.local_device_count() * args.max_batch_size_per_device
    batch_size_per_step = num_tpu_processes * samples_per_device

    model_config.train.optimizer.num_grad_acc_steps = batch_size // batch_size_per_step

    train_cfg = copy.deepcopy(model_config)
    val_cfg = copy.deepcopy(model_config)

    ###
    # Set up logging.
    ###

    logging.getLogger("smart_logging").setLevel(logging.INFO)
    logger = logging.getLogger("manyfold")
    logger.setLevel(logging.DEBUG)

    # stdout logging: logging.INFO --> stdout
    formatter = logging.Formatter(
        fmt="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # file logging: logging.DEBUG --> log_filename.log
    log_filename = "alphafold_training" if not plmfold_mode else "plmfold_training"
    file_handler = logging.FileHandler(f"{log_filename}.log")
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    ###
    # Set up datasets.
    ###

    _, seq_ids_train = parse_fasta(load_file(fasta_path_train, from_numpy=False))
    train_filenames = [
        os.path.join(os.path.join(data_dir_train, seqid), "features_dict.tfrecord")
        for seqid in seq_ids_train
    ]

    if fasta_path_val and data_dir_val:
        _, seq_ids_val = parse_fasta(load_file(fasta_path_val, from_numpy=False))
        val_filenames = [
            os.path.join(os.path.join(data_dir_val, seqid), "features_dict.tfrecord")
            for seqid in seq_ids_val
        ]
    else:
        val_filenames = train_filenames

    _train_size, _val_size = len(train_filenames), len(val_filenames)
    logging.info(f"{_train_size}/{_val_size} train/val data files found.")

    train_dataloader_params = TFDataloaderParams(
        filepaths=train_filenames,
        config=train_cfg,
        batch_dims=(num_local_devices, args.max_batch_size_per_device),
        process_msa_features=not plmfold_mode,
        split_data_across_pod_slice=False,
        ignore_errors=True,
    )

    val_dataloader_params = TFDataloaderParams(
        filepaths=val_filenames,
        config=val_cfg,
        batch_dims=(num_local_devices, args.max_batch_size_per_device),
        apply_filters=False,
        max_num_epochs=1,
        process_msa_features=not plmfold_mode,
        split_data_across_pod_slice=True,
        ignore_errors=True,
    )

    ###
    # Preload language model if needed.
    ###

    if plmfold_mode and args.load_pretrained_language_model:
        plm_config = model_config.language_model
        params_plm = get_model_haiku_params_maybe_gcp(
            model_name=plm_config.model_name,
            data_dir=plm_config.pretrained_model_dir,
        )
    else:
        params_plm = None

    ###
    # Load pre-trained model if needed.
    ###

    if args.pretrained_model:
        if not gcp_utils.is_gcp_path(pretrained_models_dir):
            assert os.path.isdir(
                pretrained_models_dir
            ), f"Directory not found {pretrained_models_dir}"
        model_params = get_model_haiku_params_maybe_gcp(
            model_name=args.pretrained_model,
            data_dir=pretrained_models_dir,
        )
    else:
        model_params = None

    ###
    # Run training!
    ###

    with TFDataloader(train_dataloader_params) as train_dataloader:
        with TFDataloader(val_dataloader_params) as val_dataloader:

            trainer = Trainer(
                cfg,
                train_dataloader=train_dataloader,
                validation_dataloader=val_dataloader,
                devices=all_devices,
            )

            logger.info("Initialize model...")
            trainer.init(params=model_params, random_seed=0, params_plm=params_plm)

            logger.info("Starting train loop...")
            trainer.train()


if __name__ == "__main__":
    train_model()
