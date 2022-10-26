# Copyright 2022 InstaDeep Ltd
#
# Licensed under the Creative Commons BY-NC-SA 4.0 License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import pickle
import random
import re
import shutil
import signal
import time
import types
from enum import Enum
from typing import Any, Callable, List, MutableMapping, Optional, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import tree
from omegaconf import OmegaConf

import manyfold.train.gcp_utils as gcp_utils
from manyfold.data.tools.utils import tmpdir_manager
from manyfold.model.model import (
    EvalModel,
    TrainModel,
    distributed_over_devices,
    fetch_from_devices,
    fetch_from_first_device,
    set_alphafold_policy,
)
from manyfold.smart_logging import Metric
from manyfold.train.dataloader_tf import TFDataloader


def add_handler(signo: signal.Signals, fn: Callable[[], Any]):
    def _wrapped(signo: signal.Signals, frame: types.FrameType):
        del signo, frame
        return fn()

    signal.signal(signo, _wrapped)


def check_params(model_params):

    dtypes = jax.tree_flatten(jax.tree_map(lambda x: x.dtype, model_params))[0]
    numels = jax.tree_flatten(jax.tree_map(lambda x: np.prod(x.shape), model_params))[0]

    is_bf16 = all([x == jnp.bfloat16 for x in dtypes])
    is_f16 = all([x == jnp.float16 for x in dtypes])
    is_f32 = all([x == jnp.float32 for x in dtypes])

    type_str = "bf16" if is_bf16 else "f16" if is_f16 else "f32" if is_f32 else None
    bytes_per_float = 2 if (is_bf16 or is_f16) else 4 if is_f32 else None

    tot_bytes = np.sum(numels) * bytes_per_float

    if type_str:

        print(
            f"Params have type {type_str}, which gives a total memory usage of "
            + f"{tot_bytes / 1024 ** 3:.3f}GiB."
        )
    else:
        print("Params have mixed dtypes!")
        print(jax.tree_map(lambda x: x.dtype, model_params))


def valuation(n: int, p: int) -> int:
    """Return p-adic valuation of n"""
    if n == 0:
        return np.inf
    v = 0
    while n % p == 0:
        v += 1
        n //= p
    return v


def replace_checkpoint_2adic(checkpoints: List[int], num_to_delete: int) -> List[int]:
    """Return indices of elements to delete in checkpoints list"""
    valuations = np.array([valuation(n, 2) for n in checkpoints])
    _, indices = zip(*sorted(zip(valuations, list(range(len(valuations))))))
    checkoints_to_delete = list(indices[:num_to_delete])
    return checkoints_to_delete


class InvalidParametersAction(Enum):
    CONTINUE = 0
    ERROR = 1
    RESTORE = 2


class Trainer:
    """Train AlphaFold."""

    def __init__(
        self,
        config: ml_collections.ConfigDict,
        train_dataloader: TFDataloader,
        validation_dataloader: TFDataloader,
        devices: Optional[Sequence[jax.xla.Device]] = None,
    ):

        if not devices:
            devices = jax.devices()
        self.devices = devices

        self.global_config = config
        self.config = config.model_config
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.logger = logging.getLogger(f"{__name__}")

        self.config.model.global_config.use_half_precision = (
            self.config.train.mixed_precision.use_half
        )

        self._eval_model = EvalModel(self.config)
        self._train_model = TrainModel(self.config)
        set_alphafold_policy(self._train_model.use_half_precision)

        self._cfg_chkpt = self.config.train.checkpointing
        self._cfg_val = self.config.train.validate

        self._num_grad_acc_steps = self.config.train.optimizer.num_grad_acc_steps

        # Placeholders for training state and exp. weighted ave. params.
        self.state, self.params_ewa, self.step = None, None, None
        self.params_plm = None

        # A boolean determind by whether we initialise from pre-trained parameters.
        # Primarily, we use this to determine whether we need to de-bias our EWA
        # of the parameters (pre-trained weights <--> do not debias).
        self._init_from_params = None

        # What to do if the parameters get NaN or Inf (InvalidParametersAction).
        self._invalid_paramaters_action = self.config.train.invalid_paramaters_action

        # Internal accumulations of over multiple sub-batches.
        self._acc_steps = 0
        self._acc_metrics = []

        # To store the model validation loss.
        self.best_lddt_loss = None

        # Handle preemption signal. Note that this must happen in the main thread.
        def _signal_handler():
            self.logger.info("Caught SIGTERM: forcing a checkpoint save.")
            self.save_checkpoint()

        add_handler(signal.SIGTERM, _signal_handler)

    @staticmethod
    def __save_checkpoint_to_file(checkpoint, checkpoint_filepath: str):
        """Save a checkpoint to disk.

        Simple utility function to ensure that checkpoints are saved to disk as pickle
        files with (i) the correct ".pkl" extension and (ii) initially to a temp file
        which is only renamed (and thus only overwrites any existing file in the same
        location) after serialization is completed successfully.
        """
        with tmpdir_manager(base_dir="/tmp") as tmp_dir:
            checkpoint_dir, filename = os.path.split(checkpoint_filepath)
            if os.path.splitext(filename)[-1] != ".pkl":
                filename += ".pkl"
            tmp_filepath = os.path.join(tmp_dir, filename)
            with open(tmp_filepath, "wb") as f:
                pickle.dump(checkpoint, f)
            output_filepath = os.path.join(checkpoint_dir, filename)
            if gcp_utils.is_gcp_path(output_filepath):
                (success, error_msg) = gcp_utils.GCPBucketFilepath.from_gcp_filepath(
                    output_filepath
                ).upload(tmp_filepath, bytes_format=True)
                if not success:
                    logging.getLogger(f"{__name__}").warning(
                        f"Failed to upload checkpoint {filename} to {output_filepath}. "
                        f"Exact error: {error_msg}"
                    )
            else:
                shutil.move(tmp_filepath, output_filepath)

    @staticmethod
    def __save_config_to_file(config, config_filepath: str):
        """Save a config to disk.

        Simple utility function to ensure that configs are saved to disk as yaml
        files with (i) the correct ".yaml" extension and (ii) initially to a temp file
        which is only renamed (and thus only overwrites any existing file in the same
        location) after saving is completed successfully.
        """

        with tmpdir_manager(base_dir="/tmp") as tmp_dir:
            config_dir, filename = os.path.split(config_filepath)
            if os.path.splitext(filename)[-1] != ".yaml":
                filename += ".yaml"
            tmp_filepath = os.path.join(tmp_dir, filename)
            with open(tmp_filepath, "w") as f:
                OmegaConf.save(config, f)
            output_filepath = os.path.join(config_dir, filename)
            if gcp_utils.is_gcp_path(output_filepath):
                (success, error_msg) = gcp_utils.GCPBucketFilepath.from_gcp_filepath(
                    output_filepath
                ).upload(tmp_filepath, bytes_format=False)
                if not success:
                    logging.getLogger(f"{__name__}").warning(
                        f"Failed to upload config {filename} to {output_filepath}. "
                        f"Exact error: {error_msg}"
                    )
            else:
                shutil.move(tmp_filepath, output_filepath)

    @staticmethod
    def __save_params_to_file(params, params_filepath: str):
        """Saves the current EWA params in .npz format."""

        def flatten(d, parent_key="", sep="//"):
            items = []
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if isinstance(v, MutableMapping):
                    items.extend(flatten(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        params = hk.data_structures.to_mutable_dict(params)

        with tmpdir_manager(base_dir="/tmp") as tmp_dir:
            params_dir, filename = os.path.split(params_filepath)
            if os.path.splitext(filename)[-1] != ".npz":
                filename += ".npz"
            tmp_filepath = os.path.join(tmp_dir, filename)
            np.savez_compressed(
                tmp_filepath, **flatten(params, parent_key="", sep="//")
            )
            output_filepath = os.path.join(params_dir, filename)
            if gcp_utils.is_gcp_path(output_filepath):
                (success, error_msg) = gcp_utils.GCPBucketFilepath.from_gcp_filepath(
                    output_filepath
                ).upload(tmp_filepath, bytes_format=True)
                if not success:
                    logging.getLogger(f"{__name__}").warning(
                        f"Failed to upload params {filename} to {output_filepath}. "
                        f"Exact error: {error_msg}"
                    )
            else:
                shutil.move(tmp_filepath, output_filepath)

    @staticmethod
    def checkpoint_paths(checkpoint_dir: str) -> Tuple[bool, str, List[str]]:
        """List all checkpoint paths, sorted in order of oldest to newest."""
        if gcp_utils.is_gcp_path(checkpoint_dir):
            (
                success,
                error_msg,
                possible_checkpoints,
            ) = gcp_utils.GCPBucketFilepath.from_gcp_filepath(checkpoint_dir).listdir()
            if not success:
                logging.error(error_msg)
                return (False, error_msg, [])
        else:
            possible_checkpoints = os.listdir(checkpoint_dir)

        def checkpoint_filter(x):
            return re.search("^checkpoint_([0-9]+)", x)

        valid_checkpoints = filter(
            lambda f: checkpoint_filter(f) is not None, possible_checkpoints
        )
        sorted_checkpoints = sorted(
            valid_checkpoints, key=lambda f: int(checkpoint_filter(f).group(1))
        )
        return (True, "", sorted_checkpoints)

    def save_checkpoint(self, checkpoint_name: Optional[str] = None):
        """Save a training checkpoint."""
        if self._skip_next_checkpoint or jax.process_index() != 0:
            self._skip_next_checkpoint = False
            return None

        (_, _, all_checkpoints) = self.checkpoint_paths(self._cfg_chkpt.checkpoint_dir)
        all_checkpoints = [
            int(re.search("checkpoint_(.+?).pkl", c).group(1)) for c in all_checkpoints
        ]

        # Retrieve current training state for non-distributed training (i.e.
        # drop the duplication along the leading device dimension).
        _state = self._train_model.to_single_state(self.state, as_numpy=True)

        if "params_plm" in _state:
            # Do not store pLM parameters from state in checkpoint.
            del _state["params_plm"]

        # Create and save new checkpoint.
        checkpoint = {
            "state": _state,
            "params_ewa": self.get_params_ewa(debias=False),
            "best_lddt_loss": self.best_lddt_loss,
            "steps": self.step,
            "init_from_params": self._init_from_params,
        }
        if not checkpoint_name:
            checkpoint_name = f"checkpoint_{self.step}"
        checkpoint_path = os.path.join(self._cfg_chkpt.checkpoint_dir, checkpoint_name)
        self.logger.debug(f"Serializing checkpoint to {checkpoint_path}")
        self.__save_checkpoint_to_file(checkpoint, checkpoint_path)
        self.logger.debug(f"Checkpoint saved to {checkpoint_path}")

        # Save config
        config_name = f"config_{self.step}"
        config_path = os.path.join(self._cfg_chkpt.checkpoint_dir, config_name)
        self.logger.debug(f"Save config to {config_path}")
        self.__save_config_to_file(self.global_config.to_dict(), config_path)
        self.logger.debug(f"Config saved to {config_path}")

        # Save params in .npz format
        params_name = f"params_{self.step}"
        params_path = os.path.join(self._cfg_chkpt.checkpoint_dir, params_name)
        self.logger.debug(f"Save params to {params_path}")
        self.__save_params_to_file(self.get_params_ewa(), params_path)
        self.logger.debug(f"Params saved to {params_path}")

        # Finally, delete older checkpoints if required.

        if (
            self._cfg_chkpt.keep_num > 0
            and len(all_checkpoints) > self._cfg_chkpt.keep_num
        ):
            checkpoints_to_delete = replace_checkpoint_2adic(
                all_checkpoints[
                    : len(all_checkpoints) - self._cfg_chkpt.keep_last_num + 1
                ],
                len(all_checkpoints) - self._cfg_chkpt.keep_num,
            )
            for checkpoint_to_delete in sorted(checkpoints_to_delete, reverse=True):
                # Delete checkpoint file
                checkpoint_path_to_delete = os.path.join(
                    self._cfg_chkpt.checkpoint_dir,
                    f"checkpoint_{all_checkpoints[checkpoint_to_delete]}.pkl",
                )
                self.logger.debug(f"Delete checkpoint from {checkpoint_path}")
                (success, error_msg) = gcp_utils.from_filepath(
                    checkpoint_path_to_delete
                ).delete()
                if not success:
                    self.logger.warning(
                        f"Failed to delete checkpoint {checkpoint_path_to_delete}. "
                        f"Exact error: {error_msg}"
                    )
                # Delete config file
                config_path_to_delete = os.path.join(
                    self._cfg_chkpt.checkpoint_dir,
                    f"config_{all_checkpoints[checkpoint_to_delete]}.yaml",
                )
                self.logger.debug(f"Delete config from {checkpoint_path}")
                (success, error_msg) = gcp_utils.from_filepath(
                    config_path_to_delete
                ).delete()
                if not success:
                    self.logger.warning(
                        f"Failed to delete config {config_path_to_delete}. "
                        f"Exact error: {error_msg}"
                    )
                # Delete params file
                params_path_to_delete = os.path.join(
                    self._cfg_chkpt.checkpoint_dir,
                    f"params_{all_checkpoints[checkpoint_to_delete]}.npz",
                )
                self.logger.debug(f"Delete params from {checkpoint_path}")
                (success, error_msg) = gcp_utils.from_filepath(
                    params_path_to_delete
                ).delete()
                if not success:
                    self.logger.warning(
                        f"Failed to params config {params_path_to_delete}. "
                        f"Exact error: {error_msg}"
                    )

    def save_params(self, file_name: Optional[str] = "params"):
        """Saves the current EWA params only.

        Unlike the checkpoints saved by save_checkpoint(...), saving only the EWA params
        will not allow training to be resumed.  However, to run the model in inference
        mode, only the EWA params are required.
        """
        params = self.get_params_ewa(debias=not self._init_from_params)

        if os.path.splitext(file_name)[-1] != ".pkl":
            file_name += ".pkl"
        path = os.path.join(self._cfg_chkpt.checkpoint_dir, file_name)
        self.logger.debug(f"Serializing params to {path}")
        self.__save_checkpoint_to_file(
            params, os.path.join(self._cfg_chkpt.checkpoint_dir, file_name)
        )

    def restore_checkpoint(self, checkpoint_name: Optional[str] = None):
        """Restore a training checkpoint."""
        if not checkpoint_name:
            (success, error_msg, checkpoint_paths) = self.checkpoint_paths(
                self._cfg_chkpt.checkpoint_dir
            )
            if not success:
                raise RuntimeError(error_msg)
            if len(checkpoint_paths) == 0:
                raise RuntimeError(
                    f"No checkpoint found in {self._cfg_chkpt.checkpoint_dir}"
                )
            checkpoint_name = checkpoint_paths[-1]
        checkpoint_path = os.path.join(self._cfg_chkpt.checkpoint_dir, checkpoint_name)
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")

        with tmpdir_manager(base_dir="/tmp") as tmp_dir:
            if gcp_utils.is_gcp_path(checkpoint_path):
                local_checkpoint_path = os.path.join(
                    tmp_dir, os.path.basename(checkpoint_path)
                )
                (success, error_msg) = gcp_utils.GCPBucketFilepath.from_gcp_filepath(
                    checkpoint_path
                ).download_to_file(local_checkpoint_path)
                if not success:
                    raise RuntimeError(
                        f"Failed to download checkpoint {checkpoint_path}."
                        f"Exact error: {error_msg}"
                    )
                checkpoint_path = local_checkpoint_path

            with open(checkpoint_path, "rb") as f:
                checkpoint = pickle.load(f)

        if self.params_plm:
            # Restore pLM parameters.
            checkpoint["state"]["params_plm"] = self.params_plm

        self.state = self._train_model.to_distributed_state(
            checkpoint["state"], jax.local_devices()
        )
        self.params_ewa = checkpoint["params_ewa"]
        self.best_lddt_loss = checkpoint["best_lddt_loss"]
        self.step = checkpoint["steps"]

        self._acc_steps = 0
        self._acc_metrics = []

        # Note that if 'init_from_params' is not in the checkpoint, we assume it
        # to be True.  This is to make the code backwards compatible with runs
        # taken before 'init_from_params' was added to the checkpoints.
        self._init_from_params = checkpoint.get("init_from_params", True)

    @staticmethod
    @jax.jit
    def __update_params_ewa(params_ewa, params, decay):
        return jax.tree_multimap(
            lambda ewa, new: decay * ewa + (1 - decay) * new, params_ewa, params
        )

    def update_params_ewa(self):
        """Update the exponenital moving average of the model parameters."""
        params = self._train_model.to_single_state(self.state["params"], as_numpy=True)
        params = jax.tree_map(lambda x: x.astype(np.float32), params)

        is_nonfinite_params = jax.tree_map(
            lambda x: not jnp.isfinite(x).all().item(), params
        )
        is_nonfinite_params = any(
            [True in p.values() for p in is_nonfinite_params.values()]
        )
        if is_nonfinite_params:
            self.logger.warning("Invalid value(s) found in the updated parameters.")

        if (
            not is_nonfinite_params
            or self._invalid_paramaters_action is InvalidParametersAction.CONTINUE
        ):
            if not self.params_ewa:
                # Initialize the EWA parameters to zero.
                self.params_ewa = jax.tree_map(lambda x: np.zeros_like(x), params)
            else:
                self.params_ewa = self.__update_params_ewa(
                    self.params_ewa, params, self._cfg_val.ewa_decay
                )
        elif self._invalid_paramaters_action is InvalidParametersAction.ERROR:
            raise Exception("Invalid value found in the updated parameters.")
        else:
            # self._invalid_paramaters_action is InvalidParametersAction.CONTINUE
            self.logger.info("Restoring last valid checkpoint.")
            self.restore_checkpoint()

    @staticmethod
    @jax.jit
    def __debias_ewa(params_ewa, step, decay):
        # Note small eps for numerical stability.
        wc = (1 - decay**step) / (1 - decay) + 1e-9
        return jax.tree_util.tree_map(lambda p: p / wc, params_ewa)

    def get_params_ewa(self, debias: Optional[bool] = None, as_numpy=True):
        """Get the exponenital moving average of the model parameters."""
        if debias is None:
            debias = self._cfg_val.ewa_debias
        params_ewa = self.params_ewa
        if debias:
            if self.step > 0:
                params_ewa = self.__debias_ewa(
                    params_ewa, self.step, self._cfg_val.ewa_decay
                )
        if as_numpy:
            params_ewa = jax.tree_util.tree_map(
                lambda p: np.asarray(p) if isinstance(p, jax.xla.DeviceArray) else p,
                params_ewa,
            )
        return hk.data_structures.to_mutable_dict(params_ewa)

    def init(self, params=None, random_seed: Optional[int] = None, params_plm=None):
        """Initialise the module for training."""

        (success, error_msg, checkpoint_paths) = self.checkpoint_paths(
            self._cfg_chkpt.checkpoint_dir
        )

        continue_from_last_checkpoint = self.config.train.continue_from_last_checkpoint

        if params_plm:
            # Update placeholder for pLM parameters.
            self.params_plm = {f"plmembed/{k}": v for k, v in params_plm.items()}

        if continue_from_last_checkpoint and (
            not success or not bool(checkpoint_paths)
        ):
            continue_from_last_checkpoint = False
            log_msg = (
                "continue_from_last_checkpoint was set but %s, "
                "will default to initialize the parameters with "
                f"{'provided set of' if params is not None else 'randomly initialized'}"
                " parameters."
            )
            if not success:
                self.logger.warning(
                    log_msg
                    % f"failed to look for checkpoints (exact error: {error_msg})"
                )
            else:
                self.logger.warning(
                    log_msg
                    % f"no checkpoint was found in {self._cfg_chkpt.checkpoint_dir}"
                )

        if not continue_from_last_checkpoint:
            self.logger.info(
                f"Creating new experiment at {self._cfg_chkpt.checkpoint_dir}"
            )
            start_time = time.time()

            if not random_seed:
                random_seed = np.random.randint(9999999)

            if params:
                self.logger.info("Initialising model parameters with provided values.")
                params = self._train_model.get_policy().cast_to_param(params)
                params = distributed_over_devices(params, devices=self.devices)
                self.state = self._train_model.init_from_params(
                    params=params,
                    random_seed=random_seed,
                    params_plm=params_plm,
                )

                self._init_from_params = True

            else:
                # Get a single sample, process it and map the params across all devices.
                t_fetch_start = time.time()
                feat = next(iter(self.train_dataloader))
                feat = tree.map_structure(lambda x: x[0, 0], feat)

                def crop_batch(batch, crop_size):
                    for k in batch:
                        try:
                            res_dim = self.config.data.eval.feat[k].index(
                                "num residues placeholder"
                            )
                            batch[k] = batch[k].take(
                                indices=np.arange(crop_size), axis=res_dim + 1
                            )
                        except ValueError:
                            pass
                    return batch

                feat = crop_batch(feat, 2)

                self.logger.info(
                    f"fetched (and cropped) first batch in {time.time() - t_fetch_start:.3f}s."
                )

                self.logger.info("Initialising model parameters with random values.")
                self.state = self._train_model.init(feat, random_seed, params_plm)

                self._init_from_params = False

            check_params(fetch_from_first_device(self.state["params"]))

            self.update_params_ewa()
            self.step = 0
            self._skip_next_checkpoint = False
            self.save_checkpoint()

            self.logger.info(
                f"Created new experiment in {time.time()-start_time:.3f}s."
            )
        else:
            # Load most recent checkpoint.
            self.restore_checkpoint(checkpoint_paths[-1])
            self._skip_next_checkpoint = True

    def log_step(self, metrics, print_summary=False, acc_steps=0):
        self._acc_metrics.append(metrics)
        if acc_steps > 0:
            # We are accumulating and have not just stepped.
            self.logger.debug(
                f"accumulated {acc_steps} steps in total,"
                + f"last step in {metrics['t_step']:.3f}s"
            )
        else:
            metrics = jax.tree_map(
                lambda *args: np.stack(args)
                if isinstance(args[0], np.ndarray)
                else np.array(args),
                *self._acc_metrics,
            )
            self._acc_metrics = []

            loss = np.mean(metrics["loss"]).astype(np.float32)
            log_str = (
                f"step {self.step} (pid={jax.process_index()}) | loss : {loss:.3f}"
            )

            self.logger.debug(str(Metric(name="loss", value=loss, step=self.step)))

            for k in metrics["output"].keys():
                if "loss" in metrics["output"][k]:
                    loss = np.mean(metrics["output"][k]["loss"]).astype(np.float32)
                    self.logger.debug(
                        str(Metric(name=f"{k}_loss", value=loss, step=self.step))
                    )
                    log_str += f",step {self.step} | {k}_loss : {loss:.3f}"

            train_time = np.sum(metrics["t_step"]).astype(np.float32)
            self.logger.debug(
                str(
                    Metric(
                        name="train_time",
                        value=train_time,
                        step=self.step,
                    )
                )
            )

            log_str += f", fetch time : {np.sum(metrics['t_fetch']):.3f}s"
            log_str += f", step time : {np.sum(metrics['t_step']):.3f}s."
            self.logger.info(log_str)

            if print_summary:
                print(log_str)

    def train(self):
        """Run the main training loop."""
        if self.state is None:
            self.init()
            set_alphafold_policy(self._train_model.use_half_precision)

        training_batch_iterator = iter(self.train_dataloader)

        while self.step < self.config.train.num_steps:
            # Run periodic validation.
            if (
                self.step % self.config.train.validate.validate_every_n_steps == 0
                and self.config.train.validate.validate_every_n_steps > 0
                and self._acc_steps == 0
                and (self.step > 0 or self.config.train.validate.validate_at_start)
                and (
                    not self.step % self._cfg_chkpt.checkpoint_every_n_steps == 0
                    or self.step == 0
                )
            ):
                self.validate(stash_params=True)

            # Fetch a new batch of data
            t = time.time()
            batch = next(training_batch_iterator)
            batch["num_iter_recycling"] = random.randint(
                0, self.config.model.num_recycle
            ) * jnp.ones_like(batch["resolution"], dtype=jnp.int32)
            t_fetch = time.time() - t
            self.logger.info(f"fetched batch in {t_fetch:.3f}s.")

            t = time.time()
            self.state, metrics = self._train_model.update_step(self.state, batch)

            metrics = fetch_from_devices(metrics, as_numpy=True)
            metrics["t_step"] = time.time() - t
            metrics["t_fetch"] = t_fetch

            self._acc_steps = (self._acc_steps + 1) % self._num_grad_acc_steps
            if self._acc_steps == 0:
                self.step += 1
                # Update the the EWA parameters.
                self.update_params_ewa()
                # Save checkpoints if needed.
                if self.step % self._cfg_chkpt.checkpoint_every_n_steps == 0:
                    # Run validation.
                    if (
                        self.step % self.config.train.validate.validate_every_n_steps
                        == 0
                        and self.config.train.validate.validate_every_n_steps > 0
                    ):
                        self.validate(stash_params=True)
                    self.save_checkpoint()

            # Logging.
            self.log_step(
                metrics=metrics,
                print_summary=False,
                acc_steps=self._acc_steps,
            )

        self.save_checkpoint()

    def validate(self, stash_params=False):
        """Validate the model."""

        self.logger.debug("Running model validation.")
        t = time.time()

        if stash_params:
            # Fetch model params from the GPU to release memory.
            self.state["params"] = fetch_from_first_device(self.state["params"])
            if "params_plm" in self.state:
                self.state["params_plm"] = fetch_from_first_device(
                    self.state["params_plm"]
                )

        max_batches = self.config.train.validate.max_batches

        # Set the policy for evaluation (likely f32 precision).
        policy = self._eval_model.get_policy()
        set_alphafold_policy(use_half=False)  # This line can be removed

        # Fetch the de-biased exponentially weighted average of the model parameters
        # and distribute them over the number of devices validation is configured for.
        # Note that by default, we pick the last devices as training automatically
        # selects the first.  This ensures that where possible, training and validati
        # are performed on different devies.
        params_ewa = self.get_params_ewa(
            debias=not self._init_from_params, as_numpy=False
        )
        params_ewa = policy.cast_to_param(params_ewa)
        params_ewa = self._eval_model.distribute_params(
            params_ewa, devices=self.devices
        )

        if "params_plm" in self.state:
            # Cast and distribute pLM parameters
            params_plm = policy.cast_to_param(self.state["params_plm"])
            params_plm = self._eval_model.distribute_params(
                params_plm, devices=self.devices
            )
        else:
            params_plm = None

        # Iterate over the validation dataset, tracking the metrics for aggregation.
        metrics = []

        t = time.time()
        for val_batch_id, batch in enumerate(self.validation_dataloader):
            if (max_batches > 0) and val_batch_id > max_batches:
                break

            self.logger.debug(f"Running validation batch {val_batch_id}.")
            results, batch_metrics = self._eval_model.predict(
                params_ewa, batch, params_plm=params_plm
            )
            batch_metrics = fetch_from_devices(batch_metrics, as_numpy=True)
            metrics.append(batch_metrics)
        val_time = time.time() - t

        # Aggregate the metrics across all batches.
        metrics = jax.tree_map(
            lambda *args: np.stack(args)
            if isinstance(args[0], np.ndarray)
            else np.array(args),
            *metrics,
        )

        loss = np.mean(metrics["loss"]).astype(np.float32)
        log_str = f"\n validation | loss : {loss:.3f}"
        self.logger.debug(str(Metric(name="val_loss", value=loss, step=self.step)))

        for k in metrics["output"].keys():
            if "loss" in metrics["output"][k]:
                loss = np.mean(metrics["output"][k]["loss"]).astype(np.float32)
                self.logger.debug(
                    str(Metric(name=f"val_{k}_loss", value=loss, step=self.step))
                )
                log_str += f", {k}_loss : {loss:.3f}"
        self.logger.debug(str(Metric(name="val_time", value=val_time, step=self.step)))
        log_str += f", validation time : {val_time:.3f}s.\n"
        self.logger.info(log_str)

        lddt_loss = np.mean(metrics["output"]["predicted_lddt"]["loss"]).astype(
            np.float32
        )
        # Update training statistics and possibly save the current parameters.
        if not self.best_lddt_loss or lddt_loss < self.best_lddt_loss:
            self.best_lddt_loss = lddt_loss
            # Don't save parameters if no training has occurred.
            if self.step > 0:
                self.logger.debug("New best performing model parameters found.")
                self.save_params()

        if stash_params:
            # Replace evaluation params with model params on the GPUs.
            del params_ewa
            self.state["params"] = distributed_over_devices(
                self.state["params"], self.devices
            )
            if "params_plm" in self.state:
                del params_plm
                self.state["params_plm"] = distributed_over_devices(
                    self.state["params_plm"], self.devices
                )

        # Restore training policy (probably mixed precision).
        set_alphafold_policy(self._train_model.use_half_precision)
        self.logger.debug("Validation complete.")
