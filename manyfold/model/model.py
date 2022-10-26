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

"""Code for constructing the model."""
import functools
import itertools

# import os
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import ml_collections
import numpy as np
import optax

from manyfold.common import confidence
from manyfold.model import common_modules, features, modules, modules_plmfold


def get_confidence_metrics(prediction_result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Post processes prediction_result to get confidence metrics."""
    confidence_metrics = {}
    confidence_metrics["plddt"] = confidence.compute_plddt(
        prediction_result["predicted_lddt"]["logits"]
    )
    if "predicted_aligned_error" in prediction_result:
        confidence_metrics.update(
            confidence.compute_predicted_aligned_error(
                logits=prediction_result["predicted_aligned_error"]["logits"],
                breaks=prediction_result["predicted_aligned_error"]["breaks"],
            )
        )
        confidence_metrics["ptm"] = confidence.predicted_tm_score(
            logits=prediction_result["predicted_aligned_error"]["logits"],
            breaks=prediction_result["predicted_aligned_error"]["breaks"],
            asym_id=None,
        )

    # Monomer models use mean pLDDT for model ranking.
    confidence_metrics["ranking_confidence"] = np.mean(confidence_metrics["plddt"])

    return confidence_metrics


def fetch_from_devices(x, as_numpy: bool = True):
    """Converts a distributed TrainingState to a single-device TrainingState."""

    def fetch_fn(x):
        if as_numpy and isinstance(x, jax.xla.DeviceArray):
            x = np.asarray(x)
        return x

    return jax.tree_util.tree_map(fetch_fn, x)


def fetch_from_first_device(x, as_numpy: bool = True):
    """Converts a distributed TrainingState to a single-device TrainingState."""

    def fetch_fn(x):
        x = x[0]
        if as_numpy and isinstance(x, jax.xla.DeviceArray):
            x = np.asarray(x)
        return x

    return jax.tree_util.tree_map(fetch_fn, x)


def distributed_over_devices(
    x,
    devices: Optional[Sequence[jax.xla.Device]] = None,
    as_sharded_array: bool = True,
):
    """Converts a single-device TrainingState to a distributed TrainingState."""
    devices = devices or jax.local_devices()

    def distribute_fn(x):
        x = [x] * len(devices)
        if as_sharded_array:
            x = jax.device_put_sharded(x, devices)
        return x

    return jax.tree_util.tree_map(distribute_fn, x)


TrainingState = Mapping[str, Union[float, np.ndarray, jax.xla.DeviceArray]]
TrainingMetrics = Mapping[str, Union[float, np.ndarray, jax.xla.DeviceArray]]


def get_policy(use_half=True, is_batch_norm=False):
    # hk.mixed_precision.set_policy(mod, policy)
    # Use mixed precision (only support A100 GPU and TPU for now)
    half = jnp.bfloat16
    full = jnp.float32
    if use_half:
        # Use mixed precision (only support A100 GPU and TPU for now)
        if is_batch_norm:
            # Compute batch norm in full precision to avoid instabilities.
            policy = jmp.Policy(compute_dtype=full, param_dtype=full, output_dtype=half)
        else:
            policy = jmp.Policy(compute_dtype=half, param_dtype=full, output_dtype=half)
    else:
        policy = jmp.Policy(compute_dtype=full, param_dtype=full, output_dtype=full)
    return policy


def set_alphafold_policy(use_half=True):
    # If we apply the policy directly to the top-level modules (e.g. modules.AlphaFold
    # or modules.AlphaFoldIteration), we can run into mis-matched dtypes when passing
    # recylced values around the model. Instead, we apply the policy independently to
    # the constituant modules as shown below.  This is not ideal, so to mitigate the
    # risk, we have separately added a manual check that all model parameters have the
    # same dtype in the init function.
    for mod in [
        common_modules.Linear,
        modules.AlphaFoldIteration,
        modules.AlphaFold,
        modules_plmfold.PLMFoldIteration,
        modules_plmfold.PLMFold,
    ]:
        policy = get_policy(use_half)
        hk.mixed_precision.set_policy(mod, policy)
        if use_half:
            # Use mixed precision (only support A100 GPU and TPU for now)
            policy_norm = get_policy(use_half=True, is_batch_norm=True)
            hk.mixed_precision.set_policy(hk.BatchNorm, policy_norm)
            hk.mixed_precision.set_policy(hk.LayerNorm, policy_norm)


class TrainModel:
    """Container for training a JAX model.

    The class is designed to be used for distributed training and prepares and exposes
    two main functions.

        - TrainModel.init: Initialise the model parameters across all devices, and
           the optimizer state.
        - TrainModel.update_step: Updates the state with a distributed gradient step.

    As a general principle, TrainModel is intended to be stateless, other than accepting
    a configuration file on initialisation to specify the model and training parameters.
    However, whilst the module is intended to be wrapped in surrounding training code, a
    few convenience functions for calling the underlying model are also exposed.

        - TrainModel.net_init: Init function for the network forward pass (not-batched).
        - TrainModel.net_apply_batched: single-device batched (i.e. vmap) call to the
            AlphaFold model.

    Finally, static methods for handling the TrainingState dictionary are provided:

        - TrainModel.is_state_synchronised: Validates that the TrainingState is the same
            on all devices.
        - TrainModel.to_single_state: Converts a distributed TrainingState to a
            single-device TrainingState.
        - TrainModel.to_distributed_state: Converts a single-device TrainingState to a
            distributed TrainingState.
    """

    def __init__(
        self,
        config: ml_collections.ConfigDict,
        devices: Optional[Sequence[jax.xla.Device]] = None,
    ):
        """
        Args:
          devices: The devices given as argument to pmap, the data
          will be sharded by pmap on these devices.
        """
        self.config = config
        self.plmfold_mode = config.model.global_config.plmfold_mode

        if not devices:
            devices = jax.devices()
        self.devices = devices
        self.num_devices = len(self.devices)
        self.batch_size_per_device = config.data.common.batch_size

        # Overwrite config flags where essential.
        self.config.data.common.use_supervised = True

        # Set up mixed-precision training.
        self.use_half_precision = self.config.train.mixed_precision.use_half
        self.get_policy = lambda: get_policy(self.use_half_precision)

        def get_init_loss_scale():
            cls = getattr(
                jmp, f"{self.config.train.mixed_precision.scale_type}LossScale"
            )
            return (
                cls(float(self.config.train.mixed_precision.scale_value))
                if cls is not jmp.NoOpLossScale
                else cls()
            )

        if self.plmfold_mode:

            def _preprocess_fn(feat_dict):
                model = modules_plmfold.PLMEmbed(self.config.language_model)
                return model(feat_dict)

            def _forward_fn(feat_dict):
                model = modules_plmfold.PLMFold(self.config.model)
                return model(
                    feat_dict,
                    is_training=True,
                    compute_loss=True,
                    ensemble_representations=False,
                    return_representations=False,
                )

        else:

            def _forward_fn(feat_dict):
                """
                Notes on the settings.

                ensemble_representations = False for training - while loop in line202 of
                modules not compatible with backwards diff.
                """
                model = modules.AlphaFold(self.config.model)
                return model(
                    feat_dict,
                    is_training=True,
                    compute_loss=True,
                    ensemble_representations=False,
                    return_representations=False,
                )

        if self.plmfold_mode:
            _preprocess_fn = hk.transform(_preprocess_fn)
            plm_model_init = _preprocess_fn.init
            plm_model_batched = jax.vmap(
                _preprocess_fn.apply, axis_name="j", in_axes=(None, 0, 0)
            )

        _forward_fn = hk.transform(_forward_fn)
        fold_model_init = _forward_fn.init
        fold_model_batched = jax.vmap(
            _forward_fn.apply, axis_name="j", in_axes=(None, 0, 0)
        )

        # Set up the loss function
        def loss_and_output(
            params,
            rng,
            loss_scale: jmp.LossScale,
            feats: features.FeatureDict,
            params_plm: Optional[hk.Params] = None,
        ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, Dict[str, float]]]:
            """Take a batched forward pass of the model and return loss and outputs.

            By default, AlphaFold returns the loss as the second element of the tuple
            when called with compute_loss = True.  To be compatible with
            jax.value_and_grad, the loss needs to be the first element and a single
            scalar value.  This function ensures these standards, by averaging the
            per-element losses for a single batched loss.

            Args:
              state: Current training state.
              features: Processed and batched FeatureDict.
            """
            if self.plmfold_mode:
                rng, rng_plm = jax.vmap(jax.random.split, out_axes=1)(rng)
                feats = plm_model_batched(params_plm, rng_plm, feats)

            output, [loss_per_chain] = fold_model_batched(
                params,
                rng,
                feats,
            )
            # SM 1.9: scale per chain losses by size after cropping.
            loss_per_chain = loss_per_chain * jnp.sqrt(feats["seq_length"][..., 0])
            loss = jnp.mean(loss_per_chain)
            # loss_scale is just for the mixed precision policy
            return loss_scale.scale(loss), (loss, output)

        self.loss_and_output = loss_and_output

        # Set up the optimizer (as per SM 1.11.3)
        def get_single_sample_gradient_transformation():
            config = self.config.train.optimizer
            optimizer = optax.clip_by_global_norm(config.clip_global_grad_norm)
            return optimizer

        self.get_single_sample_gradient_transformation = (
            get_single_sample_gradient_transformation
        )

        def get_optimizer():
            config = self.config.train.optimizer
            lr_schedule = optax.join_schedules(
                schedules=[
                    optax.linear_schedule(
                        init_value=0.0,
                        end_value=config.lr,
                        transition_steps=config.warm_up_n_steps,
                    ),
                    optax.constant_schedule(value=config.lr_decay * config.lr),
                ],
                boundaries=[config.lr_decay_after_n_steps],
            )

            optimizer = optax.chain(
                optax.scale_by_adam(b1=config.b1, b2=config.b2),
                optax.scale_by_schedule(lr_schedule),
                # optax.clip_by_global_norm(config.clip_global_grad_norm),
                # Scale updates by -1 since optax.apply_updates is *additive*
                # and we want to descend on the loss.
                optax.scale(-1.0),
                # optax.apply_every(config.num_grad_acc_steps),
            )
            optimizer = optax.MultiSteps(
                optimizer,
                every_k_schedule=config.num_grad_acc_steps,
                use_grad_mean=True,
            )
            if self.config.train.skip_nonfinite_updates:
                optimizer = optax.apply_if_finite(optimizer, max_consecutive_errors=8)
            return optimizer

        self.get_optimizer = get_optimizer

        def prepare_init_state(params, rng, params_plm=None) -> TrainingState:
            """Prepares the model state, given parameters and the rng key."""
            policy = get_policy()
            params = jax.tree_map(
                lambda x: policy.cast_to_param(jnp.asarray(x)), params
            )

            assert (
                len(
                    list(
                        itertools.groupby(
                            [p.dtype for p in jax.tree_util.tree_flatten(params)[0]]
                        )
                    )
                )
                == 1
            ), (
                "All parameters do not have the same dtype.  You may want to double"
                + "check the application of the mixed precision policy!"
            )
            optimizer_state = self.get_optimizer().init(params)
            loss_scale = get_init_loss_scale()

            state = {
                "rng": rng,
                "params": params,
                "optimizer_state": optimizer_state,
                "loss_scale": loss_scale,
            }
            if params_plm:
                state["params_plm"] = params_plm

            return state

        def _init(
            feat: features.FeatureDict, random_seed, params_plm: hk.Params = None
        ) -> TrainingState:
            """Initialise the model parameters and optimizer state."""
            rng = jax.random.PRNGKey(random_seed)

            if self.plmfold_mode:
                print("Running in plm_mode")
                if params_plm:
                    if self.config.language_model.return_all_embeddings:
                        num_layers = self.config.language_model.model.num_layers
                    else:
                        num_layers = 1
                    num_res = feat["aatype_plm"].shape[-1]
                    plm_dim = self.config.language_model.model.embed_dim

                    feat["embeddings"] = jnp.zeros(
                        (1, num_layers + 1, num_res, plm_dim)
                    )
                    feat["plm_attn_weights"] = jnp.zeros((1, num_res, num_res))

                    params_plm = {f"plmembed/{k}": v for k, v in params_plm.items()}
                    print(
                        f"\trenamed params_plm and created dummy embeddings on {feat['embeddings'].device()}"
                    )
                else:
                    print("Randomly initialising LM", end="...")
                    rng, init_rng = jax.random.split(rng)
                    params_plm = plm_model_init(init_rng, feat)
                    print("done.")

                params_plm = self.get_policy().cast_to_param(params_plm)
                params_plm = hk.data_structures.to_mutable_dict(params_plm)

            print("randomly initialising folding model", end="...")
            rng, init_rng = jax.random.split(rng)
            params = fold_model_init(init_rng, feat)
            params = hk.data_structures.to_mutable_dict(params)
            params = self.get_policy().cast_to_param(params)
            print("done.")

            # Broadcast the parameters to every *local device* (though this is
            # run on every process if using a TPU pod slice).
            print("preparing state on all local devices", end="...")

            from manyfold.train.trainer import check_params

            check_params(params)
            check_params(params_plm)

            state = prepare_init_state(params, rng, params_plm)
            del params, params_plm, rng
            state = jax.device_put_replicated(state, jax.local_devices())
            print("done.")

            # Synchronise the parameters across *all global* devices.
            # (Note that devices=jax.local_devices() would result in different
            # parameters on each slice of a TPU pod.)
            if jax.device_count() > jax.local_device_count():
                print("Synchronise the parameters across all global devices", end="...")
                state["params"] = jax.tree_map(
                    jax.pmap(lambda x: jax.lax.pmean(x, "i"), axis_name="i"),
                    state["params"],
                )
                print("done.")

            return state

        self._init = _init

        @functools.partial(
            jax.pmap, axis_name="i", in_axes=(0, None, None), devices=self.devices
        )
        def _init_from_params(
            params: hk.Params, random_seed, params_plm: hk.Params
        ) -> TrainingState:
            """Initialise the model parameters and optimizer state."""
            rng = jax.random.PRNGKey(random_seed)

            params = self.get_policy().cast_to_param(params)
            params = hk.data_structures.to_mutable_dict(params)
            if params_plm:
                params_plm = {f"plmembed/{k}": v for k, v in params_plm.items()}
                print("renamed params_plm")
                params_plm = self.get_policy().cast_to_param(params_plm)
                params_plm = hk.data_structures.to_mutable_dict(params_plm)

            return prepare_init_state(params, rng, params_plm)

        self._init_from_params = _init_from_params

        @functools.partial(
            jax.pmap,
            axis_name="i",
            donate_argnums=(0,),
            devices=self.devices,
        )
        def _update_step(
            state,
            feats: features.FeatureDict,
        ) -> (TrainingState, TrainingMetrics):
            """Updates the state using some a batch of features."""
            master_rng = state["rng"]
            params = state["params"]
            optimizer_state = state["optimizer_state"]
            loss_scale = state["loss_scale"]
            if "params_plm" in state:
                params_plm = state["params_plm"]
            else:
                params_plm = None

            rng, master_rng = jax.random.split(master_rng)
            rngs_batch = jax.random.split(rng, self.batch_size_per_device)

            grads, (loss, output) = jax.grad(loss_and_output, has_aux=True,)(
                params,
                rngs_batch,
                loss_scale,
                feats,
                params_plm,
            )

            # Grads are in "param_dtype" (likely F32) here. We cast them back to the
            # compute dtype such that we do the all-reduce below in the compute
            # precision (which is typically lower than the param precision).
            policy = get_policy()
            grads = policy.cast_to_compute(grads)
            grads = loss_scale.unscale(grads)

            # Apply and per-sample gradient transformations.
            grads, _ = self.get_single_sample_gradient_transformation().update(
                grads, optax.EmptyState()
            )

            # # Taking the mean across all replicas to keep params in sync.
            grads = jax.lax.pmean(grads, axis_name="i")

            # We compute our optimizer update in the same precision as params, even when
            # doing mixed precision training.
            grads = policy.cast_to_param(grads)

            # Calculate the parameter updates.
            updates, optimizer_state = get_optimizer().update(grads, optimizer_state)

            # Apply parameter updates.
            params = optax.apply_updates(params, updates)

            loss_outputs = {
                output_key: {
                    "loss": jax.lax.pmean(
                        jnp.mean(output[output_key]["loss"]), axis_name="i"
                    ),
                }
                for output_key in output.keys()
                if "loss" in output[output_key]
            }

            state = {
                "rng": master_rng,
                "params": params,
                "optimizer_state": optimizer_state,
                "loss_scale": loss_scale,
            }
            if self.plmfold_mode:
                state["params_plm"] = params_plm

            metrics = {
                "loss": jax.lax.pmean(jnp.mean(loss), axis_name="i"),
                "output": loss_outputs,
            }

            return state, metrics

        self._update_step = _update_step

    def init(
        self,
        feat: features.FeatureDict,
        random_seed: int,
        params_plm: hk.Params,
    ) -> TrainingState:
        """Initialise the model parameters and optimizer state."""
        return self._init(feat, random_seed, params_plm)

    def init_from_params(
        self, params, random_seed: int, params_plm: hk.Params
    ) -> TrainingState:
        """Initialise the model parameters and optimizer state."""
        return self._init_from_params(params, random_seed, params_plm)

    def update_step(
        self, state: TrainingState, feats: features.FeatureDict
    ) -> (TrainingState, TrainingMetrics):
        """Updates the state using some a batch of features."""
        return self._update_step(state, feats)

    @staticmethod
    def is_state_synchronised(
        state: TrainingState, ignore_keys: Optional[List[str]] = "rng"
    ) -> bool:
        """Validates that the TrainingState is the same on all local devices."""
        state_synchronised = True
        for k in state.keys():
            if k not in ignore_keys:
                leaf_list, _ = jax.tree_util.tree_flatten(state[k])
                if not all([(x[0] == x).all() for x in leaf_list]):
                    state_synchronised = False
        return state_synchronised

    @staticmethod
    def to_single_state(state: TrainingState, as_numpy: bool = True) -> TrainingState:
        """Converts a distributed TrainingState to a single-device TrainingState."""
        return fetch_from_first_device(state, as_numpy)

    @staticmethod
    def to_distributed_state(
        state: TrainingState,
        devices: Optional[Sequence[jax.xla.Device]] = None,
        as_sharded_array: bool = True,
    ) -> TrainingState:
        """Converts a single-device TrainingState to a distributed TrainingState."""
        return distributed_over_devices(state, devices, as_sharded_array)


class EvalModel:
    """Evaluate model parameters.

    Losses are calculated to allow for the performance to be easily evaluated.
    However, unlike TrainModel, the network is run in inference mode, and without the
    additional wrappers to calculated gradients, take update steps etc.
    """

    def __init__(
        self,
        config: ml_collections.ConfigDict,
        devices: Optional[Sequence[jax.xla.Device]] = None,
    ):
        """
        Args:
          devices: The devices given as argument to pmap, the data
          will be sharded by pmap on these devices.
        """
        self.config = config
        self.plmfold_mode = self.config.model.global_config.plmfold_mode

        if not devices:
            devices = jax.devices()
        self.devices = devices

        # Set up the inference functions.
        if self.plmfold_mode:

            def _preprocess_fn(feat_dict):
                model = modules_plmfold.PLMEmbed(self.config.language_model)
                return model(feat_dict)

            def _forward_fn(batch):
                model = modules_plmfold.PLMFold(self.config.model)
                return model(
                    batch,
                    is_training=False,
                    compute_loss=True,
                    ensemble_representations=False,
                    return_representations=False,
                )

        else:

            def _forward_fn(batch):
                model = modules.AlphaFold(self.config.model)
                return model(
                    batch,
                    is_training=False,
                    compute_loss=True,
                    ensemble_representations=False,
                    return_representations=False,
                )

        # Batched forward pass.
        net_apply = hk.transform(_forward_fn).apply
        net_apply_batched = jax.vmap(net_apply, in_axes=(None, None, 0))
        self.net_apply_batched = jax.jit(net_apply_batched)

        if self.plmfold_mode:
            plm_apply = hk.transform(_preprocess_fn).apply
            plm_apply_batched = jax.vmap(plm_apply, in_axes=(None, None, 0))
            self.plm_apply_batched = jax.jit(plm_apply_batched)

        @functools.partial(
            jax.pmap, axis_name="i", in_axes=(0, 0, None, 0), devices=self.devices
        )
        def _predict(
            params: hk.Params,
            feat: features.FeatureDict,
            random_seed: int,
            params_plm: Optional[hk.Params] = None,
        ) -> Mapping[str, Any]:
            rng = jax.random.PRNGKey(random_seed)

            if self.plmfold_mode:
                feat = plm_apply_batched(params_plm, rng, feat)

            results, loss = net_apply_batched(params, rng, feat)

            loss_outputs = {}
            for k in results.keys():
                if "loss" in results[k]:
                    loss_outputs[k] = {
                        "loss": jax.lax.pmean(
                            jnp.mean(results[k]["loss"]), axis_name="i"
                        )
                    }

            metrics = {
                "loss": loss,
                "output": loss_outputs,
            }

            return results, metrics

        self._predict = _predict

    @staticmethod
    def distribute_params(
        params: hk.Params,
        devices: Optional[Sequence[jax.xla.Device]] = None,
        as_sharded_array: bool = True,
    ):
        return distributed_over_devices(params, devices, as_sharded_array)

    def get_policy(self):
        return jmp.get_policy("f32")

    def predict(
        self,
        params: hk.Params,
        feat: features.FeatureDict,
        random_seed: Optional[int] = None,
        params_plm: Optional[hk.Params] = None,
    ) -> Mapping[str, Any]:
        """Makes a prediction on the provided features using the provided params."""
        if random_seed is None:
            random_seed = random.randint(0, 99999)
        return self._predict(params, feat, random_seed, params_plm)
