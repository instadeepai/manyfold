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

from typing import Any, Dict, List, Optional

import jax
import ml_collections
import numpy as np

from manyfold.model.model import EvalModel, fetch_from_devices, set_alphafold_policy
from manyfold.train.dataloader_tf import TFDataloaderParams
from manyfold.train.utils import get_model_haiku_params_maybe_gcp


def reshape_feat_dict(feat_dict: Dict[str, Any]) -> Dict[str, Any]:
    # Flatten first two dimensions (num-devices, batch-size).
    return jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), feat_dict)


class ValidModel:
    def __init__(self, model_name: str, model_config: ml_collections.ConfigDict):
        self.model_name = model_name
        self.model_config = model_config
        self.plmfold_mode = model_config.model.global_config.plmfold_mode
        self.params_plm = None
        self.params_plm_distrib = None

    def prepare_config(
        self,
        crop_size: int,
        batch_size: int,
        num_devices: int,
    ):
        self.model_config.data.common.crop_size = crop_size
        if self.plmfold_mode:
            self.model_config.data.common.crop_size_plm = crop_size
            self.model_config.data.eval.force_uniform_crop_plm = False

        self.model_config.data.common.use_supervised = True
        self.model_config.train.validate.batch_size = batch_size
        self.num_devices = num_devices

    def prepare_dataloader(self, filenames: List[str]) -> TFDataloaderParams:
        return TFDataloaderParams(
            filepaths=filenames,
            config=self.model_config,
            batch_dims=(self.num_devices, self.model_config.train.validate.batch_size),
            apply_filters=False,
            max_num_epochs=1,
            shuffle=False,
            is_training=False,
            is_validation_pipeline=True,
            deterministic=True,
            drop_remainder=False,
            process_msa_features=not self.plmfold_mode,
        )

    def load_params(self, params_dir: str):
        self.model_params = get_model_haiku_params_maybe_gcp(
            model_name=self.model_name,
            data_dir=params_dir,
        )

    def load_params_plm(self):
        plm_config = self.model_config.language_model
        self.params_plm = get_model_haiku_params_maybe_gcp(
            model_name=plm_config.model_name,
            data_dir=plm_config.pretrained_model_dir,
        )
        # Rename parameters for PLM
        self.params_plm = {f"plmembed/{k}": v for k, v in self.params_plm.items()}

    def set_up_devices(self):
        devices = jax.local_devices()[: self.num_devices]
        self.eval_model = EvalModel(self.model_config, devices=devices)
        # Set the policy for evaluation (likely f32 precision).
        policy = self.eval_model.get_policy()
        set_alphafold_policy(use_half=False)  # This line can be removed
        self.params_distrib = policy.cast_to_param(self.model_params)
        # Distribute the model parameters over the number of devices.
        self.params_distrib = self.eval_model.distribute_params(
            self.params_distrib, devices=devices
        )
        if self.params_plm:
            self.params_plm_distrib = policy.cast_to_param(self.params_plm)
            self.params_plm_distrib = self.eval_model.distribute_params(
                self.params_plm_distrib, devices=devices
            )

    def inference(
        self,
        batch: Dict[str, np.ndarray],
        seed_value: Optional[int] = None,
    ) -> Dict[str, Any]:
        # Get predictions for batch.
        results, _ = self.eval_model.predict(
            self.params_distrib, batch, seed_value, self.params_plm_distrib
        )
        results = fetch_from_devices(results, as_numpy=True)
        return results
