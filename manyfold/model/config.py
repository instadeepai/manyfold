# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model config."""

import os
from typing import Any, Union

import ml_collections
from ml_collections import ConfigDict, config_dict
from omegaconf import DictConfig, ListConfig, OmegaConf

from manyfold.model.tf import shape_placeholders
from manyfold.train.trainer import InvalidParametersAction

NUM_RES = shape_placeholders.NUM_RES
NUM_MSA_SEQ = shape_placeholders.NUM_MSA_SEQ
NUM_EXTRA_SEQ = shape_placeholders.NUM_EXTRA_SEQ
NUM_TEMPLATES = shape_placeholders.NUM_TEMPLATES
SINGLE = shape_placeholders.SINGLE


def convert_to_ml_dict(dct: Union[DictConfig, Any]) -> Union[ConfigDict, Any]:
    """
    This function converts the DictConfig returned by Hydra
    into a ConfigDict. The recusion allows to convert
    all the nested DictConfig elements of the config. The recursion stops
    once the reached element is not a DictConfig.
    """
    if not type(dct) is DictConfig:
        if type(dct) is ListConfig:
            return list(dct)
        return dct
    dct_ml = config_dict.ConfigDict()
    for k in list(dct.keys()):
        dct_ml[k] = convert_to_ml_dict(dct[k])
    return dct_ml


def model_config(name: str) -> ml_collections.ConfigDict:
    """Get the ConfigDict of a CASP14 model."""

    name = name.replace("model", "config_deepmind_casp14_monomer") + ".yaml"
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config/model_config"
    )
    default_cfg = OmegaConf.load(os.path.join(path, "config.yaml"))
    if name not in os.listdir(path):
        raise ValueError(f"Invalid model name {name}.")
    cfg = OmegaConf.load(os.path.join(path, name))
    merge_cfg = OmegaConf.merge(default_cfg, cfg)
    model_config = convert_to_ml_dict(merge_cfg)
    if model_config.train.invalid_paramaters_action is None:
        model_config.train.invalid_paramaters_action = InvalidParametersAction.ERROR

    return model_config


MODEL_PRESETS = {
    "monomer": (
        "model_1",
        "model_2",
        "model_3",
        "model_4",
        "model_5",
    ),
    "monomer_ptm": (
        "model_1_ptm",
        "model_2_ptm",
        "model_3_ptm",
        "model_4_ptm",
        "model_5_ptm",
    ),
}
MODEL_PRESETS["monomer_casp14"] = MODEL_PRESETS["monomer"]


#  AF-Monomer supplementary CASP config, table 5. And section 1.9
INITIAL_HEAD_WEIGHTS = {
    "distogram": 0.3,
    "experimentally_resolved": 0.0,
    "masked_msa": 2.0,
    "predicted_aligned_error": 0.1,
    "predicted_lddt": 0.01,
    "structure_module": {"weight": 1.0, "structural_violation_loss_weight": 0.0},
}


FINETUNE_HEAD_WEIGHTS = {
    "distogram": 0.3,
    "experimentally_resolved": 0.01,
    "masked_msa": 2.0,
    "predicted_aligned_error": 0.1,
    "predicted_lddt": 0.01,
    "structure_module": {"weight": 1.0, "structural_violation_loss_weight": 1.0},
}
