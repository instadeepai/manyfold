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

"""Feature pre-processing input pipeline for pLMFold."""
from typing import Optional

import tensorflow.compat.v1 as tf
import tree

from manyfold.model.tf import data_transforms, shape_placeholders

# Pylint gets confused by the curry1 decorator because it changes the number
#   of arguments to the function.
# pylint:disable=no-value-for-parameter


NUM_RES = shape_placeholders.NUM_RES
NUM_TEMPLATES = shape_placeholders.NUM_TEMPLATES


def nonensembled_map_fns(
    data_config,
    is_training: Optional[bool] = False,
    is_validation_pipeline: Optional[bool] = False,
):
    """Input pipeline functions which are not ensembled."""
    common_cfg = data_config.common

    map_fns = [
        data_transforms.cast_64bit_ints,
        data_transforms.squeeze_features,
        # Keep to not disrupt RNG.
        data_transforms.make_seq_mask,
        data_transforms.make_random_crop_to_size_seed,
        data_transforms.add_plm_features,
    ]
    if is_training or is_validation_pipeline:
        # Added for training and computing the loss during validation
        map_fns.extend(
            [
                data_transforms.make_all_atom_aatype,
                data_transforms.make_pseudo_beta(""),
            ]
        )
    if common_cfg.use_templates:
        map_fns.extend(
            [
                data_transforms.fix_templates_aatype,
                data_transforms.make_template_mask,
                data_transforms.make_pseudo_beta("template_"),
            ]
        )
    map_fns.extend(
        [
            data_transforms.make_atom14_masks,
        ]
    )

    return map_fns


def ensembled_map_fns(data_config, is_training: Optional[bool] = False):
    """Input pipeline functions that can be ensembled and averaged."""
    common_cfg = data_config.common
    eval_cfg = data_config.eval

    map_fns = []

    crop_feats = dict(eval_cfg.feat)

    if eval_cfg.fixed_size:
        map_fns.append(data_transforms.select_feat(list(crop_feats)))

        if eval_cfg.force_uniform_crop_plm:
            unclamped_fape_fraction_plm = None
        else:
            unclamped_fape_fraction_plm = eval_cfg.unclamped_fape_fraction

        map_fns.append(
            data_transforms.random_crop_to_size(
                common_cfg.crop_size_plm,
                eval_cfg.max_templates,
                crop_feats,
                eval_cfg.subsample_templates,
                unclamped_fape_fraction_plm,
                crop_plm=True,
            )
        )

        map_fns.append(
            data_transforms.random_crop_to_size(
                common_cfg.crop_size,
                eval_cfg.max_templates,
                crop_feats,
                eval_cfg.subsample_templates,
                eval_cfg.unclamped_fape_fraction,
            )
        )
        map_fns.append(
            data_transforms.make_fixed_size(
                crop_feats,
                0,  # pad_msa_clusters,
                0,  # common_cfg.max_extra_msa,
                common_cfg.crop_size,
                eval_cfg.max_templates,
                common_cfg.crop_size_plm,
            )
        )
    else:
        map_fns.append(data_transforms.crop_templates(eval_cfg.max_templates))

    return map_fns


def process_tensors_from_config(
    tensors,
    data_config,
    is_training: Optional[bool] = False,
    is_validation_pipeline: Optional[bool] = False,
):
    """Apply filters and maps to an existing dataset, based on the config."""

    def wrap_ensemble_fn(data, i):
        """Function to be mapped over the ensemble dimension."""
        d = data.copy()
        fns = ensembled_map_fns(data_config, is_training)
        fn = compose(fns)
        d["ensemble_index"] = i
        return fn(d)

    eval_cfg = data_config.eval
    tensors = compose(
        nonensembled_map_fns(data_config, is_training, is_validation_pipeline)
    )(tensors)

    tensors_0 = wrap_ensemble_fn(tensors, tf.constant(0))
    num_ensemble = eval_cfg.num_ensemble

    if isinstance(num_ensemble, tf.Tensor) or num_ensemble > 1:
        fn_output_signature = tree.map_structure(tf.TensorSpec.from_tensor, tensors_0)
        tensors = tf.map_fn(
            lambda x: wrap_ensemble_fn(tensors, x),
            tf.range(num_ensemble),
            parallel_iterations=1,
            fn_output_signature=fn_output_signature,
        )
    else:
        tensors = tree.map_structure(lambda x: x[None], tensors_0)
    return tensors


@data_transforms.curry1
def compose(x, fs):
    for f in fs:
        x = f(x)
    return x
