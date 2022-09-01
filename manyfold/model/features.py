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

"""Code to generate processed features."""
import copy
from typing import List, Mapping, Optional, Tuple

import ml_collections
import numpy as np
import tensorflow.compat.v1 as tf

from manyfold.model.tf import input_pipeline, input_pipeline_plmfold, proteins_dataset

FeatureDict = Mapping[str, np.ndarray]


def make_data_config(
    config: ml_collections.ConfigDict,
    is_training: bool = False,
    num_res: Optional[int] = None,
) -> Tuple[ml_collections.ConfigDict, List[str]]:
    """Makes a data config for the input pipeline."""
    cfg = copy.deepcopy(config.data)

    feature_names = cfg.common.unsupervised_features
    if cfg.common.use_templates:
        feature_names += cfg.common.template_features
    if cfg.common.use_supervised:
        feature_names += cfg.common.supervised_features

    # Crop to num_res if crop_size not defined i.e less than 0
    if cfg.common.crop_size < 0 and num_res:
        with cfg.unlocked():
            cfg.common.crop_size = num_res

    if is_training:
        # Add training specific features.
        cfg.eval.subsample_templates = cfg.train.subsample_templates
        cfg.eval.unclamped_fape_fraction = cfg.train.unclamped_fape_fraction
    else:
        # Use previous default behaviour for cropping.
        cfg.eval.unclamped_fape_fraction = None

    return cfg, feature_names


def tf_example_to_features(
    tf_example: tf.train.Example,
    config: ml_collections.ConfigDict,
    random_seed: int = 0,
    is_training: bool = False,
    num_res: Optional[int] = None,
) -> FeatureDict:
    """Converts tf_example to numpy feature dictionary."""
    if not num_res:
        num_res = int(tf_example.features.feature["seq_length"].int64_list.value[0])
    cfg, feature_names = make_data_config(
        config, num_res=num_res, is_training=is_training
    )

    if "deletion_matrix_int" in set(tf_example.features.feature):
        deletion_matrix_int = tf_example.features.feature[
            "deletion_matrix_int"
        ].int64_list.value
        feat = tf.train.Feature(
            float_list=tf.train.FloatList(value=map(float, deletion_matrix_int))
        )
        tf_example.features.feature["deletion_matrix"].CopyFrom(feat)
        del tf_example.features.feature["deletion_matrix_int"]

    tf_graph = tf.Graph()
    with tf_graph.as_default(), tf.device("/device:CPU:0"):
        tf.compat.v1.set_random_seed(random_seed)
        tensor_dict = proteins_dataset.create_tensor_dict(
            raw_data=tf_example.SerializeToString(), features=feature_names
        )

        if not config.model.global_config.plmfold_mode:  # process MSA features
            process_batch_fn = input_pipeline.process_tensors_from_config
        else:
            process_batch_fn = input_pipeline_plmfold.process_tensors_from_config

        processed_batch = process_batch_fn(tensor_dict, cfg, is_training)

    tf_graph.finalize()

    with tf.Session(graph=tf_graph) as sess:
        features = sess.run(processed_batch)

    return {k: v for k, v in features.items() if v.dtype != "O"}


def np_example_to_features(
    np_example: FeatureDict,
    config: ml_collections.ConfigDict,
    random_seed: int = 0,
    is_training: bool = False,
    num_res: Optional[int] = None,
) -> FeatureDict:
    """Preprocesses NumPy feature dict using TF pipeline."""
    np_example = dict(np_example)
    if not num_res:
        num_res = int(np_example["seq_length"][0])
    cfg, feature_names = make_data_config(
        config, num_res=num_res, is_training=is_training
    )

    if "deletion_matrix_int" in np_example:
        np_example["deletion_matrix"] = np_example.pop("deletion_matrix_int").astype(
            np.float32
        )

    tf_graph = tf.Graph()
    with tf_graph.as_default(), tf.device("/device:CPU:0"):
        tf.compat.v1.set_random_seed(random_seed)
        tensor_dict = proteins_dataset.np_to_tensor_dict(
            np_example=np_example, features=feature_names
        )

        if not config.model.global_config.plmfold_mode:  # process MSA features
            process_batch_fn = input_pipeline.process_tensors_from_config
        else:
            process_batch_fn = input_pipeline_plmfold.process_tensors_from_config

        processed_batch = process_batch_fn(tensor_dict, cfg, is_training)

    tf_graph.finalize()

    with tf.Session(graph=tf_graph) as sess:
        features = sess.run(processed_batch)

    return {k: v for k, v in features.items() if v.dtype != "O"}
