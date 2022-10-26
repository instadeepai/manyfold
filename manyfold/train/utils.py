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

"""Convenience functions for reading data."""

import io
import os

import haiku as hk
import numpy as np

import manyfold.train.gcp_utils as gcp_utils
from manyfold.data.tools.utils import tmpdir_manager
from manyfold.model import utils


def get_model_haiku_params(model_name: str, data_dir: str) -> hk.Params:
    """Get the Haiku parameters from a model name."""

    path = os.path.join(data_dir, f"params_{model_name}.npz")

    with open(path, "rb") as f:
        params = np.load(io.BytesIO(f.read()), allow_pickle=False)

    return utils.flat_params_to_haiku(params)


def get_model_haiku_params_maybe_gcp(model_name, data_dir):
    if gcp_utils.is_gcp_path(data_dir):
        with tmpdir_manager(base_dir="/tmp") as tmp_dir:
            # Download model locally into a temporary directory
            model_params_filename = f"params_{model_name}.npz"
            (success, error_msg) = gcp_utils.GCPBucketFilepath.from_gcp_filepath(
                os.path.join(data_dir, model_params_filename)
            ).download_to_file(os.path.join(tmp_dir, model_params_filename))
            if not success:
                raise RuntimeError(error_msg)
            model_params = get_model_haiku_params(
                model_name=model_name,
                data_dir=tmp_dir,
            )
    else:
        model_params = get_model_haiku_params(
            model_name=model_name,
            data_dir=data_dir,
        )
    return model_params
