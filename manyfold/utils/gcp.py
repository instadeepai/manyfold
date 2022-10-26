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

"""Code to read and write from a gcloud storage bucket."""
import os
import re
import tempfile
from typing import Dict, List, Tuple, Union

import numpy as np
from google.cloud import storage

BucketPath = str
BucketName = str
BucketNamePath = str
NumpyDict = Dict[str, np.ndarray]


def extract_bucket_and_path(gpath: BucketNamePath) -> Tuple[BucketPath, BucketName]:
    return re.match(r"gs://([^/]*)/(.+)", gpath).groups()


def get_bucket_and_path(
    gcp_bucket_name: BucketName, gcp_path: Union[BucketNamePath, BucketPath]
) -> Tuple[storage.Bucket, BucketPath]:
    if gcp_bucket_name == "":
        gcp_bucket_name, gcp_path = extract_bucket_and_path(gcp_path)
    storage_client = storage.Client()
    bucket = storage_client.bucket(gcp_bucket_name)
    return bucket, gcp_path


def upload(
    string_or_localpath: str,
    gcp_path: BucketPath,
    from_string: bool,
    gcp_bucket_name: BucketName = "",
    byte_format: str = "r",
) -> None:
    """Upload a binary/text file to the bucket, or directly from a string.

    If `from_string` is True, then `string_or_localpath` must be a str variable which
    will become the contents of the text file with name `gcp_path`. If `from_string` is
    False, then `string_or_localpath` must be a str variable indicating the local path
    of the binary/text file to upload.
    """
    bucket, gcp_path = get_bucket_and_path(gcp_bucket_name, gcp_path)
    blob = bucket.blob(gcp_path)
    if from_string:
        blob.upload_from_string(string_or_localpath)
    else:
        with open(string_or_localpath, byte_format) as local_file:
            blob.upload_from_file(local_file)


def download(
    gcp_path: BucketPath, gcp_bucket_name: BucketName = "", local_path: str = ""
) -> Union[None, str]:
    """Download blob, if `local_path` is provided save as file, else load as string."""
    bucket, gcp_path = get_bucket_and_path(gcp_bucket_name, gcp_path)
    blob = bucket.blob(gcp_path)
    if local_path != "":
        blob.download_to_filename(local_path)
    else:
        file_content = blob.download_as_string()
        return file_content.decode("utf-8")


def list_bucket(
    gcp_prefix: BucketPath, gcp_bucket_name: BucketName = ""
) -> List[BucketPath]:
    """Return all of the blobs on the bucket which start with the string `gcp_prefix`"""
    bucket, gcp_prefix = get_bucket_and_path(gcp_bucket_name, gcp_prefix)
    blobs = list(bucket.list_blobs(prefix=None if gcp_prefix == "" else gcp_prefix))
    return [b.name for b in blobs]


def is_blob(gcp_path: BucketPath, gcp_bucket_name: BucketName = "") -> bool:
    """Return True if the gcp_path is a blob on the bucket, else False."""
    return len(list_bucket(gcp_path, gcp_bucket_name)) == 1


def upload_numpy(
    gcp_path: BucketPath, np_obj: NumpyDict, gcp_bucket_name: BucketName = ""
) -> None:
    """Save numpy dictionary as temp file and then upload the binary to the bucket."""
    (_, fname) = tempfile.mkstemp()
    fname = f"{fname}.npy"
    np.save(fname, np_obj)
    upload(
        string_or_localpath=fname,
        gcp_path=gcp_path,
        from_string=False,
        gcp_bucket_name=gcp_bucket_name,
        byte_format="rb",
    )
    os.remove(fname)


def download_numpy(gcp_path: BucketPath, gcp_bucket_name: BucketName = "") -> NumpyDict:
    """Download the numpy binary to a temp file from the bucket, load and return it."""
    (_, fname) = tempfile.mkstemp()
    fname = f"{fname}.npy"
    download(gcp_path=gcp_path, gcp_bucket_name=gcp_bucket_name, local_path=fname)
    np_obj = np.load(fname, allow_pickle=True).item()
    os.remove(fname)
    return np_obj
