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

"""Context manager timing code, makes code much clearer when recording many timings."""
import json
import os
from timeit import default_timer
from typing import Dict, List, Union

from manyfold.utils import gcp

Entry = Union[List[float], Dict[str, float], float]


class RecordTime:
    """Record blocks of python code by placing in the `with RecordTime(...)` block.

    If the `file_path` is provided and starts with 'gs://' upload a json file to the
    gcloud bucket. If the `file_path` is provided and doesn't start with 'gs://' it will
    be written locally.

    Usage:
    # Add to a timings json file: Dict[str, float], or add the entry to a new file.
    with RecordTime("some_name", file_path):
        ...
    # OR:
    # add the entry to a dictionary, no file written to.
    with RecordTime("some_name", timings_dict=timings_dict):
        ...
    # OR:
    # If you only want one timing in the current block of code.
    with RecordTime() as rt:
        ...
    t = rt.elapsed
    # OR:
    # Add a timing dictionary to the timings json file.
    RecordTime.manually_add(file_path, "some_name", some_timings_data)
    """

    def __init__(
        self,
        name: str = "",
        file_path: str = "",
        timings_dict: Dict[str, float] = None,
        pbar=None,
        overwrite_entry=True,
    ):
        self.file_path = file_path
        self.name = name
        self.overwrite = overwrite_entry
        self.stdout = print if pbar is None else pbar.set_description
        self.timings_dict = {} if timings_dict is None else timings_dict

    def __enter__(self):
        self.stdout(f"timing: {self.name}...")
        self.t = default_timer()
        return self

    @staticmethod
    def _read(file_path):
        if file_path.startswith("gs://"):
            return json.loads(gcp.download(file_path)) if gcp.is_blob(file_path) else {}
        else:
            return json.loads(open(file_path)) if os.path.isfile(file_path) else {}

    @staticmethod
    def _write(file_path, times: Dict[str, Entry]):
        jstr = json.dumps(times, indent=4, sort_keys=True)
        if file_path.startswith("gs://"):
            gcp.upload(jstr, file_path, from_string=True)
        else:
            with open(file_path, "w") as f:
                f.write(jstr)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stdout("done")
        self.elapsed = default_timer() - self.t
        if self.file_path != "":
            times = self._read(self.file_path)
            if self.name not in times or self.overwrite:
                times[self.name] = self.elapsed
                RecordTime._write(self.file_path, times)
            else:
                self.stdout(f"{self.name} not written because overwrite is False.")
        elif self.timings_dict is not None:
            self.timings_dict[self.name] = self.elapsed

    @staticmethod
    def manually_add(file_path: str, name: str, value: Entry):
        times = RecordTime._read(file_path)
        times[name] = value
        RecordTime._write(file_path, times)
