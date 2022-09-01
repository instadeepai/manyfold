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
"""Common utilities for data pipeline tools."""
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import List, Optional, Sequence, Tuple, TypeVar

from absl import logging


@contextlib.contextmanager
def tmpdir_manager(base_dir: Optional[str] = None):
    """Context manager that deletes a temporary directory on exit."""
    tmpdir = tempfile.mkdtemp(dir=base_dir)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@contextlib.contextmanager
def timing(msg: str):
    logging.info("Started %s", msg)
    tic = time.time()
    yield
    toc = time.time()
    logging.info("Finished %s in %.3f seconds", msg, toc - tic)


T = TypeVar("T")


def split_work(list_object: Sequence[T], nb_process: int) -> Sequence[Sequence[T]]:
    """
    Split a list of objects into a list of size nb_process of lists of the same
    objects as evenly as possible.
    """
    nb_object = len(list_object)
    nb_object_per_process = nb_object // nb_process
    splitted_list_object = []
    start_index = 0
    for i in range(nb_process):
        if i == nb_process - 1:
            splitted_list_object.append(list_object[start_index:])
            break
        end_index = start_index + nb_object_per_process
        if nb_object % nb_process > i:
            end_index += 1
        splitted_list_object.append(list_object[start_index:end_index])
        start_index = end_index
    return splitted_list_object


class SubprocessManager:
    def __init__(
        self,
        exit_on_sigint: bool = True,
    ) -> None:
        self.exit_on_sigint = exit_on_sigint
        self._process = None
        signal.signal(signal.SIGTERM, self._kill_subprocess)
        signal.signal(signal.SIGINT, self._kill_subprocess)

    def _kill_subprocess(self, signal_nb, frame):
        if self._process is not None:
            self._process.kill()
            self._process.wait()

        if self.exit_on_sigint and signal_nb == int(signal.SIGINT):
            sys.exit(0)

    def run(
        self,
        commnad: List[str],
        timeout: float = 60.0,
        decode_stderr_using_ascii: bool = False,
    ) -> Tuple[bool, str, Optional[str]]:
        self._process = subprocess.Popen(
            commnad,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy(),
        )

        try:
            stdout, stderr = self._process.communicate(timeout=timeout)
            retcode = self._process.returncode
            self._process = None
        except subprocess.TimeoutExpired:
            self._kill_subprocess(None, None)
            return (False, f"Timeout of {round(timeout, 1)} expired", None)

        success = retcode == 0
        if not success:
            return (
                False,
                f"Process exited with exit code {retcode}, stderr: "
                f"{stderr.decode('ascii') if decode_stderr_using_ascii else stderr}, "
                "stdout: "
                f"{stdout.decode('ascii') if decode_stderr_using_ascii else stdout}",
                None,
            )
        return (True, "", stdout)
