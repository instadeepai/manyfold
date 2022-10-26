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
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union

# regex to handle any float representation in python
REGEX_NUMBER = "([+-]?\d+([.]\d*)?(e[+-]?\d+)?|[.]\d+(e[+-]?\d+)?)"  # noqa: W605
# regex to handle any integer representation in python
REGEX_STEP = "([0-9]+)"
# regex to handle any kind of name
REGEX_NAME = "([a-zA-Z0-9_]+)"
# compile regex once for all
METRIC_REGEX = re.compile(
    f"METRIC name={REGEX_NAME} value={REGEX_NUMBER} step={REGEX_STEP}"
)

ARTIFACT_REGEX = re.compile(f"SEND ARTIFACT located at=(\S*)\.(\S*)")  # noqa


@dataclass
class Metric:
    """Generic metric to be sent to a metric logger."""

    name: str
    value: Union[int, float]
    step: int

    def __str__(self) -> str:
        return f"METRIC name={self.name} value={self.value} step={self.step}"

    @classmethod
    def from_string(cls, msg: str):
        match = METRIC_REGEX.match(msg)
        assert match is not None
        name = match.group(1)
        value = float(match.group(2))
        step = int(match.group(6))
        return cls(name=name, value=value, step=step)


@dataclass
class Artifact:
    """
    Message sent to the logger to notify that an artifact is ready to be uploaded.
    """

    path: str

    def __str__(self) -> str:
        return f"SEND ARTIFACT located at={self.path}"

    @classmethod
    def from_string(cls, msg: str):
        match = ARTIFACT_REGEX.match(msg)
        assert match is not None
        path = match.group(1)
        extension = match.group(2)
        return cls(path=f"{path}.{extension}")


class MetricLogger(ABC):
    """Generic class that implements a metric logger.
    E.g. Neptune, W&B, Tensorboard, ..."""

    @abstractmethod
    def start(self):
        """Init things here if needed."""

    @abstractmethod
    def log(self, metric: Metric):
        """Log the metric."""

    @abstractmethod
    def log_artifact(self, path: str):
        """Log the artifact."""


class MetricLoggingHandler(logging.Handler):
    """Handler to detect when a metric is sent through the logger.
    Uses regex to do so."""

    def __init__(self, loggers: List[MetricLogger]):

        super(MetricLoggingHandler, self).__init__()
        self._loggers = loggers

        for logger in loggers:
            logger.start()

    def emit(self, record):
        formatted_msg = self.format(record)
        # Check regex for metric messages
        match = METRIC_REGEX.match(formatted_msg)
        if match is not None:
            metric = Metric.from_string(formatted_msg)
            for logger in self._loggers:
                logger.log(metric=metric)

        # Check regex for artifact messages
        match = ARTIFACT_REGEX.match(formatted_msg)
        if match is not None:
            artifact = Artifact.from_string(formatted_msg)
            for logger in self._loggers:
                logger.log_artifact(path=artifact.path)
