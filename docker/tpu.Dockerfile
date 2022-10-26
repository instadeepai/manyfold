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

FROM mambaorg/micromamba:0.22.0 as conda

# Speed up the build, and avoid unnecessary writes to disk
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1

COPY docker/environment_tpu.yaml /tmp/environment_tpu.yaml

RUN micromamba create -y --file /tmp/environment_tpu.yaml \
    && micromamba clean --all --yes \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete

FROM debian:bullseye-slim as test-image

# Let's have git installed in the container.
RUN apt update
RUN apt install -y git curl

COPY --from=conda /opt/conda/envs/. /opt/conda/envs/
ENV PATH=/opt/conda/envs/manyfold/bin/:$PATH APP_FOLDER=/app
ENV PYTHONPATH=$APP_FOLDER:$PYTHONPATH
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR $APP_FOLDER

ARG USER_ID=1000
ARG GROUP_ID=1000
ENV USER=eng
ENV GROUP=eng

COPY . $APP_FOLDER
RUN pip install --user -e ./

FROM test-image as run-image
# The run-image (default) is the same as the dev-image with the some files directly
# copied inside
