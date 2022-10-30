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

FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Speed up the build, and avoid unnecessary writes to disk
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1

# Use bash to support string substitution.
SHELL ["/bin/bash", "-c"]

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      build-essential \
      cmake \
      cuda-command-line-tools-11-3 \
      curl \
      git \
      g++ \
      parallel \
      tzdata \
      wget \
      zip \
    && rm -rf /var/lib/apt/lists/*

# Install conda package manager.
RUN curl https://repo.anaconda.com/pkgs/misc/gpgkeys/anaconda.asc | \
    gpg --dearmor > conda.gpg
RUN install -o root -g root -m 644 conda.gpg /usr/share/keyrings/conda-archive-keyring.gpg
RUN gpg --keyring /usr/share/keyrings/conda-archive-keyring.gpg \
        --no-default-keyring \
        --fingerprint 34161F5BF5EB1D4BFBBB8F0A8AEB4F8B29D82806
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/conda-archive-keyring.gpg] https://repo.anaconda.com/pkgs/misc/debrepo/conda stable main" \
  > /etc/apt/sources.list.d/conda.list
RUN apt-get update && \
    apt-get install -y conda && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root
COPY docker/environment_cuda.yaml /root/
# Install conda environment.
RUN source /opt/conda/etc/profile.d/conda.sh && \
  conda update -qy conda && \
  conda env create -f /root/environment_cuda.yaml && \
  conda activate manyfold && \
  conda clean -a

ENV PATH=/opt/conda/envs/manyfold/bin/:$PATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Apply OpenMM patch.
COPY docker/openmm.patch /root/
WORKDIR /opt/conda/envs/manyfold/lib/python3.8/site-packages
RUN patch -p0 < /root/openmm.patch

RUN mkdir /app
WORKDIR /root
COPY . /app
WORKDIR /app
RUN pip install --user -e ./

ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd --gid ${GROUP_ID} eng
RUN useradd -l --gid eng --uid ${USER_ID} --shell /bin/bash --home-dir /app eng
RUN chown -R eng /app

USER eng
RUN echo 'export PATH=/opt/conda/bin:$PATH' >> .bashrc
RUN echo 'export PATH=$HOME/bin:$HOME/.local/bin:$PATH' >> .bashrc
RUN echo 'LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/' >> .bashrc
RUN echo 'source /opt/conda/etc/profile.d/conda.sh' >> .bashrc
RUN echo "alias ls='ls --color=auto'" >> .bashrc
RUN echo "export TF_FORCE_UNIFIED_MEMORY=1" >> .bashrc
