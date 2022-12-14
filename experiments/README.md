# Training and inference

## Build the docker image on GPU/TPU

First, build the docker image for the user, which will install all dependencies needed to run the experiments.

```bash
# GPU
sudo docker build -t manyfold \
    --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) \
    -f docker/cuda.Dockerfile .

# TPU
sudo docker build -t manyfold \
    --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) \
    -f docker/tpu.Dockerfile .
```

## Run the container

Second, run the docker container in an interactive session.

```bash
# GPU
sudo docker run -it --rm --gpus all \
    --network host --name manyfold_container \
    -v ${PWD}:/app manyfold /bin/bash

# TPU
sudo docker run -it --rm --privileged \
    --network host --name manyfold_container \
    -v ${PWD}:/app manyfold /bin/bash
```

## Launch training runs

To train pLMFold/AlphaFold models use the script `experiments/train_model.py`. The training arguments are in `manyfold/model/config/config_train.yaml`. Also, the configuration files of all the models are in `manyfold/model/config/model_config`.

### 1) pLMFold training

By default, `train_model.py` trains a pLMFold model from scratch with default parameters:

```bash
python experiments/train_model.py

# which is equivalent to
python experiments/train_model.py \
    model_config="plmfold_config" \
    model_config/language_model="config_esm1b_t33_650M_UR50S" \
    args.checkpoint_dir="experiments/checkpoints/plmfold"
```

Training can be resumed from a stored checkpoint or pretrained model parameters:

```bash
# From checkpoint
python experiments/train_model.py \
    args.checkpoint_dir="<path-to-checkpoints>" \
    args.continue_from_last_checkpoint=True

# From pretrained parameters
python experiments/train_model.py \
    args.pretrained_models_dir="<path-to-pretrained-models>" \
    args.pretrained_model="model_plmfold"
```

### 2) AlphaFold training

For AlphaFold, it is required to specify the model configuration (from 1 to 5). For example, to train `model_1_ptm`:

```bash
python experiments/train_model.py \
    model_config="config_deepmind_casp14_monomer_1_ptm" \
    args.checkpoint_dir="experiments/checkpoints/alphafold/model_1_ptm"
```

To resume training from a pre-trained AlphaFold/OpenFold model (`model_1_ptm`):

```bash
python experiments/train_model.py \
    model_config="config_deepmind_casp14_monomer_1_ptm" \
    args.pretrained_models_dir="<path-to-pretrained-models>" \
    args.pretrained_model="model_1_ptm"
```

Important note: training on mixed-precision (`bfloat16`) is only supported for A100 GPU and TPU for now. To train on full-precision (`float32`), the following option needs to be added to the run call:

```bash
python experiments/train_model.py \
    model_config.train.mixed_precision.use_half=False \
    ... # other arguments/options
```

The outputs are written to `args.checkpoint_dir`, which has the following folder structure:

```LaTex
experiments/checkpoints/
    |- alphafold/
    ...
    |- plmfold/
        |- checkpoint_0.pkl
        ...
        |- config_0.yaml
        ...
        |- params_0.npz
        ...
```

## Validation

To validate a pretrained pLMFold/AlphaFold/OpenFold model use the script `experiments/validate_model.py`. The validation arguments are in `manyfold/model/config/config_val.yaml`.

The main arguments are the paths to data samples (`args.data_dir`), fasta file (`args.fasta_path`), parameters (`args.params_dir`), and results (`args.results_dir`). The number of devices and batch size per device can be controlled with the arguments `args.num_devices` and `args.batch_size`, respectively. To use Amber post-relaxation, specify the argument `args.use_relaxed_predictions=True` (this option is only available for CPU).

### 1) pLMFold validation

```bash
python experiments/validate_model.py \
    model_config="plmfold_config" \
    args.results_dir="experiments/results_cameo/plmfold" \
    args.model_name="model_plmfold" \
    args.params_dir="params/plmfold"
```

### 2) AlphaFold/OpenFold validation

For example, to validate `model_1_ptm`:

```bash
# AlphaFold
python experiments/validate_model.py \
    model_config="config_deepmind_casp14_monomer_1_ptm" \
    args.results_dir="experiments/results_cameo/alphafold" \
    args.model_name="model_1_ptm" \
    args.params_dir="params/alphafold"

# OpenFold
python experiments/validate_model.py \
    model_config="config_deepmind_casp14_monomer_1_ptm" \
    args.results_dir="experiments/results_cameo/openfold" \
    args.model_name="model_1_ptm" \
    args.params_dir="params/openfold"
```

The script `validate_model.py` assumes the target features are available in the input `.tfrecords` and computes the losses specified in the model. As outputs, the script generates for every sample: (i) a `prediction.pdb` file with the predicted structure and (ii) a `metrics.npy` file containing the confidence metrics.

For example, for the CAMEO test set, the folder structure would be as follows:

```LaTex
experiments/results_cameo/
    |- alphafold/
        |- model_1_ptm/
        |- model_2_ptm/
        ...
    |- openfold/
        |- model_1_ptm/
        ...
    |- plmfold/
        |- model_plmfold/
            |- 7EQH_A/
                |- metrics.npy
                |- prediction.pdb
            |- 7ER0_A/
            ...
```

## pLMFold inference

To run inference on the pLMFold model for the set of CAMEO sequences in a FASTA file:

```bash
python experiments/inference_plmfold.py \
    -f datasets/sequences_cameo.fasta \
    -o experiments/inference_results
```

The predicted structures are written in PDB format to `experiments/inference_results` by default. Note that custom FASTA files can be used instead. These can be created from PDB entries as:

```bash
curl https://www.rcsb.org/fasta/entry/7ZZ5 \
    -o datasets/input_sequences.fasta
```

## Timing/memory estimations

Training with 1 sample per TPU core needs a v2 (with 8GB of memory per core) for pLMFold and takes ~3.5 sec/step, compared to a v3 (16GB per core) and ~14 sec/step for AlphaFold.

The following plot shows the validation times (using `.tfrecords`) for different input sequence lengths on the CAMEO dataset:

<p align="center">
    <img src="../imgs/inference_times.png" width="400">
</p>

In terms of memory, the pLMFold model allows inference on sequences of up to 2400 residues using an A100 GPU (40GB).
