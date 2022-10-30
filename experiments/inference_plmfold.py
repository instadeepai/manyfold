import argparse
import os
import re
from typing import MutableMapping

import hydra
import numpy as np

from manyfold.common import residue_constants
from manyfold.data.parsers import parse_fasta

FeatureDict = MutableMapping[str, np.ndarray]


def single_chain_inference(fasta_filename: str, output_path: str):
    """The functionality for inference with a single sequence is placed in it's own
    function as jax wont free the device memory until the python process finishes; so
    jax is only imported in the python processes for single chain inference."""
    import haiku as hk
    import jax

    from manyfold.model import modules_plmfold
    from manyfold.model.config import convert_to_ml_dict
    from manyfold.model.model import get_confidence_metrics
    from manyfold.model.tf.data_transforms import make_atom14_masks
    from manyfold.train.utils import get_model_haiku_params_maybe_gcp
    from manyfold.validation.pdb import save_predicted_pdb

    def make_sequence_features(sequence: str, num_res: int) -> FeatureDict:
        """Constructs a feature dict of sequence features."""
        features = {}
        features["aatype"] = np.argmax(
            residue_constants.sequence_to_onehot(
                sequence=sequence,
                mapping=residue_constants.restype_order_with_x,
                map_unknown_to_x=True,
            ),
            axis=-1,
        )
        features["residue_index"] = np.array(range(num_res), dtype=np.int32)
        features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
        features["seq_mask"] = np.ones((num_res,), dtype=np.float32)
        features["aatype_plm"] = features["aatype"]
        features["residue_index_plm"] = features["residue_index"]
        features["seq_mask_plm"] = features["seq_mask"]
        features = make_atom14_masks(features)
        for k in [
            "atom14_atom_exists",
            "residx_atom14_to_atom37",
            "residx_atom37_to_atom14",
            "atom37_atom_exists",
        ]:
            features[k] = np.array(features[k])
        return features

    with hydra.initialize(config_path="../manyfold/model/config", version_base=None):
        cfg = hydra.compose(config_name="config_val", overrides=[])
    cfg = convert_to_ml_dict(cfg)

    with open(fasta_filename) as f:
        [sequence], [description] = parse_fasta(f.read())
    seq_id = re.split(r"[|\s]", description)[0]
    print(f"\nRunning {seq_id}: {len(sequence)} residues")

    feats = make_sequence_features(sequence, len(sequence))
    feats = jax.tree_map(lambda x: x[None], feats)

    def _preprocess_fn(feat_dict):
        model = modules_plmfold.PLMEmbed(cfg.model_config.language_model)
        return model(feat_dict)

    def _forward_fn(batch):
        model = modules_plmfold.PLMFold(cfg.model_config.model)
        return model(
            batch,
            is_training=False,
            compute_loss=False,
            ensemble_representations=False,
            return_representations=False,
        )

    apply_plm = hk.transform(_preprocess_fn).apply
    apply_folding = hk.transform(_forward_fn).apply

    model_params = get_model_haiku_params_maybe_gcp(
        model_name=cfg.args.model_name,
        data_dir=cfg.args.params_dir,
    )

    plm_config = cfg.model_config.language_model
    params_plm = get_model_haiku_params_maybe_gcp(
        model_name=plm_config.model_name,
        data_dir=plm_config.pretrained_model_dir,
    )
    params_plm = {f"plmembed/{k}": v for k, v in params_plm.items()}

    feats = apply_plm(params_plm, jax.random.PRNGKey(seed=0), feats)
    preds = apply_folding(model_params, jax.random.PRNGKey(seed=0), feats)

    preds.update(get_confidence_metrics(preds))
    output_dir = f"{output_path}/{seq_id}"
    os.makedirs(output_dir, exist_ok=True)
    feats = jax.tree_map(lambda x: x[0], feats)
    save_predicted_pdb(preds, feats, output_dir=output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        type=str,
        help="Path to input fasta file",
    )
    parser.add_argument(
        "-o",
        type=str,
        help="Path to output directory",
        default="experiments/inference_results",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    with open(args.f) as f:
        sequences, descriptions = parse_fasta(f.read())

    os.makedirs(args.o, exist_ok=True)

    if len(sequences) == 1:
        single_chain_inference(fasta_filename=args.f, output_path=args.o)
    else:
        # call this python file for each sequence as a subprocess
        for sequence, description in zip(sequences, descriptions):
            filename = "query.fasta"
            with open(filename, "w") as f:
                f.write(f">{description}\n{sequence}\n")
            cmd = f"python {__file__} -f {filename} -o {args.o}"

            # running a single python process per sequence forces jax to deallocate
            # the device memory after each forward pass
            v = os.system(cmd)
            os.remove(filename)
            if v != 0:
                break
