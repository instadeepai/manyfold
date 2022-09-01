import copy
from typing import Any, Dict, List, Tuple

import numpy as np

from manyfold.utils import gcp


def load_file(filepath: str, from_numpy: bool = False) -> Any:
    # Load from GCP bucket.
    if filepath.startswith("gs://"):
        obj = (
            gcp.download_numpy(gcp_path=filepath)
            if from_numpy
            else gcp.download(gcp_path=filepath)
        )
    # Load from local file.
    else:
        obj = (
            np.load(filepath, allow_pickle=True).item()
            if from_numpy
            else open(filepath, "r").read()
        )
    return obj


def filter_fasta(
    sequences: List[str],
    seq_ids: List[str],
    min_length: int = 0,
    max_length: int = 999999,
) -> Tuple[List[str], List[str]]:
    """Filter a set sequences from FASTA to have lengths within a given a range.

    Args:
      sequences: a list with the amino acid sequences.
      seq_ids: a list with the identifiers for each sample.
      min_length: minimum sequence length to filter.
      max_length: maximum sequence length to filter.

    Returns:
      filtered_sequences: a list with the new sequences after filtering.
      filtered_seq_ids: a list with the new identifiers after filtering.
    """

    def condition(string):
        return len(string) >= min_length and len(string) <= max_length

    lst = [[seq, seqid] for seq, seqid in zip(sequences, seq_ids) if condition(seq)]
    filtered_sequences, filtered_seq_ids = list(map(list, zip(*lst)))
    return filtered_sequences, filtered_seq_ids


def select_unpad_features(
    features: Dict[str, np.ndarray],
    nres: int,
) -> Dict[str, np.ndarray]:
    """Selects and unpads target features given the sequence length.

    Args:
      features: a dictionary with the target features for one sample.
      nres: number of residues in the sequence.

    Returns: a dictionary containing the selected keys and unpaded arrays.
    """

    return {
        key: features[key][0, slice(None, nres), ...]
        for key in ["aatype", "residue_index"]
    }


def unpad_prediction(
    prediction: Dict[str, Any],
    nres: int,
    nmsa: int = None,
) -> Dict[str, Any]:
    """Unpads the predictions given the sequence length.

    Args:
      prediction: a dictionary with the predictions for one sample.
      nres: number of residues in the sequence.
      nmsa: number of sequences in the MSA to keep.

    Returns: a dictionary containing the unpaded arrays.
    """

    pred = copy.deepcopy(prediction)
    s = slice(None, nres)
    for (key0, key1), slices in [
        (("distogram", "logits"), (s, s)),
        (("experimentally_resolved", "logits"), (s,)),
        (("masked_msa", "logits"), (slice(None, nmsa), s)),
        (("predicted_aligned_error", "aligned_error"), (s, s)),
        (("predicted_aligned_error", "logits"), (s, s)),
        (("predicted_lddt", "logits"), (s,)),
        (("structure_module", "final_atom_mask"), (s,)),
        (("structure_module", "final_atom_positions"), (s,)),
    ]:
        if key0 in prediction:
            pred[key0][key1] = pred[key0][key1][slices]
    return pred
