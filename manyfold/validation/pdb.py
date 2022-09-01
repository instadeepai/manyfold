import os
from typing import Any, Dict

import numpy as np

from manyfold.common import residue_constants
from manyfold.common.protein import from_pdb_string, from_prediction, to_pdb
from manyfold.relax import relax


def save_predicted_pdb(
    predictions: Dict[str, Any],
    features: Dict[str, np.ndarray],
    output_dir: str,
) -> None:
    """Gets the atom37 representation for the predicted structure and saves to file.
       It also adds the pLDDT as confidence metric for the predicted structure.

    Args:
      predictions: a dictionary with the predictions for one sample.
      features: a dictionary with the target features for one sample.
      output_dir: path to the directory where the output file is stored
        (prediction.pdb).
    """

    # Get predicted structure and save to pdb file.
    # Add the predicted LDDT in the b-factor column.
    # Note that higher predicted LDDT value means higher model confidence.
    plddt = predictions["plddt"]
    plddt_b_factors = np.repeat(
        plddt[:, None],
        residue_constants.atom_type_num,
        axis=-1,
    )
    predicted_protein = from_prediction(
        features=features,
        result=predictions,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=False,
    )

    prediction_pdb_filepath = os.path.join(output_dir, "prediction.pdb")
    with open(prediction_pdb_filepath, "w") as f:
        f.write(to_pdb(predicted_protein))


def run_relaxation_pdb(output_dir: str) -> Dict[str, np.ndarray]:
    """Runs Amber post-relaxation on the predicted structure.

    Args:
      output_dir: path to the directory where the output files are stored
        (prediction_unrelaxed.pdb and prediction.pdb).

    Returns: a dictionary containing the atom positions and mask for the predicted
      structure after relaxation.
    """

    # Load predicted structure.
    filepath_prediction = os.path.join(output_dir, "prediction.pdb")
    unrelaxed_protein = from_pdb_string(open(filepath_prediction, "r").read())

    # Run Amber relaxation.
    amber_relaxer = relax.AmberRelaxation(
        max_iterations=0,
        tolerance=2.39,
        stiffness=10.0,
        exclude_residues=[],
        max_outer_iterations=3,
        use_gpu=True,
    )
    relaxed_pdb, _, _ = amber_relaxer.process(prot=unrelaxed_protein)

    # Rename predicted structure and save relaxed structure.
    os.rename(filepath_prediction, os.path.join(output_dir, "prediction_unrelaxed.pdb"))
    with open(filepath_prediction, "w") as f:
        f.write(relaxed_pdb)
