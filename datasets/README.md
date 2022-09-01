# Datasets

This folder contains pre-generated samples for training and validation. The samples are identified by the headers in the `sequences_train.fasta` and `sequences_cameo.fasta` files.

The CAMEO test set includes 143 structures with less than 700 residues (from 03-22 to 05-22). The training set provided here contains a small sample of structures stored in the PDB before CASP14.

For both sets, the raw features for each sample are in `.tfrecord` format. These are processed through our data pipelines to generate the final features used for training and validation. For each feature type, we provide its shape and a small description of its content.

Notation: *N<sub>res</sub>* is the number of residues in the amino acid sequence, *N<sub>seq</sub>* is the number of sequences in the MSA, and *N<sub>temp</sub>* is the number of templates.

## Inference features

All models require the following features during training and inference. Those models that do not use templates can ignore the `template_*` features. Similarly, our pLMFold model ignores all `msa` features.

| Feature & Shape | Description |
|-----------------|-------------|
| `aatype` <br/> [*N<sub>res</sub>* , 21] | One-hot representation of the input amino acid sequence (20 amino acids + unknown), with the residues indexed according to `residue_contants.restypes_with_x`. |
| `between_segment_residues` <br/> [*N<sub>res</sub>* ,] | Whether there is a domain break. Always zero for chains, but keeping for compatibility with domain datasets. |
| `domain_name` <br/> [1 ,] | PDB code and chain id or some arbitrary name / description if there is no PDB code. |
| `residue_index` <br/> [*N<sub>res</sub>* ,] | Residue index given in the PDB file; if no ground truth structure exists it is just [0, 1,... *N<sub>res</sub>* - 1]. |
| `seq_length` <br/> [*N<sub>res</sub>* ,] | Number of residues in the chain, repeated along the num-res dimension. |
| `sequence` <br/> [*N<sub>res</sub>* ,] | String of amino acids in the chain, `utf-8` encoding. |
| `num_alignments` <br/> [*N<sub>res</sub>* , 1] | Number of sequences in the MSA, replicated along the num-res dimension. |
| `msa` <br/> [*N<sub>seq</sub>* , *N<sub>res</sub>* , 1] | Indices corresponding to HHBlits amino acids for each amino acid in each sequence in the MSA. |
| `deletion_matrix_int` <br/> [*N<sub>seq</sub>* , *N<sub>res</sub>* , 1] | For each sequence in the MSA, number of inserts between consecutive residues of the target sequence. |
| `template_aatype` <br/> [*N<sub>temp</sub>* , *N<sub>res</sub>* , 22] | One-hot representation of the amino acid sequence (20 amino acids + unknown + gap), with the residues indexed according to `residue_contants.restypes_with_x_and_gap`. |
| `template_all_atom_positions` <br/> [*N<sub>temp</sub>* , *N<sub>res</sub>* , 37 , 3] | Atom coordinates in `atom37` representation, for every atom in each residue in all of the templates. |
| `template_all_atom_exists` <br/> [*N<sub>temp</sub>* , *N<sub>res</sub>* , 37] | For each template, mask indicating if the coordinates of all atoms were specified in the PDB entry. |
| `template_all_atom_masks` <br/> [*N<sub>temp</sub>* , *N<sub>res</sub>* , 37] | For each template, mask indicating if the coordinates of all atoms exist in the `atom37` representation. |
| `template_domain_names` <br/> [*N<sub>temp</sub>* ,] | PDB code of the template structure, but can be named arbitrarily as it doesn't affect the pipeline. |
| `template_sequence` <br/> [*N<sub>temp</sub>* , *N<sub>res</sub>*] | Template amino-acid sequence, `utf-8` encoding. |
| `template_sum_probs` <br/> [*N<sub>temp</sub>* , 1] | Feature provided by HHSearch, used to choose the templates. |

## Training features

To compute the losses, the training and validation pipelines require the following features of the target structure.

| Feature & Shape | Description |
|-----------------|-------------|
| `pdb_cluster_size` | Size of the PDB cluster the chain structure falls into (for filtering). |
| `resolution` | Experimental resolution of the target structure as specified in the PDB entry. |
| `atom14_gt_positions` <br/> [*N<sub>res</sub>* , 14 , 3] | Atom coordinates in `atom14` representation, for every residue in the target structure. |
| `atom14_gt_exists` <br/> [*N<sub>res</sub>* , 14] | Mask indicating if the coordinates of all atoms were specified in the PDB entry. |
| `atom14_atom_exists` <br/> [*N<sub>res</sub>* , 14] | Mask indicating if the coordinates of all atoms exist in the `atom14` representation. |
| `atom14_alt_gt_positions` <br/> [*N<sub>res</sub>* , 14 , 3] | Constructs renamed atom positions for ambiguous residues. |
| `atom14_alt_gt_exists` <br/> [*N<sub>res</sub>* , 14] | `atom14_gt_exists` for the ambiguous residues. |
| `atom14_atom_is_ambiguous` <br/> [*N<sub>res</sub>* , 14] | Mask for the ambiguous atoms. |
| `all_atom_positions` <br/> [*N<sub>res</sub>* , 37 , 3] | Atom coordinates in `atom37` representation. |
| `all_atom_mask` <br/> [*N<sub>res</sub>* , 37] | Mask indicating if the coordinates of all atoms exist in the `atom37` representation. |
| `backbone_affine_tensor` <br/> [*N<sub>res</sub>* , 7] | Frames are constructed from the N, Ca, C atom coordinates of each residue, the first three numbers in the last dimension indicate the Ca position and the final four numbers are a quaternion representing the rotation for the frame. |
| `backbone_affine_mask` <br/> [*N<sub>res</sub>* ,] | Product of the masks for N, Ca and C atoms. |
| `chi_angles` <br/> [*N<sub>res</sub>* , 4, 2] | The seven torsion angles are [pre_omega, phi, psi, chi_1, chi_2, chi_3, chi_4] pre_omega is the omega torsion angle between the given amino acid and the previous amino acid. The penultimate dimension is the last four of the seven torsion angles. The final dimensions represent sin and cos. |
| `alt_chi_angles` <br/> [*N<sub>res</sub>* , 4, 2] | Mirrored torsion angles. |
| `chi_mask` <br/> [*N<sub>res</sub>* , 4] | Mask for which chi angles are present. |
| `rigidgroups_gt_frames` <br/> [*N<sub>res</sub>* , 8, 12 ] | Eight Frames corresponding to 'all_atom_positions' represented as flat twelve dimensional array. |
| `rigidgroups_gt_exists` <br/> [*N<sub>res</sub>* , 8] | Mask denoting whether the atom positions for the given frame are available in the ground truth, e.g. if they were resolved in the experiment. |
| `rigidgroups_group_exists` <br/> [*N<sub>res</sub>* , 8] | Mask denoting whether given group is in principle present for given amino acid type. |
| `rigidgroups_group_is_ambiguous` <br/> [*N<sub>res</sub>* , 8] | Mask denoting whether frame is affected by naming ambiguity. |
| `rigidgroups_alt_gt_frames` <br/> [*N<sub>res</sub>* , 8, 12] | Eight Frames with alternative atom renaming corresponding to 'all_atom_positions' represented as flat twelve dimensional array. |
