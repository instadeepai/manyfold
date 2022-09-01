from typing import Dict, Union

import jax.numpy as jnp
from typing_extensions import TypeAlias

PartialAttentionWeights: TypeAlias = jnp.ndarray
PartialSequenceEmbedding: TypeAlias = jnp.ndarray
SequenceEmbedding: TypeAlias = jnp.ndarray
Embedding: TypeAlias = jnp.ndarray
SequenceMask: TypeAlias = jnp.ndarray
AttnMartixMask: TypeAlias = jnp.ndarray
RandomBlockSelector: TypeAlias = jnp.ndarray
PRNGKey: TypeAlias = jnp.ndarray
BlockSequenceMask: TypeAlias = jnp.ndarray
BandBlockMask: TypeAlias = jnp.ndarray
AttentionWeights: TypeAlias = jnp.ndarray
TransformerOutput: TypeAlias = Dict[str, Union[SequenceEmbedding, AttentionWeights]]
