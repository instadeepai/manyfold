from typing import Dict, Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from haiku import initializers

from manyfold.model.language_model.models.types import (
    SequenceEmbedding,
    SequenceMask,
    TransformerOutput,
)


class LinearLayer(hk.Module):
    """
    Dense layer with HE initialization.
    """

    def __init__(self, output_size: int, name: str):
        """
        Args:
            output_size (int): size of the layer output
            name (str): layer's name
        """
        super().__init__(name=name)
        w_init = initializers.VarianceScaling(2.0, "fan_in", "uniform")
        b_init = initializers.VarianceScaling(2.0, "fan_in", "uniform")

        # Define layer
        self._fc = hk.Linear(output_size=output_size, w_init=w_init, b_init=b_init)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = self._fc(x)
        return y


class RobertaLMHead(hk.Module):
    """
    Roberta Language Model head. Transform final transformer layer output
    into a distribution over tokens at each position.
    """

    def __init__(self, embed_dim: int, alphabet_size: int, name: Optional[str] = None):
        """
        Args:
            embed_dim (int): Embedding dimension
            alphabet_size (int): Number of tokens in the alphabet
            name (Optional[str]): Layer's name
        """
        super(RobertaLMHead, self).__init__(name=name)
        self.embed_dim = embed_dim
        self.alphabet_size = alphabet_size

        # Define layers
        self._first_layer_norm = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="emb_layer_norm_after"
        )
        self._fc1 = LinearLayer(self.embed_dim, name="lm_head_fc_1")
        self._final_fc = LinearLayer(self.alphabet_size, name="lm_final_fc")
        self._second_layer_norm = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="lm_head_layer_norm"
        )

    def __call__(self, x: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        x = self._first_layer_norm(x)
        x = self._fc1(x)
        x = jax.nn.gelu(x, approximate=False)
        x = self._second_layer_norm(x)

        # Compute logits
        logits = self._final_fc(x)
        return {"embeddings": x, "logits": logits}


class SimpleLMHead(hk.Module):
    """
    Basic Language Model head. Transform final transformer layer output
    into a distribution over tokens at each position.
    """

    def __init__(self, embed_dim: int, alphabet_size: int, name: Optional[str] = None):
        """
        Args:
            embed_dim (int): Embedding dimension
            alphabet_size (int): Number of tokens in the alphabet
            name (Optional[str]): Layer's name
        """
        super(SimpleLMHead, self).__init__(name=name)
        self.embed_dim = embed_dim
        self.alphabet_size = alphabet_size

        # Define layers
        self._final_fc = LinearLayer(self.alphabet_size, name="lm_final_fc")

    def __call__(self, x: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        # Compute logits
        logits = self._final_fc(x)
        return {"embeddings": x, "logits": logits}


class TokensDropout(hk.Module):
    """
    Tokens dropout layer.
    """

    def __init__(
        self,
        embed_dim: int,
        pad_token_id: int,
        mask_token_id: int,
        masking_ratio: float,
        masking_prob: float,
        name: Optional[str] = None,
    ):
        """
        Args:
            embed_dim (int): Embedding dimension
            pad_token_id (int): Id of the pad token
            mask_token_id (int): Id of the pad token
            masking_ratio (float): Masking ratio
            masking_prob (float) : Probability to mask
            name (Optional[str]): Layer's name
        """
        super().__init__(name=name)
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.masking_ratio = masking_ratio
        self.masking_prob = masking_prob
        self.embed_dim = embed_dim

    def __call__(self, x: jnp.ndarray, tokens: jnp.ndarray) -> jnp.ndarray:

        padding_mask_tokens = tokens == self.pad_token_id
        tokens_repeated = jnp.repeat(
            tokens[:, :, None], repeats=self.embed_dim, axis=-1
        )
        x = jnp.where(tokens_repeated == self.mask_token_id, 0.0, x)
        mask_ratio_train = self.masking_ratio * self.masking_prob
        src_lengths = (~padding_mask_tokens).sum(-1)
        mask_ratio_observed = (tokens == self.mask_token_id).sum(-1) / src_lengths
        x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]
        return x


class TransformerLayer(hk.Module):
    """Regular transformer layer."""

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        ffn_embed_dim: int,
        add_bias_kv: bool = False,
        name: Optional[str] = None,
    ):
        """
        Args:
            num_heads (int): Number of attention heads.
            embed_dim (int): Size of embeddings.
            ffn_embed_dim (int): MLP hidden layer embedding size.
            dropout_rate (float, optional): Dropout rate applied after
                the multi-head attention. Defaults to 0.0.
            add_bias_kv (bool): Add a bias in attention layer.
            name (Optional[str], optional): Name of the transformer layer.
                Defaults to None.
        """
        super().__init__(name=name)
        # Add checks on dimensions
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"The embedding dimension should be divisible by the number of heads, "
                f"however provided embed dim is {embed_dim} and the number of heads "
                f"is {num_heads}"
            )

        # Hyperparameters internalization
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.key_size = embed_dim // num_heads
        self.dropout_rate = 0  # Hardcoded here but this might change in the future
        self.ffn_embed_dim = ffn_embed_dim
        self.add_bias_kv = add_bias_kv

        # Define layers
        self.query_layer = LinearLayer(self.embed_dim, name="query")
        self.key_layer = LinearLayer(self.embed_dim, name="key")
        self.value_layer = LinearLayer(self.embed_dim, name="value")
        self.output_layer = LinearLayer(self.embed_dim, name="mha_output")

        self.fc1 = LinearLayer(self.ffn_embed_dim, name="fc1")
        self.fc2 = LinearLayer(self.embed_dim, name="fc2")

        self.layer_norm1 = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="self_attn_layer_norm"
        )
        self.layer_norm2 = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="final_layer_norm"
        )

    @hk.transparent
    def self_attention(
        self,
        x: SequenceEmbedding,
        padding_mask_tokens: SequenceMask,
    ) -> TransformerOutput:
        """applies the attention mechanism

        Args:
            x (SequenceEmbedding): a 2D input embedding
            padding_mask_tokens (jnp.ndarray[bool]): 1D padding mask

        Returns:
           Dict[str, jnp.ndarray]: Dictionary containing the output embeddings
                and the attention weights
        """

        query_heads = self.query_layer(x)
        key_heads = self.key_layer(x)

        value_heads = self.value_layer(x)
        query_heads = query_heads.reshape(
            (*x.shape[:-1], self.num_heads, self.key_size)
        )
        key_heads = key_heads.reshape((*x.shape[:-1], self.num_heads, self.key_size))
        value_heads = value_heads.reshape(
            (*x.shape[:-1], self.num_heads, self.key_size)
        )

        if self.add_bias_kv:
            bias_k = hk.get_parameter(
                "bias_k", [1, 1, self.num_heads, self.key_size], init=jnp.zeros
            )
            batch_size = key_heads.shape[0]
            attn_bias = jnp.tile(bias_k, (batch_size, 1, 1, 1))
            key_heads = jnp.concatenate((key_heads, attn_bias), axis=1)
            padding_mask_tokens = jnp.concatenate(
                (
                    padding_mask_tokens,
                    jnp.ones(padding_mask_tokens.shape[:-1] + (1,), dtype=jnp.bool_),
                ),
                axis=-1,
            )

        attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        sqrt_key_size = np.sqrt(self.key_size).astype(x.dtype)
        attn_logits = attn_logits / sqrt_key_size

        padding_mask_att = jnp.any(padding_mask_tokens, axis=-2, keepdims=True)
        padding_mask_att = jnp.tile(padding_mask_att, (1, 1, query_heads.shape[1], 1))

        assert len(padding_mask_att.shape) == len(attn_logits.shape)
        attn_logits = jnp.where(padding_mask_att, attn_logits, -1e30)

        attn_weights = jax.nn.softmax(attn_logits)

        if self.add_bias_kv:
            bias_v = hk.get_parameter(
                "bias_v", [1, 1, self.num_heads, self.key_size], init=jnp.zeros
            )
            attn_bias = jnp.tile(bias_v, (batch_size, 1, 1, 1))
            value_heads = jnp.concatenate((value_heads, attn_bias), axis=1)

        attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)

        # Concatenate attention matrix of all heads into a single vector.
        attn_vec = jnp.reshape(attn, (*x.shape[:-1], -1))
        x = self.output_layer(attn_vec)

        return {"x": x, "attn_weights": attn_weights}

    @hk.transparent
    def mlp(self, x: SequenceEmbedding) -> SequenceEmbedding:
        """
        Applies one layer-norm, one linear layer, a Gelu activation,
        then a final linear layer

        Args:
            x (SequenceEmbedding): a 2D input embedding

        Returns:
            SequenceEmbedding: The transformed sequence embedding
        """
        x = self.layer_norm2(x)
        x = jax.nn.gelu(
            self.fc1(x),
            approximate=False,
        )
        x = self.fc2(x)
        x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
        return x

    def __call__(
        self,
        x: SequenceEmbedding,
        padding_mask_tokens: Optional[SequenceMask] = None,
    ) -> TransformerOutput:
        """Computes the output of the transformer layer.

        Args:
            x (jnp.ndarray): Input tokens.
            padding_mask_tokens (Optional[jnp.ndarray], optional): Attention mask.
                Defaults to None.
        Returns:
            Dict[str, jnp.ndarray]: Dictionary containing the output embeddings
                and the attention weights
        """

        dropout_rate = 0.0

        res = x
        x = self.layer_norm1(x)

        output = self.self_attention(
            x=x,
            padding_mask_tokens=padding_mask_tokens,
        )

        x = output["x"]
        x = hk.dropout(hk.next_rng_key(), dropout_rate, x)

        x = res + x
        x = x + self.mlp(x)

        output["x"] = x
        return output


class LearnedPositionalEmbeddings(hk.Module):
    """Position embeddings to be added to token embeddings."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        padding_idx: int,
        name: Optional[str] = None,
    ):
        """
        Args:
            vocab_size (int): Tokenizer's vocabulary size.
            embed_dim (int): Embedding size.
            padding_idx (Optional[int], optional): Index attributed to the padding
                token. Defaults to 1.
            name (Optional[str], optional): Name of the layer. Defaults to None.
        """
        super().__init__(name=name)
        self.padding_idx = padding_idx
        self._embed_layer = hk.Embed(vocab_size + padding_idx + 1, embed_dim)

    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        mask = tokens != self.padding_idx
        positions = jnp.cumsum(mask, axis=1) * mask + self.padding_idx

        return self._embed_layer(positions)


class SinusoidalPositionalEmbedding(hk.Module):

    """Sinusoidal embeddings to be added to token embeddings"""

    def __init__(
        self,
        embed_dim: int,
        padding_idx: int,
        name: Optional[str] = None,
    ):

        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx

    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        """
        Create the sinusoidal positional embeddings
        """

        bsz, seq_len = tokens.shape
        max_pos = self.padding_idx + 1 + seq_len
        weights = self._get_embedding(max_pos)
        positions = self._make_positions(tokens)

        return weights[positions.reshape(-1), :].reshape(bsz, seq_len, -1)

    @hk.transparent
    def _make_positions(self, x: jnp.ndarray) -> jnp.ndarray:
        mask = ~jnp.equal(x, self.padding_idx)
        range_buf = (
            jnp.broadcast_to(jnp.arange(x.shape[1]), x.shape) + self.padding_idx + 1
        )
        positions = jnp.broadcast_to(range_buf, x.shape)
        return positions * mask + self.padding_idx * (1 - mask)

    @hk.transparent
    def _get_embedding(self, num_embeddings: int) -> jnp.ndarray:

        half_dim = self.embed_dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
        emb = jnp.expand_dims(
            jnp.arange(num_embeddings, dtype=jnp.float32), axis=1
        ) * jnp.expand_dims(emb, 0)
        emb = jnp.reshape(
            jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1), (num_embeddings, -1)
        )

        if self.embed_dim % 2 == 1:
            # zero pad
            emb = jnp.concatenate([emb, jnp.zeros((num_embeddings, 1))], axis=1)

        if self.padding_idx is not None:
            emb = emb.at[self.padding_idx, :].set(0)

        return emb
