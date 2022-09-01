from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import haiku as hk
import jax.numpy as jnp
import jmp

from manyfold.model.language_model.models.layers import (
    LearnedPositionalEmbeddings,
    RobertaLMHead,
    SimpleLMHead,
    SinusoidalPositionalEmbedding,
    TokensDropout,
    TransformerLayer,
)
from manyfold.model.language_model.models.types import SequenceEmbedding, SequenceMask


@dataclass
class TransformerLMConfig:
    """
    Parameters to initialize a standard transformer Model.

    Args:
        alphabet_size: Token vocabulary.
        pad_token_id: Pad token index.
        mask_token_id: Mask token index.
        class_token_id: Class token index.
        eos_token_id: End of speech token index.
        prepend_cls_token (bool): Prepend beginning of speech.
        append_eos_token (bool): Append end of speech.
        attention_heads: Number of attention heads.
        embed_dim: Embedding dimension.
        ffn_embed_dim: Feed forward embedding dimension.
        num_layers: Number of transformer blocks.
        add_bias_kv: Add bias in attention layer.
        max_positions: Max position indicated to positional embedding learner.
        token_dropout (bool: Token dropout.
        masking_ratio: Masking ratio (used if token dropout is enabled).
        masking_prob: Masking probability (used of token dropout is enabled).
        embed_scale: Correction ratio applied to the embeddings to make up for the
            norm difference between the input during training and inference.
    """

    alphabet_size: int
    pad_token_id: int
    mask_token_id: int
    class_token_id: int
    eos_token_id: int
    prepend_cls_token: bool
    append_eos_token: bool
    max_positions: int = 1024
    emb_layer_norm_before: bool = False
    learned_positional_embedding: bool = True
    roberta_lm_head: bool = True
    add_bias_kv: bool = False
    attention_heads: int = 20
    embed_dim: int = 1280
    ffn_embed_dim: int = 5120
    num_layers: int = 24
    token_dropout: bool = False
    masking_ratio: float = 0.15
    masking_prob: float = 0.8
    embed_scale: float = 1.0
    use_gradient_checkpointing: bool = False


class TransformerLM(hk.Module):
    """
    Creates a standard language model
    (following architecture choices made in the ESM1b model from fair).
    """

    def __init__(
        self,
        config: TransformerLMConfig,
        name: Optional[str] = None,
    ):
        """
        Initializes the StandardTransformer model

        Args:
            config (StandardTransformerConfig): dataclass containing model
                hyperparameters.
            name (Optional[str]): Name for module (custom will break weight loading).
        """

        self._config = config
        super().__init__(name=name)

        self._embed_layer = hk.Embed(self._config.alphabet_size, self._config.embed_dim)

        if config.learned_positional_embedding:
            self._pos_embed_layer = LearnedPositionalEmbeddings(
                config.max_positions, config.embed_dim, config.pad_token_id
            )
        else:
            self._pos_embed_layer = SinusoidalPositionalEmbedding(
                config.embed_dim, config.pad_token_id
            )

        if config.roberta_lm_head:
            self._lm_head = RobertaLMHead(
                embed_dim=self._config.embed_dim,
                alphabet_size=self._config.alphabet_size,
            )
        else:
            self._lm_head = SimpleLMHead(
                embed_dim=self._config.embed_dim,
                alphabet_size=self._config.alphabet_size,
            )

    @hk.transparent
    def _transformer_layer(self, layer_idx: int) -> TransformerLayer:

        return TransformerLayer(  # type: ignore
            num_heads=self._config.attention_heads,
            embed_dim=self._config.embed_dim,
            ffn_embed_dim=self._config.ffn_embed_dim,
            add_bias_kv=self._config.add_bias_kv,
            name=f"transformer_layer_{layer_idx}",
        )

    @hk.transparent
    def apply_transformer_layers(
        self,
        x: SequenceEmbedding,
        outs: Dict[str, SequenceEmbedding],
        padding_mask_tokens: SequenceMask,
        # embeddings_layers_to_save: Tuple[int, ...],
        save_embeddings: bool,
        save_attention_weights: bool,
    ) -> Tuple[SequenceEmbedding, Dict[str, SequenceEmbedding]]:
        """
        Create the blocks of transformer layers and apply them

        Args:
            x (SequenceEmbedding): the sequence embedding
            outs (Dict[str, jnp.array]): A dictionary to carry through the
                transformer layers which stores the intermediate sequence
                embedding and attention maps.
            padding_mask_tokens (jnp.ndarray) '[batch_size, length]' :
                1D boolean mask, False represents padded inputs
            embeddings_layers_to_save (Tuple[int]): Tuple that contain the indices
                of the layers for which we want to save the embeddings. If the tuple
                is empty, then no embedding is saved during the pass.

        Returns:
            Tuple[SequenceEmbedding, Dict[str, SequenceEmbedding]]: The final sequence
            embedding and the optionnally stored intermediate results ( for inference
            mainly ).
        """

        layers: List[Callable] = [
            self._transformer_layer(layer_idx)
            for layer_idx in range(self._config.num_layers)
        ]

        if self._config.use_gradient_checkpointing:
            # the remat-ed function cannot take control flow arguments
            layers = [hk.remat(layer) for layer in layers]

        for layer_idx, layer in enumerate(layers):
            output = layer(
                x=x,
                padding_mask_tokens=padding_mask_tokens,
            )
            x = output["x"]
            # Save intermediate embeddings if needed
            # if layer_idx in embeddings_layers_to_save:
            if save_embeddings:
                outs[f"embeddings_{layer_idx}"] = output["x"]
            # Sum attention maps over heads if needed. This will be averaged when all
            # layers and all heads have been summed.
            if save_attention_weights:
                outs["attn_weights"] += jnp.sum(output["attn_weights"], axis=1)
        return x, outs

    def __call__(
        self,
        tokens: jnp.ndarray,
        # embeddings_layers_to_save: Tuple[int, ...] = (),
        save_embeddings: bool = False,
        save_attention_weights: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        """Compute the embeddings based on the input tokens.

        Args:
            tokens (jnp.ndarray): Input tokens out of the tokenizer and masking
                function (if applied). shape '[batch_size, length]'
            embeddings_layers_to_save (Tuple[int]): Tuple that contain the indices
                of the layers for which we want to save the embeddings. If the tuple
                is empty, then no embedding is saved during the pass.

        Returns:
            Mapping[str, jnp.ndarray]: Dictionary containing the final embeddings and
                logits.
        """
        # Prepare outputs dict
        outs: Dict[str, jnp.array] = {}
        if save_attention_weights:
            outs["attn_weights"] = 0
            # outs["attn_weights"] = jnp.zeros(
            #     (
            #         tokens.shape[0],
            #         self._config.max_positions,
            #         self._config.max_positions + int(self._config.add_bias_kv),
            #     )
            # )

        # Compute padding mask
        padding_mask_tokens = tokens == self._config.pad_token_id
        padding_mask_tokens = ~padding_mask_tokens[:, None, :]

        padding_mask_tokens = jnp.einsum(
            "bhT, bht->bhtT", padding_mask_tokens, padding_mask_tokens
        )

        # Compute embeddings
        x = self._embed_layer(tokens)
        # Tokens dropout if needed
        if self._config.token_dropout:
            x = TokensDropout(
                embed_dim=self._config.embed_dim,
                mask_token_id=self._config.mask_token_id,
                pad_token_id=self._config.pad_token_id,
                masking_ratio=self._config.masking_ratio,
                masking_prob=self._config.masking_prob,
            )(x, tokens)

        # RoBERTa's mask scaling factor
        x = self._config.embed_scale * x + self._pos_embed_layer(tokens)
        if self._config.emb_layer_norm_before:
            x = hk.LayerNorm(
                axis=-1,
                create_scale=True,
                create_offset=True,
                name="emb_layer_norm_before",
            )(x)

        # construct a tower of transformer layers
        x, outs = self.apply_transformer_layers(
            x=x,
            outs=outs,
            padding_mask_tokens=padding_mask_tokens,
            save_embeddings=save_embeddings,
            save_attention_weights=save_attention_weights,
        )

        # Average the attention weights over all heads and layers if needed
        if save_attention_weights:
            outs["attn_weights"] = outs["attn_weights"] / (
                self._config.num_layers * self._config.attention_heads
            )

        # RobertaLMHead
        lm_head_outs = self._lm_head(x)
        logits, embeddings = lm_head_outs["logits"], lm_head_outs["embeddings"]

        # Save final embeddings if needed
        # if self._config.num_layers in embeddings_layers_to_save:
        outs[f"embeddings_{self._config.num_layers}"] = embeddings
        # Add logits to the output dictionary
        outs["logits"] = logits
        return outs  # type: ignore


def build_transformer_forward_fn(
    model_config: TransformerLMConfig,
    embeddings_layers_to_save: Tuple[int, ...] = (),
    save_attention_weights: bool = False,
    mixed_precision: bool = False,
) -> Callable:
    """Creates the model's forward pass.

    Args:
        model_config (TransformerLMConfig): Model parameters
        embeddings_layers_to_save (Tuple[int]): Tuple that contain the indices
            of the layers for which we want to save the embeddings. If the tuple
            is empty, then no embedding is saved during the pass.
        mixed_precision: Enable mixed precision or not. Correct setting allows for mixed
            precision only on TPUs and A100 GPUs.

    Returns:
        forward_fn: The forward function
    """
    if mixed_precision:
        # Use mixed precision (only support A100 GPU and TPU for now)
        half = jnp.bfloat16
        full = jnp.float32

        policy = jmp.Policy(compute_dtype=half, param_dtype=full, output_dtype=full)
        hk.mixed_precision.set_policy(TransformerLM, policy)

        # Remove it in batch norm to avoid instabilities
        policy = jmp.Policy(compute_dtype=full, param_dtype=full, output_dtype=half)
        hk.mixed_precision.set_policy(hk.BatchNorm, policy)
        hk.mixed_precision.set_policy(hk.LayerNorm, policy)

    def forward_fn(tokens: jnp.ndarray) -> Mapping[str, jnp.ndarray]:
        """Forward pass."""
        # Run the transformer over the inputs.
        transformer = TransformerLM(model_config)
        outs = transformer(
            tokens,
            embeddings_layers_to_save=embeddings_layers_to_save,
            save_attention_weights=save_attention_weights,
        )
        return outs

    return forward_fn
