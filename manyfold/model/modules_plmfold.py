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

"""Modules and code used in the core part of PLMFold.

The structure generation code is in 'folding.py'.
"""
import functools

import haiku as hk
import jax
import jax.numpy as jnp

from manyfold.common import residue_constants
from manyfold.model import common_modules, folding, layer_stack, modules, prng
from manyfold.model.language_model.esm_jax import esm_jax
from manyfold.model.language_model.models.models import TransformerLM


class PLMFold(hk.Module):
    """PLMFold model with recycling."""

    def __init__(self, config, name="plmfold"):
        super().__init__(name=name)
        self.config = config
        self.global_config = config.global_config

    def __call__(
        self,
        batch,
        is_training,
        compute_loss=False,
        ensemble_representations=False,  # backwards compatibility
        return_representations=False,
    ):
        """Run the PLMFold model.

        Arguments:
          batch: Dictionary with inputs to the PLMFold model.
          is_training: Whether the system is in training or inference mode.
          compute_loss: Whether to compute losses (requires extra features
            to be present in the batch and knowing the true structure).
          ensemble_representations: Whether to use ensembling of representations.
          return_representations: Whether to also return the intermediate
            representations.

        Returns:
          When compute_loss is True:
            a tuple of loss and output of PLMFoldIteration.
          When compute_loss is False:
            just output of PLMFoldIteration.

          The output of PLMFoldIteration is a nested dictionary containing
          predictions from the various heads.
        """

        impl = PLMFoldIteration(self.config, self.global_config)
        batch_size, num_residues = batch["aatype"].shape

        def get_prev(ret):
            new_prev = {}
            if emb_config.recycle_pos:
                new_prev["prev_pos"] = ret["structure_module"]["final_atom_positions"]
            if emb_config.recycle_features:
                new_prev["prev_single"] = ret["representations"]["single"]
                new_prev["prev_pair"] = ret["representations"]["pair"]
            return jax.tree_map(jax.lax.stop_gradient, new_prev)

        def do_call(prev, compute_loss=compute_loss):
            ensembled_batch = batch
            non_ensembled_batch = jax.tree_map(lambda x: x, prev)

            return impl(
                ensembled_batch=ensembled_batch,
                non_ensembled_batch=non_ensembled_batch,
                is_training=is_training,
                compute_loss=compute_loss,
            )

        prev = {}
        emb_config = self.config.embeddings_and_evoformer
        if emb_config.recycle_pos:
            prev["prev_pos"] = jnp.zeros(
                [num_residues, residue_constants.atom_type_num, 3]
            )
        if emb_config.recycle_features:
            prev["prev_single"] = jnp.zeros([num_residues, emb_config.single_channel])
            prev["prev_pair"] = jnp.zeros(
                [num_residues, num_residues, emb_config.pair_channel]
            )

        if self.config.num_recycle:

            if "all_atom_positions" in batch:
                prev_dtype = batch["all_atom_positions"].dtype
                prev = jax.tree_map(lambda x: x.astype(prev_dtype), prev)
            else:
                prev_dtype = None

            if "num_iter_recycling" in batch:
                # Training time: num_iter_recycling is in batch.
                # The value for each ensemble batch is the same, so arbitrarily taking
                # 0-th.
                num_iter = batch["num_iter_recycling"][0]

                # Add insurance that we will not run more
                # recyclings than the model is configured to run.
                num_iter = jnp.minimum(num_iter, self.config.num_recycle)
            else:
                # Eval mode or tests: use the maximum number of iterations.
                num_iter = self.config.num_recycle

            body = lambda x: (  # noqa: E731
                x[0] + 1,  # pylint: disable=g-long-lambda
                get_prev(do_call(x[1], compute_loss=False)),
            )
            if hk.running_init():
                # When initializing the Haiku module, run one iteration of the
                # while_loop to initialize the Haiku modules used in `body`.
                _, prev = body((0, prev))
            else:
                _, prev = hk.while_loop(lambda x: x[0] < num_iter, body, (0, prev))

        ret = do_call(prev=prev)
        if compute_loss:
            ret = ret[0], [ret[1]]

        if not return_representations:
            del (ret[0] if compute_loss else ret)[
                "representations"
            ]  # pytype: disable=unsupported-operands
        return ret


class PLMFoldIteration(hk.Module):
    """A single recycling iteration of PLMFold architecture.

    Computes ensembled (averaged) representations from the provided features.
    These representations are then passed to the various heads
    that have been requested by the configuration file. Each head also returns a
    loss which is combined as a weighted sum to produce the total loss.
    """

    def __init__(self, config, global_config, name="plmfold_iteration"):
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config

    def __call__(
        self,
        ensembled_batch,
        non_ensembled_batch,
        is_training,
        compute_loss=False,
    ):
        def slice_batch(i):
            b = {k: v[i] for k, v in ensembled_batch.items()}
            b.update(non_ensembled_batch)
            return b

        # Compute representations for each batch element and average.
        plmformer_module = EmbeddingsAndPLMformer(
            self.config.embeddings_and_evoformer, self.global_config
        )
        batch = slice_batch(0)
        representations = plmformer_module(batch, is_training)

        heads = {}
        for head_name, head_config in sorted(self.config.heads.items()):
            if not head_config.weight:
                continue  # Do not instantiate zero-weight heads.

            head_factory = {
                "distogram": modules.DistogramHead,
                "structure_module": functools.partial(
                    folding.StructureModule, compute_loss=compute_loss
                ),
                "predicted_lddt": modules.PredictedLDDTHead,
                "predicted_aligned_error": modules.PredictedAlignedErrorHead,
                "experimentally_resolved": modules.ExperimentallyResolvedHead,
            }[head_name]
            heads[head_name] = (
                head_config,
                head_factory(head_config, self.global_config),
            )

        total_loss = 0.0
        ret = {}
        ret["representations"] = representations

        def loss(module, weight, ret, name, filter_ret=True):
            value = ret[name] if filter_ret else ret
            ret[name].update(module.loss(value, batch))
            return weight * ret[name]["loss"]

        for name, (head_config, module) in heads.items():
            # Skip PredictedLDDTHead and PredictedAlignedErrorHead until
            # StructureModule is executed.
            if name in ("predicted_lddt", "predicted_aligned_error"):
                continue
            else:
                ret[name] = module(representations, batch, is_training)
                if "representations" in ret[name]:
                    # Extra representations from the head. Used by the structure module
                    # to provide activations for the PredictedLDDTHead.
                    representations.update(ret[name].pop("representations"))
            if compute_loss:
                total_loss += loss(module, head_config.weight, ret, name)

        if self.config.heads.get("predicted_lddt.weight", 0.0):
            # Add PredictedLDDTHead after StructureModule executes.
            name = "predicted_lddt"
            # Feed all previous results to give access to structure_module result.
            head_config, module = heads[name]
            ret[name] = module(representations, batch, is_training)
            if compute_loss:
                total_loss += loss(
                    module, head_config.weight, ret, name, filter_ret=False
                )

        if "predicted_aligned_error" in self.config.heads and self.config.heads.get(
            "predicted_aligned_error.weight", 0.0
        ):
            # Add PredictedAlignedErrorHead after StructureModule executes.
            name = "predicted_aligned_error"
            # Feed all previous results to give access to structure_module result.
            head_config, module = heads[name]

            ret[name] = module(representations, batch, is_training)
            if compute_loss:
                total_loss += loss(
                    module, head_config.weight, ret, name, filter_ret=False
                )

        if compute_loss:
            return ret, total_loss
        return ret


class PLMformerIteration(hk.Module):
    """Single iteration (block) of PLMformer stack."""

    def __init__(self, config, global_config, name="plmformer_iteration"):
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config

    def __call__(self, activations, masks, is_training=True, safe_key=None):
        """Builds PLMformerIteration module.

        Arguments:
          activations: Dictionary containing activations:
            * 'single': single activations, shape [1, N_res, c_m].
            * 'pair': pair activations, shape [N_res, N_res, c_z].
          masks: Dictionary of masks:
            * 'single': single mask, shape [1, N_res].
            * 'pair': pair mask, shape [N_res, N_res].
          is_training: Whether the module is in training mode.
          safe_key: prng.SafeKey encapsulating rng key.

        Returns:
          Outputs, same shape/type as act.
        """
        c = self.config
        gc = self.global_config

        single_act, pair_act = activations["single"], activations["pair"]

        if safe_key is None:
            safe_key = prng.SafeKey(hk.next_rng_key())

        single_mask, pair_mask = masks["single"], masks["pair"]

        dropout_wrapper_fn = functools.partial(
            modules.dropout_wrapper, is_training=is_training, global_config=gc
        )

        safe_key, *sub_keys = safe_key.split(10)
        sub_keys = iter(sub_keys)

        outer_module = modules.OuterStackMean(
            config=c.outer_stack_mean,
            global_config=self.global_config,
            num_output_channel=int(pair_act.shape[-1]),
            name="outer_stack_mean",
        )

        pair_act = dropout_wrapper_fn(
            outer_module,
            single_act,
            single_mask,
            safe_key=next(sub_keys),
            output_act=pair_act,
        )

        single_row_atten_config = c.msa_row_attention_with_pair_bias

        single_act = dropout_wrapper_fn(
            modules.MSARowAttentionWithPairBias(
                single_row_atten_config,
                gc,
                name="single_row_attention_with_pair_bias",
            ),
            single_act,
            single_mask,
            safe_key=next(sub_keys),
            pair_act=pair_act,
        )

        single_act = dropout_wrapper_fn(
            modules.Transition(c.msa_transition, gc, name="single_transition"),
            single_act,
            single_mask,
            safe_key=next(sub_keys),
        )

        pair_act = dropout_wrapper_fn(
            modules.TriangleMultiplication(
                c.triangle_multiplication_outgoing,
                gc,
                name="triangle_multiplication_outgoing",
            ),
            pair_act,
            pair_mask,
            safe_key=next(sub_keys),
        )
        pair_act = dropout_wrapper_fn(
            modules.TriangleMultiplication(
                c.triangle_multiplication_incoming,
                gc,
                name="triangle_multiplication_incoming",
            ),
            pair_act,
            pair_mask,
            safe_key=next(sub_keys),
        )

        if not c.no_triangle_attention:
            pair_act = dropout_wrapper_fn(
                modules.TriangleAttention(
                    c.triangle_attention_starting_node,
                    gc,
                    name="triangle_attention_starting_node",
                ),
                pair_act,
                pair_mask,
                safe_key=next(sub_keys),
            )
            pair_act = dropout_wrapper_fn(
                modules.TriangleAttention(
                    c.triangle_attention_ending_node,
                    gc,
                    name="triangle_attention_ending_node",
                ),
                pair_act,
                pair_mask,
                safe_key=next(sub_keys),
            )

        pair_act = dropout_wrapper_fn(
            modules.Transition(c.pair_transition, gc, name="pair_transition"),
            pair_act,
            pair_mask,
            safe_key=next(sub_keys),
        )

        return {"single": single_act, "pair": pair_act}


class EmbeddingsAndPLMformer(hk.Module):
    """Embeds the input data and runs PLMformer.

    Produces the single and pair representations.
    """

    def __init__(self, config, global_config, name="plmformer"):
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config

    def __call__(self, batch, is_training, safe_key=None):

        c = self.config
        gc = self.global_config

        if safe_key is None:
            safe_key = prng.SafeKey(hk.next_rng_key())

        # Embed single sequences.
        # Slice PLM embeddings to the crop size expected in the PLMformer.
        start_index = batch["residue_index"][0] - batch["residue_index_plm"][0]
        crop_size = len(batch["residue_index"])

        plm_embeddings = batch["embeddings"]
        if c.use_weighted_embeddings_for_single_channel:
            embedding_weights = hk.get_parameter(
                "embedding_weights",
                shape=(plm_embeddings.shape[0],),
                init=hk.initializers.Constant(1),
            )
            embedding_weights = jax.nn.softmax(embedding_weights)
            plm_embeddings = (embedding_weights[:, None, None] * plm_embeddings).sum(0)

        plm_embeddings = jax.lax.dynamic_slice_in_dim(
            plm_embeddings,
            start_index=start_index,
            slice_size=crop_size,
            axis=0,
        )

        plm_embeddings = hk.Sequential(
            [
                common_modules.Linear(
                    2056, initializer="relu", name="preprocess_1d_layer1"
                ),
                jax.nn.relu,
                common_modules.Linear(
                    2056, initializer="relu", name="preprocess_1d_layer2"
                ),
                jax.nn.relu,
                common_modules.Linear(c.single_channel, name="preprocess_1d_layer3"),
            ]
        )(plm_embeddings)

        single_activations = jnp.expand_dims(plm_embeddings, axis=0)
        mask_1d = jnp.expand_dims(batch["seq_mask"], axis=0)

        if c.use_out_plm_attention_for_pair_init:
            pair_embeddings = batch["plm_attn_weights"]
            pair_embeddings = jax.lax.dynamic_slice_in_dim(
                pair_embeddings,
                start_index=start_index,
                slice_size=crop_size,
                axis=-1,
            )
            pair_embeddings = jax.lax.dynamic_slice_in_dim(
                pair_embeddings,
                start_index=start_index,
                slice_size=crop_size,
                axis=-2,
            )
            pair_embeddings = jnp.expand_dims(pair_embeddings, -1)
            pair_activations = common_modules.Linear(
                c.pair_channel, name="pair_input_proj"
            )(pair_embeddings)
        else:
            left_single = common_modules.Linear(c.pair_channel, name="left_single")(
                plm_embeddings
            )
            right_single = common_modules.Linear(c.pair_channel, name="right_single")(
                plm_embeddings
            )
            pair_activations = left_single[:, None] + right_single[None]

        mask_2d = batch["seq_mask"][:, None] * batch["seq_mask"][None, :]

        # Inject previous outputs for recycling.
        if c.recycle_pos:
            prev_pseudo_beta = modules.pseudo_beta_fn(
                batch["aatype"], batch["prev_pos"], None
            )
            dgram = modules.dgram_from_positions(
                prev_pseudo_beta, **self.config.prev_pos
            )
            pair_activations += common_modules.Linear(
                c.pair_channel, name="prev_pos_linear"
            )(dgram)

        if c.recycle_features:
            prev_single = hk.LayerNorm(
                axis=[-1],
                create_scale=True,
                create_offset=True,
                name="prev_single_norm",
            )(batch["prev_single"])
            single_activations = single_activations.at[0].add(prev_single)

            pair_activations += hk.LayerNorm(
                axis=[-1], create_scale=True, create_offset=True, name="prev_pair_norm"
            )(batch["prev_pair"])

        # Relative position encoding.
        if c.max_relative_feature:
            # Add one-hot-encoded clipped residue distances to the pair activations.
            pos = batch["residue_index"]
            offset = pos[:, None] - pos[None, :]
            rel_pos = jax.nn.one_hot(
                jnp.clip(
                    offset + c.max_relative_feature,
                    a_min=0,
                    a_max=2 * c.max_relative_feature,
                ),
                2 * c.max_relative_feature + 1,
            )
            pair_activations += common_modules.Linear(
                c.pair_channel, name="pair_activations"
            )(rel_pos)

        plmformer_input = {
            "single": single_activations,
            "pair": pair_activations,
        }

        plmformer_masks = {
            "single": mask_1d,
            "pair": mask_2d,
        }

        # Main trunk of the network.
        plmformer_iteration = PLMformerIteration(
            c.evoformer, gc, name="plmformer_iteration"
        )

        def plmformer_fn(x):
            act, safe_key = x
            safe_key, safe_subkey = safe_key.split()
            plmformer_output = plmformer_iteration(
                activations=act,
                masks=plmformer_masks,
                is_training=is_training,
                safe_key=safe_subkey,
            )
            return (plmformer_output, safe_key)

        if gc.use_remat:
            plmformer_fn = hk.remat(plmformer_fn)

        plmformer_stack = layer_stack.layer_stack(c.evoformer_num_block)(plmformer_fn)
        plmformer_output, safe_key = plmformer_stack((plmformer_input, safe_key))

        single_activations = plmformer_output["single"]
        pair_activations = plmformer_output["pair"]

        single_activations = common_modules.Linear(
            c.single_channel, name="single_activations"
        )(single_activations[0])

        output = {
            "single": single_activations,
            "pair": pair_activations,
        }

        return output


class PLMEmbed(hk.Module):
    """Extract protein language model embeddings for the input sequence."""

    def __init__(self, config, name="plmembed"):
        super().__init__(name=name)
        self.config = config

    def __call__(self, batch):
        """Builds PLMembed module.

        Arguments:
          batch: Dictionary of batch features.  Required are:
            'aatype': [N_res_pLM] input sequence as array of amino acid tokens (batch["aatype_plm"]).
            'seq_mask': [N_res_pLM] binary mask of input (batch["seq_mask_plm"]).

        Returns:
          batch: Updated dictionary of batch features, containing batch['embeddings'] with shape
            [N_res_pLM, config.embed_dim]."""
        sequence, mask = batch["aatype_plm"], batch["seq_mask_plm"]

        # Generate the TransformerLMConfig and tokenizer.
        config, tokenizer = esm_jax.get_config_and_tokenizer(self.config, "model")

        # Convert the sequence from AF index tokens to the LM index tokens.
        af_tokens_to_plm = jnp.zeros(tokenizer.vocabulary_size, dtype=jnp.int32)
        for k, af_idx in residue_constants.restype_order_with_x_and_gap.items():
            af_tokens_to_plm = af_tokens_to_plm.at[af_idx].set(tokenizer.token_to_id(k))
        sequence_plm_tokens = af_tokens_to_plm[sequence]

        # Add pad tokens to masked positions.
        sequence_plm_tokens = jnp.where(
            mask,
            sequence_plm_tokens,
            jnp.ones_like(sequence_plm_tokens, dtype=jnp.int32)
            * tokenizer.pad_token_id,
        )

        # Generate embeddings.
        outs = TransformerLM(config)(
            sequence_plm_tokens,
            save_embeddings=self.config.return_all_embeddings,
            save_attention_weights=self.config.return_all_attention_weights,
        )

        if self.config.return_all_embeddings:
            batch["embeddings"] = jnp.concatenate(
                [v for k, v in outs.items() if "embeddings_" in k], 0
            )[None]
        else:
            batch["embeddings"] = outs[f"embeddings_{config.num_layers}"]

        if self.config.return_all_attention_weights:
            batch["plm_attn_weights"] = outs["attn_weights"]

        return batch
