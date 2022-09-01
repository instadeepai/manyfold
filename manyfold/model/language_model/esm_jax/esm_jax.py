from typing import Dict, Tuple

from manyfold.model.language_model.models.models import TransformerLMConfig
from manyfold.model.language_model.utils.tokenizers import FixedSizeBasicTokenizer


def get_config_and_tokenizer(
    hyperparam_dict: Dict,
    model_name: str,
) -> Tuple[TransformerLMConfig, FixedSizeBasicTokenizer]:
    hyperparam_dict = hyperparam_dict[model_name]
    tokens = hyperparam_dict["tokens"]

    unk_token = "<unk>"
    pad_token = "<pad>"
    mask_token = "<mask>"
    cls_token = "<cls>"
    eos_token = "<eos>"
    extra_special_tokens = []
    if "<null_1>" in tokens:
        extra_special_tokens.append("<null_1>")
    if "<null_0>" in tokens:
        extra_special_tokens.append("<null_0>")
    if "<sep>" in tokens:
        extra_special_tokens.append("<sep>")

    special_tokens = [unk_token, pad_token, mask_token, cls_token, eos_token]
    special_tokens = special_tokens + extra_special_tokens

    standard_tokens = list(set(tokens).difference(set(special_tokens)))

    tokenizer = FixedSizeBasicTokenizer(
        fixed_length=1024,
        standard_tokens=standard_tokens,
        unk_token=unk_token,
        pad_token=pad_token,
        mask_token=mask_token,
        class_token=cls_token,
        eos_token=eos_token,
        prepend_cls_token=True,
        append_eos_token=False,
        extra_special_tokens=extra_special_tokens,
        tokens_to_ids={tok: i for i, tok in enumerate(tokens)},
    )

    config = TransformerLMConfig(
        alphabet_size=tokenizer.vocabulary_size,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        class_token_id=tokenizer.class_token_id,
        eos_token_id=tokenizer.eos_token_id,
        prepend_cls_token=True,
        append_eos_token=False,
        max_positions=1024,
        emb_layer_norm_before=hyperparam_dict["emb_layer_norm_before"],
        roberta_lm_head=hyperparam_dict["roberta_lm_head"],
        add_bias_kv=hyperparam_dict["add_bias_kv"],
        learned_positional_embedding=hyperparam_dict["learned_positional_embedding"],
        attention_heads=hyperparam_dict["attention_heads"],
        embed_dim=hyperparam_dict["embed_dim"],
        ffn_embed_dim=hyperparam_dict["ffn_embed_dim"],
        num_layers=hyperparam_dict["num_layers"],
        token_dropout=hyperparam_dict["token_dropout"],
        masking_ratio=0.15,
        masking_prob=0.8,
        embed_scale=hyperparam_dict["embed_scale"],
    )

    return config, tokenizer
