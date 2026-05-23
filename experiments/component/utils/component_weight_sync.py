"""Cross-backend weight sync for component-level experiments.

Components are real framework modules (Megatron / MindFormers), so parameter
names differ significantly between PTA and MSA. This module uses heuristic
name mapping plus fallback exact-match to bridge the gap.
"""
import numpy as np
from pathlib import Path


# Component-type specific name mappings (PTA name -> MSA name)
_COMPONENT_NAME_MAPS = {
    "embedding_layer": {
        "word_embeddings.weight": "word_embeddings.embedding_table",
        "position_embeddings.weight": "position_embeddings.embedding_table",
        "tokentype_embeddings.weight": "tokentype_embeddings.embedding_table",
    },
    "self_attention_block": {
        # Megatron query_key_value -> MindFormers qkv_proj (varies by version)
        "query_key_value.weight": "qkv_proj.weight",
        "query_key_value.bias": "qkv_proj.bias",
        "dense.weight": "proj.weight",
        "dense.bias": "proj.bias",
        # Alternative naming
        "q_proj.weight": "q_proj.weight",
        "k_proj.weight": "k_proj.weight",
        "v_proj.weight": "v_proj.weight",
        "o_proj.weight": "o_proj.weight",
    },
    "ffn_block": {
        "mlp.w1.weight": "w1.weight",
        "mlp.w2.weight": "w2.weight",
        "mlp.w3.weight": "w3.weight",
        "mlp.gate_proj.weight": "gate_proj.weight",
        "mlp.up_proj.weight": "up_proj.weight",
        "mlp.down_proj.weight": "down_proj.weight",
        # Megatron standard naming
        "dense_h_to_4h.weight": "dense_h_to_4h.weight",
        "dense_4h_to_h.weight": "dense_4h_to_h.weight",
    },
    "decoder_block": {
        # Self-attention sub-module
        "self_attention.query_key_value.weight": "self_attention.qkv_proj.weight",
        "self_attention.query_key_value.bias": "self_attention.qkv_proj.bias",
        "self_attention.dense.weight": "self_attention.proj.weight",
        "self_attention.dense.bias": "self_attention.proj.bias",
        # MLP sub-module
        "mlp.dense_h_to_4h.weight": "mlp.dense_h_to_4h.weight",
        "mlp.dense_4h_to_h.weight": "mlp.dense_4h_to_h.weight",
        "mlp.w1.weight": "mlp.w1.weight",
        "mlp.w2.weight": "mlp.w2.weight",
        "mlp.w3.weight": "mlp.w3.weight",
        # LayerNorm / RMSNorm
        "input_layernorm.weight": "input_layernorm.weight",
        "input_layernorm.bias": "input_layernorm.bias",
        "post_attention_layernorm.weight": "post_attention_layernorm.weight",
        "post_attention_layernorm.bias": "post_attention_layernorm.bias",
        "input_norm.weight": "input_norm.weight",
        "input_norm.bias": "input_norm.bias",
        "post_attention_norm.weight": "post_attention_norm.weight",
        "post_attention_norm.bias": "post_attention_norm.bias",
    },
    "output_layer": {
        "weight": "weight",
        "bias": "bias",
    },
    "moe_ffn_block": {
        # Same as ffn_block
        "mlp.w1.weight": "w1.weight",
        "mlp.w2.weight": "w2.weight",
        "mlp.w3.weight": "w3.weight",
        "mlp.gate_proj.weight": "gate_proj.weight",
        "mlp.up_proj.weight": "up_proj.weight",
        "mlp.down_proj.weight": "down_proj.weight",
    },
    "mla_self_attention_block": {
        # MLA specific projections
        "q_down_proj.weight": "q_down_proj.weight",
        "q_up_proj.weight": "q_up_proj.weight",
        "kv_down_proj.weight": "kv_down_proj.weight",
        "k_up_proj.weight": "k_up_proj.weight",
        "v_up_proj.weight": "v_up_proj.weight",
        "dense.weight": "proj.weight",
    },
}


def _get_param_dict(module, backend: str) -> dict:
    """Extract parameters as a dict {name: numpy_array}."""
    params = {}
    if backend == "pta":
        for name, param in module.named_parameters():
            params[name] = param.detach().cpu().numpy()
    elif backend == "msa":
        for param in module.get_parameters():
            params[param.name] = param.asnumpy()
    return params


# Global replacement rules applied to all PTA param names
_GLOBAL_REPLACEMENTS = [
    ("word_embeddings.weight", "word_embeddings.embedding_table"),
    ("position_embeddings.weight", "position_embeddings.embedding_table"),
    ("tokentype_embeddings.weight", "tokentype_embeddings.embedding_table"),
    ("query_key_value", "qkv_proj"),
    ("self_attention.dense", "self_attention.proj"),
    (".dense.", ".proj."),
    (".dense_weight", ".proj_weight"),
    (".dense_bias", ".proj_bias"),
]


def _apply_name_mapping(pta_name: str, comp_name: str) -> str:
    """Map PTA param name to MSA param name using component-specific rules."""
    name_map = _COMPONENT_NAME_MAPS.get(comp_name, {})
    # Direct lookup
    if pta_name in name_map:
        return name_map[pta_name]
    # Global heuristic replacements
    msa_name = pta_name
    for old, new in _GLOBAL_REPLACEMENTS:
        msa_name = msa_name.replace(old, new)
    if msa_name != pta_name:
        return msa_name
    # Fallback: exact match
    return pta_name


def save_pta_component_weights(pta_comp, comp_name: str, iteration: int, out_dir: Path):
    """Save PTA component parameters to .npz file."""
    params = _get_param_dict(pta_comp, "pta")
    if not params:
        return
    comp_dir = out_dir / comp_name
    comp_dir.mkdir(parents=True, exist_ok=True)
    np.savez(comp_dir / f"iter_{iteration:03d}_pta_weights.npz", **params)


def set_msa_component_weights_from_pta(msa_comp, comp_name: str, iteration: int, out_dir: Path):
    """Load PTA weights from .npz and set them on MSA component.

    Returns (num_matched, num_unmatched) tuple.
    """
    import mindspore as ms
    weight_path = out_dir / comp_name / f"iter_{iteration:03d}_pta_weights.npz"
    if not weight_path.exists():
        return 0, 0

    data = np.load(weight_path)
    msa_params = {p.name: p for p in msa_comp.get_parameters()}
    matched = 0
    unmatched = 0

    for pta_name, w in data.items():
        msa_name = _apply_name_mapping(pta_name, comp_name)
        msa_param = msa_params.get(msa_name)
        if msa_param is not None:
            msa_param.set_data(ms.Tensor(w))
            matched += 1
        else:
            # Try exact match as last resort
            msa_param = msa_params.get(pta_name)
            if msa_param is not None:
                msa_param.set_data(ms.Tensor(w))
                matched += 1
            else:
                unmatched += 1
                # Log unmatched for debugging
                print(f"    [weight sync] unmatched: PTA '{pta_name}' -> MSA '{msa_name}' not found")

    total = matched + unmatched
    if unmatched > 0:
        available = list(msa_params.keys())
        print(f"    [weight sync] {comp_name} iter {iteration}: {matched} matched, {unmatched} unmatched")
        print(f"    [weight sync] available MSA params: {available[:10]}...")
        if total > 0 and unmatched / total > 0.5:
            print(f"    [weight sync] WARNING: >50% params unmatched. "
                  f"This often indicates a structural mismatch (e.g., MoE/MLA fallback to standard module). "
                  f"RQ1 diff for this component may not be meaningful.")

    return matched, unmatched


def sync_component_weights_same_backend(comp_a, comp_b, backend: str):
    """Copy weights from comp_a to comp_b within the same backend (RQ2 baseline->perturbed)."""
    if backend == "pta":
        for (_, p_src), (_, p_dst) in zip(comp_a.named_parameters(), comp_b.named_parameters()):
            p_dst.data.copy_(p_src.data)
    elif backend == "msa":
        for p_src, p_dst in zip(comp_a.get_parameters(), comp_b.get_parameters()):
            p_dst.set_data(p_src.data)
