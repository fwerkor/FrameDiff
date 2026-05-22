"""Cross-backend deterministic weight initialization.

Uses numpy as the bridge to ensure PTA and MSA operators have identical weights.
"""
import numpy as np
from pathlib import Path


# PTA parameter name -> MSA parameter name mapping
_PARAM_NAME_MAP = {
    "embedding": {"weight": "embedding_table"},
    "layernorm": {"weight": "gamma", "bias": "beta"},
    "rmsnorm": {"weight": "gamma"},
}


def _hash_seed(base: str) -> int:
    return int(abs(hash(base)) % (2 ** 31))


def sync_weights(pta_op, msa_op, op_name: str, iteration: int):
    """Copy PTA weights into MSA operator using numpy as bridge.

    For each parameter in the PTA operator, find the corresponding MSA parameter
    by name (with mapping if needed) and set it to the same numpy-generated value.
    """
    name_map = _PARAM_NAME_MAP.get(op_name, {})

    # Collect MSA params by name
    msa_params = {p.name: p for p in msa_op.get_parameters()}

    for pta_name, pta_param in pta_op.named_parameters():
        msa_name = name_map.get(pta_name, pta_name)
        msa_param = msa_params.get(msa_name)
        if msa_param is None:
            continue

        # Use identical seed for both backends
        seed = _hash_seed(f"{op_name}_{pta_name}_{iteration}")
        np.random.seed(seed)
        w = np.random.randn(*pta_param.shape).astype(np.float32)

        # Set PTA weight
        import torch
        pta_param.data.copy_(torch.from_numpy(w))

        # Set MSA weight
        import mindspore as ms
        msa_param.set_data(ms.Tensor(w))


def save_pta_weights(pta_op, op_name: str, iteration: int, out_dir: Path):
    """Save PTA operator weights to .npz file so MSA can load them later."""
    weights = {}
    for name, param in pta_op.named_parameters():
        weights[name] = param.detach().cpu().numpy()
    weight_dir = out_dir / op_name
    weight_dir.mkdir(parents=True, exist_ok=True)
    np.savez(weight_dir / f"iter_{iteration:03d}_pta_weights.npz", **weights)


def set_msa_weights_from_pta(msa_op, op_name: str, iteration: int, out_dir: Path):
    """Load PTA weights from .npz file and set them on MSA operator."""
    import mindspore as ms
    weight_path = out_dir / op_name / f"iter_{iteration:03d}_pta_weights.npz"
    if not weight_path.exists():
        return False
    data = np.load(weight_path)
    name_map = _PARAM_NAME_MAP.get(op_name, {})
    msa_params = {p.name: p for p in msa_op.get_parameters()}
    for pta_name, w in data.items():
        msa_name = name_map.get(pta_name, pta_name)
        msa_param = msa_params.get(msa_name)
        if msa_param is not None:
            msa_param.set_data(ms.Tensor(w))
    return True
