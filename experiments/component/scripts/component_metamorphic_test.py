#!/usr/bin/env python3
"""RQ2: Component-level metamorphic testing (single backend).

This script ONLY generates inputs, adds perturbation, runs components,
and saves baseline/perturbed output tensors.
Metrics computation is done separately by analyze_rq2.py.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.config_loader import get_config, reset_config
from common.tensor_manager import TensorManager
from common.tensor_io import save_tensor, to_torch

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from component_registry import COMPONENT_REGISTRY, _get_pta_transformer_config, _get_msa_transformer_config


def add_uniform_perturbation(tensor, sigma):
    """每个数值加同一个固定的单向扰动 sigma."""
    if tensor.dtype in (torch.int64, torch.int32):
        return (tensor.float() + sigma).round().to(tensor.dtype)
    return tensor + sigma


def _get_hidden_size(config_path):
    try:
        cfg = _get_pta_transformer_config(config_path)
        return cfg.hidden_size
    except Exception:
        return 1024


def _prepare_inputs(comp_name, backend, x, config_path):
    if comp_name == "embedding_layer":
        seq_len, batch = x.shape
        position_ids = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0).expand(batch, -1).t()
        return (x, position_ids)

    if comp_name in ("self_attention_block", "decoder_block", "mla_self_attention_block"):
        seq_len, batch, hidden_size = x.shape
        attention_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        attention_mask = attention_mask.expand(batch, 1, seq_len, seq_len)
        if backend == "pta":
            try:
                from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
                cfg = _get_pta_transformer_config(config_path)
                kv_channels = getattr(cfg, "kv_channels", cfg.hidden_size // cfg.num_attention_heads)
                rotary = RotaryEmbedding(kv_channels=kv_channels, rotary_percent=1.0)
                rotary_pos_emb = rotary.forward(max_seq_len=seq_len)
                return (x, attention_mask, rotary_pos_emb)
            except Exception:
                return (x, attention_mask)
        else:
            # MSA inference modules use rotary_pos_cos / rotary_pos_sin
            try:
                from mindformers.parallel_core.inference.base_models.common.embeddings.rope_utils import get_rope
                import mindspore.common.dtype as mstype
                cfg = _get_msa_transformer_config(config_path)
                rotary = get_rope(
                    config=cfg,
                    hidden_dim=cfg.kv_channels,
                    rotary_percent=cfg.partial_rotary_factor,
                    rotary_base=cfg.rotary_base,
                    rotary_dtype=getattr(mstype, cfg.rotary_dtype, mstype.float32),
                    position_embedding_type=cfg.position_embedding_type,
                    original_max_position_embeddings=cfg.max_position_embeddings,
                    rotary_cos_format=cfg.rotary_cos_format,
                )
                # Get prefill cos/sin for full sequence length
                import mindspore as ms
                cos_cache, sin_cache = rotary.get_cos_sin_for_prefill()
                # Slice to actual sequence length
                cos = cos_cache[:seq_len]
                sin = sin_cache[:seq_len]
                return (x, attention_mask, cos, sin)
            except Exception:
                return (x, attention_mask)

    return (x,)


def build_component(comp_name: str, backend: str, config_path: str | None = None):
    """Build a component instance for the given backend."""
    entry = COMPONENT_REGISTRY[comp_name]
    builder = entry[backend]
    return builder(config_path)


def run_component_instance(comp, comp_name: str, backend: str, x, config_path: str | None = None):
    """Run a pre-built component instance with the given input."""
    inputs = _prepare_inputs(comp_name, backend, x, config_path)

    if backend == "pta":
        device = torch.device("npu" if torch.npu.is_available() else "cpu")
        comp = comp.to(device)
        inputs = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in inputs)
        with torch.no_grad():
            out = comp(*inputs)
        return out
    else:
        import mindspore as ms
        from mindspore import Tensor as MSTensor
        ctx = ms.get_context("device_target")
        if ctx is None:
            ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
        inputs = tuple(MSTensor(t.numpy()) if isinstance(t, torch.Tensor) else t for t in inputs)
        out = comp(*inputs)
        return out


def run_rq2(backend_filter: str = "both", config_path: str | None = None):
    reset_config()
    cfg = get_config("component")
    out_dir = Path(cfg["experiment"]["output_dir"]) / "rq2_meta"
    num_iter = cfg["experiment"]["num_iterations"]
    seed = cfg["experiment"]["seed"]
    sigmas = [float(s) for s in cfg["perturbation"]["sigmas"]]
    tm = TensorManager(seed=seed, device="cpu")

    hidden_size = _get_hidden_size(config_path)
    backends = ["pta", "msa"] if backend_filter == "both" else [backend_filter]

    for comp_name in COMPONENT_REGISTRY.keys():
        print(f"\n=== Component: {comp_name} ===")

        for sigma in sigmas:
            print(f"  sigma={sigma}")
            comp_dir = out_dir / comp_name / f"sigma_{sigma}"

            for i in range(num_iter):
                if comp_name == "embedding_layer":
                    x = tm.generate_input_ids(f"{comp_name}_input", i, (32, 2))
                else:
                    x = tm.generate(f"{comp_name}_input", i, (32, 2, hidden_size))
                x_perturbed = add_uniform_perturbation(x, sigma)

                for backend in backends:
                    try:
                        # Build component once, run twice (baseline + perturbed)
                        comp = build_component(comp_name, backend, config_path)
                        baseline = run_component_instance(comp, comp_name, backend, x, config_path)
                        perturbed = run_component_instance(comp, comp_name, backend, x_perturbed, config_path)

                        baseline_t = to_torch(baseline)
                        perturbed_t = to_torch(perturbed)
                        if isinstance(baseline_t, (tuple, list)):
                            baseline_t = baseline_t[0]
                        if isinstance(perturbed_t, (tuple, list)):
                            perturbed_t = perturbed_t[0]

                        save_tensor(baseline_t, comp_dir / f"iter_{i:03d}_{backend}_baseline.pt")
                        save_tensor(perturbed_t, comp_dir / f"iter_{i:03d}_{backend}_perturbed.pt")
                        print(f"    {backend} iter {i}: saved")
                    except Exception as e:
                        print(f"    {backend} iter {i} error: {e}")

    print(f"\nAll tensors saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["pta", "msa", "both"], default="both")
    parser.add_argument("--config", type=str, default=None, help="Path to model config YAML")
    args = parser.parse_args()
    run_rq2(backend_filter=args.backend, config_path=args.config)
