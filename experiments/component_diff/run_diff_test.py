#!/usr/bin/env python3
"""RQ1: Component-level differential testing (PTA vs MSA).

This script ONLY generates inputs, runs components, and saves output tensors.
Metrics computation is done separately by analyze_rq1.py.
"""
import argparse
from pathlib import Path

import torch

from experiments.frame_diff_common.config_loader import get_config
from experiments.frame_diff_common.tensor_manager import TensorManager
from experiments.frame_diff_common.tensor_io import save_tensor, to_torch
from experiments.component_diff.component_registry import COMPONENT_REGISTRY, _get_pta_transformer_config


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
            return (x, attention_mask)

    return (x,)


def run_component(comp_name: str, backend: str, x, config_path: str | None = None):
    entry = COMPONENT_REGISTRY[comp_name]
    builder = entry[backend]
    comp = builder(config_path)
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
            ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
        inputs = tuple(MSTensor(t.numpy()) if isinstance(t, torch.Tensor) else t for t in inputs)
        out = comp(*inputs)
        return out


def run_rq1(backend_filter: str = "both"):
    cfg = get_config("component")
    out_dir = Path(cfg["experiment"]["output_dir"]) / "rq1_diff"
    num_iter = cfg["experiment"]["num_iterations"]
    seed = cfg["experiment"]["seed"]
    tm = TensorManager(seed=seed, device="cpu")

    config_path = None
    hidden_size = _get_hidden_size(config_path)
    backends = ["pta", "msa"] if backend_filter == "both" else [backend_filter]

    for comp_name in COMPONENT_REGISTRY.keys():
        print(f"\n=== Component: {comp_name} ===")
        comp_dir = out_dir / comp_name

        for i in range(num_iter):
            if comp_name == "embedding_layer":
                x = tm.generate_input_ids(f"{comp_name}_input", i, (32, 2))
            else:
                x = tm.generate(f"{comp_name}_input", i, (32, 2, hidden_size))

            for backend in backends:
                try:
                    out = run_component(comp_name, backend, x)
                    out_t = to_torch(out)
                    if isinstance(out_t, (tuple, list)):
                        out_t = out_t[0]
                    save_tensor(out_t, comp_dir / f"iter_{i:03d}_{backend}_output.pt")
                    print(f"  iter {i} {backend}: saved")
                except Exception as e:
                    print(f"  iter {i} {backend} error: {e}")

    print(f"\nAll tensors saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["pta", "msa", "both"], default="both")
    args = parser.parse_args()
    run_rq1(backend_filter=args.backend)
