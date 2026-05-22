"""Component registry: 7 components with PTA/MSA backends."""
import os
import sys
import types
from pathlib import Path
from typing import Callable, Any


# ---------------------------------------------------------------------------
# NPU compatibility: mock torch.cuda before any Megatron imports
# ---------------------------------------------------------------------------
def _mock_cuda_for_npu():
    import torch
    _npu_is_available = getattr(torch, 'npu', None) and getattr(torch.npu, 'is_available', None)
    _npu_current_device = getattr(torch, 'npu', None) and getattr(torch.npu, 'current_device', None)
    if hasattr(torch, 'npu'):
        torch.cuda = torch.npu
    if hasattr(torch, 'cuda'):
        torch.cuda._is_compiled = lambda: True
        torch.cuda.is_available = lambda: _npu_is_available() if _npu_is_available else False
        torch.cuda.current_device = lambda: _npu_current_device() if (_npu_is_available and _npu_is_available()) else 0
        if not hasattr(torch.cuda, 'nvtx'):
            nvtx_mod = types.ModuleType("nvtx")
            nvtx_mod.range_push = lambda *a, **k: None
            nvtx_mod.range_pop = lambda *a, **k: None
            nvtx_mod.mark = lambda *a, **k: None
            torch.cuda.nvtx = nvtx_mod


_mock_cuda_for_npu()


def _is_pta():
    try:
        import torch_npu
        return True
    except Exception:
        return False


def _is_msa():
    try:
        import mindspore
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

BACKEND = "pta" if "torch" in sys.modules or _is_pta() else "msa"


# ---------------------------------------------------------------------------
# PTA environment initialization (distributed + seed)
# ---------------------------------------------------------------------------
_PTA_INIT_DONE = False


def init_pta_env():
    """Initialize PTA distributed environment for Megatron-Core. Idempotent."""
    global _PTA_INIT_DONE
    if _PTA_INIT_DONE:
        return

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        for port in range(29501, 29600):
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.bind(("127.0.0.1", port))
                os.environ["MASTER_PORT"] = str(port)
                s.close()
                break
            except OSError:
                s.close()
                continue
    os.environ.setdefault("PYTORCH_NPU_FORCE_FALLBACK", "1")

    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if not dist.is_initialized():
        dist.init_process_group(backend="hccl", rank=0, world_size=1)

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )
    try:
        model_parallel_cuda_manual_seed(42)
    except Exception:
        pass
    _PTA_INIT_DONE = True


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def _get_pta_transformer_config(config_path: str | None = None):
    """Build Megatron TransformerConfig from YAML."""
    from ruamel.yaml import YAML
    from megatron.core.transformer.transformer_layer import TransformerConfig

    yaml = YAML()
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "frame_diff_common" / "model_configs" / "qwen2.yaml")

    with open(config_path, "r") as f:
        raw = yaml.load(f)

    cfg_dict = dict(raw.get("TransformerConfig", {}))
    if cfg_dict.get("init_method") == "torch.nn.init.xavier_uniform_":
        cfg_dict["init_method"] = __import__("torch").nn.init.xavier_uniform_
    # Remove fields not supported by Megatron TransformerConfig
    for key in ("position_embedding_type",):
        cfg_dict.pop(key, None)
    cfg = TransformerConfig(**cfg_dict)

    # Patch num_moe_experts to avoid None % int in MoELayer
    if getattr(cfg, "num_moe_experts", None) is None:
        cfg.num_moe_experts = 1
    # Patch moe_router_topk to avoid topk > num_experts
    if getattr(cfg, "num_moe_experts", 1) == 1 and getattr(cfg, "moe_router_topk", 0) != 1:
        cfg.moe_router_topk = 1
    return cfg


def _get_msa_transformer_config(config_path: str | None = None):
    """Build MindFormers TransformerConfig from YAML."""
    from ruamel.yaml import YAML
    from mindformers.parallel_core.transformer_config import TransformerConfig

    yaml = YAML()
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "frame_diff_common" / "model_configs" / "qwen2.yaml")

    with open(config_path, "r") as f:
        raw = yaml.load(f)

    cfg_dict = dict(raw.get("TransformerConfig", {}))
    valid_fields = set(TransformerConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in cfg_dict.items() if k in valid_fields}
    if filtered.get("num_attention_heads", 0) in (0, None):
        filtered["num_attention_heads"] = 1
    if filtered.get("num_query_groups") in (0, None):
        filtered["num_query_groups"] = filtered.get("num_attention_heads", 1)
    if filtered.get("tensor_model_parallel_size", 0) in (0, None):
        filtered["tensor_model_parallel_size"] = 1
    if filtered.get("pipeline_model_parallel_size", 0) in (0, None):
        filtered["pipeline_model_parallel_size"] = 1
    if filtered.get("num_layers", 0) in (0, None):
        filtered["num_layers"] = 1
    if filtered.get("kv_channels") in (0, None):
        hs = filtered.get("hidden_size", 0)
        nh = filtered.get("num_attention_heads", 1)
        filtered["kv_channels"] = hs // nh if nh else hs
    return TransformerConfig(**filtered)


def _get_total_config(config_path: str | None = None) -> dict:
    """Read full YAML including extra_config."""
    from ruamel.yaml import YAML

    yaml = YAML()
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "frame_diff_common" / "model_configs" / "qwen2.yaml")

    with open(config_path, "r") as f:
        raw = yaml.load(f)

    return dict(raw)


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------

def build_pta_embedding_layer(config_path: str | None = None):
    init_pta_env()
    from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding

    total_cfg = _get_total_config(config_path)
    transformer_cfg = _get_pta_transformer_config(config_path)
    extra = total_cfg.get("extra_config", {})

    return LanguageModelEmbedding(
        config=transformer_cfg,
        vocab_size=extra.get("padded_vocab_size", extra.get("vocab_size", 32000)),
        max_sequence_length=extra.get("seq_length", 2048),
        position_embedding_type=extra.get("position_embedding_type", "rope"),
    )


def build_msa_embedding_layer(config_path: str | None = None):
    from mindformers.parallel_core.training_graph.base_models.common.embeddings.language_model_embedding import LanguageModelEmbedding

    total_cfg = _get_total_config(config_path)
    transformer_cfg = _get_msa_transformer_config(config_path)
    extra = total_cfg.get("extra_config", {})

    return LanguageModelEmbedding(
        config=transformer_cfg,
        vocab_size=extra.get("padded_vocab_size", extra.get("vocab_size", 32000)),
        max_sequence_length=extra.get("seq_length", 2048),
        position_embedding_type=extra.get("position_embedding_type", "rope"),
    )


def build_pta_transformer_block(config_path: str | None = None):
    init_pta_env()
    import mindspeed.megatron_adaptor
    from megatron.core.transformer.transformer_block import TransformerBlock
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

    transformer_cfg = _get_pta_transformer_config(config_path)
    total_cfg = _get_total_config(config_path)
    spec_cfg = total_cfg.get("get_gpt_layer_local_spec", {})

    return TransformerBlock(
        config=transformer_cfg,
        spec=get_gpt_layer_local_spec(
            num_experts=spec_cfg.get("num_experts", None),
            moe_grouped_gemm=spec_cfg.get("moe_grouped_gemm", False),
            qk_layernorm=spec_cfg.get("qk_layernorm", False),
            multi_latent_attention=spec_cfg.get("multi_latent_attention", False),
            fp8=spec_cfg.get("fp8", None),
            moe_use_legacy_grouped_gemm=spec_cfg.get("moe_use_legacy_grouped_gemm", False),
            normalization=spec_cfg.get("normalization", "RMSNorm"),
            qk_l2_norm=spec_cfg.get("qk_l2_norm", False),
        ),
        pre_process=True,
        post_process=True,
    )


def build_msa_transformer_block(config_path: str | None = None):
    from mindformers.parallel_core.training_graph.transformer.transformer_block import TransformerBlock
    from mindformers.parallel_core.training_graph.base_models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

    transformer_cfg = _get_msa_transformer_config(config_path)
    total_cfg = _get_total_config(config_path)
    spec_cfg = total_cfg.get("get_gpt_layer_local_spec", {})

    return TransformerBlock(
        config=transformer_cfg,
        spec=get_gpt_layer_local_spec(
            num_experts=spec_cfg.get("num_experts", None),
            moe_grouped_gemm=spec_cfg.get("moe_grouped_gemm", False),
            qk_layernorm=spec_cfg.get("qk_layernorm", False),
            multi_latent_attention=spec_cfg.get("multi_latent_attention", False),
            fp8=spec_cfg.get("fp8", None),
            moe_use_legacy_grouped_gemm=spec_cfg.get("moe_use_legacy_grouped_gemm", False),
            normalization=spec_cfg.get("normalization", "RMSNorm"),
            qk_l2_norm=spec_cfg.get("qk_l2_norm", False),
        ),
        pre_process=False,
        post_process=False,
    )


def build_pta_self_attention_block(config_path: str | None = None):
    block = build_pta_transformer_block(config_path)
    return block.layers[0].self_attention


def build_msa_self_attention_block(config_path: str | None = None):
    block = build_msa_transformer_block(config_path)
    return block.layers[0].self_attention


def build_pta_ffn_block(config_path: str | None = None):
    block = build_pta_transformer_block(config_path)
    return block.layers[0].mlp


def build_msa_ffn_block(config_path: str | None = None):
    block = build_msa_transformer_block(config_path)
    return block.layers[0].mlp


def build_pta_decoder_block(config_path: str | None = None):
    block = build_pta_transformer_block(config_path)
    return block.layers[0]


def build_msa_decoder_block(config_path: str | None = None):
    block = build_msa_transformer_block(config_path)
    return block.layers[0]


def build_pta_output_layer(config_path: str | None = None):
    init_pta_env()
    from megatron.core import tensor_parallel

    transformer_cfg = _get_pta_transformer_config(config_path)
    total_cfg = _get_total_config(config_path)
    extra = total_cfg.get("extra_config", {})
    vocab_size = extra.get("padded_vocab_size", extra.get("vocab_size", 32000))

    return tensor_parallel.ColumnParallelLinear(
        transformer_cfg.hidden_size,
        vocab_size,
        config=transformer_cfg,
        init_method=transformer_cfg.init_method,
        bias=False,
        skip_bias_add=False,
        gather_output=False,
        skip_weight_param_allocation=False,
    )


def build_msa_output_layer(config_path: str | None = None):
    from mindformers.parallel_core.inference.tensor_parallel.layers import ColumnParallelLinear

    transformer_cfg = _get_msa_transformer_config(config_path)
    total_cfg = _get_total_config(config_path)
    extra = total_cfg.get("extra_config", {})
    vocab_size = extra.get("padded_vocab_size", extra.get("vocab_size", 32000))

    kwargs = dict(
        config=transformer_cfg,
        init_method=transformer_cfg.init_method,
        bias=False,
        skip_bias_add=False,
        gather_output=False,
        skip_weight_param_allocation=False,
    )
    try:
        return ColumnParallelLinear(transformer_cfg.hidden_size, vocab_size, **kwargs)
    except TypeError:
        kwargs.pop("grad_output_buffer", None)
        kwargs.pop("embedding_activation_buffer", None)
        return ColumnParallelLinear(transformer_cfg.hidden_size, vocab_size, **kwargs)


def build_pta_moe_ffn_block(config_path: str | None = None):
    """MoE FFN Block: same as FFN block for MoE models."""
    return build_pta_ffn_block(config_path)


def build_msa_moe_ffn_block(config_path: str | None = None):
    return build_msa_ffn_block(config_path)


def build_pta_mla_self_attention_block(config_path: str | None = None):
    """MLA Self-Attention: DeepSeekV3 specific. Fallback to standard SA."""
    return build_pta_self_attention_block(config_path)


def build_msa_mla_self_attention_block(config_path: str | None = None):
    return build_msa_self_attention_block(config_path)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

COMPONENT_REGISTRY: dict[str, dict[str, Callable]] = {
    "embedding_layer": {
        "pta": build_pta_embedding_layer,
        "msa": build_msa_embedding_layer,
    },
    "self_attention_block": {
        "pta": build_pta_self_attention_block,
        "msa": build_msa_self_attention_block,
    },
    "ffn_block": {
        "pta": build_pta_ffn_block,
        "msa": build_msa_ffn_block,
    },
    "decoder_block": {
        "pta": build_pta_decoder_block,
        "msa": build_msa_decoder_block,
    },
    "output_layer": {
        "pta": build_pta_output_layer,
        "msa": build_msa_output_layer,
    },
    "moe_ffn_block": {
        "pta": build_pta_moe_ffn_block,
        "msa": build_msa_moe_ffn_block,
    },
    "mla_self_attention_block": {
        "pta": build_pta_mla_self_attention_block,
        "msa": build_msa_mla_self_attention_block,
    },
}


def get_component_builder(name: str, backend: str):
    if name not in COMPONENT_REGISTRY:
        raise ValueError(f"Unknown component: {name}. Available: {list(COMPONENT_REGISTRY.keys())}")
    if backend not in COMPONENT_REGISTRY[name]:
        raise ValueError(f"Unknown backend: {backend}. Available: {list(COMPONENT_REGISTRY[name].keys())}")
    return COMPONENT_REGISTRY[name][backend]
