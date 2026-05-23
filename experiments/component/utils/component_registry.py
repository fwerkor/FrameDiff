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


# ---------------------------------------------------------------------------
# MSA compatibility: remove old mindformers from PYTHONPATH so pip-installed
# mindformers 1.9.0 (with parallel_core.inference modules) is used instead.
# ---------------------------------------------------------------------------
if '/mindformers' in sys.path:
    sys.path.remove('/mindformers')


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
# MSA environment initialization
# ---------------------------------------------------------------------------
_MSA_INIT_DONE = False


def init_msa_env():
    """Initialize MindSpore context for MSA inference. Idempotent."""
    global _MSA_INIT_DONE
    if _MSA_INIT_DONE:
        return
    import mindspore as ms
    # Use PYNATIVE_MODE so inference modules (which don't use Morph) work.
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
    _MSA_INIT_DONE = True


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def _get_pta_transformer_config(config_path: str | None = None):
    """Build Megatron TransformerConfig from YAML."""
    from ruamel.yaml import YAML
    from megatron.core.transformer.transformer_layer import TransformerConfig

    yaml = YAML()
    if config_path is None:
        config_path = str(Path(__file__).parent.parent.parent / "model_config" / "qwen2.yaml")

    with open(config_path, "r") as f:
        raw = yaml.load(f)

    cfg_dict = dict(raw.get("TransformerConfig", raw.get("MLATransformerConfig", {})))
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
        config_path = str(Path(__file__).parent.parent.parent / "model_config" / "qwen2.yaml")

    with open(config_path, "r") as f:
        raw = yaml.load(f)

    cfg_dict = dict(raw.get("TransformerConfig", raw.get("MLATransformerConfig", {})))
    valid_fields = set(TransformerConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in cfg_dict.items() if k in valid_fields}

    # Ensure critical fields have valid values for inference modules
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
    if filtered.get("ffn_hidden_size", 0) in (0, None):
        filtered["ffn_hidden_size"] = filtered.get("hidden_size", 1024) * 4
    # Inference MLP only supports 'silu' activation; use silu + gated_linear_unit
    # for SwiGLU-equivalent behavior.
    if filtered.get("hidden_act") in (0, None):
        filtered["hidden_act"] = "silu"
    if filtered.get("gated_linear_unit") in (0, None):
        filtered["gated_linear_unit"] = True
    if filtered.get("normalization") in (0, None):
        filtered["normalization"] = "RMSNorm"
    if filtered.get("layernorm_epsilon") in (0, None):
        filtered["layernorm_epsilon"] = 1e-6
    # Inference DotProductAttention supports fp32; FlashAttention requires fp16/bf16.
    # Default to False for cross-backend fp32 consistency.
    if filtered.get("use_flash_attention") in (0, None):
        filtered["use_flash_attention"] = False
    if filtered.get("params_dtype") in (0, None):
        filtered["params_dtype"] = "float32"
    if filtered.get("compute_dtype") in (0, None):
        filtered["compute_dtype"] = "float32"
    if filtered.get("rotary_base") in (0, None):
        filtered["rotary_base"] = 10000
    if filtered.get("partial_rotary_factor") in (0, None):
        filtered["partial_rotary_factor"] = 1.0
    if filtered.get("position_embedding_type") in (0, None):
        filtered["position_embedding_type"] = "rope"
    if filtered.get("rotary_dtype") in (0, None):
        filtered["rotary_dtype"] = "float32"
    if filtered.get("rotary_cos_format") in (0, None):
        filtered["rotary_cos_format"] = "rotate_half"
    if filtered.get("max_position_embeddings") in (0, None):
        filtered["max_position_embeddings"] = filtered.get("seq_length", 2048)
    if filtered.get("qk_layernorm") in (0, None):
        filtered["qk_layernorm"] = False
    if filtered.get("add_qkv_bias") in (0, None):
        filtered["add_qkv_bias"] = False
    if filtered.get("add_bias_linear") in (0, None):
        filtered["add_bias_linear"] = False
    if filtered.get("moe_layer_freq") in (0, None):
        filtered["moe_layer_freq"] = 1
    if filtered.get("first_k_dense_replace") in (0, None):
        filtered["first_k_dense_replace"] = 0
    if filtered.get("num_moe_experts") in (0, None):
        filtered["num_moe_experts"] = 1
    if filtered.get("moe_router_topk") in (0, None):
        filtered["moe_router_topk"] = 1
    if filtered.get("moe_router_pre_softmax") in (0, None):
        filtered["moe_router_pre_softmax"] = True
    if filtered.get("multi_latent_attention") in (0, None):
        filtered["multi_latent_attention"] = False
    if filtered.get("use_fused_mla") in (0, None):
        filtered["use_fused_mla"] = False
    if filtered.get("use_alltoall") in (0, None):
        filtered["use_alltoall"] = False
    if filtered.get("sandwich_norm") in (0, None):
        filtered["sandwich_norm"] = False
    if filtered.get("attn_reduce_scatter") in (0, None):
        filtered["attn_reduce_scatter"] = False
    if filtered.get("attn_allgather") in (0, None):
        filtered["attn_allgather"] = False
    if filtered.get("attn_allreduce") in (0, None):
        filtered["attn_allreduce"] = True
    if filtered.get("ffn_allgather") in (0, None):
        filtered["ffn_allgather"] = False
    if filtered.get("ffn_allreduce") in (0, None):
        filtered["ffn_allreduce"] = True

    return TransformerConfig(**filtered)


def _get_total_config(config_path: str | None = None) -> dict:
    """Read full YAML including extra_config."""
    from ruamel.yaml import YAML

    yaml = YAML()
    if config_path is None:
        config_path = str(Path(__file__).parent.parent.parent / "model_config" / "qwen2.yaml")

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
    init_msa_env()
    from mindformers.parallel_core.inference.base_models.common.embeddings.language_model_embedding import LanguageModelEmbedding
    from mindformers.parallel_core.process_group_config import default_model_comm_pgs

    total_cfg = _get_total_config(config_path)
    transformer_cfg = _get_msa_transformer_config(config_path)
    extra = total_cfg.get("extra_config", {})

    return LanguageModelEmbedding(
        config=transformer_cfg,
        vocab_size=extra.get("padded_vocab_size", extra.get("vocab_size", 32000)),
        max_sequence_length=extra.get("seq_length", 2048),
        position_embedding_type=extra.get("position_embedding_type", "rope"),
        model_comm_pgs=default_model_comm_pgs,
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
    init_msa_env()
    from mindformers.parallel_core.inference.transformer.transformer_block import TransformerBlock
    from mindformers.parallel_core.inference.base_models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from mindformers.parallel_core.process_group_config import default_model_comm_pgs

    transformer_cfg = _get_msa_transformer_config(config_path)
    total_cfg = _get_total_config(config_path)
    spec_cfg = total_cfg.get("get_gpt_layer_local_spec", {})

    # Align with real model usage in lm_fullnet/utils/runtime/core/graph.py:
    # pre_process=True, post_process=True, post_layer_norm=True.
    # For inference modules, force num_experts=None to avoid MoE complexity
    # (MoE grouped_gemm=False is not supported in inference).
    return TransformerBlock(
        config=transformer_cfg,
        spec=get_gpt_layer_local_spec(
            num_experts=None,
            moe_grouped_gemm=spec_cfg.get("moe_grouped_gemm", False),
            qk_layernorm=spec_cfg.get("qk_layernorm", False),
            gated_linear_unit=spec_cfg.get("gated_linear_unit", True),
            multi_latent_attention=spec_cfg.get("multi_latent_attention", False),
            normalization=spec_cfg.get("normalization", "RMSNorm"),
            qk_l2_norm=spec_cfg.get("qk_l2_norm", False),
            # Prefer transformer_cfg for use_flash_attention (inference DotProductAttention
            # supports fp32, while FlashAttention on Ascend requires fp16/bf16).
            use_flash_attention=getattr(transformer_cfg, "use_flash_attention", spec_cfg.get("use_flash_attention", False)),
            sandwich_norm=spec_cfg.get("sandwich_norm", False),
            use_alltoall=spec_cfg.get("use_alltoall", False),
            use_fused_mla=spec_cfg.get("use_fused_mla", False),
        ),
        post_layer_norm=True,
        pre_process=True,
        post_process=True,
        model_comm_pgs=default_model_comm_pgs,
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
    init_msa_env()
    from mindformers.parallel_core.inference.tensor_parallel.layers import ColumnParallelLinear
    from mindformers.parallel_core.process_group_config import default_model_comm_pgs

    transformer_cfg = _get_msa_transformer_config(config_path)
    total_cfg = _get_total_config(config_path)
    extra = total_cfg.get("extra_config", {})
    vocab_size = extra.get("padded_vocab_size", extra.get("vocab_size", 32000))

    return ColumnParallelLinear(
        input_size=transformer_cfg.hidden_size,
        output_size=vocab_size,
        config=transformer_cfg,
        init_method=transformer_cfg.init_method,
        bias=False,
        gather_output=False,
        skip_weight_param_allocation=False,
        transpose_b=True,
        compute_dtype=transformer_cfg.compute_dtype,
        tp_group=default_model_comm_pgs.tp,
    )


def build_pta_moe_ffn_block(config_path: str | None = None):
    """MoE FFN Block: build with num_experts > 1 if config does not already enable MoE."""
    init_pta_env()
    import mindspeed.megatron_adaptor
    from megatron.core.transformer.transformer_block import TransformerBlock
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

    transformer_cfg = _get_pta_transformer_config(config_path)
    total_cfg = _get_total_config(config_path)
    spec_cfg = total_cfg.get("get_gpt_layer_local_spec", {})

    # Ensure MoE is enabled in config
    num_moe_experts = getattr(transformer_cfg, "num_moe_experts", 1)
    if num_moe_experts <= 1:
        transformer_cfg.num_moe_experts = 8
    moe_topk = min(getattr(transformer_cfg, "moe_router_topk", 2), transformer_cfg.num_moe_experts)
    transformer_cfg.moe_router_topk = moe_topk

    block = TransformerBlock(
        config=transformer_cfg,
        spec=get_gpt_layer_local_spec(
            num_experts=transformer_cfg.num_moe_experts,
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
    return block.layers[0].mlp


def build_msa_moe_ffn_block(config_path: str | None = None):
    """MoE FFN Block: attempt true MoE; fallback to standard FFN if inference MoE is unavailable."""
    init_msa_env()
    from mindformers.parallel_core.inference.transformer.transformer_block import TransformerBlock
    from mindformers.parallel_core.inference.base_models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from mindformers.parallel_core.process_group_config import default_model_comm_pgs

    transformer_cfg = _get_msa_transformer_config(config_path)
    total_cfg = _get_total_config(config_path)
    spec_cfg = total_cfg.get("get_gpt_layer_local_spec", {})

    # Try to use num_experts from config; default to 8 if not set
    num_experts = spec_cfg.get("num_experts", None)
    if num_experts is None or num_experts <= 1:
        num_experts = getattr(transformer_cfg, "num_moe_experts", 1)
        if num_experts <= 1:
            num_experts = 8

    try:
        block = TransformerBlock(
            config=transformer_cfg,
            spec=get_gpt_layer_local_spec(
                num_experts=num_experts,
                moe_grouped_gemm=spec_cfg.get("moe_grouped_gemm", False),
                qk_layernorm=spec_cfg.get("qk_layernorm", False),
                gated_linear_unit=spec_cfg.get("gated_linear_unit", True),
                multi_latent_attention=spec_cfg.get("multi_latent_attention", False),
                normalization=spec_cfg.get("normalization", "RMSNorm"),
                qk_l2_norm=spec_cfg.get("qk_l2_norm", False),
                use_flash_attention=getattr(transformer_cfg, "use_flash_attention", spec_cfg.get("use_flash_attention", False)),
                sandwich_norm=spec_cfg.get("sandwich_norm", False),
                use_alltoall=spec_cfg.get("use_alltoall", False),
                use_fused_mla=spec_cfg.get("use_fused_mla", False),
            ),
            post_layer_norm=True,
            pre_process=True,
            post_process=True,
            model_comm_pgs=default_model_comm_pgs,
        )
        return block.layers[0].mlp
    except Exception as e:
        # Inference MoE may not be supported; fallback to standard FFN
        print(f"    [MSA MoE] build failed ({e}), falling back to standard FFN")
        block = build_msa_transformer_block(config_path)
        comp = block.layers[0].mlp
        comp._is_fallback = True
        return comp


def build_pta_mla_self_attention_block(config_path: str | None = None):
    """MLA Self-Attention: DeepSeekV3 specific. Force multi_latent_attention=True."""
    init_pta_env()
    import mindspeed.megatron_adaptor
    from megatron.core.transformer.transformer_block import TransformerBlock
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

    transformer_cfg = _get_pta_transformer_config(config_path)
    total_cfg = _get_total_config(config_path)
    spec_cfg = total_cfg.get("get_gpt_layer_local_spec", {})

    try:
        block = TransformerBlock(
            config=transformer_cfg,
            spec=get_gpt_layer_local_spec(
                num_experts=spec_cfg.get("num_experts", None),
                moe_grouped_gemm=spec_cfg.get("moe_grouped_gemm", False),
                qk_layernorm=spec_cfg.get("qk_layernorm", False),
                multi_latent_attention=True,
                fp8=spec_cfg.get("fp8", None),
                moe_use_legacy_grouped_gemm=spec_cfg.get("moe_use_legacy_grouped_gemm", False),
                normalization=spec_cfg.get("normalization", "RMSNorm"),
                qk_l2_norm=spec_cfg.get("qk_l2_norm", False),
            ),
            pre_process=True,
            post_process=True,
        )
        return block.layers[0].self_attention
    except Exception as e:
        print(f"    [PTA MLA] build failed ({e}), falling back to standard SA")
        block = build_pta_transformer_block(config_path)
        comp = block.layers[0].self_attention
        comp._is_fallback = True
        return comp


def build_msa_mla_self_attention_block(config_path: str | None = None):
    """MLA Self-Attention: force multi_latent_attention=True; fallback to standard SA on error."""
    init_msa_env()
    from mindformers.parallel_core.inference.transformer.transformer_block import TransformerBlock
    from mindformers.parallel_core.inference.base_models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from mindformers.parallel_core.process_group_config import default_model_comm_pgs

    transformer_cfg = _get_msa_transformer_config(config_path)
    total_cfg = _get_total_config(config_path)
    spec_cfg = total_cfg.get("get_gpt_layer_local_spec", {})

    try:
        block = TransformerBlock(
            config=transformer_cfg,
            spec=get_gpt_layer_local_spec(
                num_experts=None,
                moe_grouped_gemm=spec_cfg.get("moe_grouped_gemm", False),
                qk_layernorm=spec_cfg.get("qk_layernorm", False),
                gated_linear_unit=spec_cfg.get("gated_linear_unit", True),
                multi_latent_attention=True,
                normalization=spec_cfg.get("normalization", "RMSNorm"),
                qk_l2_norm=spec_cfg.get("qk_l2_norm", False),
                use_flash_attention=getattr(transformer_cfg, "use_flash_attention", spec_cfg.get("use_flash_attention", False)),
                sandwich_norm=spec_cfg.get("sandwich_norm", False),
                use_alltoall=spec_cfg.get("use_alltoall", False),
                use_fused_mla=spec_cfg.get("use_fused_mla", False),
            ),
            post_layer_norm=True,
            pre_process=True,
            post_process=True,
            model_comm_pgs=default_model_comm_pgs,
        )
        return block.layers[0].self_attention
    except Exception as e:
        print(f"    [MSA MLA] build failed ({e}), falling back to standard SA")
        block = build_msa_transformer_block(config_path)
        comp = block.layers[0].self_attention
        comp._is_fallback = True
        return comp


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
