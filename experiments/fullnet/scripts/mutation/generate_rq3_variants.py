#!/usr/bin/env python3
"""Generate prepared RQ3 fullnet variants from model configs.

The generated layout is:

    experiments/mutated_config/<model>/<variant>/
        mutating.json
        mutated_config.yaml

Each mutating.json keeps the legacy numeric node records required by Graph.load
and adds RQ3 metadata/runtime_overrides at the top level.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
MODEL_CONFIG_ROOT = EXPERIMENTS_ROOT / "model_config"
MUTATED_CONFIG_ROOT = EXPERIMENTS_ROOT / "mutated_config"


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def active_sections(model_doc: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    if "MLATransformerConfig" in model_doc:
        section_name = "MLATransformerConfig"
    else:
        section_name = "TransformerConfig"
    return (
        section_name,
        copy.deepcopy(model_doc.get(section_name, {}) or {}),
        copy.deepcopy(model_doc.get("extra_config", {}) or {}),
        copy.deepcopy(model_doc.get("get_gpt_layer_local_spec", {}) or {}),
    )


def get_yaml_config(doc: dict[str, Any]) -> dict[str, Any]:
    return doc.setdefault("base_config", {}).setdefault("config", {})


def set_yaml_config_key(doc: dict[str, Any], key: str, value: Any) -> None:
    get_yaml_config(doc)[key] = value


def set_yaml_base_key(doc: dict[str, Any], key: str, value: Any) -> None:
    doc.setdefault("base_config", {})[key] = value


def set_input_hidden(doc: dict[str, Any], hidden_size: int) -> None:
    input_cfg = doc.setdefault("input", {})
    shape = input_cfg.get("shape")
    if isinstance(shape, list) and shape:
        shape[-1] = hidden_size


def config_section_name(config_doc: dict[str, Any]) -> str:
    if "MLATransformerConfig" in config_doc:
        return "MLATransformerConfig"
    return "TransformerConfig"


def set_json_transformer_key(mutating_doc: dict[str, Any], key: str, value: Any) -> None:
    for node_id, record in mutating_doc.items():
        if not str(node_id).isdigit() or not isinstance(record, dict):
            continue
        after = record.get("after")
        if not isinstance(after, dict):
            continue
        section = config_section_name(after)
        section_cfg = after.setdefault(section, {})
        if isinstance(section_cfg, dict):
            section_cfg[key] = value
            record["mutated"] = True


def set_json_extra_key(mutating_doc: dict[str, Any], key: str, value: Any) -> None:
    for node_id, record in mutating_doc.items():
        if not str(node_id).isdigit() or not isinstance(record, dict):
            continue
        after = record.get("after")
        if not isinstance(after, dict):
            continue
        extra_cfg = after.setdefault("extra_config", {})
        if isinstance(extra_cfg, dict):
            extra_cfg[key] = value
            record["mutated"] = True


def set_json_spec_key(mutating_doc: dict[str, Any], key: str, value: Any) -> None:
    for node_id, record in mutating_doc.items():
        if not str(node_id).isdigit() or not isinstance(record, dict):
            continue
        after = record.get("after")
        if not isinstance(after, dict):
            continue
        spec_cfg = after.setdefault("get_gpt_layer_local_spec", {})
        if isinstance(spec_cfg, dict):
            spec_cfg[key] = value
            record["mutated"] = True


def set_config_key(key: str, value: Any, *, extra_key: str | None = None, spec_key: str | None = None) -> Callable:
    def _apply(yaml_doc: dict[str, Any], mutating_doc: dict[str, Any]) -> None:
        set_yaml_config_key(yaml_doc, key, value)
        set_json_transformer_key(mutating_doc, key, value)
        if extra_key:
            set_json_extra_key(mutating_doc, extra_key, value)
        if spec_key:
            set_json_spec_key(mutating_doc, spec_key, value)

    return _apply


def set_base_key(key: str, value: Any, *, extra_key: str | None = None) -> Callable:
    def _apply(yaml_doc: dict[str, Any], mutating_doc: dict[str, Any]) -> None:
        set_yaml_base_key(yaml_doc, key, value)
        if extra_key:
            set_json_extra_key(mutating_doc, extra_key, value)

    return _apply


def set_yaml_extra_key(doc: dict[str, Any], key: str, value: Any) -> None:
    doc.setdefault("extra_config", {})[key] = value


def set_extra_config_key(key: str, value: Any) -> Callable:
    def _apply(yaml_doc: dict[str, Any], mutating_doc: dict[str, Any]) -> None:
        set_yaml_extra_key(yaml_doc, key, value)
        set_json_extra_key(mutating_doc, key, value)

    return _apply


def set_base_position_embedding(value: str) -> Callable:
    def _apply(yaml_doc: dict[str, Any], mutating_doc: dict[str, Any]) -> None:
        set_yaml_base_key(yaml_doc, "position_embedding_type", value)
        set_json_extra_key(mutating_doc, "position_embedding_type", value)

    return _apply


def round_down_multiple(value: int, factor: int) -> int:
    factor = max(1, factor)
    return max(factor, (value // factor) * factor)


def simple_before(features: dict[str, Any], key: str) -> Any:
    for source in ("yaml_config", "transformer", "extra", "spec"):
        cfg = features.get(source, {})
        if isinstance(cfg, dict) and key in cfg:
            return cfg[key]
    return None


class Variant:
    def __init__(
        self,
        name: str,
        category: str,
        scope: str,
        mutation_key: str,
        after: Any,
        *,
        apply: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
        runtime_overrides: dict[str, Any] | None = None,
        shape_preserving: bool | str = True,
        crash_risk: str = "low",
        notes: str = "",
        condition: Callable[[dict[str, Any]], bool] | None = None,
        before: Callable[[dict[str, Any]], Any] | Any = None,
    ) -> None:
        self.name = name
        self.category = category
        self.scope = scope
        self.mutation_key = mutation_key
        self.after = after
        self.apply = apply
        self.runtime_overrides = runtime_overrides or {}
        self.shape_preserving = shape_preserving
        self.crash_risk = crash_risk
        self.notes = notes
        self.condition = condition or (lambda _features: True)
        self.before = before

    def before_value(self, features: dict[str, Any]) -> Any:
        if callable(self.before):
            return self.before(features)
        if self.before is not None:
            return self.before
        return simple_before(features, self.mutation_key)


def default_overrides() -> dict[str, Any]:
    return {
        "env": {},
        "runtime_args": [],
        "launcher": {},
        "optimizer_env": {},
        "runtime_control": {},
    }


def runtime_args_variant(name: str, args: list[str], key: str, *, risk: str = "medium", notes: str = "") -> Variant:
    overrides = default_overrides()
    overrides["runtime_args"] = args
    return Variant(name, "runtime_optimization", "runtime_args", key, True, runtime_overrides=overrides, crash_risk=risk, notes=notes)


def launcher_variant(name: str, launcher: dict[str, Any], key: str, after: Any, *, risk: str = "medium", condition=None, notes: str = "") -> Variant:
    overrides = default_overrides()
    overrides["launcher"] = launcher
    return Variant(name, "parallel_strategy", "launcher", key, after, runtime_overrides=overrides, crash_risk=risk, condition=condition, notes=notes)


def optimizer_variant(name: str, env: dict[str, Any], key: str, after: Any) -> Variant:
    overrides = default_overrides()
    overrides["optimizer_env"] = env
    return Variant(name, "training_hyperparameter", "optimizer", key, after, runtime_overrides=overrides)


def env_variant(name: str, env: dict[str, Any], key: str, after: Any) -> Variant:
    overrides = default_overrides()
    overrides["env"] = env
    return Variant(name, "determinism_control", "env", key, after, runtime_overrides=overrides)


def runtime_control_variant(name: str, control: dict[str, Any], key: str, after: Any) -> Variant:
    overrides = default_overrides()
    overrides["runtime_control"] = control
    return Variant(name, "training_config", "runtime_control", key, after, runtime_overrides=overrides)


def config_optimizer_variant(
    name: str,
    env: dict[str, Any],
    key: str,
    after: Any,
    *,
    apply: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
    risk: str = "medium",
) -> Variant:
    overrides = default_overrides()
    overrides["optimizer_env"] = env
    return Variant(
        name,
        "training_config",
        "config",
        key,
        after,
        apply=apply,
        runtime_overrides=overrides,
        crash_risk=risk,
    )


def has_moe(features: dict[str, Any]) -> bool:
    cfg = features["transformer"]
    spec = features["spec"]
    extra = features["extra"]
    return any(
        as_int(source.get(key), 0) > 0
        for source in (cfg, spec, extra)
        for key in ("num_moe_experts", "num_experts", "expert_model_parallel_size")
    ) or any(key in cfg for key in ("moe_router_topk", "moe_router_load_balancing_type", "moe_aux_loss_coeff"))


def has_mla(features: dict[str, Any]) -> bool:
    return features["section_name"] == "MLATransformerConfig" or any(
        key in features["transformer"]
        for key in ("q_lora_rank", "kv_lora_rank", "qk_head_dim", "qk_pos_emb_head_dim", "v_head_dim")
    )


def has_swiglu(features: dict[str, Any]) -> bool:
    return any(
        key in source
        for source in (features["transformer"], features["extra"])
        for key in ("gated_linear_unit", "swiglu", "use_fused_swiglu", "bias_swiglu_fusion", "bias_activation_fusion", "use_fused_mlp")
    )


def model_is(name: str) -> Callable[[dict[str, Any]], bool]:
    return lambda features: features.get("model_name") == name


def never_generate(_features: dict[str, Any]) -> bool:
    return False


def tp_valid(tp: int) -> Callable[[dict[str, Any]], bool]:
    def _check(features: dict[str, Any]) -> bool:
        cfg = features["yaml_config"]
        for key in ("hidden_size", "ffn_hidden_size", "num_attention_heads", "num_query_groups"):
            value = as_int(cfg.get(key), 0)
            if value > 0 and value % tp != 0:
                return False
        return True

    return _check


def has_heads_groups(features: dict[str, Any]) -> bool:
    cfg = features["yaml_config"]
    return as_int(cfg.get("num_attention_heads"), 0) > 0 and as_int(cfg.get("num_query_groups"), 0) > 0


def query_groups_half(features: dict[str, Any]) -> int:
    groups = as_int(features["yaml_config"].get("num_query_groups"), 1)
    return max(1, groups // 2)


def query_groups_half_valid(features: dict[str, Any]) -> bool:
    heads = as_int(features["yaml_config"].get("num_attention_heads"), 0)
    value = query_groups_half(features)
    return heads > 0 and heads % value == 0


def attention_heads_half(features: dict[str, Any]) -> int:
    heads = as_int(features["yaml_config"].get("num_attention_heads"), 1)
    return max(1, heads // 2)


def attention_heads_half_valid(features: dict[str, Any]) -> bool:
    cfg = features["yaml_config"]
    new_heads = attention_heads_half(features)
    groups = as_int(cfg.get("num_query_groups"), new_heads)
    hidden = as_int(cfg.get("hidden_size"), 0)
    return new_heads > 0 and hidden % new_heads == 0 and groups <= new_heads and new_heads % max(1, groups) == 0


def hidden_scaled_apply(features: dict[str, Any]) -> Callable:
    cfg = features["yaml_config"]
    heads = max(1, as_int(cfg.get("num_attention_heads"), 1))
    new_hidden = round_down_multiple(int(as_int(cfg.get("hidden_size"), heads) * 0.75), heads)

    def _apply(yaml_doc: dict[str, Any], mutating_doc: dict[str, Any]) -> None:
        set_yaml_config_key(yaml_doc, "hidden_size", new_hidden)
        set_input_hidden(yaml_doc, new_hidden)
        set_json_transformer_key(mutating_doc, "hidden_size", new_hidden)

    return _apply


def ffn_scaled_apply(features: dict[str, Any]) -> Callable:
    cfg = features["yaml_config"]
    new_ffn = round_down_multiple(int(as_int(cfg.get("ffn_hidden_size"), 1024) * 0.75), 8)
    return set_config_key("ffn_hidden_size", new_ffn)


def build_variants(features: dict[str, Any]) -> list[Variant]:
    cfg = features["yaml_config"]
    heads = as_int(cfg.get("num_attention_heads"), 1)
    base_kv = max(1, as_int(cfg.get("kv_channels"), as_int(cfg.get("hidden_size"), heads) // max(1, heads)))
    variants: list[Variant] = [
        # Factor 1: runtime optimization mechanisms.
        runtime_args_variant("runtime_flash_attn_on", ["--use-flash-attn"], "use_flash_attn", risk="high"),
        runtime_args_variant("runtime_softmax_fp32_on", ["--attention-softmax-in-fp32"], "attention_softmax_in_fp32"),
        runtime_args_variant("runtime_masked_softmax_fusion_off", ["--no-masked-softmax-fusion"], "masked_softmax_fusion"),
        Variant("mlp_swiglu_fusion_on", "runtime_optimization", "config", "use_fused_swiglu", True, apply=set_config_key("use_fused_swiglu", True, extra_key="use_fused_swiglu"), condition=has_swiglu, crash_risk="medium", notes="may_be_suppressed_by_runtime_stabilizer"),
        Variant("mlp_swiglu_fusion_off", "runtime_optimization", "config", "use_fused_swiglu", False, apply=set_config_key("use_fused_swiglu", False, extra_key="use_fused_swiglu"), condition=has_swiglu, crash_risk="medium", notes="may_be_suppressed_by_runtime_stabilizer"),
        runtime_args_variant("runtime_use_fused_rmsnorm_on", ["--use-fused-rmsnorm"], "use_fused_rmsnorm", risk="medium"),
        runtime_args_variant("runtime_use_fused_rotary_pos_emb_on", ["--use-fused-rotary-pos-emb"], "use_fused_rotary_pos_emb", risk="medium"),
        runtime_args_variant("runtime_no_gradient_accumulation_fusion", ["--no-gradient-accumulation-fusion"], "no_gradient_accumulation_fusion", risk="medium"),
        Variant("moe_grouped_gemm_on", "runtime_optimization", "config", "moe_grouped_gemm", True, apply=set_config_key("moe_grouped_gemm", True, spec_key="moe_grouped_gemm"), condition=has_moe, crash_risk="high"),
        Variant("moe_use_fused_moe_token_permute_and_unpermute_on", "runtime_optimization", "config", "use_fused_moe_token_permute_and_unpermute", True, apply=set_config_key("use_fused_moe_token_permute_and_unpermute", True, extra_key="use_fused_moe_token_permute_and_unpermute"), condition=has_moe, crash_risk="high"),
        runtime_args_variant("runtime_recompute_activation_function_on", ["--recompute-activation-function"], "recompute_activation_function", risk="medium"),
        runtime_args_variant("runtime_recompute_granularity_full", ["--recompute-granularity", "full"], "recompute_granularity", risk="medium"),
        runtime_args_variant("runtime_recompute_method_uniform", ["--recompute-method", "uniform"], "recompute_method", risk="medium"),
        runtime_args_variant("runtime_recompute_num_layers_1", ["--recompute-num-layers", "1"], "recompute_num_layers", risk="medium"),

        # Factor 2: parallel strategies.
        launcher_variant("parallel_tp1", {"TARGET_TENSOR_PARALLEL_SIZE": 1}, "tensor_parallel_size", 1, risk="low", condition=tp_valid(1)),
        launcher_variant("parallel_tp2", {"TARGET_TENSOR_PARALLEL_SIZE": 2}, "tensor_parallel_size", 2, risk="medium", condition=tp_valid(2)),
        launcher_variant("parallel_tp4", {"TARGET_TENSOR_PARALLEL_SIZE": 4}, "tensor_parallel_size", 4, risk="high", condition=tp_valid(4)),
        launcher_variant("parallel_expert_parallel_2", {"TARGET_EXPERT_PARALLEL_SIZE": 2}, "expert_parallel_size", 2, risk="high", condition=has_moe),
        launcher_variant("parallel_pipeline_parallel_2", {"TARGET_PIPELINE_PARALLEL_SIZE": 2}, "pipeline_parallel_size", 2, risk="high", condition=lambda f: as_int(f["yaml_config"].get("num_layers"), 0) >= 2),
        launcher_variant("parallel_data_parallel_2", {"TARGET_TENSOR_PARALLEL_SIZE": 1, "TARGET_PIPELINE_PARALLEL_SIZE": 1, "TARGET_EXPERT_PARALLEL_SIZE": 1, "TARGET_CONTEXT_PARALLEL_SIZE": 1, "TARGET_WORLD_SIZE": 2, "TARGET_NPUS_PER_NODE": 2, "ENABLE_DATA_PARALLEL": True}, "data_parallel_size", 2, risk="high"),
        runtime_args_variant("runtime_sequence_parallel_on", ["--sequence-parallel"], "sequence_parallel", risk="medium"),
        launcher_variant("parallel_context_parallel_2", {"TARGET_CONTEXT_PARALLEL_SIZE": 2, "TARGET_WORLD_SIZE": 2, "TARGET_NPUS_PER_NODE": 2}, "context_parallel_size", 2, risk="high"),

        # Factor 3: training configuration.
        Variant("moe_aux_loss_coeff_0", "training_config", "config", "moe_aux_loss_coeff", 0, apply=set_config_key("moe_aux_loss_coeff", 0), condition=has_moe, crash_risk="medium"),
        Variant("moe_aux_loss_coeff_1e_2", "training_config", "config", "moe_aux_loss_coeff", 0.01, apply=set_config_key("moe_aux_loss_coeff", 0.01), condition=has_moe, crash_risk="medium"),
        Variant("config_moe_device_level_aux_loss_coeff_0_03", "training_config", "config", "moe_device_level_aux_loss_coeff", 0.03, apply=set_config_key("moe_device_level_aux_loss_coeff", 0.03), condition=has_moe, crash_risk="medium"),
        Variant("config_moe_comm_aux_loss_coeff_0_01", "training_config", "config", "moe_comm_aux_loss_coeff", 0.01, apply=set_config_key("moe_comm_aux_loss_coeff", 0.01), condition=has_moe, crash_risk="medium"),
        Variant("config_seq_aux_on", "training_config", "config", "seq_aux", True, apply=set_config_key("seq_aux", True), condition=has_moe, crash_risk="medium"),
        config_optimizer_variant("train_lr_1e_3", {"LMSV_RQ3_LR": "1e-3"}, "lr", 1e-3, apply=set_extra_config_key("lr", 1e-3), risk="medium"),
        config_optimizer_variant("train_weight_decay_0", {"LMSV_RQ3_WEIGHT_DECAY": "0"}, "weight_decay", 0, apply=set_extra_config_key("weight_decay", 0), risk="medium"),
        Variant("config_layernorm_eps_1e_7", "training_config", "config", "layernorm_epsilon", 1e-7, apply=set_config_key("layernorm_epsilon", 1e-7), crash_risk="medium"),

        # Factor 4: numerical precision control.
        runtime_args_variant("precision_bf16_on", ["--bf16"], "bf16", risk="medium"),
        runtime_args_variant("precision_fp16_on", ["--fp16"], "fp16", risk="high"),
        runtime_args_variant("precision_accumulate_grads_fp32_on", ["--accumulate-allreduce-grads-in-fp32"], "accumulate_allreduce_grads_in_fp32", risk="medium"),
        runtime_args_variant("precision_fp32_residual_on", ["--fp32-residual-connection"], "fp32_residual_connection", risk="medium"),
        runtime_args_variant("runtime_use_distributed_optimizer_on", ["--use-distributed-optimizer"], "use_distributed_optimizer", risk="high"),
        runtime_args_variant("runtime_reuse_fp32_param_on", ["--reuse-fp32-param"], "reuse_fp32_param", risk="high"),
        Variant("config_normalization_layernorm", "precision_control", "config", "normalization", "LayerNorm", apply=set_config_key("normalization", "LayerNorm", spec_key="normalization"), crash_risk="medium"),
        Variant("config_qk_layernorm_on", "precision_control", "config", "qk_layernorm", True, apply=set_config_key("qk_layernorm", True, spec_key="qk_layernorm"), crash_risk="medium"),
        Variant("config_embedding_multiplier_scale_78_38", "precision_control", "config", "embedding_multiplier_scale", 78.38, apply=set_config_key("embedding_multiplier_scale", 78.38), condition=model_is("grok1"), crash_risk="high", notes="Grok-1 numeric scaling variant"),
        Variant("config_output_multiplier_scale_0_57", "precision_control", "config", "output_multiplier_scale", 0.57, apply=set_config_key("output_multiplier_scale", 0.57), condition=model_is("grok1"), crash_risk="high", notes="Grok-1 numeric scaling variant"),

        # Factor 5: model structure configuration.
        Variant("config_first_k_dense_replace_0", "model_structure", "config", "first_k_dense_replace", 0, apply=set_config_key("first_k_dense_replace", 0), condition=has_moe, crash_risk="high"),
        Variant("config_moe_layer_freq_2", "model_structure", "config", "moe_layer_freq", 2, apply=set_config_key("moe_layer_freq", 2), condition=has_moe, crash_risk="high"),
        Variant("config_n_shared_experts_2", "model_structure", "config", "n_shared_experts", 2, apply=set_config_key("n_shared_experts", 2), condition=has_moe, crash_risk="high"),
        Variant("config_num_moe_experts_8", "model_structure", "config", "num_moe_experts", 8, apply=set_config_key("num_moe_experts", 8, spec_key="num_experts"), condition=has_moe, crash_risk="high"),
        Variant("config_moe_intermediate_size_1536", "model_structure", "config", "moe_intermediate_size", 1536, apply=set_config_key("moe_ffn_hidden_size", 1536), condition=has_moe, crash_risk="high", before=lambda f: simple_before(f, "moe_ffn_hidden_size")),
        Variant("moe_router_load_balancing_none", "model_structure", "config", "moe_router_load_balancing_type", "none", apply=set_config_key("moe_router_load_balancing_type", "none"), condition=has_moe, crash_risk="high"),
        Variant("moe_router_load_balancing_aux_loss", "model_structure", "config", "moe_router_load_balancing_type", "aux_loss", apply=set_config_key("moe_router_load_balancing_type", "aux_loss"), condition=has_moe, crash_risk="medium"),
        Variant("moe_router_pre_softmax_on", "model_structure", "config", "moe_router_pre_softmax", True, apply=set_config_key("moe_router_pre_softmax", True), condition=has_moe, crash_risk="medium"),
        Variant("moe_router_pre_softmax_off", "model_structure", "config", "moe_router_pre_softmax", False, apply=set_config_key("moe_router_pre_softmax", False), condition=has_moe, crash_risk="medium"),
        Variant("config_input_jitter_on", "model_structure", "config", "input_jitter", True, apply=set_config_key("input_jitter", True), condition=model_is("deepseekv3"), crash_risk="high", notes="DeepSeekV3 MoE router jitter variant"),
        Variant("linear_bias_on", "model_structure", "config", "add_bias_linear", True, apply=set_config_key("add_bias_linear", True), crash_risk="medium", notes="MoE models may force add_bias_linear=false"),
        Variant("linear_bias_off", "model_structure", "config", "add_bias_linear", False, apply=set_config_key("add_bias_linear", False), crash_risk="low"),
        Variant("attention_qkv_bias_on", "model_structure", "config", "add_qkv_bias", True, apply=set_config_key("add_qkv_bias", True), crash_risk="medium"),
        Variant("attention_qkv_bias_off", "model_structure", "config", "add_qkv_bias", False, apply=set_config_key("add_qkv_bias", False), crash_risk="medium"),
        Variant("config_position_embedding_learned", "model_structure", "config", "position_embedding_type", "learned_absolute", apply=set_base_position_embedding("learned_absolute"), condition=never_generate, crash_risk="high", notes="disabled: current script constraints only allow rope prepared variants"),
        Variant("mla_qk_head_dim_scaled_0_75", "model_structure", "config", "qk_head_dim", round_down_multiple(int(base_kv * 0.75), 8), apply=set_config_key("qk_head_dim", round_down_multiple(int(base_kv * 0.75), 8)), condition=has_mla, crash_risk="high", shape_preserving=False),
        Variant("mla_v_head_dim_scaled_0_75", "model_structure", "config", "v_head_dim", round_down_multiple(int(base_kv * 0.75), 8), apply=set_config_key("v_head_dim", round_down_multiple(int(base_kv * 0.75), 8)), condition=has_mla, crash_risk="high", shape_preserving=False),
        Variant("mla_q_lora_rank_scaled_0_5", "model_structure", "config", "q_lora_rank", 96, apply=set_config_key("q_lora_rank", 96), condition=has_mla, crash_risk="high", shape_preserving=False),
        Variant("mla_kv_lora_rank_scaled_0_5", "model_structure", "config", "kv_lora_rank", 32, apply=set_config_key("kv_lora_rank", 32), condition=has_mla, crash_risk="high", shape_preserving=False),
        Variant("mla_qk_rope_head_dim_128", "model_structure", "config", "qk_rope_head_dim", 128, apply=lambda y, j: (set_yaml_config_key(y, "qk_rope_head_dim", 128), set_yaml_config_key(y, "qk_head_dim", 128), set_json_transformer_key(j, "qk_rope_head_dim", 128), set_json_transformer_key(j, "qk_head_dim", 128)), condition=has_mla, crash_risk="high", shape_preserving=False),
        Variant("mla_qk_nope_head_dim_128", "model_structure", "config", "qk_nope_head_dim", 128, apply=lambda y, j: (set_yaml_config_key(y, "qk_nope_head_dim", 128), set_yaml_config_key(y, "qk_pos_emb_head_dim", 128), set_json_transformer_key(j, "qk_nope_head_dim", 128), set_json_transformer_key(j, "qk_pos_emb_head_dim", 128)), condition=has_mla, crash_risk="high", shape_preserving=False),
        Variant("mla_kv_channels_128", "model_structure", "config", "kv_channels", 128, apply=set_config_key("kv_channels", 128), condition=has_mla, crash_risk="high", shape_preserving=False),
        Variant("config_num_query_groups_1", "model_structure", "config", "num_query_groups", 1, apply=set_config_key("num_query_groups", 1), condition=has_heads_groups, crash_risk="medium"),
        Variant("config_num_query_groups_half", "model_structure", "config", "num_query_groups", query_groups_half(features), apply=set_config_key("num_query_groups", query_groups_half(features)), condition=query_groups_half_valid, crash_risk="medium"),
        Variant("config_num_query_groups_equal_heads", "model_structure", "config", "num_query_groups", heads, apply=set_config_key("num_query_groups", heads), condition=has_heads_groups, crash_risk="medium"),
        Variant("config_hidden_size_scaled_0_75", "model_structure", "config", "hidden_size", round_down_multiple(int(as_int(cfg.get("hidden_size"), heads) * 0.75), max(1, heads)), apply=hidden_scaled_apply(features), crash_risk="high", shape_preserving=False),
        Variant("config_ffn_hidden_size_scaled_0_75", "model_structure", "config", "ffn_hidden_size", round_down_multiple(int(as_int(cfg.get("ffn_hidden_size"), 1024) * 0.75), 8), apply=ffn_scaled_apply(features), crash_risk="high", shape_preserving=False),
        Variant("config_num_attention_heads_half", "model_structure", "config", "num_attention_heads", attention_heads_half(features), apply=set_config_key("num_attention_heads", attention_heads_half(features)), condition=attention_heads_half_valid, crash_risk="high", shape_preserving=False),
        Variant("mlp_gated_linear_unit_toggle", "model_structure", "config", "gated_linear_unit", not bool(cfg.get("gated_linear_unit", False)), apply=set_config_key("gated_linear_unit", not bool(cfg.get("gated_linear_unit", False))), condition=lambda f: "gated_linear_unit" in f["yaml_config"], crash_risk="high", shape_preserving=False),
    ]
    return variants

def annotate_mutating_doc(mutating_doc: dict[str, Any], variant: Variant, features: dict[str, Any]) -> None:
    overrides = copy.deepcopy(default_overrides())
    for key, value in variant.runtime_overrides.items():
        if key in overrides and isinstance(overrides[key], dict) and isinstance(value, dict):
            overrides[key].update(value)
        else:
            overrides[key] = copy.deepcopy(value)
    mutating_doc.update(
        {
            "variant_name": variant.name,
            "category": variant.category,
            "scope": variant.scope,
            "single_point": True,
            "shape_preserving": variant.shape_preserving,
            "crash_risk": variant.crash_risk,
            "mutation": {
                "key": variant.mutation_key,
                "before": variant.before_value(features),
                "after": variant.after,
            },
            "apply_to": ["pta", "msa"],
            "runtime_overrides": overrides,
            "notes": variant.notes,
        }
    )


def update_yaml_metadata(yaml_doc: dict[str, Any], variant: Variant, features: dict[str, Any]) -> None:
    metadata = yaml_doc.setdefault("metadata", {})
    metadata.update(
        {
            "variant_name": variant.name,
            "category": variant.category,
            "scope": variant.scope,
            "single_point": True,
            "shape_preserving": variant.shape_preserving,
            "crash_risk": variant.crash_risk,
            "mutation": {
                "key": variant.mutation_key,
                "before": variant.before_value(features),
                "after": variant.after,
            },
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "generated_by": "FrameDiff RQ3 fullnet prepared variant generator",
        }
    )


def copy_ancestor_canonical(model_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    ancestor_dir = model_dir / "ancestor"
    source_yaml = ancestor_dir / "mutated_config.yaml"
    source_json = ancestor_dir / "mutating.json"
    if not source_yaml.exists() or not source_json.exists():
        raise FileNotFoundError(
            f"{ancestor_dir} must contain canonical mutating.json and mutated_config.yaml"
        )
    ancestor_yaml = load_yaml(source_yaml)
    ancestor_json = read_json(source_json)

    ancestor_json.setdefault("variant_name", "ancestor")
    ancestor_json.setdefault("category", "ancestor")
    ancestor_json.setdefault("scope", "none")
    ancestor_json.setdefault("single_point", True)
    ancestor_json.setdefault("shape_preserving", True)
    ancestor_json.setdefault("crash_risk", "low")
    ancestor_json.setdefault("mutation", {"key": "ancestor", "before": None, "after": None})
    ancestor_json.setdefault("apply_to", ["pta", "msa"])
    ancestor_json.setdefault("runtime_overrides", default_overrides())
    ancestor_json.setdefault("notes", "baseline ancestor")
    ancestor_yaml.setdefault("metadata", {}).setdefault("variant_name", "ancestor")

    write_yaml(ancestor_dir / "mutated_config.yaml", ancestor_yaml)
    write_json(ancestor_dir / "mutating.json", ancestor_json)
    return ancestor_yaml, ancestor_json


def feature_summary(model_doc: dict[str, Any], ancestor_yaml: dict[str, Any]) -> dict[str, Any]:
    section_name, transformer, extra, spec = active_sections(model_doc)
    yaml_cfg = copy.deepcopy(get_yaml_config(ancestor_yaml))
    features = {
        "section_name": section_name,
        "transformer": transformer,
        "extra": extra,
        "spec": spec,
        "yaml_config": yaml_cfg,
    }
    return {
        "section_name": section_name,
        "has_moe": has_moe(features),
        "has_mla": has_mla(features),
        "has_rope": str(ancestor_yaml.get("base_config", {}).get("position_embedding_type", extra.get("position_embedding_type", ""))).lower() == "rope",
        "has_gqa": has_heads_groups(features),
        "has_swiglu": has_swiglu(features),
        "has_qkv_bias": any(key in transformer for key in ("add_qkv_bias", "qkv_bias")),
        "has_linear_bias": "add_bias_linear" in transformer,
    }


def generate_model(model_name: str) -> dict[str, Any]:
    model_dir = MUTATED_CONFIG_ROOT / model_name
    ancestor_yaml, ancestor_json = copy_ancestor_canonical(model_dir)
    model_doc = load_yaml(MODEL_CONFIG_ROOT / f"{model_name}.yaml")
    section_name, transformer, extra, spec = active_sections(model_doc)
    features = {
        "model_name": model_name,
        "section_name": section_name,
        "transformer": transformer,
        "extra": extra,
        "spec": spec,
        "yaml_config": copy.deepcopy(get_yaml_config(ancestor_yaml)),
    }

    for child in model_dir.iterdir():
        if child.is_dir() and child.name != "ancestor":
            shutil.rmtree(child)

    variants = []
    skipped = []
    for variant in build_variants(features):
        if not variant.condition(features):
            skipped.append(
                {
                    "variant": variant.name,
                    "category": variant.category,
                    "scope": variant.scope,
                    "mutation_key": variant.mutation_key,
                    "reason": "not_applicable_to_model_or_parallel_constraints",
                    "crash_risk": variant.crash_risk,
                    "shape_preserving": variant.shape_preserving,
                }
            )
            continue
        variant_yaml = copy.deepcopy(ancestor_yaml)
        variant_json = copy.deepcopy(ancestor_json)
        if variant.apply:
            variant.apply(variant_yaml, variant_json)
        update_yaml_metadata(variant_yaml, variant, features)
        annotate_mutating_doc(variant_json, variant, features)

        variant_dir = model_dir / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        write_yaml(variant_dir / "mutated_config.yaml", variant_yaml)
        write_json(variant_dir / "mutating.json", variant_json)
        variants.append(
            {
                "variant": variant.name,
                "category": variant.category,
                "scope": variant.scope,
                "mutation_key": variant.mutation_key,
                "crash_risk": variant.crash_risk,
                "shape_preserving": variant.shape_preserving,
            }
        )

    manifest = {
        "model": model_name,
        "variant_count": len(variants) + 1,
        "variants": [{"variant": "ancestor", "category": "ancestor", "scope": "none", "mutation_key": "ancestor", "crash_risk": "low", "shape_preserving": True}, *variants],
        "skipped_or_disabled_variants": skipped,
        "model_detected_features": feature_summary(model_doc, ancestor_yaml),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(model_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    manifests = []
    for config_path in sorted(MODEL_CONFIG_ROOT.glob("*.yaml")):
        model_name = config_path.stem
        if not (MUTATED_CONFIG_ROOT / model_name / "ancestor").is_dir():
            print(f"[skip] missing ancestor: {model_name}")
            continue
        manifest = generate_model(model_name)
        manifests.append(manifest)
        print(
            f"[ok] {model_name}: variants={manifest['variant_count']} "
            f"skipped={len(manifest['skipped_or_disabled_variants'])}"
        )
    summary = {
        "model_count": len(manifests),
        "models": [item["model"] for item in manifests],
        "total_variant_count": sum(item["variant_count"] for item in manifests),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(MUTATED_CONFIG_ROOT / "manifest.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
