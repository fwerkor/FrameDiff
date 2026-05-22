"""Shared helpers for runtime mutation/forward scripts."""

from __future__ import annotations

import math
import os
import random
import hashlib


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


DEFAULT_TRANSFORMER_CONFIG = {
    "tensor_model_parallel_size": 1,
    "pipeline_model_parallel_size": 1,
    "num_layers": 24,
    "hidden_size": 896,
    "ffn_hidden_size": 4864,
    "num_attention_heads": 14,
    "num_query_groups": 2,
    "attention_dropout": 0.0,
    "init_method_std": 0.01,
    "hidden_dropout": 0.0,
    "normalization": "RMSNorm",
    "layernorm_epsilon": 1e-6,
}

REQUIRED_TRANSFORMER_FIELDS = {
    "tensor_model_parallel_size": 1,
    "pipeline_model_parallel_size": 1,
    "init_method_std": 0.01,
    "normalization": "RMSNorm",
    "layernorm_epsilon": 1e-6,
    "attention_dropout": 0.0,
    "hidden_dropout": 0.0,
}

PROBLEMATIC_TRANSFORMER_FIELDS = [
    "params_dtype",
    "bf16",
    "fp16",
    "attention_softmax_in_fp32",
    "masked_softmax_fusion",
    "sequence_parallel",
    "gated_linear_unit",
    "multi_latent_attention",
]

MOE_INDICATORS = ["num_moe_experts", "moe_router_topk", "moe_grouped_gemm", "moe_ffn_hidden_size"]
TRANSFORMER_INT_FIELDS = [
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "num_layers",
    "hidden_size",
    "ffn_hidden_size",
    "num_attention_heads",
    "num_query_groups",
    "num_moe_experts",
    "n_shared_experts",
    "moe_router_topk",
    "topk_group",
    "moe_ffn_hidden_size",
    "moe_layer_freq",
    "kv_channels",
    "max_position_embeddings",
    "rope_scaling_factor",
    "rope_scaling_original_max_position_embeddings",
]
TRANSFORMER_FLOAT_FIELDS = [
    "attention_dropout",
    "hidden_dropout",
    "init_method_std",
    "layernorm_epsilon",
    "moe_aux_loss_coeff",
]


def resolve_repo_path(path: str | None) -> str | None:
    """Resolve relative project paths against the repository root when needed."""
    if not path:
        return path
    if os.path.isabs(path) or os.path.exists(path):
        return path
    return os.path.join(REPO_ROOT, path)


def seed_all(seed=42, *, np_module=None, torch_module=None, torch_npu_module=None, ms_module=None):
    """Seed the RNGs used by runtime scripts across supported backends."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if np_module is not None:
        np_module.random.seed(seed)
    if torch_module is not None:
        torch_module.manual_seed(seed)
        torch_module.use_deterministic_algorithms(True)
    if torch_npu_module is not None:
        torch_npu_module.npu.manual_seed_all(seed)
        torch_npu_module.npu.manual_seed(seed)
    if ms_module is not None:
        ms_module.set_seed(seed)


def mix_seed(base_seed: int, *parts) -> int:
    """Derive a deterministic non-linear seed from a base seed and context parts."""
    payload = "::".join([str(base_seed), *[str(part) for part in parts]]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


def make_rng(base_seed: int, *parts) -> random.Random:
    """Create an isolated RNG for a specific deterministic context."""
    return random.Random(mix_seed(base_seed, *parts))


def parse_numbers_simple(input_str):
    if not input_str:
        return []
    try:
        return [float(num) if "." in num else int(num) for num in input_str.split(",") if num.strip()]
    except ValueError as exc:
        print(f"解析错误: {exc}")
        return []


def coerce_optional_int(value):
    """Convert integer-like scalars while preserving None/invalid text."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized or normalized.lower() in {"none", "null"}:
            return None
        try:
            return int(normalized)
        except ValueError:
            try:
                float_value = float(normalized)
            except ValueError:
                return value
            return int(float_value) if float_value.is_integer() else value
    return value


def coerce_optional_float(value):
    """Convert float-like scalars while preserving None/invalid text."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized or normalized.lower() in {"none", "null"}:
            return None
        try:
            return float(normalized)
        except ValueError:
            return value
    return value


def normalize_transformer_scalar_types(config: dict) -> dict:
    """Normalize common TransformerConfig scalar fields loaded from YAML/JSON."""
    normalized = dict(config or {})
    for field in TRANSFORMER_INT_FIELDS:
        if field in normalized:
            normalized[field] = coerce_optional_int(normalized[field])
    for field in TRANSFORMER_FLOAT_FIELDS:
        if field in normalized:
            normalized[field] = coerce_optional_float(normalized[field])
    return normalized


def is_active_moe_config(config: dict) -> bool:
    """Return True when a config enables MoE experts."""
    if not config:
        return False
    for field in ("num_moe_experts", "num_experts"):
        value = coerce_optional_int(config.get(field))
        if isinstance(value, int) and value > 0:
            return True
    return False


def has_moe_indicators(config: dict) -> bool:
    """Return True when a config contains any enabled MoE-specific fields."""
    if is_active_moe_config(config):
        return True
    if not config:
        return False
    for field in MOE_INDICATORS:
        if field not in config:
            continue
        value = coerce_optional_int(config[field])
        if value is None or value is False:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0:
            continue
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "false", "0"}:
            continue
        return True
    return False


def enforce_moe_bias_constraint(config: dict, *, log: bool = False) -> dict:
    """Disable linear bias for active MoE configs before TransformerConfig validates it."""
    if not has_moe_indicators(config):
        return config
    if config.get("add_bias_linear") is not False:
        if log:
            print("  检测到MoE配置，强制设置 add_bias_linear = False")
        config["add_bias_linear"] = False
    if "disable_bias_linear" in config:
        config["disable_bias_linear"] = True
    return config


def load_random_model_configs(yaml_loader, module, config_dir, count: int = 2):
    """Load configs from explicit module paths or randomly from a config dir."""
    configs = []
    file_names = []

    if module:
        module_paths = [path.strip() for path in str(module).split(",") if path.strip()]
        for path in module_paths:
            try:
                file_name = os.path.basename(path)
                print(f"正在加载模块文件：{path}")
                with open(path, "r", encoding="utf-8") as handle:
                    config = yaml_loader.load(handle)
                config["_source_file"] = file_name
                configs.append(config)
                file_names.append(file_name)
            except FileNotFoundError:
                print(f"警告：文件不存在，跳过 {path}")
            except Exception as exc:
                print(f"YAML解析错误（{path}）：{exc}")
        return configs, file_names

    available_configs = []
    if os.path.exists(config_dir):
        for file in os.listdir(config_dir):
            if file.endswith(".yaml") and file != "note.txt":
                available_configs.append(file)

    if len(available_configs) < count:
        raise ValueError(f"需要至少{count}个配置文件，但只找到 {len(available_configs)} 个")

    selected_files = random.sample(available_configs, count)
    print(f"随机选择的配置文件: {selected_files}")
    for config_file in selected_files:
        config_path = os.path.join(config_dir, config_file)
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml_loader.load(handle)
        config["_source_file"] = config_file
        configs.append(config)

    return configs, selected_files


def extract_transformer_config_from_yaml(yaml_config: dict) -> dict:
    """Extract a sanitized TransformerConfig dict from a model yaml."""
    if "TransformerConfig" in yaml_config:
        base_config = yaml_config["TransformerConfig"].copy()
    elif "MLATransformerConfig" in yaml_config:
        # Support DeepSeekV3 MLA configuration
        base_config = yaml_config["MLATransformerConfig"].copy()
    else:
        base_config = DEFAULT_TRANSFORMER_CONFIG.copy()

    for field, default_value in REQUIRED_TRANSFORMER_FIELDS.items():
        base_config.setdefault(field, default_value)

    for field in PROBLEMATIC_TRANSFORMER_FIELDS:
        base_config.pop(field, None)

    if has_moe_indicators(base_config):
        enforce_moe_bias_constraint(base_config, log=True)
    else:
        base_config.setdefault("add_bias_linear", False)

    return normalize_transformer_scalar_types(base_config)


def prepare_mutation_transformer_config(yaml_config: dict) -> dict:
    """Shared config post-processing used by mutate_submodule-auto scripts."""
    base_config = extract_transformer_config_from_yaml(yaml_config)
    base_config["gradient_accumulation_fusion"] = False
    if "num_query_groups" not in base_config:
        base_config["num_query_groups"] = base_config.get("num_attention_heads", 2)
    if base_config["num_query_groups"] > base_config["num_attention_heads"]:
        base_config["num_query_groups"] = base_config["num_attention_heads"]
    return base_config


def prepare_strict_mutation_transformer_config(yaml_config: dict) -> dict:
    """Extra alignment used by mutate_submodule to avoid attention shape issues."""
    base_config = prepare_mutation_transformer_config(yaml_config)

    if base_config["num_attention_heads"] <= 0:
        base_config["num_attention_heads"] = 1
    if base_config["num_query_groups"] <= 0:
        base_config["num_query_groups"] = 1
    if base_config["num_query_groups"] > base_config["num_attention_heads"]:
        base_config["num_query_groups"] = base_config["num_attention_heads"]

    heads = base_config["num_attention_heads"]
    hidden = base_config["hidden_size"]

    if hidden % heads != 0:
        aligned_hidden = heads * math.ceil(hidden / heads)
        print(f"  调整hidden_size {hidden} -> {aligned_hidden} 以整除num_attention_heads {heads}")
        base_config["hidden_size"] = aligned_hidden

    if heads % base_config["num_query_groups"] != 0:
        new_groups = math.gcd(heads, base_config["num_query_groups"]) or 1
        if new_groups != base_config["num_query_groups"]:
            print(
                f"  调整num_query_groups {base_config['num_query_groups']} -> {new_groups} "
                f"以整除num_attention_heads {heads}"
            )
            base_config["num_query_groups"] = new_groups

    return base_config


def extract_graph_transformer_config_from_yaml(yaml_config: dict) -> dict:
    """Graph scripts support both TransformerConfig and MLATransformerConfig."""
    if "TransformerConfig" in yaml_config:
        transformed = {"TransformerConfig": yaml_config["TransformerConfig"]}
    elif "MLATransformerConfig" in yaml_config:
        transformed = {"TransformerConfig": yaml_config["MLATransformerConfig"]}
    else:
        transformed = yaml_config

    base_config = extract_transformer_config_from_yaml(transformed)
    if "num_query_groups" not in base_config or base_config["num_query_groups"] is None:
        base_config["num_query_groups"] = base_config.get("num_attention_heads", 2)
    if base_config["num_query_groups"] > base_config["num_attention_heads"]:
        base_config["num_query_groups"] = base_config["num_attention_heads"]
    return base_config
