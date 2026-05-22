from __future__ import annotations

from pathlib import Path

from .paths import MODEL_CONFIG_DIR


ICSE_MODEL_NAMES = {
    "qwen2",
    "llama2",
    "baichuan2",
    "chatglm3",
    "glm4",
    "yi",
    "codellama",
    "pangu",
    "deepseekv3",
    "mixtral",
    "grok1",
}


def available_models(model_config_dir: Path = MODEL_CONFIG_DIR) -> list[str]:
    """Return model names backed by language-model YAML configs."""
    if not model_config_dir.is_dir():
        return []
    return sorted(path.stem for path in model_config_dir.glob("*.yaml") if path.stem in ICSE_MODEL_NAMES)


def validate_models(models: list[str], model_config_dir: Path = MODEL_CONFIG_DIR) -> list[str]:
    """Validate model aliases and return normalized names."""
    normalized = [str(item).strip() for item in models if str(item).strip()]
    if not normalized:
        raise ValueError("至少需要选择一个语言模型，例如 qwen2 或 glm4")

    known = set(available_models(model_config_dir))
    unknown = [model for model in normalized if model.endswith(".yaml") is False and model not in known]
    if unknown:
        known_text = ", ".join(sorted(known)) or "<none>"
        raise ValueError(f"未知模型: {', '.join(unknown)}。可选模型: {known_text}")
    return normalized
