from pathlib import Path
import yaml

_CONFIG_CACHE = {}


def _resolve_output_dir(cfg: dict, config_path: Path) -> dict:
    """Ensure output_dir is resolved relative to the experiment root."""
    out_dir = cfg.get("experiment", {}).get("output_dir", "res")
    out_path = Path(out_dir)
    if not out_path.is_absolute():
        # Resolve relative to the experiment/ directory (parent of common/)
        experiment_root = config_path.parent.parent.parent
        out_path = experiment_root / out_path
    cfg["experiment"]["output_dir"] = str(out_path.resolve())
    return cfg


def get_config(config_name: str = "operator") -> dict:
    if config_name not in _CONFIG_CACHE:
        if config_name == "operator":
            config_path = Path(__file__).parent.parent / "operator" / "configs" / "operator_experiment.yaml"
        elif config_name == "component":
            config_path = Path(__file__).parent.parent / "component" / "configs" / "component_experiment.yaml"
        else:
            raise ValueError(f"Unknown config name: {config_name}")
        with open(config_path, "r") as f:
            raw_cfg = yaml.safe_load(f)
        _CONFIG_CACHE[config_name] = _resolve_output_dir(raw_cfg, config_path)
    return _CONFIG_CACHE[config_name]


def reset_config():
    _CONFIG_CACHE.clear()
