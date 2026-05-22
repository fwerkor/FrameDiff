from pathlib import Path
import yaml

_CONFIGS = {}


def _resolve_output_dir(cfg: dict, config_path: Path) -> dict:
    """Ensure output_dir is resolved relative to the experiments root."""
    out_dir = cfg.get("experiment", {}).get("output_dir", "res")
    out_path = Path(out_dir)
    if not out_path.is_absolute():
        experiments_root = config_path.parents[1]
        out_path = experiments_root / out_path
    cfg["experiment"]["output_dir"] = str(out_path.resolve())
    return cfg


def get_config(config_name: str = "operator") -> dict:
    if config_name not in _CONFIGS:
        if config_name == "operator":
            config_path = Path(__file__).parent.parent / "operator_diff" / "config.yaml"
        elif config_name == "component":
            config_path = Path(__file__).parent.parent / "component_diff" / "config.yaml"
        else:
            raise ValueError(f"Unknown config name: {config_name}")
        with open(config_path, "r") as f:
            raw_cfg = yaml.safe_load(f)
        _CONFIGS[config_name] = _resolve_output_dir(raw_cfg, config_path)
    return _CONFIGS[config_name]


def reset_config():
    _CONFIGS.clear()
