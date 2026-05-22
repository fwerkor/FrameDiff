from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
CONFIG_PATH = PROJECT_ROOT / "config.json"
CONFIG_EXAMPLE_PATH = PROJECT_ROOT / "config.json.example"
OUTPUT_ROOT = PROJECT_ROOT / "output"
MODEL_CONFIG_DIR = WORKSPACE_ROOT / "frame_diff_common" / "model_configs"
