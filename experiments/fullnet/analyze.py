#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from fullnet_core.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["analyze", *(__import__("sys").argv[1:])]))
