#!/usr/bin/env python3
from __future__ import annotations

from frame_fullnet.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["analyze", *(__import__("sys").argv[1:])]))
