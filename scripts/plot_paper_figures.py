#!/usr/bin/env python3
"""Regenerate paper Figures 5, 6, and 7 from their retained source data."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    commands = (
        [sys.executable, str(ROOT / "scripts" / "plot_fig5.py")],
        [sys.executable, str(ROOT / "scripts" / "plot_fig6.py")],
        [sys.executable, str(ROOT / "scripts" / "plot_fig7.py")],
    )
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
