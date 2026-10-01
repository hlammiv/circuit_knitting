#!/usr/bin/env python3
"""Plot paper Figure 7 from the cached fit and bootstrap analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .fit_fig7 import (
        DEFAULT_CACHE,
        DEFAULT_DATA,
        DEFAULT_FIGURE,
        DEFAULT_JSON,
        load_analysis_cache,
        load_data,
        make_figure,
    )
except ImportError:  # Direct execution: python scripts/plot_fig7.py
    from fit_fig7 import (
        DEFAULT_CACHE,
        DEFAULT_DATA,
        DEFAULT_FIGURE,
        DEFAULT_JSON,
        load_analysis_cache,
        load_data,
        make_figure,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()
    data = load_data(args.data_dir)
    summary = json.loads(args.json.read_text())
    internals = load_analysis_cache(args.cache, data, summary)
    make_figure(data, args.data_dir, summary, internals, args.figure)
    print(args.figure)


if __name__ == "__main__":
    main()
