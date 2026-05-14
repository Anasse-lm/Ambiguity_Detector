#!/usr/bin/env python3
"""CLI: load raw CSV/XLSX, clean, multi-label stratified split, write artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without installation: add src to path
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from req_ambiguity.preprocessing.pipeline import run_preprocessing_from_train_config
from req_ambiguity.utils.config import find_project_root, load_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess labeled requirement dataset.")
    parser.add_argument(
        "--config",
        default="configs/train.yaml",
        help="Path to train.yaml (relative to project root or absolute).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Repository root (default: auto-detect via configs/train.yaml).",
    )
    parser.add_argument(
        "--raw-data",
        default=None,
        help="Optional path to override train.yaml paths.raw_data (CSV or Excel).",
    )
    args = parser.parse_args()

    root = args.project_root or find_project_root()
    cfg = load_yaml(args.config, root=root)
    if args.raw_data is not None:
        cfg.setdefault("paths", {})["raw_data"] = args.raw_data
    summary = run_preprocessing_from_train_config(cfg, project_root=root)
    print("Preprocessing complete.")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
