#!/usr/bin/env python3
"""Train DeBERTa-v3-base multi-label ambiguity classifier."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Before importing transformers (via req_ambiguity), pin HF cache inside the repo.
_hf = (_ROOT / ".hf_cache").resolve()
_hf.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(_hf)
os.environ["HF_HUB_CACHE"] = str(_hf / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(_hf / "transformers")
os.environ["XDG_CACHE_HOME"] = str(_hf / "xdg")
os.environ["MPLCONFIGDIR"] = str((_ROOT / ".mplconfig").resolve())
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from req_ambiguity.modeling.train import train_from_config
from req_ambiguity.utils.config import find_project_root, load_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ambiguity classifier (DeBERTa multi-label).")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm batch bars (epoch summary lines still print).",
    )
    args = parser.parse_args()

    root = args.project_root or find_project_root()
    cfg = load_yaml(args.config, root=root)
    summary = train_from_config(cfg, project_root=root, show_progress=not args.no_progress)
    print("Training complete.")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
