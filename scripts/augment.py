#!/usr/bin/env python3
"""CLI: augment processed training split (train only) using placeholder policy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from req_ambiguity.augmentation.augmenter import run_augmentation_from_train_config
from req_ambiguity.augmentation.policy import load_placeholder_policy, resolve_placeholders_path
from req_ambiguity.utils.config import find_project_root, load_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Augment ambiguous training rows with placeholders.")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to train.yaml.")
    parser.add_argument("--project-root", type=Path, default=None, help="Repository root.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even when augmentation.enabled is false (for ablation / experiments).",
    )
    args = parser.parse_args()

    root = args.project_root or find_project_root()
    cfg = load_yaml(args.config, root=root)

    aug = cfg.get("augmentation") or {}
    if not aug.get("enabled", False) and not args.force:
        print(
            "Augmentation is disabled in configs/train.yaml (augmentation.enabled: false).\n"
            "Enable it or pass --force to generate augmented/train.csv anyway."
        )
        return 0

    label_cols: list[str] = list(cfg["label_cols"])
    ph_path = resolve_placeholders_path(cfg, project_root=root)
    label_map, legal = load_placeholder_policy(ph_path, label_cols, project_root=root)

    summary = run_augmentation_from_train_config(
        cfg,
        project_root=root,
        label_to_placeholders=label_map,
        legal_tokens=legal,
    )
    print("Augmentation complete.")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
