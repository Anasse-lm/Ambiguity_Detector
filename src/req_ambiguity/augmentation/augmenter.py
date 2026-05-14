"""Train-set augmentation: append taxonomy placeholders for ambiguous rows only."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from req_ambiguity.utils.config import resolve_path


@dataclass(frozen=True)
class AugmentationRecord:
    source_row_index: int
    variant_index: int
    active_labels: list[str]
    appended_tokens: list[str]
    original_text: str
    augmented_text: str


def _active_labels(row: pd.Series, label_cols: list[str]) -> list[str]:
    return [c for c in label_cols if int(row[c]) == 1]


def _num_variants(active: list[str], *, max_variants: int, variants_cap: int) -> int:
    """More variants when multiple ambiguity types fire; capped by config."""
    base = 2 if len(active) >= 2 else 1
    return min(max_variants, variants_cap, base)


def augment_story_append_markers(
    text: str,
    active_labels: list[str],
    label_to_placeholders: dict[str, list[str]],
    rng: random.Random,
) -> tuple[str, list[str]]:
    """
    Append one randomly chosen legal placeholder per active label (order follows active_labels).
    Does not rewrite the original sentence body.
    """
    insertions: list[str] = []
    for label in active_labels:
        pool = label_to_placeholders.get(label) or []
        if not pool:
            continue
        insertions.append(rng.choice(pool))
    if not insertions:
        return text, []
    augmented = f"{text} {' '.join(insertions)}"
    return augmented, insertions


def run_augmentation(
    df_train: pd.DataFrame,
    *,
    text_column: str,
    label_cols: list[str],
    label_to_placeholders: dict[str, list[str]],
    random_seed: int,
    aug_cfg: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[AugmentationRecord]]:
    """
    Augment only rows with at least one positive label. Original rows are kept.

    Adds boolean column `augmented` (False for originals, True for synthetic rows).
    """
    strategy = str(aug_cfg.get("strategy", "append_markers"))
    if strategy != "append_markers":
        raise ValueError(f"Unsupported augmentation strategy: {strategy!r}")

    max_variants = int(aug_cfg.get("max_variants_per_story", 2))
    variants_cap = int(aug_cfg.get("variants_if_enabled", 2))

    rng = random.Random(random_seed)
    records: list[AugmentationRecord] = []
    synthetic_rows: list[dict[str, Any]] = []

    for idx, row in df_train.iterrows():
        active = _active_labels(row, label_cols)
        if not active:
            continue

        n_variants = _num_variants(active, max_variants=max_variants, variants_cap=variants_cap)
        original_text = str(row[text_column])

        for v in range(n_variants):
            new_text, tokens = augment_story_append_markers(
                original_text,
                active,
                label_to_placeholders,
                rng,
            )
            if not tokens:
                continue
            new_row = row.to_dict()
            new_row[text_column] = new_text
            new_row["augmented"] = True
            synthetic_rows.append(new_row)
            records.append(
                AugmentationRecord(
                    source_row_index=int(idx),
                    variant_index=v,
                    active_labels=list(active),
                    appended_tokens=list(tokens),
                    original_text=original_text,
                    augmented_text=new_text,
                )
            )

    base = df_train.copy()
    base["augmented"] = False

    if synthetic_rows:
        aug_df = pd.DataFrame(synthetic_rows)
        combined = pd.concat([base, aug_df], ignore_index=True)
        combined = combined.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    else:
        combined = base

    return combined, records


def write_augmentation_artifacts(
    *,
    combined_train: pd.DataFrame,
    records: list[AugmentationRecord],
    legal_tokens: frozenset[str],
    augmented_dir: Path,
    reports_dir: Path,
    label_cols: list[str],
    aug_cfg: Mapping[str, Any],
    random_seed: int,
) -> dict[str, Any]:
    augmented_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    out_train = augmented_dir / "train.csv"
    combined_train.to_csv(out_train, index=False)

    meta_rows = [
        {
            "source_row_index": r.source_row_index,
            "variant_index": r.variant_index,
            "active_labels": ";".join(r.active_labels),
            "appended_tokens": ";".join(r.appended_tokens),
            "original_text": r.original_text,
            "augmented_text": r.augmented_text,
        }
        for r in records
    ]
    meta_path = reports_dir / "augmentation_metadata.csv"
    pd.DataFrame(meta_rows).to_csv(meta_path, index=False)

    legal_path = reports_dir / "legal_placeholders.txt"
    legal_path.write_text("\n".join(sorted(legal_tokens)) + "\n", encoding="utf-8")

    summary = {
        "random_seed": random_seed,
        "strategy": aug_cfg.get("strategy"),
        "max_variants_per_story": aug_cfg.get("max_variants_per_story"),
        "variants_if_enabled": aug_cfg.get("variants_if_enabled"),
        "n_original_train": int((combined_train["augmented"].eq(False)).sum()),
        "n_synthetic": len(records),
        "n_combined_train": len(combined_train),
        "output_train_csv": str(out_train),
        "metadata_csv": str(meta_path),
        "legal_placeholders_txt": str(legal_path),
        "label_cols": list(label_cols),
    }
    with (reports_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def run_augmentation_from_train_config(
    cfg: Mapping[str, Any],
    *,
    project_root: Path,
    label_to_placeholders: dict[str, list[str]],
    legal_tokens: frozenset[str],
) -> dict[str, Any]:
    """Load processed train split, augment, write data/augmented + reports."""
    paths = cfg["paths"]
    label_cols: list[str] = list(cfg["label_cols"])
    text_column = str(paths["text_column"])
    processed_dir = resolve_path(paths["processed_dir"], root=project_root)
    augmented_dir = resolve_path(paths["augmented_dir"], root=project_root)

    aug = cfg.get("augmentation") or {}
    reports_rel = aug.get("reports_dir", "outputs/reports/augmentation/")
    reports_dir = resolve_path(str(reports_rel), root=project_root)

    train_path = processed_dir / "train.csv"
    if not train_path.is_file():
        raise FileNotFoundError(
            f"Missing {train_path}. Run preprocessing (scripts/preprocess.py) first."
        )

    df_train = pd.read_csv(train_path, encoding="utf-8")
    seed = int(cfg.get("random_seed", 42))

    combined, records = run_augmentation(
        df_train,
        text_column=text_column,
        label_cols=label_cols,
        label_to_placeholders=label_to_placeholders,
        random_seed=seed,
        aug_cfg=aug,
    )

    return write_augmentation_artifacts(
        combined_train=combined,
        records=records,
        legal_tokens=legal_tokens,
        augmented_dir=augmented_dir,
        reports_dir=reports_dir,
        label_cols=label_cols,
        aug_cfg=aug,
        random_seed=seed,
    )
