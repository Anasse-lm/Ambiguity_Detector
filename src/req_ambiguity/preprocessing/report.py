"""Label distribution tables and preprocessing summary artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def document_distribution(
    df: pd.DataFrame,
    label_cols: list[str],
    split_name: str,
    *,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """
    Build per-label counts for the full split and for ambiguous-only rows.

    Ambiguous-only: any label in label_cols is 1.
    """
    ambiguous_mask = (df[label_cols].sum(axis=1) > 0).astype(bool)
    ambiguous_only = df.loc[ambiguous_mask]
    n_total = len(df)
    n_ambig = max(len(ambiguous_only), 1)

    rows: list[dict[str, Any]] = []
    for col in label_cols:
        all_count = int(df[col].sum())
        amb_count = int(ambiguous_only[col].sum())
        rows.append(
            {
                "Label": col,
                "Count_All": all_count,
                "Pct_All": round(all_count / n_total * 100, 2) if n_total else 0.0,
                "Count_AmbiguousOnly": amb_count,
                "Pct_AmbiguousOnly": round(amb_count / n_ambig * 100, 2),
            }
        )

    dist_df = pd.DataFrame(rows)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dist_df.to_csv(save_path, index=False)

    return dist_df


def write_preprocessing_summary(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
