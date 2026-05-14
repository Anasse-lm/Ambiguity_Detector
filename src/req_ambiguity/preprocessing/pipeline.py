"""End-to-end preprocessing from raw file to cleaned CSV splits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from req_ambiguity.preprocessing.clean import normalize_story_text
from req_ambiguity.preprocessing.io import load_raw_dataframe, validate_schema
from req_ambiguity.preprocessing.report import document_distribution, write_preprocessing_summary
from req_ambiguity.preprocessing.split import multilabel_stratified_three_way, save_split_csvs
from req_ambiguity.utils.config import resolve_path


def _normalize_binary_labels(df: pd.DataFrame, label_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in label_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        out[col] = (out[col] > 0.5).astype(int)
    return out


def run_preprocessing_from_train_config(
    cfg: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """
    Load raw data, clean, stratified split, write CSVs and reports.

    Expects keys: paths (raw_data, text_column, processed_dir), label_cols,
    split (train_ratio, val_ratio, test_ratio), random_seed,
    preprocessing (min_story_chars, reports_dir).
    """
    paths = cfg["paths"]
    label_cols: list[str] = list(cfg["label_cols"])
    text_column: str = str(paths["text_column"])
    raw_path = resolve_path(paths["raw_data"], root=project_root)
    processed_dir = resolve_path(paths["processed_dir"], root=project_root)

    pre = cfg.get("preprocessing") or {}
    min_chars = int(pre.get("min_story_chars", 5))
    reports_dir = resolve_path(pre.get("reports_dir", "outputs/reports/preprocessing/"), root=project_root)

    split_cfg = cfg.get("split") or {}
    train_ratio = float(split_cfg.get("train_ratio", 0.8))
    val_ratio = float(split_cfg.get("val_ratio", 0.1))
    test_ratio = float(split_cfg.get("test_ratio", 0.1))
    seed = int(cfg.get("random_seed", 42))

    df_raw = load_raw_dataframe(raw_path)
    validate_schema(df_raw, text_column=text_column, label_cols=label_cols)
    n_raw = len(df_raw)

    df = df_raw.dropna(subset=[text_column]).copy()
    df[text_column] = df[text_column].astype(str).map(normalize_story_text)
    df = df[df[text_column].str.len() > min_chars].reset_index(drop=True)

    df = _normalize_binary_labels(df, label_cols)

    train_df, val_df, test_df = multilabel_stratified_three_way(
        df,
        text_column=text_column,
        label_cols=label_cols,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=seed,
    )

    save_split_csvs(train_df, val_df, test_df, processed_dir)

    document_distribution(df, label_cols, "Full_Cleaned", save_path=reports_dir / "distribution_full.csv")
    document_distribution(train_df, label_cols, "Train", save_path=reports_dir / "distribution_train.csv")
    document_distribution(val_df, label_cols, "Val", save_path=reports_dir / "distribution_val.csv")
    document_distribution(test_df, label_cols, "Test", save_path=reports_dir / "distribution_test.csv")

    summary: dict[str, Any] = {
        "raw_path": str(raw_path),
        "text_column": text_column,
        "label_cols": label_cols,
        "random_seed": seed,
        "min_story_chars": min_chars,
        "n_raw_rows": n_raw,
        "n_after_drop_and_clean": len(df),
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "processed_dir": str(processed_dir),
        "split_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
    }
    write_preprocessing_summary(reports_dir / "summary.json", summary)

    log_path = reports_dir / "preprocessing_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_lines = [
        "Preprocessing run",
        f"  raw_path: {raw_path}",
        f"  rows_raw: {n_raw}",
        f"  rows_after_clean: {len(df)}",
        f"  train/val/test: {len(train_df)}/{len(val_df)}/{len(test_df)}",
        f"  processed_dir: {processed_dir}",
    ]
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    return summary
