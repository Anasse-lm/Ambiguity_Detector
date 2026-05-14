"""Load CSV / Excel datasets and validate required columns."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw_dataframe(path: Path | str) -> pd.DataFrame:
    """
    Load a labeled dataset from CSV (.csv) or Excel (.xlsx, .xls).

    CSV is read as UTF-8. Excel uses pandas/openpyxl defaults.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p, encoding="utf-8")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    raise ValueError(f"Unsupported dataset extension {suffix!r}; use .csv, .xlsx, or .xls")


def validate_schema(
    df: pd.DataFrame,
    *,
    text_column: str,
    label_cols: list[str],
) -> None:
    """Raise ValueError if required columns are missing."""
    missing: list[str] = []
    if text_column not in df.columns:
        missing.append(text_column)
    for col in label_cols:
        if col not in df.columns:
            missing.append(col)
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + f". Available columns: {list(df.columns)}"
        )
