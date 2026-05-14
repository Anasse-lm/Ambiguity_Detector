"""Multi-label stratified train / validation / test splits."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def multilabel_stratified_three_way(
    df: pd.DataFrame,
    *,
    text_column: str,
    label_cols: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train / val / test with approximate multi-label balance.

    Uses iterative stratification (Sechidis et al., 2011) via
    `iterative-stratification`. Falls back to a single random split if
    the dataset is too small for stratification.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    holdout_ratio = val_ratio + test_ratio
    if holdout_ratio <= 0 or holdout_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be in (0, 1).")

    X = df[text_column].values.reshape(-1, 1)
    y = df[label_cols].values.astype(np.int64)

    n = len(df)
    if n < 4:
        raise ValueError(f"Need at least 4 rows for a 3-way split, got {n}")

    try:
        splitter1 = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=holdout_ratio,
            random_state=random_seed,
        )
        train_idx, holdout_idx = next(splitter1.split(X, y))

        # Within holdout: test gets (test_ratio / holdout_ratio) of holdout rows.
        test_fraction_within_holdout = test_ratio / holdout_ratio

        splitter2 = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=test_fraction_within_holdout,
            random_state=random_seed,
        )
        X_h = X[holdout_idx]
        y_h = y[holdout_idx]
        val_rel, test_rel = next(splitter2.split(X_h, y_h))
        # iterstrat returns (train_idx, test_idx) for the inner split.
        val_idx = holdout_idx[val_rel]
        test_idx = holdout_idx[test_rel]
    except Exception as exc:  # pragma: no cover - defensive path
        warnings.warn(
            f"Multi-label stratified split failed ({exc!r}); falling back to random split.",
            UserWarning,
            stacklevel=2,
        )
        rng = np.random.default_rng(random_seed)
        perm = rng.permutation(n)
        n_train = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))
        train_idx = perm[:n_train]
        val_idx = perm[n_train : n_train + n_val]
        test_idx = perm[n_train + n_val :]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, val_df, test_df


def save_split_csvs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
