"""Multilabel classification metrics (macro/micro F1, per-label AUC)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def probs_and_preds_from_logits(
    logits: np.ndarray,
    *,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(np.int64)
    return probs, preds


def find_optimal_threshold(
    y_true: np.ndarray,
    logits: np.ndarray,
    label_names: list[str],
    metric: str = "macro_f1"
) -> float:
    """Finds the probability threshold (0.1 to 0.9) that maximizes the target metric on validation set."""
    probs = 1.0 / (1.0 + np.exp(-logits))
    best_threshold = 0.5
    best_score = -1.0
    
    # Test thresholds from 0.1 to 0.9 in 0.05 increments
    thresholds = np.arange(0.1, 0.95, 0.05)
    for t in thresholds:
        preds = (probs >= t).astype(np.int64)
        if metric == "macro_f1":
            score = float(f1_score(y_true, preds, average="macro", zero_division=0))
        elif metric == "micro_f1":
            score = float(f1_score(y_true, preds, average="micro", zero_division=0))
        else:
            raise ValueError(f"Unsupported metric for tuning: {metric}")
            
        if score > best_score:
            best_score = score
            best_threshold = float(t)
            
    return best_threshold


def multilabel_metrics(
    y_true: np.ndarray,
    logits: np.ndarray,
    *,
    label_names: list[str],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """y_true, logits shape (N, K)."""
    probs, preds = probs_and_preds_from_logits(logits, threshold=threshold)
    out: dict[str, Any] = {
        "threshold": threshold,
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, preds, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, preds, average="macro", zero_division=0)),
        "per_label": {},
    }
    per: dict[str, Any] = {}
    for i, name in enumerate(label_names):
        yt = y_true[:, i]
        yp = preds[:, i]
        pr = probs[:, i]
        roc = None
        pr_auc = None
        if np.unique(yt).size >= 2:
            try:
                roc = float(roc_auc_score(yt, pr))
            except ValueError:
                roc = None
            try:
                pr_auc = float(average_precision_score(yt, pr))
            except ValueError:
                pr_auc = None
        per[name] = {
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "roc_auc": roc,
            "pr_auc": pr_auc,
        }
    out["per_label"] = per
    return out
