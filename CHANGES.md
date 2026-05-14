# Implementation Changes Summary

This document summarizes the changes made to the training pipeline to support the methodology requirements of the master thesis on AI-based ambiguity detection.

## 1. Early Stopping on Validation Macro F1
**Rationale:** To prevent overfitting while explicitly tracking class-balanced performance rather than overall loss.
- Enabled `early_stopping` in `configs/train.yaml`.
- Early stopping triggered by `macro_f1` improvement stagnation (patience=3).
- Outputs the exact stopping epoch to `outputs/results/early_stopping_epoch.txt`.

## 2. Configurable Positive Class Weighting
**Rationale:** The user story dataset suffers from extreme class imbalance (e.g., TechnicalAmbiguity at ~0.3%). Using raw dynamic weighting causes gradient explosion and NaN loss.
- Added `loss.pos_weight_cap: 50.0` to `configs/train.yaml`.
- Implemented tensor capping in `train.py` for BCEWithLogitsLoss.
- Final utilized weights are exported to `outputs/results/pos_weights.txt`.

## 3. Post-Training Threshold Tuning
**Rationale:** High threshold variance exists across ambiguity types due to differing label supports. A static 0.5 threshold heavily penalizes minority classes.
- Conducts an automated threshold sweep (0.10 to 0.90) on the validation logits.
- Selects the *per-label* thresholds that individually maximize F1.
- Global optimal threshold exported to `outputs/results/optimal_global_threshold.txt`.
- Per-label threshold dictionary exported to `outputs/results/optimal_thresholds.json`.
- Plots validation macro F1 vs. threshold curve at `outputs/figures/threshold_curve.png`.

## 4. Per-Label Test Diagnostics
**Rationale:** Aggregate reporting masks minority class failure.
- Created `src/req_ambiguity/evaluation/per_label_diagnostics.py`.
- Generates `per_label_test_report.csv` compiling Support, Precision, Recall, F1, AUC, and exact Thresholds.
- Outputs individual 2x2 confusion matrices (`outputs/figures/confusion_matrices/`).
- Outputs probability density distributions (`outputs/figures/prob_distributions/`).
- Auto-generates `diagnostic_summary.txt` that sorts labels from weakest to strongest and diagnoses errors (e.g., "Over-prediction", "Under-prediction").

## 5. Augmentation Ablation Pipeline
**Rationale:** To provide empirical justification for synthetic data generation in the thesis Results chapter.
- Replaced `data.train_split` with a boolean `use_augmented_data` flag in `configs/train.yaml`.

---

## Augmentation Ablation Results (Pending Execution)
*Instructions: Run the pipeline twice in Kaggle, once with `use_augmented_data: false` and once with `true`. After both runs complete, run `per_label_diagnostics.py` and record the Test F1 scores here to copy into the thesis.*

| Label | F1 (Original Data) | F1 (Augmented Data) | Delta (Δ) |
|---|---|---|---|
| SemanticAmbiguity | `[TODO]` | `[TODO]` | `[TODO]` |
| ScopeAmbiguity | `[TODO]` | `[TODO]` | `[TODO]` |
| ActorAmbiguity | `[TODO]` | `[TODO]` | `[TODO]` |
| AcceptanceAmbiguity | `[TODO]` | `[TODO]` | `[TODO]` |
| DependencyAmbiguity | `[TODO]` | `[TODO]` | `[TODO]` |
| PriorityAmbiguity | `[TODO]` | `[TODO]` | `[TODO]` |
| TechnicalAmbiguity | `[TODO]` | `[TODO]` | `[TODO]` |
