# Implementation Changes Summary

This document summarizes the changes made to the training pipeline to support the methodology requirements of the master thesis on AI-based ambiguity detection.

## 1. Augmentation Removal
**Rationale:** Augmentation was entirely excluded from the training phase to keep the trained classifier free of synthetic placeholder artifacts. Because this classifier is also used to evaluate and re-score placeholder-containing refined outputs in the downstream verification stage, training without augmentation protects the verification step from shortcut-learning risk and guarantees the classifier is evaluated purely on its ability to generalize to natural, human-written ambiguity patterns.

## 2. Early Stopping on Validation Macro F1
**Rationale:** To prevent overfitting while explicitly tracking class-balanced performance rather than overall loss.
- Enabled `early_stopping` in `configs/train.yaml`.
- Early stopping triggered by `macro_f1` improvement stagnation (patience=3).
- Outputs the exact stopping epoch to `outputs/results/early_stopping_epoch.txt`.

## 3. Configurable Positive Class Weighting
**Rationale:** The user story dataset suffers from extreme class imbalance (e.g., TechnicalAmbiguity at ~0.3%). Using raw dynamic weighting causes gradient explosion and NaN loss.
- Supports `pos_weight_strategy: "cap"` (with `pos_weight_cap: 50.0`) and `"sqrt"`.
- Final utilized weights are exported to `outputs/results/pos_weights.txt`.

## 4. Post-Training Threshold Tuning
**Rationale:** High threshold variance exists across ambiguity types due to differing label supports. A static 0.5 threshold heavily penalizes minority classes.
- Conducts an automated threshold sweep (0.10 to 0.90) on the validation logits.
- Selects the *per-label* thresholds that individually maximize F1.

## 5. Training Efficiency Improvements
**Rationale:** To support exhaustive multi-seed and hyperparameter sweeps, extreme efficiencies were enabled.
- Implemented `torch.cuda.amp` (Automatic Mixed Precision).
- Implemented Gradient Accumulation (`gradient_accumulation_steps`).
- Pre-tokenized the dataset at startup to prevent Tokenizer bottlenecks.
- Efficiency speedup per epoch: `[TODO: record seconds before vs after]`

---

## Multi-Seed Results
*Canonical macro F1 and standard deviation across seeds 42, 43, 44.*

- **Macro F1:** `[TODO: Mean] ± [TODO: Std]`
- **Micro F1:** `[TODO: Mean] ± [TODO: Std]`

## Hyperparameter Sensitivity Findings
*Identified via independent hyperparameter sweeping with early stopping.*

- **Most Sensitive Hyperparameters:** `[TODO: identify from hparam_sensitivity_summary.txt]`
- **Least Sensitive Hyperparameters:** `[TODO: identify from hparam_sensitivity_summary.txt]`
