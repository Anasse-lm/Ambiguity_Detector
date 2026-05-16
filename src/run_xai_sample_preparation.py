#!/usr/env/bin python3
import os
import argparse
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from tqdm.auto import tqdm

import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from req_ambiguity.utils.config import load_yaml, find_project_root, resolve_path
from req_ambiguity.evaluation.per_label_diagnostics import load_model_and_data, run_inference

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    root = find_project_root()
    cfg = load_yaml("configs/train.yaml", root=root)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    samples_dir = root / "outputs/xai/samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Verify no training/val leakage
    data_dir = resolve_path(cfg["paths"]["processed_dir"], root=root)
    df_train = pd.read_csv(data_dir / "train.csv")
    df_val = pd.read_csv(data_dir / "val.csv")
    df_test = pd.read_csv(data_dir / "test.csv")
    
    train_ids = set(df_train["StoryID"].astype(str)) if "StoryID" in df_train.columns else set(df_train.index.astype(str))
    val_ids = set(df_val["StoryID"].astype(str)) if "StoryID" in df_val.columns else set(df_val.index.astype(str))
    
    test_ids = set(df_test["StoryID"].astype(str)) if "StoryID" in df_test.columns else set(df_test.index.astype(str))
    
    overlap_train = test_ids.intersection(train_ids)
    overlap_val = test_ids.intersection(val_ids)
    print(f"Verification: Found {len(overlap_train)} overlap with train, {len(overlap_val)} overlap with val.")
    
    if len(overlap_train) > 0 or len(overlap_val) > 0:
        if "StoryID" in df_test.columns:
            df_test = df_test[~df_test["StoryID"].astype(str).isin(overlap_train.union(overlap_val))]
        else:
            df_test = df_test[~df_test.index.astype(str).isin(overlap_train.union(overlap_val))]
        print(f"Removed leaking IDs. Remaining test size: {len(df_test)}")

    model, test_loader, label_cols = load_model_and_data(cfg, root, device)

    print("Running inference to get predictions...")
    y_true, logits = run_inference(model, test_loader, device)
    probs = 1.0 / (1.0 + np.exp(-logits))

    thresholds_path = root / "outputs/results/optimal_thresholds.json"
    if thresholds_path.exists():
        with open(thresholds_path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)
    else:
        thresholds = {l: 0.5 for l in label_cols}

    records = []
    for i in range(len(df_test)):
        true_labels = []
        pred_labels = []
        tp_labels = []
        
        for j, label in enumerate(label_cols):
            t = thresholds.get(label, 0.5)
            is_true = (y_true[i, j] == 1)
            is_pred = (probs[i, j] >= t)
            
            if is_true: true_labels.append(label)
            if is_pred: pred_labels.append(label)
            if is_true and is_pred: tp_labels.append(label)

        rec = {
            "StoryID": str(df_test.index[i] if "StoryID" not in df_test.columns else df_test["StoryID"].iloc[i]),
            "StoryText": df_test[cfg["paths"]["text_column"]].iloc[i],
            "true_labels": ",".join(true_labels),
            "predicted_labels": ",".join(pred_labels),
            "is_true_positive_for_label": ",".join(tp_labels)
        }
        for j, label in enumerate(label_cols):
            p = float(probs[i, j])
            rec[f"prob_{label}"] = p
            rec[f"tp_{label}"] = (label in tp_labels)
            
            # Confidence banding
            if p >= 0.66:
                band = "High"
            elif p >= 0.33:
                band = "Medium"
            else:
                band = "Low"
            rec[f"conf_band_{label}"] = band
            
        records.append(rec)

    df_annotated = pd.DataFrame(records)

    # Sampling
    np.random.seed(42)

    # Sample A: Visualization (5-10 TP per label)
    sample_a_idx = set()
    for label in label_cols:
        candidates = df_annotated[df_annotated[f"tp_{label}"]]
        n = min(10, len(candidates))
        if n > 0:
            sample_a_idx.update(candidates.sample(n, random_state=42).index)
    df_sample_a = df_annotated.loc[list(sample_a_idx)]
    df_sample_a.to_csv(samples_dir / "visualization_candidates.csv", index=False)
    print(f"Sample A (Visualization): {len(df_sample_a)} stories")

    # Sample B: Bridge Validation (30 TP per label)
    sample_b_idx = set()
    for label in label_cols:
        candidates = df_annotated[df_annotated[f"tp_{label}"]]
        n = min(30, len(candidates))
        if n > 0:
            sample_b_idx.update(candidates.sample(n, random_state=42).index)
    df_sample_b = df_annotated.loc[list(sample_b_idx)]
    df_sample_b.to_csv(samples_dir / "bridge_validation_sample.csv", index=False)
    print(f"Sample B (Bridge Validation): {len(df_sample_b)} stories")

    # Sample C: Faithfulness (100-150 per label stratified by confidence)
    sample_c_idx = set()
    for label in label_cols:
        for band in ["High", "Medium", "Low"]:
            candidates = df_annotated[df_annotated[f"conf_band_{label}"] == band]
            # 50 per band = 150 per label
            n = min(50, len(candidates))
            if n > 0:
                sample_c_idx.update(candidates.sample(n, random_state=42).index)
                
    df_sample_c = df_annotated.loc[list(sample_c_idx)]
    df_sample_c.to_csv(samples_dir / "faithfulness_sample.csv", index=False)
    print(f"Sample C (Faithfulness): {len(df_sample_c)} stories")

    print(f"Sample preparation complete. Manifests in {samples_dir}")

if __name__ == "__main__":
    main()
