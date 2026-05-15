#!/usr/env/bin python3
"""
Diagnostic script for per-label evaluation on the test set.
Applies per-label optimal thresholds and generates detailed reports for the thesis.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

import sys
_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from req_ambiguity.modeling.classifier import DeBERTaAmbiguityClassifier
from req_ambiguity.preprocessing.tokenize import UserStoryDataset
from req_ambiguity.utils.config import load_yaml, find_project_root, resolve_path


def load_model_and_data(cfg, root: Path, device: torch.device):
    from transformers import DebertaV2Tokenizer
    
    paths = cfg["paths"]
    best_ckpt_path = resolve_path(paths["best_checkpoint"], root=root)
    if not best_ckpt_path.exists():
        raise FileNotFoundError(f"Best checkpoint missing: {best_ckpt_path}")
        
    metadata_path = best_ckpt_path.with_name("best_model_metadata.json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    state_dict = torch.load(best_ckpt_path, map_location=device, weights_only=True)
    model_name = metadata["model_name"]
    label_cols = metadata["label_cols"]
    max_length = metadata["max_length"]
    
    model = DeBERTaAmbiguityClassifier(model_name, num_labels=len(label_cols), dropout=0.0).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    test_csv = resolve_path(paths["processed_dir"], root=root) / "test.csv"
    df_test = pd.read_csv(test_csv, encoding="utf-8")
    
    test_ds = UserStoryDataset.from_dataframe(
        df_test, text_column=paths["text_column"], label_cols=label_cols, tokenizer=tokenizer, max_length=max_length
    )
    test_loader = DataLoader(
        test_ds, batch_size=int(cfg.get("batch_size", 16)), shuffle=False, 
        collate_fn=lambda batch: {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        }
    )
    
    return model, test_loader, label_cols


def run_inference(model, loader, device):
    logits_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            logits_list.append(logits.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            
    return np.concatenate(labels_list, axis=0), np.concatenate(logits_list, axis=0)


def generate_diagnostics(y_true, logits, label_cols, thresholds_dict, output_dir: Path):
    probs = 1.0 / (1.0 + np.exp(-logits))
    
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    cm_dir = figures_dir / "confusion_matrices"
    dist_dir = figures_dir / "prob_distributions"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    cm_dir.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)
    
    report_data = []
    weakness_logs = []
    
    for i, label in enumerate(label_cols):
        yt = y_true[:, i]
        pr = probs[:, i]
        
        # 1. Apply specific threshold
        t = thresholds_dict.get(label, 0.5)
        yp = (pr >= t).astype(int)
        
        # 2. Compute metrics
        sup_pos = int(np.sum(yt == 1))
        sup_neg = int(np.sum(yt == 0))
        
        prec = precision_score(yt, yp, zero_division=0)
        rec = recall_score(yt, yp, zero_division=0)
        f1 = f1_score(yt, yp, zero_division=0)
        
        auc = None
        if np.unique(yt).size >= 2:
            auc = roc_auc_score(yt, pr)
            
        report_data.append({
            "Label": label,
            "Support_Positive": sup_pos,
            "Support_Negative": sup_neg,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "AUC-ROC": auc,
            "Optimal_Threshold": t
        })
        
        # Tracking for diagnostic summary
        weakness_logs.append((f1, prec, rec, label))
        
        # 3. Confusion Matrix Plot
        cm = confusion_matrix(yt, yp)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
        fig, ax = plt.subplots()
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        ax.set_title(f"Confusion Matrix: {label}")
        plt.tight_layout()
        plt.savefig(cm_dir / f"{label}_cm.png", dpi=150)
        plt.close(fig)
        
        # 4. Probability Distribution Histograms
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(pr[yt == 0], bins=20, alpha=0.5, label='True: Negative', color='red', density=True)
        if sup_pos > 0:
            ax.hist(pr[yt == 1], bins=20, alpha=0.5, label='True: Positive', color='blue', density=True)
        ax.axvline(t, color='black', linestyle='--', label=f'Threshold: {t:.2f}')
        ax.set_title(f"Probability Distribution: {label}")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Density")
        ax.legend()
        plt.tight_layout()
        plt.savefig(dist_dir / f"{label}_prob_dist.png", dpi=150)
        plt.close(fig)
        
        # 5. ROC Curve Plot
        if np.unique(yt).size >= 2:
            from sklearn.metrics import roc_curve, auc as calc_auc
            fpr, tpr, _ = roc_curve(yt, pr)
            roc_auc = calc_auc(fpr, tpr)
            
            roc_dir = figures_dir / "roc_curves"
            roc_dir.mkdir(parents=True, exist_ok=True)
            
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'Receiver Operating Characteristic: {label}')
            ax.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(roc_dir / f"{label}_roc.png", dpi=150)
            plt.close(fig)
        
    # Export CSV Report
    df_report = pd.DataFrame(report_data)
    df_report.to_csv(results_dir / "per_label_test_report.csv", index=False)
    
    # Generate Diagnostic Summary
    weakness_logs.sort(key=lambda x: x[0])  # Sort by F1 ascending
    
    with (results_dir / "diagnostic_summary.txt").open("w", encoding="utf-8") as f:
        f.write("=== PER-LABEL DIAGNOSTIC SUMMARY ===\n")
        f.write("Labels ordered from weakest to strongest (based on F1):\n\n")
        
        for f1, prec, rec, label in weakness_logs:
            f.write(f"Label: {label}\n")
            f.write(f"  F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}\n")
            
            if f1 < 0.60:
                f.write("  Diagnosis: ")
                if prec < 0.40 and rec < 0.40:
                    f.write("General capacity issue. Model struggles to distinguish this class. Check if Support is too low or if features overlap heavily.\n")
                elif prec < 0.50:
                    f.write("Over-prediction. Model flags too many False Positives. Consider raising the threshold or increasing negative examples.\n")
                elif rec < 0.50:
                    f.write("Under-prediction. Model misses True Positives (False Negatives). Consider lowering the threshold or applying higher pos_weight.\n")
                else:
                    f.write("Moderate performance. Can be improved with more data.\n")
            else:
                f.write("  Diagnosis: Acceptable performance.\n")
            f.write("\n")
            
    print(f"Diagnostics generated in {results_dir} and {figures_dir}")

def main():
    root = find_project_root()
    cfg = load_yaml("configs/train.yaml", root=root)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running on {device}")
    
    model, test_loader, label_cols = load_model_and_data(cfg, root, device)
    
    print("Running inference on test set...")
    y_true, logits = run_inference(model, test_loader, device)
    
    thresholds_path = root / "outputs" / "results" / "optimal_thresholds.json"
    if thresholds_path.exists():
        with thresholds_path.open("r", encoding="utf-8") as f:
            thresholds_dict = json.load(f)
    else:
        print("Warning: optimal_thresholds.json not found. Defaulting to 0.5 for all.")
        thresholds_dict = {}
        
    print("Generating diagnostic reports and figures...")
    generate_diagnostics(y_true, logits, label_cols, thresholds_dict, root / "outputs")

if __name__ == "__main__":
    main()
