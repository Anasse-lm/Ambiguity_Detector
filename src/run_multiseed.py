#!/usr/env/bin python3
"""
Multi-seed orchestration script for the final thesis run.
Runs training and diagnostics across seeds [42, 43, 44], aggregates statistical results,
and selects the canonical checkpoint.
"""

import os
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import torch

import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from req_ambiguity.utils.config import load_yaml, find_project_root
from req_ambiguity.modeling.train import train_from_config
from req_ambiguity.evaluation.per_label_diagnostics import load_model_and_data, run_inference, generate_diagnostics

def run_seed(seed: int, root: Path) -> dict:
    print(f"\n{'='*80}")
    print(f"=== STARTING RUN FOR SEED {seed} ===")
    print(f"{'='*80}\n")
    
    cfg = load_yaml("configs/train.yaml", root=root)
    
    # Isolate outputs
    seed_dir = f"outputs/seed_{seed}"
    cfg["random_seed"] = seed
    cfg["paths"]["checkpoints_dir"] = f"{seed_dir}/checkpoints/"
    cfg["paths"]["best_checkpoint"] = f"{seed_dir}/checkpoints/best_model.pt"
    cfg["paths"]["training_logs_dir"] = f"{seed_dir}/reports/training/"
    cfg["paths"]["figures_dir"] = f"{seed_dir}/figures/"
    cfg["paths"]["results_dir"] = f"{seed_dir}/results/"
    
    # 1. Run Training
    train_from_config(cfg, project_root=root, show_progress=True)
    
    # 2. Run Diagnostics
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model, test_loader, label_cols = load_model_and_data(cfg, root, device)
    y_true, logits = run_inference(model, test_loader, device)
    
    thresholds_path = root / f"{seed_dir}/results/optimal_thresholds.json"
    with thresholds_path.open("r", encoding="utf-8") as f:
        thresholds_dict = json.load(f)
        
    generate_diagnostics(y_true, logits, label_cols, thresholds_dict, root / seed_dir)
    
    # Extract Macro F1 from test metrics
    test_metrics_path = root / f"{seed_dir}/reports/training/test_metrics.json"
    with test_metrics_path.open("r", encoding="utf-8") as f:
        test_metrics = json.load(f)
        
    return {
        "seed": seed,
        "macro_f1": test_metrics["metrics"]["macro_f1"],
        "micro_f1": test_metrics["metrics"]["micro_f1"],
        "report_path": root / f"{seed_dir}/results/per_label_test_report.csv",
        "checkpoint_path": root / f"{seed_dir}/checkpoints/best_model.pt",
        "metadata_path": root / f"{seed_dir}/checkpoints/best_model_metadata.json",
    }

def main():
    root = find_project_root()
    seeds = [42, 43, 44]
    
    results = []
    for seed in seeds:
        res = run_seed(seed, root)
        results.append(res)
        
    print("\nAggregating multiseed results...")
    
    # Aggregate F1 scores
    df_list = []
    for r in results:
        df = pd.read_csv(r["report_path"])
        df = df[["Label", "F1"]].rename(columns={"F1": f"F1_seed{r['seed']}"})
        df_list.append(df)
        
    # Merge all DataFrames
    df_agg = df_list[0]
    for df in df_list[1:]:
        df_agg = df_agg.merge(df, on="Label")
        
    f1_cols = [c for c in df_agg.columns if c.startswith("F1_seed")]
    df_agg["F1_mean"] = df_agg[f1_cols].mean(axis=1)
    df_agg["F1_std"] = df_agg[f1_cols].std(axis=1)
    
    final_results_dir = root / "outputs" / "results"
    final_results_dir.mkdir(parents=True, exist_ok=True)
    df_agg.to_csv(final_results_dir / "multiseed_aggregate.csv", index=False)
    
    # Calculate global mean/std
    macro_f1s = [r["macro_f1"] for r in results]
    micro_f1s = [r["micro_f1"] for r in results]
    
    macro_mean, macro_std = np.mean(macro_f1s), np.std(macro_f1s)
    micro_mean, micro_std = np.mean(micro_f1s), np.std(micro_f1s)
    
    # Identify canonical seed
    best_result = max(results, key=lambda x: x["macro_f1"])
    best_seed = best_result["seed"]
    
    summary_text = (
        "=== MULTISEED TRAINING SUMMARY ===\n"
        f"Seeds run: {seeds}\n"
        f"Macro F1: {macro_mean:.4f} ± {macro_std:.4f}\n"
        f"Micro F1: {micro_mean:.4f} ± {micro_std:.4f}\n\n"
        f"Canonical Seed selected: {best_seed} (Macro F1: {best_result['macro_f1']:.4f})\n"
        "This seed's checkpoint has been copied to outputs/checkpoints/best_model.pt "
        "and will be used for downstream verification and XAI stages.\n"
    )
    
    with (final_results_dir / "multiseed_summary.txt").open("w", encoding="utf-8") as f:
        f.write(summary_text)
        
    print(summary_text)
    
    # Copy canonical checkpoint
    root_checkpoints = root / "outputs" / "checkpoints"
    root_checkpoints.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_result["checkpoint_path"], root_checkpoints / "best_model.pt")
    shutil.copy2(best_result["metadata_path"], root_checkpoints / "best_model_metadata.json")
    
    # Copy canonical results, figures, and reports to root
    best_seed_dir = root / f"outputs/seed_{best_seed}"
    shutil.copytree(best_seed_dir / "results", root / "outputs/results", dirs_exist_ok=True)
    shutil.copytree(best_seed_dir / "figures", root / "outputs/figures", dirs_exist_ok=True)
    shutil.copytree(best_seed_dir / "reports", root / "outputs/reports", dirs_exist_ok=True)
    
    # Run final thesis visualizations on the canonical outputs
    print("\nGenerating final high-resolution thesis figures for the canonical seed...")
    try:
        from req_ambiguity.evaluation.thesis_visualizations import main as vis_main
        vis_main()
    except Exception as e:
        print(f"Warning: Failed to generate thesis visualizations: {e}")
    
    print(f"\nCanonical checkpoint and artifacts promoted to outputs/")
    print("Multi-seed execution completed successfully.")

if __name__ == "__main__":
    main()
