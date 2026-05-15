#!/usr/env/bin python3
"""
Hyperparameter Sensitivity Analysis Script.
Sweeps key hyperparameters independently while holding others at defaults.
Uses early stopping to save time and avoids saving massive artifacts.
"""

import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import copy

import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from req_ambiguity.utils.config import load_yaml, find_project_root
from req_ambiguity.modeling.train import train_from_config

def plot_sensitivity(df, hparam, default_val, out_dir):
    plt.figure(figsize=(8, 5))
    
    # Sort for plotting line chart
    df_plot = df.sort_values("Value")
    
    # If the values are strings or sparse, a line plot with markers is best
    plt.plot(df_plot["Value"].astype(str), df_plot["Val_Macro_F1"], marker='o', linestyle='-', linewidth=2)
    
    # Highlight default value
    idx_default = df_plot.index[df_plot['Value'] == default_val].tolist()[0]
    plt.scatter([str(default_val)], [df_plot.loc[idx_default, "Val_Macro_F1"]], color='red', s=100, zorder=5, label='Default')
    
    plt.title(f"Sensitivity Analysis: {hparam}")
    plt.xlabel(hparam)
    plt.ylabel("Validation Macro F1")
    plt.ylim(max(0, df_plot["Val_Macro_F1"].min() - 0.05), min(1.0, df_plot["Val_Macro_F1"].max() + 0.05))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_dir / f"hparam_sensitivity_{hparam}.png", dpi=150)
    plt.close()

def main():
    root = find_project_root()
    base_cfg = load_yaml("configs/train.yaml", root=root)
    
    # Ensure seed is explicitly 42
    base_cfg["random_seed"] = 42
    
    # Define sweep grid
    grid = {
        "learning_rate": [1e-5, 2e-5, 3e-5, 5e-5],
        "batch_size": [8, 16, 32],
        "dropout_rate": [0.1, 0.2, 0.3],
        "max_length": [64, 128, 256],
        "warmup_ratio": [0.0, 0.05, 0.1, 0.2]
    }
    
    results = []
    
    # Store default values to avoid re-running the default config multiple times
    default_vals = {k: base_cfg.get(k, grid[k][0]) for k in grid.keys()}
    
    # Run the absolute default config first
    print("\n=== RUNNING DEFAULT CONFIGURATION ===")
    cfg = copy.deepcopy(base_cfg)
    t0 = time.perf_counter()
    out = train_from_config(cfg, project_root=root, show_progress=False, save_artifacts=False)
    default_time = time.perf_counter() - t0
    
    # Record the default result for ALL hparams
    for hparam, val in default_vals.items():
        results.append({
            "Hyperparameter": hparam,
            "Value": val,
            "Val_Macro_F1": out["best_val_score"],
            "Val_Micro_F1": out.get("test_micro_f1", 0.0), # using best val micro approx if available
            "Epochs_Run": out["best_epoch"],
            "Time_Sec": default_time,
            "Is_Default": True
        })
    
    default_macro_f1 = out["best_val_score"]
    
    for hparam, values in grid.items():
        print(f"\n{'='*80}")
        print(f"=== SWEEPING: {hparam} ===")
        print(f"{'='*80}\n")
        
        for val in values:
            if val == default_vals[hparam]:
                continue # Already ran default
                
            print(f"Testing {hparam} = {val}...")
            cfg = copy.deepcopy(base_cfg)
            cfg[hparam] = val
            
            # Special case for batch_size if we need to adjust accumulation
            if hparam == "batch_size" and val == 32:
                # If memory is an issue, force accumulation instead of physical batch size
                cfg["batch_size"] = 16
                cfg["gradient_accumulation_steps"] = 2
                
            t0 = time.perf_counter()
            try:
                out = train_from_config(cfg, project_root=root, show_progress=False, save_artifacts=False)
                run_time = time.perf_counter() - t0
                results.append({
                    "Hyperparameter": hparam,
                    "Value": val,
                    "Val_Macro_F1": out["best_val_score"],
                    "Val_Micro_F1": out.get("test_micro_f1", 0.0),
                    "Epochs_Run": out["best_epoch"],
                    "Time_Sec": run_time,
                    "Is_Default": False
                })
            except Exception as e:
                print(f"Run failed for {hparam}={val}: {str(e)}")
                # Continue with sweep
                
    # Save CSV
    df = pd.DataFrame(results)
    results_dir = root / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "hparam_sensitivity.csv", index=False)
    
    # Generate Plots
    figures_dir = root / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    
    sensitivities = []
    
    for hparam in grid.keys():
        df_sub = df[df["Hyperparameter"] == hparam].copy()
        if len(df_sub) > 1:
            plot_sensitivity(df_sub, hparam, default_vals[hparam], figures_dir)
            # Calculate sensitivity (max - min)
            spread = df_sub["Val_Macro_F1"].max() - df_sub["Val_Macro_F1"].min()
            sensitivities.append((hparam, spread))
            
    # Write Summary Text
    sensitivities.sort(key=lambda x: x[1], reverse=True)
    
    with (results_dir / "hparam_sensitivity_summary.txt").open("w", encoding="utf-8") as f:
        f.write("=== HYPERPARAMETER SENSITIVITY SUMMARY ===\n\n")
        f.write("Ranked from most sensitive to least sensitive (by Val Macro F1 spread):\n\n")
        for hparam, spread in sensitivities:
            f.write(f"1. {hparam}: spread = {spread:.4f}\n")
            
    print(f"\nSensitivity sweep complete! Results in {results_dir} and {figures_dir}")

if __name__ == "__main__":
    main()
