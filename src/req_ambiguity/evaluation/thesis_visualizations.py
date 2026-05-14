#!/usr/env/bin python3
"""
Generates publication-ready figures for the master thesis using the tracked training history and test diagnostics.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[3]

def main():
    # Use a professional, clean theme suitable for academic papers
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # Paths
    history_path = _ROOT / "outputs" / "reports" / "training" / "history.csv"
    test_report_path = _ROOT / "outputs" / "results" / "per_label_test_report.csv"
    figures_dir = _ROOT / "outputs" / "figures" / "thesis_ready"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating thesis visualizations in {figures_dir}...")
    
    # 1. Plot Learning Curves (Loss & Macro Metrics)
    if history_path.exists():
        df_hist = pd.read_csv(history_path)
        epochs = df_hist['epoch']
        
        # Loss Curve
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, df_hist['train_loss'], label='Train Loss', color='#e74c3c', linewidth=2)
        plt.plot(epochs, df_hist['val_loss'], label='Validation Loss', color='#2980b9', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('BCE Loss')
        plt.title('Training vs Validation Loss over Time')
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "learning_curve_loss.png", dpi=300)
        plt.close()
        
        # Macro Metrics Curve
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, df_hist['val_macro_f1'], label='Macro F1', color='#27ae60', linewidth=2)
        plt.plot(epochs, df_hist['val_macro_precision'], label='Macro Precision', color='#8e44ad', linewidth=2, linestyle='--')
        plt.plot(epochs, df_hist['val_macro_recall'], label='Macro Recall', color='#f39c12', linewidth=2, linestyle='-.')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Validation Aggregate Metrics over Time')
        plt.ylim(0, 1.05)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "macro_metrics_evolution.png", dpi=300)
        plt.close()
        
        # Per-Label F1 Curve
        # Find columns starting with val_f1_
        f1_cols = [c for c in df_hist.columns if c.startswith('val_f1_')]
        if f1_cols:
            plt.figure(figsize=(10, 6))
            colors = sns.color_palette("husl", len(f1_cols))
            for col, color in zip(f1_cols, colors):
                label_name = col.replace("val_f1_", "")
                plt.plot(epochs, df_hist[col], label=label_name, color=color, linewidth=2, alpha=0.8)
                
            plt.xlabel('Epoch')
            plt.ylabel('F1 Score')
            plt.title('Per-Label Validation F1 Score over Time')
            plt.ylim(0, 1.05)
            # Put legend outside the plot
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(figures_dir / "per_label_f1_evolution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
    else:
        print(f"Warning: {history_path} not found. Cannot plot learning curves.")

    # 2. Final Test Performance Bar Chart
    if test_report_path.exists():
        df_test = pd.read_csv(test_report_path)
        # Melt DataFrame for seaborn grouped bar chart
        df_melt = df_test.melt(id_vars='Label', value_vars=['Precision', 'Recall', 'F1'], 
                               var_name='Metric', value_name='Score')
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_melt, x='Label', y='Score', hue='Metric', palette='viridis')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.05)
        plt.title('Final Test Set Performance per Ambiguity Type')
        plt.tight_layout()
        plt.savefig(figures_dir / "final_performance_bar_chart.png", dpi=300)
        plt.close()
    else:
        print(f"Warning: {test_report_path} not found. Run per_label_diagnostics.py first to generate the test report.")

    print("Done! High-resolution thesis figures saved successfully.")

if __name__ == "__main__":
    main()
