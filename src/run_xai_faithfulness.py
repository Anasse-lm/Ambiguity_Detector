#!/usr/env/bin python3
import os
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm

import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from req_ambiguity.utils.config import load_yaml, find_project_root
from req_ambiguity.evaluation.per_label_diagnostics import load_model_and_data
from req_ambiguity.xai.integrated_gradients import AmbiguityExplainer

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    root = find_project_root()
    cfg = load_yaml("configs/train.yaml", root=root)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    model, test_loader, label_cols = load_model_and_data(cfg, root, device)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    explainer = AmbiguityExplainer(model, tokenizer, device, label_cols)

    sample_c_path = root / "outputs/xai/samples/faithfulness_sample.csv"
    if not sample_c_path.exists():
        print("Sample C not found.")
        return
        
    df = pd.read_csv(sample_c_path)

    results_dir = root / "outputs/xai/results"
    figures_dir = root / "outputs/xai/figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    records = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Faithfulness Evaluation"):
        story_id = row["StoryID"]
        text = row["StoryText"]
        tp_labels = str(row["predicted_labels"]).split(",") if pd.notna(row["predicted_labels"]) else []
        tp_labels = [l for l in tp_labels if l]

        for label in tp_labels:
            if f"conf_band_{label}" in row:
                conf_band = row[f"conf_band_{label}"]
            else:
                conf_band = "Unknown"
                
            label_idx = label_cols.index(label)
            
            encoding = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            
            with torch.no_grad():
                orig_logits = model(input_ids, attention_mask)
                orig_prob = torch.sigmoid(orig_logits)[0, label_idx].item()
                
            _, attributions = explainer.explain(text, label, story_id=str(story_id))
            
            special_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
            
            scores_with_idx = []
            for i, (tok_id, attr) in enumerate(zip(input_ids[0].tolist(), attributions)):
                if tok_id not in special_ids:
                    scores_with_idx.append((i, abs(attr)))
            
            scores_with_idx.sort(key=lambda x: x[1], reverse=True)
            top_k_indices = [idx for idx, _ in scores_with_idx[:5]]
            
            comp_input_ids = input_ids.clone()
            for idx in top_k_indices:
                comp_input_ids[0, idx] = tokenizer.pad_token_id
            
            with torch.no_grad():
                comp_logits = model(comp_input_ids, attention_mask)
                comp_prob = torch.sigmoid(comp_logits)[0, label_idx].item()
            comprehensiveness = orig_prob - comp_prob
            
            suff_input_ids = input_ids.clone()
            for i, tok_id in enumerate(input_ids[0].tolist()):
                if tok_id not in special_ids and i not in top_k_indices:
                    suff_input_ids[0, i] = tokenizer.pad_token_id
            
            with torch.no_grad():
                suff_logits = model(suff_input_ids, attention_mask)
                suff_prob = torch.sigmoid(suff_logits)[0, label_idx].item()
            sufficiency = orig_prob - suff_prob
            
            records.append({
                "StoryID": story_id,
                "Label": label,
                "ConfBand": conf_band,
                "OriginalProb": orig_prob,
                "Comprehensiveness": comprehensiveness,
                "Sufficiency": sufficiency
            })

    out_df = pd.DataFrame(records)
    
    # Per label aggregation
    agg_records = []
    for label in label_cols:
        ldf = out_df[out_df["Label"] == label]
        if len(ldf) > 0:
            agg_records.append({
                "Label": label,
                "N_Stories": len(ldf),
                "Comprehensiveness_Mean": ldf["Comprehensiveness"].mean(),
                "Comprehensiveness_Std": ldf["Comprehensiveness"].std(),
                "Sufficiency_Mean": ldf["Sufficiency"].mean(),
                "Sufficiency_Std": ldf["Sufficiency"].std()
            })
            
    agg_df = pd.DataFrame(agg_records)
    agg_df.to_csv(results_dir / "faithfulness_per_label.csv", index=False)
    
    # Per confidence band aggregation
    conf_agg_records = []
    for band in ["High", "Medium", "Low", "Unknown"]:
        bdf = out_df[out_df["ConfBand"] == band]
        if len(bdf) > 0:
            conf_agg_records.append({
                "ConfBand": band,
                "N_Stories": len(bdf),
                "Comprehensiveness_Mean": bdf["Comprehensiveness"].mean(),
                "Comprehensiveness_Std": bdf["Comprehensiveness"].std(),
                "Sufficiency_Mean": bdf["Sufficiency"].mean(),
                "Sufficiency_Std": bdf["Sufficiency"].std()
            })
    pd.DataFrame(conf_agg_records).to_csv(results_dir / "faithfulness_per_confidence_band.csv", index=False)

    overall_mean_comp = agg_df["Comprehensiveness_Mean"].mean() if len(agg_df) > 0 else 0
    print(f"\nOverall Mean Comprehensiveness: {overall_mean_comp:.4f}")
    if overall_mean_comp > 0.20:
        print("FALSIFIABLE VERIFICATION PASSED: Mean Comprehensiveness > 0.20")
    else:
        print("FALSIFIABLE VERIFICATION FAILED: Mean Comprehensiveness too low")
        
    if len(agg_df) > 0:
        plt.figure(figsize=(10, 6))
        x = np.arange(len(agg_df))
        width = 0.35
        plt.bar(x - width/2, agg_df["Comprehensiveness_Mean"], width, yerr=agg_df["Comprehensiveness_Std"], label='Comprehensiveness')
        plt.bar(x + width/2, agg_df["Sufficiency_Mean"], width, yerr=agg_df["Sufficiency_Std"], label='Sufficiency')
        plt.ylabel('Score')
        plt.title('XAI Faithfulness per Label')
        plt.xticks(x, agg_df["Label"], rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "faithfulness_per_label.png", dpi=150)
        plt.close()

if __name__ == "__main__":
    main()
