#!/usr/env/bin python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from itertools import combinations
import torch

import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from req_ambiguity.utils.config import find_project_root, load_yaml
from req_ambiguity.evaluation.per_label_diagnostics import load_model_and_data
from req_ambiguity.xai.integrated_gradients import AmbiguityExplainer

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    if len(s1.union(s2)) == 0:
        return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))

def main():
    root = find_project_root()
    cfg = load_yaml("configs/train.yaml", root=root)
    results_dir = root / "outputs/xai/results"
    figures_dir = root / "outputs/xai/figures"
    
    bridge_path = results_dir / "bridge_validation_per_story.csv"
    if not bridge_path.exists():
        print("Bridge validation results not found.")
        return
        
    df = pd.read_csv(bridge_path)
    
    records = []
    for label in df["Label"].unique():
        ldf = df[df["Label"] == label]
        
        words = []
        for ev in ldf["EvidenceTokens"].dropna():
            words.extend(ev.split(","))
            
        counter = Counter([w.strip() for w in words if w.strip()])
        total_stories = len(ldf)
        
        top_words = counter.most_common(15)
        for w, count in top_words:
            records.append({
                "Label": label,
                "Word": w,
                "Frequency": count,
                "Pct_Of_Stories": count / total_stories if total_stories > 0 else 0
            })
            
        if top_words:
            plt.figure(figsize=(8, 5))
            words_plot = [w for w, c in top_words[::-1]]
            counts_plot = [c for w, c in top_words[::-1]]
            plt.barh(words_plot, counts_plot, color="skyblue")
            plt.title(f"Top 15 Attributed Words for {label}")
            plt.xlabel("Frequency")
            plt.tight_layout()
            plt.savefig(figures_dir / f"top_words_{label.replace('/', '_')}.png", dpi=150)
            plt.close()

    pd.DataFrame(records).to_csv(results_dir / "top_attributed_words_per_label.csv", index=False)
    
    # Setup for Jaccard Overlap
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model, test_loader, label_cols = load_model_and_data(cfg, root, device)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    explainer = AmbiguityExplainer(model, tokenizer, device, label_cols)

    sample_b_path = root / "outputs/xai/samples/bridge_validation_sample.csv"
    if sample_b_path.exists():
        df_b = pd.read_csv(sample_b_path)
        
        overlap_records = []
        for _, row in df_b.iterrows():
            story_id = row["StoryID"]
            text = row["StoryText"]
            tp_labels = str(row["is_true_positive_for_label"]).split(",") if pd.notna(row["is_true_positive_for_label"]) else []
            tp_labels = [l for l in tp_labels if l]
            
            if len(tp_labels) >= 2:
                # get evidence tokens for each label
                ev_map = {}
                for l in tp_labels:
                    top_k = explainer.top_evidence_tokens(text, l, top_k=5, story_id=str(story_id))
                    ev_map[l] = [t for t, _ in top_k]
                    
                for l1, l2 in combinations(tp_labels, 2):
                    score = jaccard_similarity(ev_map[l1], ev_map[l2])
                    overlap_records.append({
                        "StoryID": story_id,
                        "Label1": l1,
                        "Label2": l2,
                        "JaccardOverlap": score
                    })
        
        if overlap_records:
            overlap_df = pd.DataFrame(overlap_records)
            # Make undirected pairs
            overlap_df["Pair"] = overlap_df.apply(lambda r: tuple(sorted([r["Label1"], r["Label2"]])), axis=1)
            mean_overlap = overlap_df.groupby("Pair")["JaccardOverlap"].mean().reset_index()
            mean_overlap["Label1"] = mean_overlap["Pair"].apply(lambda x: x[0])
            mean_overlap["Label2"] = mean_overlap["Pair"].apply(lambda x: x[1])
            mean_overlap.drop(columns=["Pair"], inplace=True)
            mean_overlap.to_csv(results_dir / "cross_label_overlap.csv", index=False)
            print("\nCross-Label Overlap Summary:")
            print(mean_overlap.to_string())

    print("Summary generation complete.")

if __name__ == "__main__":
    main()
