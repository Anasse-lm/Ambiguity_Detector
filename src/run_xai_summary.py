#!/usr/env/bin python3
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from req_ambiguity.utils.config import find_project_root

def main():
    root = find_project_root()
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
            plt.savefig(figures_dir / f"top_words_{label}.png", dpi=150)
            plt.close()

    pd.DataFrame(records).to_csv(results_dir / "top_attributed_words_per_label.csv", index=False)
    
    failure_records = []
    for _, row in df.head(10).iterrows():
        failure_records.append({
            "StoryID": row["StoryID"],
            "Label": row["Label"],
            "FailureType": "SimulatedFailure",
            "EvidenceTokens": row["EvidenceTokens"],
            "Description": "Placeholder for a failure case."
        })
    pd.DataFrame(failure_records).to_csv(results_dir / "failure_cases.csv", index=False)
    print("Summary generation complete.")

if __name__ == "__main__":
    main()
