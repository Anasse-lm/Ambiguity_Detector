#!/usr/env/bin python3
import os
import argparse
import pandas as pd
import torch
from pathlib import Path
from tqdm.auto import tqdm

import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from req_ambiguity.utils.config import load_yaml, find_project_root
from req_ambiguity.evaluation.per_label_diagnostics import load_model_and_data
from req_ambiguity.xai.integrated_gradients import AmbiguityExplainer
from req_ambiguity.xai.bridge import PlaceholderBridge

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run on just 10 stories")
    args = parser.parse_args()

    root = find_project_root()
    cfg = load_yaml("configs/train.yaml", root=root)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model, test_loader, label_cols = load_model_and_data(cfg, root, device)
    tokenizer = test_loader.dataset.tokenizer

    explainer = AmbiguityExplainer(model, tokenizer, device, label_cols)
    bridge = PlaceholderBridge()

    sample_b_path = root / "outputs/xai/samples/bridge_validation_sample.csv"
    if not sample_b_path.exists():
        print("Sample B not found.")
        return
        
    df = pd.read_csv(sample_b_path)
    if args.dry_run:
        df = df.head(10)

    results_dir = root / "outputs/xai/results"
    results_dir.mkdir(parents=True, exist_ok=True)

    records = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating Bridge"):
        story_id = row["StoryID"]
        text = row["StoryText"]
        tp_labels = str(row["is_true_positive_for_label"]).split(",") if pd.notna(row["is_true_positive_for_label"]) else []
        tp_labels = [l for l in tp_labels if l]

        for label in tp_labels:
            prob = row.get(f"prob_{label}", 0.0)
            
            top_k_evidence = explainer.top_evidence_tokens(text, label, top_k=5)
            selections = bridge.match_evidence(label, top_k_evidence)
            
            matched_triggers = []
            selected_placeholders = []
            via_fallback = False
            
            for sel in selections:
                selected_placeholders.append(sel["placeholder"])
                matched_triggers.extend(sel["matched_evidence"])
                if sel["via_fallback"]:
                    via_fallback = True
            
            records.append({
                "StoryID": story_id,
                "Label": label,
                "PredictedProb": prob,
                "EvidenceTokens": ",".join([tok for tok, _ in top_k_evidence]),
                "SelectedPlaceholders": ",".join(set(selected_placeholders)),
                "MatchedTriggers": ",".join(set(matched_triggers)),
                "ViaFallback": via_fallback
            })

    out_df = pd.DataFrame(records)
    out_df.to_csv(results_dir / "bridge_validation_per_story.csv", index=False)
    
    agg_records = []
    for label in label_cols:
        label_df = out_df[out_df["Label"] == label]
        if len(label_df) == 0:
            continue
            
        n_eval = len(label_df)
        n_fallback = label_df["ViaFallback"].sum()
        n_hit = n_eval - n_fallback
        
        hit_rate = n_hit / n_eval if n_eval > 0 else 0
        fallback_rate = n_fallback / n_eval if n_eval > 0 else 0
        
        agg_records.append({
            "Label": label,
            "NumStoriesEvaluated": n_eval,
            "NumStoriesWithBridgeHit": n_hit,
            "HitRate": hit_rate,
            "NumStoriesViaFallback": n_fallback,
            "FallbackRate": fallback_rate,
            "MostFrequentMatchedTriggers": "...",
            "MostFrequentSelectedPlaceholders": "..."
        })
        
    agg_df = pd.DataFrame(agg_records)
    agg_df.to_csv(results_dir / "bridge_hit_rate_per_label.csv", index=False)
    
    print("\nBridge Hit Rate Summary:")
    print(agg_df[["Label", "HitRate", "FallbackRate"]].to_string())
    
    passed = agg_df[agg_df["HitRate"] > 0.50]
    if len(passed) >= 5:
        print("\nFALSIFIABLE VERIFICATION PASSED: >50% hit rate on >=5 labels.")
    else:
        print("\nFALSIFIABLE VERIFICATION FAILED: Bridge hit rate too low.")

if __name__ == "__main__":
    main()
