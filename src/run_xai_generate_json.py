#!/usr/env/bin python3
import os
import argparse
import pandas as pd
import json
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
    
    model, test_loader, label_cols = load_model_and_data(cfg, root, device)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    explainer = AmbiguityExplainer(model, tokenizer, device, label_cols)
    bridge = PlaceholderBridge()

    sample_b_path = root / "outputs/xai/samples/bridge_validation_sample.csv"
    sample_c_path = root / "outputs/xai/samples/faithfulness_sample.csv"
    
    dfs = []
    if sample_b_path.exists(): dfs.append(pd.read_csv(sample_b_path))
    if sample_c_path.exists(): dfs.append(pd.read_csv(sample_c_path))
    
    if not dfs:
        print("No samples found.")
        return
        
    df = pd.concat(dfs).drop_duplicates(subset=["StoryID"])
    if args.dry_run:
        df = df.head(10)

    out_dir = root / "outputs/xai/json"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_jsonl_path = root / "outputs/xai/all_xai_outputs.jsonl"

    valid_count = 0
    with all_jsonl_path.open("w", encoding="utf-8") as f_jsonl:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating JSON"):
            story_id = row["StoryID"]
            text = row["StoryText"]
            tp_labels = str(row["is_true_positive_for_label"]).split(",") if pd.notna(row["is_true_positive_for_label"]) else []
            tp_labels = [l for l in tp_labels if l]

            label_explanations = {}
            total_cands = 0
            used_fallback = []

            for label in tp_labels:
                label_idx = label_cols.index(label)
                prob = row.get(f"prob_{label}", 0.0)
                
                tokens, attributions = explainer.explain(text, label, story_id=str(story_id))
                
                top_k = explainer.top_evidence_tokens(text, label, top_k=5, story_id=str(story_id))
                
                top_evidence_tokens = [{"token": t, "score": s, "position": 0} for t, s in top_k]
                word_level_attributions = [{"word": t.replace(" ", ""), "score": float(s)} for t, s in zip(tokens, attributions) if t not in tokenizer.all_special_tokens]
                
                selections = bridge.match_evidence(label, top_k)
                
                for sel in selections:
                    total_cands += 1
                    if sel["via_fallback"]:
                        used_fallback.append(sel["placeholder"])
                
                label_explanations[label] = {
                    "label_index": label_idx,
                    "predicted_probability": float(prob),
                    "top_evidence_tokens": top_evidence_tokens,
                    "word_level_attributions": word_level_attributions,
                    "bridge_selections": selections
                }

            record = {
                "story_id": str(story_id),
                "original_text": str(text),
                "predicted_labels": tp_labels,
                "label_explanations": label_explanations,
                "bridge_summary": {
                    "total_candidates": total_cands,
                    "used_fallback": used_fallback
                }
            }

            with (out_dir / f"{story_id}.json").open("w", encoding="utf-8") as sf:
                json.dump(record, sf, indent=2)
                
            f_jsonl.write(json.dumps(record) + "\n")
            valid_count += 1
            
    print(f"Generated {valid_count} JSON records.")

if __name__ == "__main__":
    main()
