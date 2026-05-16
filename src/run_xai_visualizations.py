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

from req_ambiguity.utils.config import load_yaml, find_project_root, resolve_path
from req_ambiguity.evaluation.per_label_diagnostics import load_model_and_data
from req_ambiguity.xai.integrated_gradients import AmbiguityExplainer
from req_ambiguity.xai.visualization import render_html_heatmap, render_png_heatmap, render_text_annotation

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    root = find_project_root()
    cfg = load_yaml("configs/train.yaml", root=root)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model, test_loader, label_cols = load_model_and_data(cfg, root, device)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    explainer = AmbiguityExplainer(model, tokenizer, device, label_cols)

    sample_a_path = root / "outputs/xai/samples/visualization_candidates.csv"
    if not sample_a_path.exists():
        print(f"Sample A not found at {sample_a_path}. Run sample prep first.")
        return
        
    df = pd.read_csv(sample_a_path)

    vis_dir = root / "outputs/xai/visualizations"
    html_dir = vis_dir / "html"
    png_dir = vis_dir / "png"
    text_dir = vis_dir / "text"
    html_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    index_html = ["<html><head><title>XAI Visualizations</title></head><body style='font-family: sans-serif;'><h1>XAI Visualization Contact Sheet</h1>"]

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating Visualizations"):
        story_id = row["StoryID"]
        text = row["StoryText"]
        tp_labels = str(row["is_true_positive_for_label"]).split(",") if pd.notna(row["is_true_positive_for_label"]) else []
        tp_labels = [l for l in tp_labels if l]

        for label in tp_labels:
            tokens, attributions = explainer.explain(text, label, story_id=str(story_id))
            
            base_name = f"{story_id}__{label.replace('/', '_')}"
            
            html_path = html_dir / f"{base_name}.html"
            render_html_heatmap(tokens, attributions, html_path)
            
            png_path = png_dir / f"{base_name}.png"
            render_png_heatmap(tokens, attributions, png_path)
            
            txt_path = text_dir / f"{base_name}.txt"
            render_text_annotation(tokens, attributions, txt_path)

            index_html.append(f"<h2>{story_id} - {label}</h2>")
            with html_path.open("r", encoding="utf-8") as f:
                snippet = f.read()
            index_html.append(snippet)
            index_html.append("<hr>")

    index_html.append("</body></html>")
    with (vis_dir / "index.html").open("w", encoding="utf-8") as f:
        f.write("\n".join(index_html))

    print(f"Visualizations saved to {vis_dir}")

if __name__ == "__main__":
    main()
