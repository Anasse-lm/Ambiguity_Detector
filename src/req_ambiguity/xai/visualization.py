from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# A pale red for positive (pushing up), pale blue for negative (pushing down)
ATTRIBUTION_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "attribution_cmap", ["#e8f4f8", "#ffffff", "#fce8e6"]
)

def _get_color(score: float, max_score: float) -> str:
    if max_score == 0:
        norm_score = 0
    else:
        norm_score = max(min(score / max_score, 1.0), -1.0)
    
    # Map to [0, 1] for the colormap
    mapped = (norm_score + 1) / 2
    rgba = ATTRIBUTION_CMAP(mapped)
    return mcolors.to_hex(rgba)

def render_html_heatmap(tokens: List[str], attributions: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    max_score = np.max(np.abs(attributions)) if len(attributions) > 0 else 1.0
    
    html = ["<div style='font-family: monospace; line-height: 2.0; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #fff;'>"]
    for tok, score in zip(tokens, attributions):
        clean_tok = tok.replace(" ", " ") if " " in tok else tok
        clean_tok = clean_tok.replace("<", "&lt;").replace(">", "&gt;")
        
        # skip special tokens for clean display
        if clean_tok in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>"]:
            continue
            
        color = _get_color(float(score), float(max_score))
        html.append(f"<span style='background-color: {color}; padding: 2px 4px; margin: 0 1px; border-radius: 3px; color: #000;' title='score: {score:.4f}'>{clean_tok}</span>")
        
    html.append("</div>")
    
    with out_path.open("w", encoding="utf-8") as f:
        f.write("".join(html))

def render_png_heatmap(tokens: List[str], attributions: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis("off")
    
    max_score = np.max(np.abs(attributions)) if len(attributions) > 0 else 1.0
    
    x, y = 0.01, 0.8
    for tok, score in zip(tokens, attributions):
        clean_tok = tok.replace(" ", " ")
        if clean_tok in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>"]:
            continue
            
        color = _get_color(float(score), float(max_score))
        t = ax.text(x, y, clean_tok, fontsize=12, bbox=dict(facecolor=color, edgecolor='#ccc', boxstyle='round,pad=0.2'))
        
        fig.canvas.draw()
        bbox = t.get_window_extent()
        width = bbox.width / fig.dpi
        
        x += (width / 10) + 0.02
        if x > 0.95:
            x = 0.01
            y -= 0.3
            
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def render_text_annotation(tokens: List[str], attributions: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for tok, score in zip(tokens, attributions):
            f.write(f"{tok}: {score:.4f}\n")
