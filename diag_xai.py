import torch
from transformers import AutoTokenizer
import yaml
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent / "src"))
from req_ambiguity.modeling.classifier import DeBERTaAmbiguityClassifier
from req_ambiguity.xai.integrated_gradients import AmbiguityExplainer
from req_ambiguity.xai.visualization import render_html_heatmap

def run_diag_xai():
    with open('configs/train.yaml', 'r') as f:
        train_config = yaml.safe_load(f)
        
    label_cols = train_config['label_cols']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = train_config['model_name']
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DeBERTaAmbiguityClassifier(model_name, len(label_cols))
    model.load_state_dict(torch.load('outputs/checkpoints/best_model.pt', map_location=device))
    model.eval()
    model.to(device)
    
    explainer = AmbiguityExplainer(model, tokenizer, device, label_cols)
    
    story = "As a doctor, I would like to update system to save time so that I can ensure quality."
    print("Explaining SemanticAmbiguity...")
    try:
        attributions = explainer.explain_label(story, "SemanticAmbiguity")
        print("Tokens:", attributions['tokens'])
        print("Attributions (len):", len(attributions['attributions']))
        print("Top evidence:", attributions['top_evidence_tokens'])
        
        out_path = Path("outputs/refinement/temp_heatmap.html")
        render_html_heatmap(attributions['tokens'], np.array(attributions['attributions']), out_path)
        with open(out_path, 'r') as f:
            html_content = f.read()
            print("HTML generated size:", len(html_content))
            print("HTML snippet:", html_content[:200])
    except Exception as e:
        print("Error during XAI:", e)

if __name__ == "__main__":
    run_diag_xai()
