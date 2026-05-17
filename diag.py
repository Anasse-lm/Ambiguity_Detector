import torch
from transformers import AutoTokenizer
import yaml
import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))
from req_ambiguity.modeling.classifier import DeBERTaAmbiguityClassifier

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_diag():
    train_config = load_config('configs/train.yaml')
    label_cols = train_config['label_cols']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = train_config['model_name']
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DeBERTaAmbiguityClassifier(model_name, len(label_cols))
    
    # Check what is in the checkpoint before loading
    ckpt = torch.load('outputs/checkpoints/best_model.pt', map_location=device)
    print("Checkpoint type:", type(ckpt))
    if isinstance(ckpt, dict):
        print("Checkpoint keys:", list(ckpt.keys())[:10], "...")
        print("Model keys:", list(model.state_dict().keys())[:10], "...")
        
        # Load state dict
        model.load_state_dict(ckpt)
    
    model.eval()
    model.to(device)
    
    with open('outputs/results/optimal_thresholds.json', 'r') as f:
        thresholds = json.load(f)
        
    print("Thresholds:", thresholds)
        
    test_stories = [
        "As a user, I want to update records.", # Not ambiguous
        "As a doctor, I would like to update system to save time so that I can ensure quality.", # Ambiguous
        "As a caregiver, I want to update patient info with fast performance as soon as possible." # Ambiguous
    ]
    
    for story in test_stories:
        print(f"\nStory: {story}")
        inputs = tokenizer(story, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        for i, col in enumerate(label_cols):
            prob = float(probs[i])
            flag = "*" if prob >= thresholds[col] else " "
            print(f"  {flag} {col}: {prob:.4f} (thresh: {thresholds[col]:.4f})")

    print("\nTesting a well-written story:")
    good_story = "As an authenticated system administrator, I want to update the patient allergy records in the central database to ensure clinical safety."
    inputs = tokenizer(good_story, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    probs = torch.sigmoid(logits).cpu().numpy()[0]
    for i, col in enumerate(label_cols):
        prob = float(probs[i])
        flag = "*" if prob >= thresholds[col] else " "
        print(f"  {flag} {col}: {prob:.4f} (thresh: {thresholds[col]:.4f})")

if __name__ == "__main__":
    run_diag()
