import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.req_ambiguity.refinement.prompt_builder import PromptBuilder

def main():
    # Setup paths
    root_dir = Path(__file__).parent.parent.parent
    xai_record_path = root_dir / "outputs" / "xai" / "json" / "US-C1-005.json"
    output_dir = root_dir / "outputs" / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "sample_assembled_prompt.txt"
    
    if not xai_record_path.exists():
        print(f"Error: Could not find XAI record at {xai_record_path}")
        # Try to find any other record
        json_dir = root_dir / "outputs" / "xai" / "json"
        if json_dir.exists():
            files = list(json_dir.glob("*.json"))
            if files:
                xai_record_path = files[0]
                print(f"Using alternative record: {xai_record_path}")
            else:
                print("No JSON records found.")
                return
        else:
            print("XAI JSON directory does not exist.")
            return

    # Load XAI record
    with open(xai_record_path, 'r', encoding='utf-8') as f:
        raw_record = json.load(f)
        
    # Transform raw JSON into the format PromptBuilder expects
    xai_record = {
        "story_text": raw_record.get("original_text", ""),
        "labels": {},
        "bridge_selections": []
    }
    
    if "label_explanations" in raw_record:
        for label, exp in raw_record["label_explanations"].items():
            # Mock above_threshold based on presence in label_explanations
            xai_record["labels"][label] = {
                "above_threshold": True,
                "top_evidence_tokens": [t["token"].replace(" ", "").replace("_", "") for t in exp.get("top_evidence_tokens", [])]
            }
            # Extract bridge selections
            for bs in exp.get("bridge_selections", []):
                xai_record["bridge_selections"].append(bs["placeholder"])
                
    # We must ensure bridge_selections exists for PromptBuilder
    if not xai_record['bridge_selections']:
        # Mocking bridge selections for the test if not present
        print("Adding mock bridge_selections for testing...")
        xai_record['bridge_selections'] = ["<TBD_ACTION_SPECIFICATION>", "<TBD_SCOPE_ENTITY>"]

    # Instantiate PromptBuilder
    builder = PromptBuilder()
    
    # Extract variables
    original_story = xai_record["story_text"]
    active_labels = [label for label, info in xai_record["labels"].items() if info.get("above_threshold", False)]
    
    seen = set()
    evidence_tokens_ordered = []
    for label in active_labels:
        tokens = xai_record["labels"][label].get("top_evidence_tokens", [])[:5]
        for tok in tokens:
            if tok not in seen:
                seen.add(tok)
                evidence_tokens_ordered.append(tok)
                
    allowed_placeholders = xai_record['bridge_selections']

    # Render prompt
    prompt_text = builder.render_for_inspection(original_story, active_labels, evidence_tokens_ordered, allowed_placeholders)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(prompt_text)
        
    print(f"Success! Assembled prompt saved to {output_file}")
    print("\n--- ASSEMBLED PROMPT PREVIEW ---")
    print(prompt_text[:1000] + "\n...\n[TRUNCATED FOR PREVIEW]\n...\n" + prompt_text[-500:])

if __name__ == "__main__":
    main()
