import yaml
from pathlib import Path
from typing import List, Tuple, Dict

class PlaceholderBridge:
    def __init__(self, trigger_map_path="configs/trigger_map.yaml", placeholders_path="configs/placeholders.yaml"):
        root = Path(__file__).resolve().parents[3]
        
        with open(root / trigger_map_path, "r", encoding="utf-8") as f:
            self.trigger_map = yaml.safe_load(f)
            
        with open(root / placeholders_path, "r", encoding="utf-8") as f:
            self.placeholders = yaml.safe_load(f)

    def match_evidence(self, label: str, evidence_tokens: List[Tuple[str, float]]) -> List[Dict]:
        """
        Match IG evidence tokens against the trigger map for the given label.
        Returns a list of dicts: placeholder, match_score, matched_evidence, via_fallback.
        """
        results = []
        if label not in self.placeholders:
            return results

        placeholder_defs = self.placeholders[label]
        extracted_tokens = [tok.lower() for tok, score in evidence_tokens]
        
        matched_any = False
        for pdef in placeholder_defs:
            placeholder_name = pdef["name"]
            
            triggers = []
            if placeholder_name in self.trigger_map:
                trigger_items = self.trigger_map[placeholder_name].get("triggers", [])
                for t in trigger_items:
                    if isinstance(t, dict) and "word" in t:
                        triggers.append(t["word"].lower())
                    elif isinstance(t, str):
                        triggers.append(t.lower())
                        
            # Simple substring/token matching
            matched_evidence = []
            for tok in extracted_tokens:
                for t in triggers:
                    if t in tok or tok in t:
                        matched_evidence.append(tok)
                        
            if matched_evidence:
                matched_any = True
                results.append({
                    "placeholder": placeholder_name,
                    "match_score": 1.0,
                    "matched_evidence": list(set(matched_evidence)),
                    "via_fallback": False
                })
                
        # If no triggers match, fallback to the first placeholder defined for that label
        if not matched_any and placeholder_defs:
            results.append({
                "placeholder": placeholder_defs[0]["name"],
                "match_score": 0.0,
                "matched_evidence": [],
                "via_fallback": True
            })
            
        return results
