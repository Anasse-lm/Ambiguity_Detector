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

        # Build lookup: label -> list of placeholder names
        self.label_to_placeholders: Dict[str, List[str]] = {}
        for label, ph_list in self.placeholders.items():
            if isinstance(ph_list, list):
                self.label_to_placeholders[label] = [ph["name"] for ph in ph_list]

        # Build lookup: placeholder name -> set of trigger words
        self.placeholder_to_triggers: Dict[str, List[str]] = {}
        for ph_name, trigger_def in self.trigger_map.items():
            if not isinstance(trigger_def, dict):
                continue  # skip metadata keys like 'version'
            triggers = []
            for t in trigger_def.get("triggers", []):
                if isinstance(t, dict) and "word" in t:
                    triggers.append(t["word"].lower())
                elif isinstance(t, str):
                    triggers.append(t.lower())
            self.placeholder_to_triggers[ph_name] = triggers

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

    def select_placeholders(self, active_labels: List[str], evidence_tokens_per_label: Dict[str, List[str]]) -> List[str]:
        """
        For each active label, find placeholders whose trigger words overlap with
        the IG-derived evidence tokens for that label.

        Parameters
        ----------
        active_labels : list of predicted label names above threshold
        evidence_tokens_per_label : dict mapping label -> list of cleaned token strings

        Returns
        -------
        Deduplicated list of matched placeholder names.
        """
        selected = []
        for label in active_labels:
            candidate_placeholders = self.label_to_placeholders.get(label, [])
            evidence_tokens_for_label = [t.lower() for t in evidence_tokens_per_label.get(label, [])]

            for placeholder in candidate_placeholders:
                triggers = self.placeholder_to_triggers.get(placeholder, [])
                # Check if any IG evidence token overlaps (substring) with any trigger
                if any(
                    trigger in token or token in trigger
                    for token in evidence_tokens_for_label
                    for trigger in triggers
                ):
                    selected.append(placeholder)

        return list(set(selected))  # deduplicate
