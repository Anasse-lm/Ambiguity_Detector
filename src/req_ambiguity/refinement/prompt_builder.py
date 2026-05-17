import json
import yaml
from typing import List

class PromptBuilder:
    """
    Constructs the prompt for the Refinement module based on the
    strict 4-component specification.
    """
    def __init__(self, placeholders_path: str, few_shot_path: str):
        with open(placeholders_path, 'r', encoding='utf-8') as f:
            self.placeholders_map = yaml.safe_load(f)
            
        with open(few_shot_path, 'r', encoding='utf-8') as f:
            self.few_shot_data = yaml.safe_load(f)
            
    def _build_system_role(self) -> str:
        return """You are a requirements engineering assistant. Refine ambiguous software requirements to reduce ambiguity, WITHOUT inventing information not present in the original. When information is missing, insert a placeholder from the provided controlled vocabulary — NEVER substitute an invented detail. Follow IEEE 29148 quality criteria: the refined requirement should be unambiguous, verifiable, and complete.

Domain conventions to respect:
- Common SE verbs like 'send', 'receive', 'set', 'shutdown' have shared domain meaning and do NOT need to be flagged as ambiguous.
- 'shall' is the only acceptable deterministic mandate verb under ISO/IEC/IEEE 29148. Do not insert placeholders for 'shall'.

Return your answer as valid JSON only.
"""

    def _build_few_shot_examples(self) -> str:
        examples_str = "\n### FEW-SHOT EXAMPLES ###\n"
        for ex in self.few_shot_data.get('examples', []):
            examples_str += f"\n{ex['name']}\n"
            examples_str += f"Original: \"{ex['original']}\"\n"
            examples_str += f"Active labels: {', '.join(ex['active_labels'])}\n"
            examples_str += f"Evidence tokens: {json.dumps(ex['evidence_tokens'])}\n"
            examples_str += f"Allowed placeholders: {', '.join(ex['allowed_placeholders'])}\n"
            examples_str += f"Refined:\n{json.dumps(ex['refined'], indent=2)}\n"
        return examples_str

    def _build_current_request(self, original_story: str, active_labels: List[str], evidence_tokens: List[str], allowed_placeholders: List[str]) -> str:
        # Build descriptions of allowed placeholders
        allowed_descriptions = []
        for label in active_labels:
            if label in self.placeholders_map:
                for ph in self.placeholders_map[label]:
                    if ph['name'] in allowed_placeholders:
                        allowed_descriptions.append(f"{ph['name']}: {ph['description'].strip()}")
                        
        req_str = "\n### CURRENT REQUEST ###\n"
        req_str += f"Original: \"{original_story}\"\n"
        req_str += f"Active labels: {', '.join(active_labels)}\n"
        req_str += f"Evidence tokens identified by explainability module: {', '.join(evidence_tokens)}\n"
        req_str += "Allowed placeholders for this refinement:\n"
        for desc in set(allowed_descriptions):
            req_str += f"  - {desc}\n"
            
        return req_str

    def _build_output_spec(self) -> str:
        return """
### OUTPUT SPECIFICATION ###
Return ONLY valid JSON with these three keys:
  - 'refined_story': the refined requirement text
  - 'placeholders_used': array of placeholder names actually inserted
  - 'clarification_questions': array of questions to resolve each placeholder

Do not wrap in markdown code fences. Do not add prose outside the JSON.
"""

    def build_prompt(self, original_story: str, active_labels: List[str], evidence_tokens: List[str], allowed_placeholders: List[str], previous_error: str = None) -> str:
        prompt = self._build_system_role()
        prompt += self._build_few_shot_examples()
        prompt += self._build_current_request(original_story, active_labels, evidence_tokens, allowed_placeholders)
        prompt += self._build_output_spec()
        
        if previous_error:
            prompt += f"\nYour previous response had the following issue: {previous_error}. Please correct and return only valid JSON with legal placeholders.\n"
            
        return prompt
