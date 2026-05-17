import json
import yaml
from pathlib import Path
from typing import List, Dict, Any

class PromptBuilder:
    """
    Constructs the prompt for the Refinement module based on a
    strict template-driven approach using configs/refinement_prompt_template.txt.
    """
    def __init__(self, template_path: str = "configs/refinement_prompt_template.txt", placeholders_path: str = "configs/placeholders.yaml"):
        # Load the prompt template
        template_file = Path(template_path)
        if not template_file.exists():
            raise FileNotFoundError(f"Prompt template missing: {template_path}")
        with open(template_file, 'r', encoding='utf-8') as f:
            self.template = f.read()
            
        # Load the placeholders mapping to build descriptions
        placeholders_file = Path(placeholders_path)
        if not placeholders_file.exists():
            raise FileNotFoundError(f"Placeholders file missing: {placeholders_path}")
        with open(placeholders_file, 'r', encoding='utf-8') as f:
            placeholders_map = yaml.safe_load(f)
            
        self.placeholder_descriptions = {}
        for category, ph_list in placeholders_map.items():
            if isinstance(ph_list, list):
                for ph in ph_list:
                    self.placeholder_descriptions[ph['name']] = ph.get('description', '').strip()
                
    def build_prompt(self, original_story: str, active_labels: List[str], evidence_tokens: List[str], allowed_placeholders: List[str], previous_error: str = None) -> str:
        """
        Builds the prompt by filling the 4 slots in the template:
        {ORIGINAL_STORY}, {ACTIVE_LABELS}, {EVIDENCE_TOKENS}, {ALLOWED_PLACEHOLDERS}
        """
        active_labels_str = ", ".join(active_labels)
        evidence_tokens_str = ", ".join(f'"{tok}"' for tok in evidence_tokens)
        
        # d. Extract allowed placeholders
        lines = []
        for placeholder_name in allowed_placeholders:
            desc = self.placeholder_descriptions.get(placeholder_name, "(no description available)")
            lines.append(f"{placeholder_name}: {desc}")
        allowed_placeholders_str = "\n".join(lines)
        
        # e. Fill the template
        base_prompt = self.template.format(
            ORIGINAL_STORY=original_story,
            ACTIVE_LABELS=active_labels_str,
            EVIDENCE_TOKENS=evidence_tokens_str,
            ALLOWED_PLACEHOLDERS=allowed_placeholders_str,
        )
        
        if previous_error:
            # Delegate to build_retry_prompt logic if previous_error is provided
            retry_instruction = (
                "\n\n============================================================\n"
                "RETRY INSTRUCTION\n"
                "============================================================\n"
                "Your previous response had the following issue:\n"
                f"{previous_error}\n\n"
                "Please correct the issue and return only valid JSON with legal "
                "placeholders from the allowed list."
            )
            return base_prompt + retry_instruction
            
        return base_prompt

    def build_retry_prompt(self, original_story: str, active_labels: List[str], evidence_tokens: List[str], allowed_placeholders: List[str], previous_error: str) -> str:
        """
        Builds the prompt and appends a retry instruction for validation failures.
        """
        base_prompt = self.build_prompt(original_story, active_labels, evidence_tokens, allowed_placeholders)
        retry_instruction = (
            "\n\n============================================================\n"
            "RETRY INSTRUCTION\n"
            "============================================================\n"
            "Your previous response had the following issue:\n"
            f"{previous_error}\n\n"
            "Please correct the issue and return only valid JSON with legal "
            "placeholders from the allowed list."
        )
        return base_prompt + retry_instruction
        
    def render_for_inspection(self, original_story: str, active_labels: List[str], evidence_tokens: List[str], allowed_placeholders: List[str]) -> str:
        """
        Return the assembled prompt that would be sent to the LLM.
        Used for testing and for thesis appendix documentation.
        """
        return self.build_prompt(original_story, active_labels, evidence_tokens, allowed_placeholders)
