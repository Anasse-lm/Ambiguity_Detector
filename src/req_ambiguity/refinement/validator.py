import json
import re
import yaml
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ValidationResult:
    passed: bool
    parse_error: Optional[str]
    schema_errors: List[str]
    illegal_placeholders: List[str]
    inconsistency_warnings: List[str]
    parsed_json: Optional[dict]

class RefinementValidator:
    """
    Validates LLM outputs against 4 strict checks:
    1. JSON validity
    2. Schema conformance
    3. Placeholder legality (no hallucinations)
    4. Self-consistency
    """
    def __init__(self, placeholders_path: str):
        with open(placeholders_path, 'r', encoding='utf-8') as f:
            self.placeholders_map = yaml.safe_load(f)
            
        # Build the flat set of ALL legal placeholder tokens globally for validation
        self.all_legal_placeholders = set()
        for label, placeholders in self.placeholders_map.items():
            if isinstance(placeholders, list):
                for ph in placeholders:
                    self.all_legal_placeholders.add(ph['name'])
                    
    def validate(self, text: str) -> ValidationResult:
        # CHECK 1: JSON validity
        raw_text = text.strip()
        raw_text = re.sub(r"^```json\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()
        raw_text = re.sub(r"^```\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()
        
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as e:
            return ValidationResult(
                passed=False,
                parse_error=str(e),
                schema_errors=[],
                illegal_placeholders=[],
                inconsistency_warnings=[],
                parsed_json=None
            )
            
        # CHECK 2: Schema conformance
        schema_errors = []
        required_keys = {
            'refined_story': str,
            'placeholders_used': list,
            'clarification_questions': list
        }
        
        if not isinstance(parsed, dict):
            return ValidationResult(False, None, ["Root element must be a JSON object"], [], [], parsed)
            
        for key, expected_type in required_keys.items():
            if key not in parsed:
                schema_errors.append(f"Missing required key: '{key}'")
            elif not isinstance(parsed[key], expected_type):
                schema_errors.append(f"Key '{key}' must be of type {expected_type.__name__}")
                
        if schema_errors:
            return ValidationResult(False, None, schema_errors, [], [], parsed)
            
        # CHECK 3: Placeholder legality
        refined = parsed['refined_story']
        found_placeholders = set(re.findall(r"<TBD_[A-Z_]+>", refined))
        illegal_placeholders = list(found_placeholders - self.all_legal_placeholders)
        
        # CHECK 4: Self-consistency
        declared_placeholders = set(parsed['placeholders_used'])
        inconsistency_warnings = []
        
        undeclared = list(found_placeholders - declared_placeholders)
        if undeclared:
            inconsistency_warnings.append(f"Placeholders found in text but not declared in array: {undeclared}")
            
        missing_in_text = list(declared_placeholders - found_placeholders)
        if missing_in_text:
            inconsistency_warnings.append(f"Placeholders declared in array but missing from text: {missing_in_text}")
            
        passed = len(schema_errors) == 0 and len(illegal_placeholders) == 0
        
        return ValidationResult(
            passed=passed,
            parse_error=None,
            schema_errors=schema_errors,
            illegal_placeholders=illegal_placeholders,
            inconsistency_warnings=inconsistency_warnings,
            parsed_json=parsed
        )
