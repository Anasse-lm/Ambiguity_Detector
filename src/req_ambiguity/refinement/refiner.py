import logging
from dataclasses import dataclass
from typing import List, Dict, Any

from req_ambiguity.refinement.backends.base import RefinementBackend, RefinementRequest
from req_ambiguity.refinement.prompt_builder import PromptBuilder
from req_ambiguity.refinement.validator import RefinementValidator

logger = logging.getLogger(__name__)

@dataclass
class RefinementOutcome:
    first_attempt_passed: bool
    final_attempt_passed: bool
    attempts_used: int
    validated_response: Dict[str, Any]
    all_attempt_logs: List[Dict[str, Any]]

class Refiner:
    """
    Orchestrates the LLM refinement loop.
    Extracts evidence from the XAI record, builds the prompt, calls the backend,
    validates the output, and automatically retries if validation fails.
    """
    def __init__(self, backend: RefinementBackend, prompt_builder: PromptBuilder, validator: RefinementValidator, config: dict):
        self.backend = backend
        self.prompt_builder = prompt_builder
        self.validator = validator
        self.config = config
        
    def refine(self, story_id: str, xai_record: dict) -> RefinementOutcome:
        original_story = xai_record['original_text']
        active_labels = xai_record.get('predicted_labels', [])
        
        # Extract evidence tokens and allowed placeholders from the XAI record
        evidence_tokens = set()
        allowed_placeholders = set()
        
        for label, exp in xai_record.get('label_explanations', {}).items():
            for tok in exp.get('top_evidence_tokens', []):
                # Clean up DeBERTa token prefix (the ' ' character)
                clean_tok = tok['token'].replace(' ', '').strip()
                if clean_tok:
                    evidence_tokens.add(clean_tok)
                    
            for b in exp.get('bridge_selections', []):
                allowed_placeholders.add(b['placeholder'])
                
        evidence_tokens = list(evidence_tokens)
        allowed_placeholders = list(allowed_placeholders)
        
        all_attempt_logs = []
        previous_error = None
        max_retries = self.config.get('max_retries', 3)
        
        first_attempt_passed = False
        final_attempt_passed = False
        validated_response = None
        attempts_used = 0
        
        for attempt in range(1, max_retries + 1):
            attempts_used = attempt
            prompt_text = self.prompt_builder.build_prompt(
                original_story=original_story,
                active_labels=active_labels,
                evidence_tokens=evidence_tokens,
                allowed_placeholders=allowed_placeholders,
                previous_error=previous_error
            )
            
            request = RefinementRequest(
                prompt_text=prompt_text,
                model_name=self.config.get('model_name', 'gemini-1.5-pro'),
                temperature=self.config.get('temperature', 0.2),
                max_output_tokens=self.config.get('max_output_tokens', 1024),
                top_p=self.config.get('top_p', 0.9)
            )
            
            try:
                response = self.backend.call(request)
            except Exception as e:
                logger.warning(f"Backend call failed for {story_id} on attempt {attempt}: {str(e)}")
                all_attempt_logs.append({
                    "attempt": attempt,
                    "request": prompt_text,
                    "error": str(e)
                })
                previous_error = f"Backend error: {str(e)}"
                continue
                
            validation_result = self.validator.validate(response.text)
            
            all_attempt_logs.append({
                "attempt": attempt,
                "request": prompt_text,
                "response": response.text,
                "validation": {
                    "passed": validation_result.passed,
                    "parse_error": validation_result.parse_error,
                    "schema_errors": validation_result.schema_errors,
                    "illegal_placeholders": validation_result.illegal_placeholders,
                    "inconsistency_warnings": validation_result.inconsistency_warnings
                }
            })
            
            if validation_result.passed:
                if attempt == 1:
                    first_attempt_passed = True
                final_attempt_passed = True
                validated_response = validation_result.parsed_json
                break
            else:
                # Build error message for retry loop
                error_parts = []
                if validation_result.parse_error:
                    error_parts.append(f"JSON Parse Error: {validation_result.parse_error}")
                if validation_result.schema_errors:
                    error_parts.append(f"Schema Errors: {', '.join(validation_result.schema_errors)}")
                if validation_result.illegal_placeholders:
                    error_parts.append(f"Illegal Placeholders Invented: {', '.join(validation_result.illegal_placeholders)}")
                    
                previous_error = " | ".join(error_parts)
                logger.info(f"Story {story_id} failed validation on attempt {attempt}: {previous_error}")
                
        if not final_attempt_passed:
            logger.error(f"Story {story_id} failed all {max_retries} refinement attempts.")
            
        return RefinementOutcome(
            first_attempt_passed=first_attempt_passed,
            final_attempt_passed=final_attempt_passed,
            attempts_used=attempts_used,
            validated_response=validated_response,
            all_attempt_logs=all_attempt_logs
        )
