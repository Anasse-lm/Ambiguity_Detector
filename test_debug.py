import sys
import os
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))
from req_ambiguity.refinement.validator import RefinementValidator

def main():
    validator = RefinementValidator('configs/placeholders.yaml')
    
    mock_llm_response = """
{
  "refined_story": "As a investor, I want to <TBD_ACTION_SPECIFICATION> account so that I can improve security by <TBD_URGENCY>",
  "placeholders_used": ["<TBD_ACTION_SPECIFICATION>", "<TBD_URGENCY>"],
  "clarification_questions": [
    "What specific action?",
    "When exactly?"
  ]
}
"""
    print(f"All legal placeholders: {len(validator.all_legal_placeholders)}")
    res = validator.validate(mock_llm_response)
    print("Passed:", res.passed)
    print("Schema Errors:", res.schema_errors)
    print("Parse Error:", res.parse_error)
    print("Illegal Placeholders:", res.illegal_placeholders)
    print("Inconsistency:", res.inconsistency_warnings)

if __name__ == "__main__":
    main()
