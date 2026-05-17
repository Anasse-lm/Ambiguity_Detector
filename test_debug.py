import sys
import os
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from req_ambiguity.refinement.refiner import Refiner
from req_ambiguity.refinement.prompt_builder import PromptBuilder
from req_ambiguity.refinement.backends.gemini import GeminiBackend
from req_ambiguity.refinement.validator import RefinementValidator

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY")
        return
        
    builder = PromptBuilder()
    backend = GeminiBackend(max_retries=1, retry_delay_seconds=1.0)
    validator = RefinementValidator('configs/placeholders.yaml')
    
    refiner = Refiner(
        prompt_builder=builder,
        backend=backend,
        validator=validator,
        config={"max_retries": 1}
    )
    
    xai_record_path = Path(__file__).parent / "outputs" / "xai" / "json" / "US-C1-005.json"
    with open(xai_record_path, 'r') as f:
        xai_record = json.load(f)
        
    print("Testing refiner...")
    outcome = refiner.refine("US-C1-005", xai_record)
    
    print("\n--- OUTCOME ---")
    print(f"Passed: {outcome.passed}")
    print(f"Parsed JSON: {outcome.parsed_json}")
    print(f"Raw Response: {outcome.raw_response}")
    
    if not outcome.passed:
        for log in outcome.attempt_logs:
            print("\nAttempt Error:", log.get('validation_error'))

if __name__ == "__main__":
    main()
