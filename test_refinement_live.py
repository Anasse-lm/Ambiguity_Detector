"""
test_refinement_live.py — run from project root
Shows exactly what Gemini returns and why the validator rejects it.

Usage:
    PYTHONPATH=src GEMINI_API_KEY=<your_key> venv/bin/python test_refinement_live.py
"""
import sys, os
from pathlib import Path
sys.path.insert(0, "src")

# ── patch API key from env ──────────────────────────────────────────────────
api_key = os.environ.get("GEMINI_API_KEY", "")
if not api_key:
    print("ERROR: set GEMINI_API_KEY env var before running this script.")
    sys.exit(1)

from req_ambiguity.refinement.prompt_builder import PromptBuilder
from req_ambiguity.refinement.backends.gemini import GeminiBackend
from req_ambiguity.refinement.cache import CachedBackend
from req_ambiguity.refinement.validator import RefinementValidator
from req_ambiguity.refinement.backends.base import RefinementRequest
import yaml

# ── build components ────────────────────────────────────────────────────────
ref_config = yaml.safe_load(open("configs/refinement.yaml"))
backend_raw = GeminiBackend(max_retries=1, retry_delay_seconds=1.0)
backend = CachedBackend(backend_raw, ref_config["cache_dir"], cache_enabled=False)  # disable cache for live test
pb = PromptBuilder()
validator = RefinementValidator("configs/placeholders.yaml")

print(f"\n{'='*60}")
print(f"Legal placeholders ({len(validator.all_legal_placeholders)}):")
for p in sorted(validator.all_legal_placeholders):
    print(f"  {p}")
print(f"{'='*60}\n")

# ── test story ──────────────────────────────────────────────────────────────
story = "As a user, I want to manage account so that I can improve security as soon as possible"
active_labels = ["SemanticAmbiguity", "ScopeAmbiguity"]
evidence_tokens = ["manage", "improve", "security", "soon", "possible"]
allowed_placeholders = ["<TBD_ACTION_SPECIFICATION>", "<TBD_SCOPE_ENTITY>", "<TBD_URGENCY>"]

prompt = pb.build_prompt(
    original_story=story,
    active_labels=active_labels,
    evidence_tokens=evidence_tokens,
    allowed_placeholders=allowed_placeholders,
)

print("PROMPT (last 300 chars):")
print("..." + prompt[-300:])
print(f"\n{'='*60}\n")

request = RefinementRequest(
    prompt_text=prompt,
    model_name=ref_config.get("model_name", "gemini-1.5-pro"),
    temperature=ref_config.get("temperature", 0.2),
    max_output_tokens=ref_config.get("max_output_tokens", 1024),
    top_p=ref_config.get("top_p", 0.9),
)

print("Calling Gemini...")
response = backend.call(request)
print(f"\nRAW RESPONSE:\n{response.text}\n")
print(f"{'='*60}\n")

# ── validate ────────────────────────────────────────────────────────────────
result = validator.validate(response.text)
print(f"VALIDATION RESULT:")
print(f"  passed               : {result.passed}")
print(f"  parse_error          : {result.parse_error}")
print(f"  schema_errors        : {result.schema_errors}")
print(f"  illegal_placeholders : {result.illegal_placeholders}")
print(f"  inconsistency_warnings: {result.inconsistency_warnings}")
print(f"  parsed_json          : {result.parsed_json}")
