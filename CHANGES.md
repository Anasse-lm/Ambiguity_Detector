# Phase 9: Streamlit Interactive Demo App

## Implementation Summary
The interactive Streamlit demonstration app brings together the DeBERTa ambiguity classifier, Captum XAI integrated gradients, and Gemini LLM refinement into a single web interface. 

It implements a unified batch review workflow across three input modes: Single Story, Multiple Stories (newline separated), and Document Upload (CSV, TXT, DOCX). It automatically handles model pre-warming to minimize CPU latency on the first request and enforces strict validation bounds (e.g., maximum 50 stories per batch) to ensure system stability.

## Architecture and Framing
**LLM Scoping**: The demo uses Google Gemini as its single LLM backend, consistent with the empirical evaluation in earlier phases. The underlying `RefinementBackend` interface is pluggable, supporting future multi-LLM comparison as identified future work. The user-facing interface does not expose a model picker to maintain consistency between what the demo shows and what was empirically evaluated.

**Batch Evaluation**: The batch upload capability demonstrates the system handles realistic refinement workflows even though the empirical evaluation was conducted on the per-story batch evaluator on 210 stories.

## Infrastructure for Future User Studies
The session logging mechanism records every user interaction in a structured SQLite log including input mode, batch progression, accept/regenerate/skip decisions, and timing. **This thesis does not conduct a formal user study** because recruiting practicing requirements engineers was not feasible within the thesis scope. The log infrastructure exists as future-work scaffolding: a future researcher could attach a user study protocol to the existing logging interface without modifying the demo. Aggregate session statistics derived from the log are reported in the thesis Discussion chapter as descriptive evidence about system usage, not as evaluative claims about user preferences.

## How to Launch
1. Open your terminal in the `Solution` directory.
2. (Optional but recommended) Export your API key:
   `export GEMINI_API_KEY="your-api-key"`
3. Install dependencies:
   `pip install -r requirements.txt`
4. Run the app:
   `streamlit run app/streamlit_demo.py`

## Defense-Day Checklist
- [ ] Ensure `GEMINI_API_KEY` is active.
- [ ] Run `streamlit run app/streamlit_demo.py` beforehand to allow the model to pre-warm on the CPU.
- [ ] Load the Single Story mode and use the pre-loaded examples.
- [ ] Prepare a small 5-10 row CSV file if demonstrating the Document Upload feature.
- [ ] Familiarize yourself with the "New Session" button to clear state between examiner questions.

## Sample Document Formats
- **CSV**: Ensure there is a column named `StoryText`, `story`, or `text`.
- **TXT**: Separate stories with a blank line (double newline).
- **DOCX**: Each paragraph is parsed as a single story.

## Known Limitations
- **CPU Latency**: Because the DeBERTa model runs on CPU natively on Mac, processing a story takes ~10-15 seconds.
- **50-Story Cap**: To prevent API quota exhaustion and long processing delays, batches are capped at 50 stories.
- **Single-LLM Backend**: Currently hardcoded to the Gemini flash model via API, with no open-source LLM fallback.

# Phase 10: Prompt Architecture Refactor (v1.2)

## Implementation Summary
- Extracted the programmatic refinement prompt into a fixed, inspectable template at `configs/refinement_prompt_template.txt`.
- `PromptBuilder` now loads this template at initialization and fills four runtime variable slots: `{ORIGINAL_STORY}`, `{ACTIVE_LABELS}`, `{EVIDENCE_TOKENS}`, and `{ALLOWED_PLACEHOLDERS}`.
- The template includes three worked few-shot examples covering `SemanticAmbiguity`, `ActorAmbiguity`+`ScopeAmbiguity`, and `TechnicalAmbiguity`+`PriorityAmbiguity`.
- Enforces six critical rules: no fact invention, restricted placeholder vocabulary, evidence-token grounding, structure preservation, IEEE 29148 quality criteria, and strictly formatted JSON output.
- Incorporates domain conventions explicitly excluding common software engineering verbs and "shall" from being flagged as ambiguous.
- Added `src/debug/test_prompt_assembly.py` utility for inspecting the assembled prompt without querying the LLM.

# Phase 11: Structural-Template-Attribution Diagnostic

## Implementation Summary
- Added `configs/structural_tokens.yaml` listing 23 English user-story template scaffold tokens (as, a, an, the, i, want, to, so, that, in, order, for, of, be, able, and special tokens [CLS]/[SEP]/[PAD]/[UNK]).
- Added `src/req_ambiguity/xai/attribution_diagnostic.py` implementing `AttributionDiagnostic` class:
  - Computes structural-vs-content attribution fraction from per-token IG scores.
  - Prints a formatted terminal report per story (token counts, attribution mass split, per-label breakdown, top-5 filtered/unfiltered evidence tokens, HEALTHY/BORDERLINE/WARNING assessment).
  - Thresholds: ≤33% structural = HEALTHY, 33–50% = BORDERLINE, >50% = WARNING.
  - Returns a diagnostic dict for optional session logging.
- Modified `src/req_ambiguity/xai/bridge.py`:
  - Loads `structural_tokens.yaml` at `__init__` time.
  - Filters structural tokens from IG evidence before trigger-map lookup in both `match_evidence()` and `select_placeholders()`.
  - Prints stdout log of filtered tokens for traceability.
- Wired diagnostic into `app/streamlit_demo.py` after the XAI step, logging results to the session DB under `STORY_DIAGNOSTIC` event type.
- Added `src/debug/test_diagnostic.py` for batch inspection of three sample XAI files.
- Verified: `outputs/debug/diagnostic_samples.txt` generated successfully with three reports.

# Phase 12: Structural Token Filter Audit and Strengthening

## Issues Found
- Original `structural_tokens.yaml` had only 23 entries; punctuation tokens (`,`),
  modal verbs (`can`, `have`, `will`), and pronouns (`it`, `this`, `they`) leaked through.
- Normalization function did not strip punctuation, so `","` was never matched.
- Trigger map had no documentation of structural-override entries.

## Changes Made
- **`configs/structural_tokens.yaml`**: Expanded from 23 to 52+ entries. Added pronouns,
  modals, auxiliaries, connectors, punctuation, and special tokens. Added explanatory
  header documenting the precedence rule and intentional optionality-trigger overlap.
- **`src/req_ambiguity/xai/bridge.py`**: Full rewrite with robust `normalize_token()`
  (strips BPE prefixes + punctuation), pre-normalized structural set at init,
  `_filter_structural()` helper, and verbose per-label `_print_filter_report()`.
- **`configs/trigger_map.yaml`**: Added `# NOTE: filtered by structural_tokens.yaml`
  comments to 10 overlapping trigger entries (may, should, could, if, when, while,
  with, it, this, they).

## Verification Results (30 stories)
- Mean structural tokens filtered per story: **7.53**
- Bridge fill rate: **100%** (0 of 30 stories returned empty selections)
- Top filtered tokens: `as` (54), `a` (46), `,` as empty string (37), `to` (33)
- Per-label dominant selections: `<TBD_ROLE>` (ActorAmbiguity), `<TBD_SCOPE_ENTITY>`
  (ScopeAmbiguity), `<TBD_ACTION_SPECIFICATION>` (SemanticAmbiguity)

## New Debug Files
- `outputs/debug/filter_audit_report.txt`
- `outputs/debug/filter_verification_report.txt`
- `src/debug/verify_filter.py`
