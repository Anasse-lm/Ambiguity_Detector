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
