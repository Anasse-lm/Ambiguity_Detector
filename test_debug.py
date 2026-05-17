import sys
import os
from pathlib import Path

# Add project root and src to sys.path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

import streamlit as st

class MockSessionState(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        return None
    def __setattr__(self, key, value):
        self[key] = value

st.session_state = MockSessionState()
st.session_state.current_session_id = "test"
st.session_state.api_key = "dummy"

class MockLogger:
    def log_event(self, *args, **kwargs): pass
st.session_state.session_log = MockLogger()

from app.streamlit_demo import run_pipeline

story = "As a user, I want to update records so that I can process the system quickly."
print("Running pipeline...")
try:
    outputs = run_pipeline(story)
    print("Pipeline finished successfully.")
    
    print("\n--- ACTIVE LABELS ---")
    print(outputs.get('active_labels', []))
    
    print("\n--- XAI RESULTS ---")
    if 'xai' in outputs:
        for label, exp in outputs['xai'].items():
            print(f"Label: {label}")
            print(f"Top Evidence Tokens: {exp.get('top_evidence_tokens', [])}")
            
    print("\n--- BRIDGE SELECTIONS ---")
    bridge_sel = outputs.get('bridge_selections', [])
    print(bridge_sel)
    
except Exception as e:
    import traceback
    traceback.print_exc()
