import sys
import os
import time
import json
import uuid
import datetime
from pathlib import Path
import streamlit as st
import torch
import pandas as pd
import random
import yaml
import matplotlib.pyplot as plt

# Add src to python path so we can import modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from req_ambiguity.modeling.classifier import DeBERTaAmbiguityClassifier
from transformers import AutoTokenizer
from req_ambiguity.xai.integrated_gradients import AmbiguityExplainer
import importlib
import sys
if 'req_ambiguity.xai.integrated_gradients' in sys.modules:
    importlib.reload(sys.modules['req_ambiguity.xai.integrated_gradients'])
if 'req_ambiguity.xai.visualization' in sys.modules:
    importlib.reload(sys.modules['req_ambiguity.xai.visualization'])
from req_ambiguity.xai.integrated_gradients import AmbiguityExplainer

from req_ambiguity.xai.bridge import PlaceholderBridge
from req_ambiguity.refinement.backends.base import RefinementRequest
from req_ambiguity.refinement.backends.gemini import GeminiBackend
from req_ambiguity.refinement.cache import CachedBackend
from req_ambiguity.refinement.prompt_builder import PromptBuilder
from req_ambiguity.refinement.validator import RefinementValidator
from req_ambiguity.refinement.refiner import Refiner
from req_ambiguity.verification.verifier import Verifier
from req_ambiguity.xai.visualization import render_html_heatmap

from req_ambiguity.session.session_log import SessionLog
from req_ambiguity.session.input_ingestion import (
    parse_single_story, parse_multiple_stories,
    parse_csv_upload, parse_txt_upload, parse_docx_upload,
    validate_stories
)
from req_ambiguity.reporting.reports import (
    per_story_report, session_summary_report,
    clarification_questions_report, refined_requirements_report
)

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# Initialization
st.set_page_config(page_title="Ambiguity Detection Pipeline", layout="wide")

# Load CSS
css_path = Path(__file__).parent / "static" / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if 'session_log' not in st.session_state:
    st.session_state.session_log = SessionLog()

if 'demo_config' not in st.session_state:
    st.session_state.demo_config = load_config('configs/demo.yaml')
    
demo_config = st.session_state.demo_config

# State Management Task 13
def init_state():
    defaults = {
        'api_key': os.environ.get("GEMINI_API_KEY", ""),
        'current_session_id': None,
        'current_input_mode': 'Single story',
        'current_batch_id': None,
        'story_queue': [],
        'current_queue_position': 0,
        'current_story_id': None,
        'current_story_text': "",
        'current_pipeline_outputs': {},
        'regeneration_count_for_current_story': 0,
        'session_active': False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# Task 5: Model loading with pre-warming
@st.cache_resource
def load_classifier_and_tokenizer():
    # Cache busted for MPS
    train_config = load_config('configs/train.yaml')
    label_cols = train_config['label_cols']
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model_name = train_config['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DeBERTaAmbiguityClassifier(model_name, len(label_cols))
    model.load_state_dict(torch.load('outputs/checkpoints/best_model.pt', map_location=device))
    model = model.to(device)
    model.eval()
    
    with open('outputs/results/optimal_thresholds.json', 'r') as f:
        thresholds = json.load(f)
        
    # Pre-warm
    dummy = demo_config.get('pre_warming_dummy_text', "As a user, I want to update records.")
    inputs = tokenizer(dummy, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        
    return model, tokenizer, thresholds, label_cols, device

@st.cache_resource
def load_xai_explainer():
    # Cache busted v4
    model, tokenizer, _, label_cols, device = load_classifier_and_tokenizer()
    return AmbiguityExplainer(model, tokenizer, device, label_cols)

@st.cache_resource
def load_placeholder_bridge():
    # Cache busted v4
    return PlaceholderBridge('configs/trigger_map.yaml', 'configs/placeholders.yaml')

# Load models
model, tokenizer, thresholds, label_cols, device = load_classifier_and_tokenizer()
explainer = load_xai_explainer()
bridge = load_placeholder_bridge()

# Setup Refiner
def get_refiner():
    ref_config = load_config('configs/refinement.yaml')
    gemini_backend = GeminiBackend(
        max_retries=ref_config.get('max_retries', 3),
        retry_delay_seconds=ref_config.get('retry_delay_seconds', 2.0)
    )
    backend = CachedBackend(gemini_backend, ref_config['cache_dir'], ref_config.get('cache_enabled', True))
    prompt_builder = PromptBuilder()
    validator = RefinementValidator('configs/placeholders.yaml')
    return Refiner(backend, prompt_builder, validator, ref_config)

verifier = Verifier(model, tokenizer, device, label_cols)

def start_new_session(input_mode):
    if st.session_state.current_session_id:
        st.session_state.session_log.end_session(st.session_state.current_session_id)
    
    st.session_state.current_session_id = st.session_state.session_log.start_session(input_mode)
    st.session_state.current_input_mode = input_mode
    st.session_state.current_batch_id = None
    st.session_state.story_queue = []
    st.session_state.current_queue_position = 0
    st.session_state.current_story_id = None
    st.session_state.current_story_text = ""
    st.session_state.current_pipeline_outputs = {}
    st.session_state.regeneration_count_for_current_story = 0
    st.session_state.session_active = True

if not st.session_state.session_active:
    start_new_session(st.session_state.current_input_mode)

# TASK 6: HEADER & UI SCALFFOLDING
st.markdown("""
<div class="branded-header">
    <h1>Ambiguity Detection and Refinement Pipeline</h1>
    <p>Master Thesis Demonstration</p>
    <span class="version">v1.0 — Built 2026</span>
</div>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.header("Configuration")
    
    # Task 4: API Key dual mode
    env_key = os.environ.get("GEMINI_API_KEY", "")
    if env_key:
        st.success("API key loaded from environment")
        st.session_state.api_key = env_key
    else:
        st.session_state.api_key = st.text_input("Gemini API Key", type="password", value=st.session_state.api_key)
        if not st.session_state.api_key:
            st.warning("Enter your Gemini API key to enable refinement.")
            
    st.divider()
    if st.button("New Session", use_container_width=True):
        start_new_session(st.session_state.current_input_mode)
        st.success("New session started")
        st.rerun()

    # Batch Progress
    if st.session_state.current_input_mode in ["Multiple stories", "Upload document"] and st.session_state.current_batch_id:
        st.header("Batch Progress")
        batch_prog = st.session_state.session_log.get_batch_progress(st.session_state.current_batch_id)
        if batch_prog:
            total = batch_prog['total_stories']
            reviewed = batch_prog['stories_reviewed']
            st.progress(reviewed / max(total, 1))
            st.text(f"Currently reviewing: story {st.session_state.current_queue_position + 1} of {total}")

# MAIN DASHBOARD METRICS
summary = st.session_state.session_log.get_session_summary(st.session_state.current_session_id)
st.markdown('<div class="metric-container">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Session ID", st.session_state.current_session_id[:8])
col2.metric("Stories Processed", summary.get('total_stories', 0))
col3.metric("Refinements Accepted", summary.get('accepted', 0))
col4.metric("Stories Skipped", summary.get('skipped', 0))
st.markdown('</div>', unsafe_allow_html=True)

# INPUT SECTION
st.markdown('<div class="custom-card"><h3>Input Phase</h3>', unsafe_allow_html=True)
input_mode = st.radio("Select Input Mode", ["Single story", "Multiple stories", "Upload document"], index=["Single story", "Multiple stories", "Upload document"].index(st.session_state.current_input_mode), horizontal=True)

if input_mode != st.session_state.current_input_mode:
    start_new_session(input_mode)

# Function to run pipeline
def run_pipeline(story_text, is_regeneration=False, old_outputs=None):
    if not story_text.strip():
        st.error("Please enter a user story to analyze.")
        return None
        
    outputs = old_outputs or {}
    
    # 1. Classification
    if not is_regeneration:
        try:
            inputs = tokenizer(story_text, return_tensors='pt', padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            
            detection = {}
            active_labels = []
            for i, col in enumerate(label_cols):
                prob = float(probs[i])
                detection[col] = prob
                if prob >= thresholds[col]:
                    active_labels.append(col)
                    
            outputs['detection'] = detection
            outputs['active_labels'] = active_labels
        except Exception as e:
            st.session_state.session_log.log_event(st.session_state.current_session_id, "ERROR", {"stage": "classification", "error": str(e)})
            st.error("Could not analyze the story. Please check that the input is valid and try again.")
            return None
            
        st.session_state.session_log.log_event(st.session_state.current_session_id, "DETECTION_DONE", {"active_labels": active_labels})
    
    # 2. XAI
    if not is_regeneration and outputs.get('active_labels'):
        try:
            xai_results = {}
            for label in outputs['active_labels']:
                attributions = explainer.explain_label(story_text, label, top_k=2)
                xai_results[label] = {
                    "tokens": attributions['tokens'],
                    "scores": attributions['attributions'],
                    "top_evidence_tokens": attributions['top_evidence_tokens']
                }
            outputs['xai'] = xai_results
            
            # Bridge
            bridge_selections = []
            evidence_tokens = set()
            allowed_placeholders = set()
            
            for label in outputs['active_labels']:
                exp = outputs.get('xai', {}).get(label, {})
                evidence = exp.get('top_evidence_tokens', [])
                for t in evidence:
                    evidence_tokens.add(t[0].replace(' ', '').strip())
                    
                matched_list = bridge.match_evidence(label, evidence)
                for matched in matched_list:
                    p = matched['placeholder']
                    allowed_placeholders.add(p)
                    bridge_selections.append({
                        "label": label,
                        "placeholder": p,
                        "trigger": ", ".join(matched['matched_evidence']) if matched['matched_evidence'] else "fallback"
                    })
                    
            evidence_tokens = [t for t in list(evidence_tokens) if t]
            allowed_placeholders = list(allowed_placeholders)
            outputs['bridge_selections'] = bridge_selections
            print(f"[DEBUG] Bridge Selections Generated: {bridge_selections}", flush=True)
            outputs['bridge_evidence_tokens'] = evidence_tokens
            outputs['bridge_allowed_placeholders'] = allowed_placeholders
            st.session_state.session_log.log_event(st.session_state.current_session_id, "EXPLANATION_RENDERED", {"labels": outputs['active_labels']})
        except Exception as e:
            st.session_state.session_log.log_event(st.session_state.current_session_id, "ERROR", {"stage": "xai", "error": str(e)})
            st.error("Could not generate explanation visualization. The story was classified successfully but attribution failed.")
            
    # 3. Refinement
    if outputs.get('active_labels'):
        try:
            if not st.session_state.api_key:
                st.error("Refinement is unavailable: Missing API key. Please check your API key and connection.")
                return outputs
                
            os.environ["GEMINI_API_KEY"] = st.session_state.api_key
            
            refiner = get_refiner()
            
            prompt = refiner.prompt_builder.build_prompt(
                original_story=story_text,
                active_labels=outputs['active_labels'],
                evidence_tokens=outputs.get('bridge_evidence_tokens', []),
                allowed_placeholders=outputs.get('bridge_allowed_placeholders', [])
            )
            
            if is_regeneration:
                prompt += "\n\nThe previous refinement was rejected. Please propose an alternative refinement using the same placeholder vocabulary but with different word choices or phrasing."
                
            request = RefinementRequest(
                prompt_text=prompt,
                model_name=refiner.config.get('model_name', 'gemini-1.5-pro'),
                temperature=refiner.config.get('temperature', 0.2),
                max_output_tokens=refiner.config.get('max_output_tokens', 1024),
                top_p=refiner.config.get('top_p', 0.9)
            )
            
            response = refiner.backend.call(request)
            validated = refiner.validator.validate(response.text)
            
            if validated.passed:
                outputs['refinement'] = validated.parsed_json
                st.session_state.session_log.log_event(st.session_state.current_session_id, "REFINEMENT_GENERATED", {"status": "success", "is_regeneration": is_regeneration})
            else:
                outputs['refinement'] = {"error": "Failed validation rules."}
        except Exception as e:
            st.session_state.session_log.log_event(st.session_state.current_session_id, "ERROR", {"stage": "refinement", "error": str(e)})
            st.error(f"Refinement is unavailable: {str(e)}. Please check your API key and connection.")
            return outputs
            
    # 4. Verification
    if 'refinement' in outputs and 'refined_story' in outputs['refinement']:
        try:
            v_res = verifier.verify(story_text, outputs['refinement']['refined_story'])
            outputs['verification'] = {
                "aggregate_delta": v_res.aggregate_delta,
                "improved": v_res.improved,
                "probs_before": [float(p) for p in v_res.probs_before],
                "probs_after": [float(p) for p in v_res.probs_after]
            }
        except Exception as e:
            st.session_state.session_log.log_event(st.session_state.current_session_id, "ERROR", {"stage": "verification", "error": str(e)})
            st.error("Could not compute before/after verification.")
            
    return outputs# Input Area
validation_warnings = []
trigger_batch = False

if input_mode == "Single story":
    st.markdown("**Try a quick-fill example:**")
    try:
        df = pd.read_csv("data/processed/test.csv")
        # Get processed stories to avoid showing them again
        session_data = st.session_state.session_log.get_session_data(st.session_state.current_session_id)
        used_stories = [s['original_text'] for s in session_data['stories']]
        available_df = df[~df['StoryText'].isin(used_stories)]
        if len(available_df) >= 4:
            sampled = available_df.sample(4)['StoryText'].tolist()
        else:
            sampled = df.sample(min(4, len(df)))['StoryText'].tolist()
            
        cols = st.columns(4)
        for i, example_text in enumerate(sampled):
            if cols[i].button(example_text[:50] + "...", key=f"ex_{i}", help=example_text):
                st.session_state['quick_fill'] = example_text
    except Exception as e:
        st.warning("Could not load examples from test.csv")

    current_val = st.session_state.get('quick_fill', "")
    story_input = st.text_area("User Story", value=current_val, height=100, placeholder="e.g. As a user, I want to update records...")
    
    if st.button("🔍 Analyze and Refine", type="primary", disabled=not st.session_state.api_key, help="Enter your Gemini API key in the sidebar to enable analysis."):
        st.session_state.story_queue = [{"id": str(uuid.uuid4()), "text": story_input.strip()}]
        st.session_state.current_queue_position = 0
        trigger_batch = True
        
elif input_mode == "Multiple stories":
    stories_input = st.text_area("Paste multiple stories here, separated by blank lines. Maximum 50 stories per batch.", height=250)
    if st.button("🔍 Start Batch", type="primary", disabled=not st.session_state.api_key, help="Enter your Gemini API key in the sidebar to enable analysis."):
        stories = parse_multiple_stories(stories_input)
        valid_stories, validation_warnings = validate_stories(stories, demo_config['max_stories_per_batch'], demo_config['min_story_length_chars'], demo_config['max_story_length_chars'])
        st.session_state.story_queue = [{"id": str(uuid.uuid4()), "text": s} for s in valid_stories]
        st.session_state.current_queue_position = 0
        trigger_batch = True
        
elif input_mode == "Upload document":
    uploaded_file = st.file_uploader("Upload CSV, TXT, or DOCX", type=["csv", "txt", "docx"])
    with st.expander("Format Help"):
        st.markdown("CSV: one story per row in a column named 'StoryText', 'story', or 'text'.\nTXT: one story per blank-line-separated block.\nDOCX: one story per paragraph.\nMaximum 50 stories per batch.")
    if st.button("🔍 Start Batch", type="primary", disabled=not st.session_state.api_key, help="Enter your Gemini API key in the sidebar to enable analysis.") and uploaded_file:
        try:
            filename = uploaded_file.name.lower()
            if filename.endswith(".csv"):
                stories = parse_csv_upload(uploaded_file.getvalue())
            elif filename.endswith(".txt"):
                stories = parse_txt_upload(uploaded_file.getvalue())
            elif filename.endswith(".docx"):
                stories = parse_docx_upload(uploaded_file.getvalue())
            else:
                stories = []
            valid_stories, validation_warnings = validate_stories(stories, demo_config['max_stories_per_batch'], demo_config['min_story_length_chars'], demo_config['max_story_length_chars'])
            st.session_state.story_queue = [{"id": str(uuid.uuid4()), "text": s} for s in valid_stories]
            st.session_state.current_queue_position = 0
            trigger_batch = True
        except Exception as e:
            st.session_state.session_log.log_event(st.session_state.current_session_id, "ERROR", {"stage": "upload", "error": str(e)})
            st.error(f"Could not parse the uploaded file. Check format. ({e})")
            
st.markdown('</div>', unsafe_allow_html=True)

if validation_warnings:
    for w in validation_warnings:
        st.warning(w)

if trigger_batch and st.session_state.story_queue:
    if input_mode != "Single story":
        st.session_state.current_batch_id = st.session_state.session_log.start_batch(st.session_state.current_session_id, input_mode, "upload", len(st.session_state.story_queue))
    
    first_item = st.session_state.story_queue[0]
    st.session_state.current_story_id = first_item['id']
    st.session_state.current_story_text = first_item['text']
    st.session_state.regeneration_count_for_current_story = 0
    hardware_name = "Apple Silicon GPU (MPS)" if device.type == 'mps' else "NVIDIA GPU" if device.type == 'cuda' else "CPU"
    with st.spinner(f"Processing your story — running hardware acceleration on {hardware_name}."):
        outputs = run_pipeline(first_item['text'])
        st.session_state.current_pipeline_outputs = outputs
        st.session_state.session_log.log_story(st.session_state.current_session_id, first_item['id'], first_item['text'], outputs, st.session_state.current_batch_id, 0)
    st.rerun()

# Display current story results
if st.session_state.current_story_text and st.session_state.current_pipeline_outputs:
    st.divider()
    outputs = st.session_state.current_pipeline_outputs
    
    st.markdown('<div class="custom-card"><h3>Detection Results</h3>', unsafe_allow_html=True)
    if 'detection' in outputs:
        html_chips = []
        for i, (label, prob) in enumerate(outputs['detection'].items()):
            if prob >= 0.9:
                chip_class = "danger"
                icon = "⚠"
            elif prob >= thresholds[label]:
                chip_class = "warn"
                icon = "⚠"
            else:
                chip_class = "pass"
                icon = "✓"
                
            display_label = label.replace('Ambiguity', '')
            html_chips.append(f"""
            <div class="label-chip {chip_class}">
                <div>{icon} {display_label}</div>
                <div class="prob">{prob:.2f} (Threshold: {thresholds[label]:.2f})</div>
            </div>
            """)
            
        st.markdown(" ".join(html_chips), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if outputs.get('active_labels'):
        st.markdown('<div class="custom-card"><h3>Explanation (Integrated Gradients)</h3>', unsafe_allow_html=True)
        if 'xai' in outputs:
            for label in outputs['active_labels']:
                prob = outputs['detection'][label]
                st.markdown(f"**{label} ({prob:.2f})**")
                st.caption("Tokens contributing to this prediction")
                exp = outputs['xai'].get(label)
                if exp:
                    html_path = Path("outputs/refinement/temp_heatmap.html")
                    html_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    severity_color = "#B91C1C" # danger (red)
                        
                    render_html_heatmap(exp['tokens'], exp['scores'], html_path, positive_color=severity_color)
                    with open(html_path, 'r', encoding='utf-8') as f:
                        st.components.v1.html(f.read(), height=100)
                        
                    # Show Placeholders from Bridge
                    label_placeholders = [bs['placeholder'] for bs in outputs.get('bridge_selections', []) if bs['label'] == label]
                    if label_placeholders:
                        st.markdown("**Mapped Placeholders:**")
                        chips = " ".join([f"<span class='label-chip pass' style='font-family: monospace; padding: 2px 6px;'>{p}</span>" for p in set(label_placeholders)])
                        st.markdown(chips, unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
                        
        st.markdown('<div class="custom-card"><h3>Refined Story</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="story-box"><div class="story-box-title">Original Story</div>', unsafe_allow_html=True)
            st.write(st.session_state.current_story_text)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="story-box"><div class="story-box-title">Refined Story</div>', unsafe_allow_html=True)
            if 'refinement' in outputs and 'refined_story' in outputs['refinement']:
                ref_text = outputs['refinement']['refined_story']
                # Highlight placeholders
                for p in bridge.vocabulary:
                    ref_text = ref_text.replace(p, f"<span class='placeholder-highlight'>{p}</span>")
                st.markdown(ref_text, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
                
        if 'refinement' in outputs and 'placeholders_used' in outputs['refinement']:
            st.markdown("**Placeholders inserted:**")
            # Build compact chips with description from config
            _pb_descriptions = getattr(get_refiner().prompt_builder, 'placeholder_descriptions', {})
            ph_chips = []
            for p in outputs['refinement']['placeholders_used']:
                desc = _pb_descriptions.get(p, "")
                # Truncate long descriptions to keep chip compact
                short_desc = (desc[:90] + "\u2026") if len(desc) > 90 else desc
                ph_chips.append(
                    f"<span class='placeholder-chip'>"
                    f"<span class='ph-name'>{p}</span>"
                    f"<span class='ph-desc'>{short_desc}</span>"
                    f"</span>"
                )
            st.markdown("".join(ph_chips), unsafe_allow_html=True)
                
            st.markdown("**Clarification questions:**")
            for i, q in enumerate(outputs['refinement'].get('clarification_questions', [])):
                st.markdown(f"{i+1}. {q}")
        st.markdown('</div>', unsafe_allow_html=True)
                
        # Verification
        if 'verification' in outputs:
            st.markdown('<div class="custom-card"><h3>Ambiguity Reduction</h3>', unsafe_allow_html=True)
            df_ver = pd.DataFrame({
                "Before": outputs['verification']['probs_before'],
                "After": outputs['verification']['probs_after']
            }, index=[l.replace('Ambiguity', '') for l in label_cols])
            st.bar_chart(df_ver)
            
            delta = outputs['verification']['aggregate_delta']
            st.write(f"Aggregate change: **{delta:.4f}** (negative = reduced)")
            if outputs['verification']['improved']:
                st.markdown("<div style='color: var(--success); font-weight: bold;'>✓ Refinement reduced ambiguity</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='color: var(--warning); font-weight: bold;'>⚠ Refinement did not reduce ambiguity</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
                
        st.divider()
        
        # Action Buttons
        cols = st.columns(4)
        if cols[0].button("✓ Accept"):
            st.session_state.session_log.log_event(st.session_state.current_session_id, "REFINEMENT_ACCEPTED", {"story_id": st.session_state.current_story_id}, story_id=st.session_state.current_story_id)
            st.session_state.session_log.update_story_status(st.session_state.current_story_id, "accepted")
            st.success("Refinement accepted")
            if st.session_state.current_input_mode == "Single story":
                st.session_state.current_story_text = ""
                st.session_state.current_pipeline_outputs = {}
            st.rerun()
            
        if cols[1].button("↻ Regenerate", disabled=(st.session_state.regeneration_count_for_current_story >= demo_config['max_regenerations_per_story'])):
            st.session_state.regeneration_count_for_current_story += 1
            st.session_state.session_log.log_event(st.session_state.current_session_id, "REFINEMENT_REGENERATED", {"story_id": st.session_state.current_story_id, "attempt": st.session_state.regeneration_count_for_current_story}, story_id=st.session_state.current_story_id)
            with st.spinner("Regenerating refinement..."):
                outputs = run_pipeline(st.session_state.current_story_text, is_regeneration=True, old_outputs=st.session_state.current_pipeline_outputs)
                st.session_state.current_pipeline_outputs = outputs
            st.rerun()
            
        if cols[2].button("→ Skip"):
            st.session_state.session_log.log_event(st.session_state.current_session_id, "STORY_SKIPPED", {"story_id": st.session_state.current_story_id}, story_id=st.session_state.current_story_id)
            st.session_state.session_log.update_story_status(st.session_state.current_story_id, "skipped")
            
            if st.session_state.current_input_mode != "Single story":
                st.session_state.current_queue_position += 1
                if st.session_state.current_queue_position < len(st.session_state.story_queue):
                    next_item = st.session_state.story_queue[st.session_state.current_queue_position]
                    st.session_state.current_story_id = next_item['id']
                    st.session_state.current_story_text = next_item['text']
                    st.session_state.regeneration_count_for_current_story = 0
                    with st.spinner("Processing next story..."):
                        outputs = run_pipeline(next_item['text'])
                        st.session_state.current_pipeline_outputs = outputs
                        st.session_state.session_log.log_story(st.session_state.current_session_id, next_item['id'], next_item['text'], outputs, st.session_state.current_batch_id, st.session_state.current_queue_position)
                else:
                    st.session_state.session_log.log_event(st.session_state.current_session_id, "BATCH_COMPLETED", {"batch_id": st.session_state.current_batch_id})
                    st.session_state.current_story_text = ""
                    st.session_state.current_pipeline_outputs = {}
                    st.success("Batch complete!")
            else:
                st.session_state.current_story_text = ""
                st.session_state.current_pipeline_outputs = {}
            st.rerun()
            
        if st.session_state.current_input_mode != "Single story":
            if cols[3].button("Next Story →"):
                st.session_state.current_queue_position += 1
                if st.session_state.current_queue_position < len(st.session_state.story_queue):
                    next_item = st.session_state.story_queue[st.session_state.current_queue_position]
                    st.session_state.current_story_id = next_item['id']
                    st.session_state.current_story_text = next_item['text']
                    st.session_state.regeneration_count_for_current_story = 0
                    with st.spinner("Processing next story..."):
                        outputs = run_pipeline(next_item['text'])
                        st.session_state.current_pipeline_outputs = outputs
                        st.session_state.session_log.log_story(st.session_state.current_session_id, next_item['id'], next_item['text'], outputs, st.session_state.current_batch_id, st.session_state.current_queue_position)
                else:
                    st.session_state.session_log.log_event(st.session_state.current_session_id, "BATCH_COMPLETED", {"batch_id": st.session_state.current_batch_id})
                    st.session_state.current_story_text = ""
                    st.session_state.current_pipeline_outputs = {}
                    st.success("Batch complete!")
                st.rerun()
    else:
        st.success("No ambiguity detected! This user story passes the quality threshold.")

# REPORTS SECTION
st.divider()
st.markdown('<div class="custom-card"><h3>Export Results</h3>', unsafe_allow_html=True)
st.write("Download your session artifacts or generated reports here.")
cols = st.columns(4)

if st.session_state.current_story_id:
    report_bytes = per_story_report(st.session_state.current_session_id, st.session_state.current_story_id)
    if report_bytes:
        cols[0].download_button("📄 Per-story Report", data=report_bytes, file_name=f"report_{st.session_state.current_story_id}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        
sum_bytes = session_summary_report(st.session_state.current_session_id)
if sum_bytes:
    cols[1].download_button("📊 Session Summary", data=sum_bytes, file_name=f"session_summary_{st.session_state.current_session_id[:8]}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    
cq_bytes = clarification_questions_report(st.session_state.current_session_id)
if cq_bytes:
    cols[2].download_button("❓ Clarifications", data=cq_bytes, file_name=f"clarification_questions_{st.session_state.current_session_id[:8]}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    
summary = st.session_state.session_log.get_session_summary(st.session_state.current_session_id)
if summary['accepted'] >= demo_config.get('refined_requirements_doc_min_stories', 5):
    req_bytes = refined_requirements_report(st.session_state.current_session_id)
    if req_bytes:
        cols[3].download_button("📑 Refined Requirements", data=req_bytes, file_name=f"refined_requirements_{st.session_state.current_session_id[:8]}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
else:
    cols[3].button("📑 Refined Requirements", disabled=True, help="Accept at least 5 stories to generate this report.")
st.markdown('</div>', unsafe_allow_html=True)
