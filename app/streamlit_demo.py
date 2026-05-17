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
import yaml
import matplotlib.pyplot as plt

# Add src to python path so we can import modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from req_ambiguity.modeling.classifier import DeBERTaAmbiguityClassifier
from transformers import AutoTokenizer
from req_ambiguity.xai.integrated_gradients import AmbiguityExplainer
from req_ambiguity.xai.bridge import PlaceholderBridge
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
    train_config = load_config('configs/train.yaml')
    label_cols = train_config['label_cols']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = train_config['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DeBERTaAmbiguityClassifier(model_name, len(label_cols))
    model.load_state_dict(torch.load('outputs/checkpoints/best_model.pt', map_location=device))
    model.eval()
    
    with open('outputs/results/optimal_thresholds.json', 'r') as f:
        thresholds = json.load(f)
        
    # Pre-warm
    dummy = demo_config.get('pre_warming_dummy_text', "As a user, I want to update records.")
    inputs = tokenizer(dummy, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        
    return model, tokenizer, thresholds, label_cols, device

@st.cache_resource
def load_xai_explainer():
    model, tokenizer, _, label_cols, device = load_classifier_and_tokenizer()
    return AmbiguityExplainer(model, tokenizer, device, label_cols)

@st.cache_resource
def load_placeholder_bridge():
    return PlaceholderBridge('configs/placeholders.yaml', 'configs/trigger_map.yaml')

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
    prompt_builder = PromptBuilder('configs/placeholders.yaml', ref_config['few_shot_examples_path'])
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
st.title("AI-Based Ambiguity Detection and Refinement Pipeline")
st.subheader("Master Thesis Demonstration")

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
            
    st.text(f"Model Checkpoint: outputs/checkpoints/best_model.pt")
    st.text(f"Trigger Map Version: v1.2")
    
    st.header("Input Mode")
    input_mode = st.radio("Select Input Mode", ["Single story", "Multiple stories", "Upload document"], index=["Single story", "Multiple stories", "Upload document"].index(st.session_state.current_input_mode))
    
    if input_mode != st.session_state.current_input_mode:
        start_new_session(input_mode)
        
    example_selected = None
    if input_mode == "Single story":
        st.header("Example Stories")
        examples = [
            "",
            "As a doctor, I would like to update system to save time so that I can ensure quality.",
            "As a user, I want to access records in order to reduce errors.",
            "As a caregiver, I want to update patient info with fast performance as soon as possible.",
            "As a bank customer, I need to download statements continually so that I can manage finances."
        ]
        example_selected = st.selectbox("Choose an example to load", examples)

    if input_mode in ["Multiple stories", "Upload document"] and st.session_state.current_batch_id:
        st.header("Batch Progress")
        batch_prog = st.session_state.session_log.get_batch_progress(st.session_state.current_batch_id)
        if batch_prog:
            total = batch_prog['total_stories']
            reviewed = batch_prog['stories_reviewed']
            st.progress(reviewed / max(total, 1))
            st.text(f"Currently reviewing: story {st.session_state.current_queue_position + 1} of {total}")
            
    st.header("Session")
    st.text(f"Session ID: {st.session_state.current_session_id[:8]}")
    summary_placeholder = st.empty()
    
    if st.button("New Session"):
        start_new_session(input_mode)
        st.success("New session started")
        st.rerun()
        
    st.header("Reports")
    if st.session_state.current_story_id:
        report_bytes = per_story_report(st.session_state.current_session_id, st.session_state.current_story_id)
        if report_bytes:
            st.download_button("Download Per-story Report", data=report_bytes, file_name=f"report_{st.session_state.current_story_id}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            
    sum_bytes = session_summary_report(st.session_state.current_session_id)
    if sum_bytes:
        st.download_button("Download Session Summary", data=sum_bytes, file_name=f"session_summary_{st.session_state.current_session_id[:8]}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        
    cq_bytes = clarification_questions_report(st.session_state.current_session_id)
    if cq_bytes:
        st.download_button("Download Clarification Questions", data=cq_bytes, file_name=f"clarification_questions_{st.session_state.current_session_id[:8]}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        
    summary = st.session_state.session_log.get_session_summary(st.session_state.current_session_id)
    if summary['accepted'] >= demo_config.get('refined_requirements_doc_min_stories', 5):
        req_bytes = refined_requirements_report(st.session_state.current_session_id)
        if req_bytes:
            st.download_button("Download Refined Requirements", data=req_bytes, file_name=f"refined_requirements_{st.session_state.current_session_id[:8]}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

# Function to run pipeline
def run_pipeline(story_text, is_regeneration=False, old_outputs=None):
    if not story_text.strip():
        st.error("Please enter a user story to analyze.")
        return None
        
    outputs = old_outputs or {}
    
    # 1. Classification
    if not is_regeneration:
        try:
            inputs = tokenizer(story_text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
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
                attributions = explainer.explain_label(story_text, label)
                xai_results[label] = {
                    "tokens": attributions['tokens'],
                    "scores": attributions['attributions'],
                    "top_evidence_tokens": attributions['top_evidence_tokens']
                }
            outputs['xai'] = xai_results
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
            
            # Bridge
            bridge_selections = []
            for label in outputs['active_labels']:
                exp = outputs.get('xai', {}).get(label, {})
                top_tokens = [t[0] for t in exp.get('top_evidence_tokens', [])]
                matched = bridge.match_trigger(label, top_tokens)
                if matched:
                    bridge_selections.append({
                        "label": label,
                        "placeholder": matched['placeholder'],
                        "trigger": matched['trigger']
                    })
                    
            xai_record = {
                "original_text": story_text,
                "predicted_labels": outputs['active_labels'],
                "label_explanations": outputs.get('xai', {}),
                "bridge_selections": bridge_selections
            }
            
            prompt = refiner.prompt_builder.build_prompt(xai_record)
            if is_regeneration:
                prompt += "\n\nThe previous refinement was rejected. Please propose an alternative refinement using the same placeholder vocabulary but with different word choices or phrasing."
                
            response_text = refiner.backend.call(prompt)
            validated = refiner.validator.validate(response_text)
            
            if validated:
                outputs['refinement'] = validated
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
            
    return outputs

# Input Area
validation_warnings = []
trigger_batch = False

if input_mode == "Single story":
    story_input = st.text_area("User Story", value=example_selected if example_selected else "", height=100)
    if st.button("Analyze and Refine"):
        st.session_state.story_queue = [{"id": str(uuid.uuid4()), "text": story_input.strip()}]
        st.session_state.current_queue_position = 0
        trigger_batch = True
        
elif input_mode == "Multiple stories":
    stories_input = st.text_area("Paste multiple stories here, separated by blank lines. Maximum 50 stories per batch.", height=250)
    if st.button("Start Batch"):
        stories = parse_multiple_stories(stories_input)
        valid_stories, validation_warnings = validate_stories(stories, demo_config['max_stories_per_batch'], demo_config['min_story_length_chars'], demo_config['max_story_length_chars'])
        st.session_state.story_queue = [{"id": str(uuid.uuid4()), "text": s} for s in valid_stories]
        st.session_state.current_queue_position = 0
        trigger_batch = True
        
elif input_mode == "Upload document":
    uploaded_file = st.file_uploader("Upload CSV, TXT, or DOCX", type=["csv", "txt", "docx"])
    st.markdown("CSV: one story per row in a column named 'StoryText', 'story', or 'text'.\nTXT: one story per blank-line-separated block.\nDOCX: one story per paragraph.\nMaximum 50 stories per batch.")
    if st.button("Start Batch") and uploaded_file:
        file_bytes = uploaded_file.read()
        stories = []
        if uploaded_file.name.endswith(".csv"):
            stories = parse_csv_upload(file_bytes)
        elif uploaded_file.name.endswith(".txt"):
            stories = parse_txt_upload(file_bytes)
        elif uploaded_file.name.endswith(".docx"):
            stories = parse_docx_upload(file_bytes)
            
        valid_stories, validation_warnings = validate_stories(stories, demo_config['max_stories_per_batch'], demo_config['min_story_length_chars'], demo_config['max_story_length_chars'])
        st.session_state.story_queue = [{"id": str(uuid.uuid4()), "text": s} for s in valid_stories]
        st.session_state.current_queue_position = 0
        trigger_batch = True

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
    with st.spinner("Processing your story — this takes a few seconds on CPU."):
        outputs = run_pipeline(first_item['text'])
        st.session_state.current_pipeline_outputs = outputs
        st.session_state.session_log.log_story(st.session_state.current_session_id, first_item['id'], first_item['text'], outputs, st.session_state.current_batch_id, 0)
    st.rerun()

# Display current story results
if st.session_state.current_story_text and st.session_state.current_pipeline_outputs:
    st.divider()
    outputs = st.session_state.current_pipeline_outputs
    
    st.header("Detection Results")
    if 'detection' in outputs:
        cols = st.columns(7)
        for i, (label, prob) in enumerate(outputs['detection'].items()):
            color = "orange" if prob >= thresholds[label] else "green"
            cols[i % 7].markdown(f"<div style='background-color: {color}; padding: 5px; border-radius: 5px; text-align: center; color: white;'><b>{label.replace('Ambiguity', '')}</b><br>{prob:.2f}</div>", unsafe_allow_html=True)
    
    if outputs.get('active_labels'):
        st.header("Explanation (Integrated Gradients)")
        if 'xai' in outputs:
            for label in outputs['active_labels']:
                st.subheader(f"{label} ({outputs['detection'][label]:.2f})")
                exp = outputs['xai'].get(label)
                if exp:
                    html_path = Path("outputs/refinement/temp_heatmap.html")
                    html_path.parent.mkdir(parents=True, exist_ok=True)
                    render_html_heatmap(exp['tokens'], exp['scores'], html_path)
                    with open(html_path, 'r', encoding='utf-8') as f:
                        st.components.v1.html(f.read(), height=150)
                        
        st.header("Refinement")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Story")
            st.write(st.session_state.current_story_text)
            
        with col2:
            st.subheader("Refined Story")
            if 'refinement' in outputs and 'refined_story' in outputs['refinement']:
                ref_text = outputs['refinement']['refined_story']
                # Highlight placeholders
                for p in bridge.vocabulary:
                    ref_text = ref_text.replace(p, f"<span style='background-color: #fff3cd; padding: 2px 4px; border-radius: 3px;'>{p}</span>")
                st.markdown(ref_text, unsafe_allow_html=True)
                
        if 'refinement' in outputs and 'placeholders_used' in outputs['refinement']:
            st.markdown("**Placeholders inserted:**")
            for p in outputs['refinement']['placeholders_used']:
                st.markdown(f"- {p}")
                
            st.markdown("**Clarification questions:**")
            for i, q in enumerate(outputs['refinement'].get('clarification_questions', [])):
                st.markdown(f"{i+1}. {q}")
                
        # Verification
        if 'verification' in outputs:
            st.header("Verification (Before vs After)")
            df_ver = pd.DataFrame({
                "Before": outputs['verification']['probs_before'],
                "After": outputs['verification']['probs_after']
            }, index=[l.replace('Ambiguity', '') for l in label_cols])
            st.bar_chart(df_ver)
            
            delta = outputs['verification']['aggregate_delta']
            st.write(f"Aggregate delta: **{delta:.4f}**")
            if outputs['verification']['improved']:
                st.success("✓ Refinement reduced ambiguity")
            else:
                st.warning("⚠ Refinement did not reduce ambiguity")
                
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

# Session Summary Panel Update
summary = st.session_state.session_log.get_session_summary(st.session_state.current_session_id)
with summary_placeholder.container():
    if summary['started_at']:
        try:
            started = pd.to_datetime(summary['started_at'])
            elapsed = datetime.datetime.now() - started
            # remove microseconds
            elapsed = elapsed - datetime.timedelta(microseconds=elapsed.microseconds)
            st.text(f"Session active for: {elapsed}")
        except:
            pass
    st.text(f"Input mode: {summary['input_mode']}")
    st.text(f"Stories processed: {summary['stories_processed']}")
    st.text(f"Refinements accepted: {summary['accepted']}")
    st.text(f"Refinements regenerated: {summary['regenerated']}")
    st.text(f"Stories skipped: {summary['skipped']}")
    if st.session_state.current_batch_id:
        batch_prog = st.session_state.session_log.get_batch_progress(st.session_state.current_batch_id)
        if batch_prog:
            st.text(f"Batch progress: {batch_prog['stories_reviewed']}/{batch_prog['total_stories']} reviewed")
