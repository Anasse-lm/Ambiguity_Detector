#!/usr/bin/env python3
import os
import argparse
import yaml
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import datetime
import zipfile

from req_ambiguity.modeling.classifier import DeBERTaAmbiguityClassifier
from req_ambiguity.refinement.backends.gemini import GeminiBackend
from req_ambiguity.refinement.cache import CachedBackend
from req_ambiguity.refinement.prompt_builder import PromptBuilder
from req_ambiguity.refinement.validator import RefinementValidator
from req_ambiguity.refinement.refiner import Refiner
from req_ambiguity.verification.verifier import Verifier

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--story-list', type=str, default='outputs/xai/samples/bridge_validation_sample.csv')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set. Refinement requires the API key.")

    ref_config = load_config('configs/refinement.yaml')
    train_config = load_config('configs/train.yaml')
    label_cols = train_config['label_cols']

    os.makedirs('outputs/refinement/traces', exist_ok=True)
    os.makedirs('outputs/refinement/results', exist_ok=True)
    os.makedirs('outputs/refinement/figures', exist_ok=True)
    os.makedirs(ref_config['cache_dir'], exist_ok=True)

    gemini_backend = GeminiBackend(
        max_retries=ref_config.get('max_retries', 3),
        retry_delay_seconds=ref_config.get('retry_delay_seconds', 2.0)
    )
    backend = CachedBackend(gemini_backend, ref_config['cache_dir'], ref_config.get('cache_enabled', True))

    prompt_builder = PromptBuilder('configs/placeholders.yaml', ref_config['few_shot_examples_path'])
    validator = RefinementValidator('configs/placeholders.yaml')
    refiner = Refiner(backend, prompt_builder, validator, ref_config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = train_config['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DeBERTaAmbiguityClassifier(model_name, len(label_cols))
    model.load_state_dict(torch.load('outputs/checkpoints/best_model.pt', map_location=device))
    verifier = Verifier(model, tokenizer, device, label_cols)

    df = pd.read_csv(args.story_list)
    if args.dry_run:
        df = df.head(10)
        print("DRY RUN: Processing only 10 stories.")

    total_processed = 0
    compliance_records = []
    verification_records = []
    placeholder_usage = []
    failure_cases = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Refining & Verifying"):
        story_id = row['StoryID']
        xai_json_path = f"outputs/xai/json/{story_id}.json"
        
        if not os.path.exists(xai_json_path):
            continue
            
        with open(xai_json_path, 'r', encoding='utf-8') as f:
            xai_record = json.load(f)
            
        outcome = refiner.refine(story_id, xai_record)
        
        trace = {
            "story_id": story_id,
            "original_text": xai_record['original_text'],
            "xai_record_reference": xai_json_path,
            "refinement_outcome": {
                "first_attempt_passed": outcome.first_attempt_passed,
                "final_attempt_passed": outcome.final_attempt_passed,
                "attempts_used": outcome.attempts_used,
                "validated_response": outcome.validated_response
            },
            "all_attempt_logs": outcome.all_attempt_logs,
            "verification_result": None
        }

        illegal_ph = []
        if not outcome.final_attempt_passed and outcome.all_attempt_logs:
            validation_dict = outcome.all_attempt_logs[-1].get("validation", {})
            illegal_ph = validation_dict.get("illegal_placeholders", [])

        used_placeholders = []
        if outcome.final_attempt_passed and outcome.validated_response:
            used_placeholders = outcome.validated_response.get("placeholders_used", [])
            for p in used_placeholders:
                for active_label in xai_record.get('predicted_labels', []):
                    placeholder_usage.append({"Placeholder": p, "AmbiguityLabel": active_label})
                    
            v_res = verifier.verify(xai_record['original_text'], outcome.validated_response['refined_story'])
            trace['verification_result'] = {
                "refined_stripped": v_res.refined_stripped,
                "per_label_delta": v_res.per_label_delta,
                "aggregate_delta": v_res.aggregate_delta,
                "improved": v_res.improved
            }
            
            v_record = {
                "StoryID": story_id,
                "AggregateBefore": float(np.mean(v_res.probs_before)),
                "AggregateAfter": float(np.mean(v_res.probs_after)),
                "AggregateDelta": v_res.aggregate_delta,
                "Improved": v_res.improved
            }
            for col in label_cols:
                v_record[f"{col}_Before"] = float(v_res.probs_before[label_cols.index(col)])
                v_record[f"{col}_After"] = float(v_res.probs_after[label_cols.index(col)])
                v_record[f"{col}_Delta"] = v_res.per_label_delta[col]
            verification_records.append(v_record)
            
            if not v_res.improved:
                failure_cases.append({
                    "StoryID": story_id,
                    "Original": xai_record['original_text'],
                    "Refined": outcome.validated_response['refined_story'],
                    "AggregateDelta": v_res.aggregate_delta,
                    "Notes": "Aggregate probability increased."
                })

        compliance_records.append({
            "StoryID": story_id,
            "FirstAttemptPassed": outcome.first_attempt_passed,
            "FinalAttemptPassed": outcome.final_attempt_passed,
            "AttemptsUsed": outcome.attempts_used,
            "PlaceholdersUsed": "|".join(used_placeholders),
            "IllegalPlaceholders": "|".join(illegal_ph)
        })
        
        with open(f"outputs/refinement/traces/{story_id}.json", 'w', encoding='utf-8') as f:
            json.dump(trace, f, indent=2)
            
        total_processed += 1

    if total_processed == 0:
        print("No stories processed.")
        return

    df_comp = pd.DataFrame(compliance_records)
    df_comp.to_csv("outputs/refinement/results/compliance_report.csv", index=False)
    
    first_pass_rate = df_comp['FirstAttemptPassed'].mean() * 100
    final_pass_rate = df_comp['FinalAttemptPassed'].mean() * 100
    failed_all = (1 - df_comp['FinalAttemptPassed'].mean()) * 100
    
    with open("outputs/refinement/results/compliance_summary.txt", 'w', encoding='utf-8') as f:
        f.write(f"First-attempt compliance rate: {first_pass_rate:.1f}%\n")
        f.write(f"Final-attempt compliance rate: {final_pass_rate:.1f}%\n")
        f.write(f"Failed all retries: {failed_all:.1f}%\n")
        
    mean_agg_delta = 0
    pct_improved = 0
    if len(verification_records) > 0:
        df_ver = pd.DataFrame(verification_records)
        df_ver.to_csv("outputs/refinement/results/verification_report.csv", index=False)
        mean_agg_delta = df_ver['AggregateDelta'].mean()
        pct_improved = df_ver['Improved'].mean() * 100
        
        with open("outputs/refinement/results/verification_summary.txt", 'w', encoding='utf-8') as f:
            f.write(f"Mean aggregate delta: {mean_agg_delta:.4f}\n")
            f.write(f"Percentage improved: {pct_improved:.1f}%\n")
            for col in label_cols:
                f.write(f"{col} mean delta: {df_ver[f'{col}_Delta'].mean():.4f}\n")
                
        plt.figure()
        df_ver['AggregateDelta'].hist(bins=20)
        plt.title('Distribution of Aggregate Deltas')
        plt.savefig('outputs/refinement/figures/aggregate_delta_distribution.png')
        plt.close()
        
        plt.figure(figsize=(12, 6))
        delta_cols = [f"{col}_Delta" for col in label_cols]
        df_ver[delta_cols].boxplot()
        plt.xticks(rotation=45)
        plt.title('Per-Label Delta Distributions')
        plt.tight_layout()
        plt.savefig('outputs/refinement/figures/per_label_delta_boxplot.png')
        plt.close()

    df_fail = pd.DataFrame(failure_cases)
    if not df_fail.empty:
        df_fail.to_csv("outputs/refinement/results/failure_cases.csv", index=False)
        
    df_ph = pd.DataFrame(placeholder_usage)
    if not df_ph.empty:
        usage_counts = df_ph.groupby(['Placeholder', 'AmbiguityLabel']).size().reset_index(name='TimesUsed')
        usage_counts['PctOfRefinements'] = (usage_counts['TimesUsed'] / total_processed) * 100
        usage_counts.to_csv("outputs/refinement/results/placeholder_usage.csv", index=False)
        
        plt.figure(figsize=(10, 8))
        usage_counts.groupby('Placeholder')['TimesUsed'].sum().sort_values().plot(kind='barh')
        plt.title('Placeholder Usage Frequency')
        plt.tight_layout()
        plt.savefig('outputs/refinement/figures/placeholder_usage_bar.png')
        plt.close()

    print("\n=== FALSIFIABLE VERIFICATION CRITERIA ===")
    c1 = first_pass_rate >= 85
    c2 = mean_agg_delta < 0
    c3 = pct_improved >= 70
    c4 = failed_all <= 5
    
    print(f"[{'X' if c1 else ' '}] First-attempt placeholder compliance >= 85% (Actual: {first_pass_rate:.1f}%)")
    print(f"[{'X' if c2 else ' '}] Mean aggregate delta < 0 (Actual: {mean_agg_delta:.4f})")
    print(f"[{'X' if c3 else ' '}] Percentage of stories improved >= 70% (Actual: {pct_improved:.1f}%)")
    print(f"[{'X' if c4 else ' '}] No more than 5% of stories fail all retries (Actual: {failed_all:.1f}%)")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = f"outputs/refinement_artifacts_{timestamp}.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk('outputs/refinement'):
            for file in files:
                zipf.write(os.path.join(root, file))
        zipf.write('configs/refinement.yaml')
        zipf.write('configs/placeholders.yaml')
        zipf.write('configs/trigger_map.yaml')
        zipf.write('configs/few_shot_examples.yaml')
        zipf.write('requirements.txt')
        for root, _, files in os.walk('src/req_ambiguity/refinement'):
            for file in files:
                if file.endswith('.py'):
                    zipf.write(os.path.join(root, file))
        for root, _, files in os.walk('src/req_ambiguity/verification'):
            for file in files:
                if file.endswith('.py'):
                    zipf.write(os.path.join(root, file))
        zipf.write('src/run_xai_refinement.py')
        
        readme_content = f"""Refinement Artifacts

Total stories evaluated: {total_processed}
First-attempt compliance rate: {first_pass_rate:.1f}%
Final-attempt compliance rate: {final_pass_rate:.1f}%
Mean aggregate delta: {mean_agg_delta:.4f}
Percentage improved: {pct_improved:.1f}%

Criteria passed: {sum([c1, c2, c3, c4])}/4
"""
        zipf.writestr('README.md', readme_content)
        
    print(f"\nArtifact zip created at: {zip_path}")

if __name__ == "__main__":
    main()
