#!/bin/bash
set -e
start_total=$(date +%s)
mkdir -p outputs/xai
echo "Starting XAI Pipeline execution..." > outputs/xai/TIMING_REPORT.txt

run_script() {
    script_name=$1
    echo "Running $script_name..."
    start=$(date +%s)
    python3 src/$script_name
    end=$(date +%s)
    echo "$script_name: $((end-start)) seconds" >> outputs/xai/TIMING_REPORT.txt
}

# Clear previous outputs for a clean run if needed, but we want to keep the IG cache!
mkdir -p outputs/xai/cache/ig

run_script run_xai_sample_preparation.py
run_script run_xai_visualizations.py
run_script run_xai_bridge_validation.py
run_script run_xai_faithfulness.py
run_script run_xai_summary.py
run_script run_xai_generate_json.py
run_script run_xai_package_artifacts.py

end_total=$(date +%s)
echo "Total Time: $((end_total-start_total)) seconds" >> outputs/xai/TIMING_REPORT.txt
