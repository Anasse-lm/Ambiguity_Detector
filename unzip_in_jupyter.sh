#!/bin/bash
set -e

# Recombine parts if they exist
if ls model_part_* 1> /dev/null 2>&1; then
    echo "Found split archive parts. Recombining into full_model_transfer.zip..."
    cat model_part_* > full_model_transfer.zip
    echo "Recombination complete."
fi

# Make sure we're in the right directory where the zip was uploaded
if [ ! -f "full_model_transfer.zip" ]; then
    echo "Error: full_model_transfer.zip not found in the current directory."
    echo "Please ensure you have uploaded it (or its parts) to your Jupyter environment."
    exit 1
fi

echo "Unzipping full model and results..."
unzip -o full_model_transfer.zip

echo "Unzip complete! The models and thresholds are now in outputs/"
echo "You can now run the XAI pipeline with: !bash run_xai_pipeline.sh"
