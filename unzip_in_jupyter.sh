#!/bin/bash
set -e

# Make sure we're in the right directory where the zip was uploaded
if [ ! -f "remote_outputs_transfer.zip" ]; then
    echo "Error: remote_outputs_transfer.zip not found in the current directory."
    echo "Please ensure you have uploaded it to your Jupyter environment."
    exit 1
fi

echo "Unzipping outputs (excluding heavy models)..."
unzip -o remote_outputs_transfer.zip

echo "Unzip complete!"
echo "NOTE: Because the heavy .pt model was excluded from the zip,"
echo "you MUST make sure the model is on your remote server in the correct place before running the pipeline."
echo "If you already have it in outputs/seed_42/, run this:"
echo "cp outputs/seed_42/checkpoints/best_model.pt outputs/checkpoints/best_model.pt"
echo ""
echo "After that, you can run the XAI pipeline with: !bash run_xai_pipeline.sh"
