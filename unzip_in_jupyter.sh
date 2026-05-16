#!/bin/bash
set -e

# Make sure we're in the right directory where the zip was uploaded
if [ ! -f "remote_outputs_transfer.zip" ]; then
    echo "Error: remote_outputs_transfer.zip not found in the current directory."
    echo "Please ensure you have uploaded it to your Jupyter environment."
    exit 1
fi

echo "Unzipping lightweight outputs (metadata, thresholds, figures, reports)..."
unzip -o remote_outputs_transfer.zip

echo "Unzip complete!"
echo "Now run the following command to link your massive model checkpoint:"
echo "cp outputs/seed_42/checkpoints/best_model.pt outputs/checkpoints/best_model.pt"
echo ""
echo "After that, you can run the XAI pipeline with: !bash run_xai_pipeline.sh"
