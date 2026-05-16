#!/bin/bash
set -e

# Make sure we're in the right directory where the zip was uploaded
if [ ! -f "remote_outputs_transfer.zip" ]; then
    echo "Error: remote_outputs_transfer.zip not found in the current directory."
    echo "Please ensure you have uploaded it to your Jupyter environment."
    exit 1
fi

echo "Unzipping outputs..."
unzip -o remote_outputs_transfer.zip

echo "Unzip complete! The outputs directory is now populated."
echo "You can now run the XAI pipeline with: !bash run_xai_pipeline.sh"
