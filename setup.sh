#!/usr/bin/env bash

echo "Setting up environment for Thesis Training Run..."

# FastAI / Kaggle usually has most of these, but we ensure they are upgraded/installed
# sentencepiece is absolutely critical for DeBERTa tokenizers!
pip install -qU torch transformers pandas numpy scikit-learn matplotlib seaborn PyYAML tqdm sentencepiece

echo "Dependencies installed successfully!"
echo "You can now run your training scripts:"
echo "  !python src/run_hparam_sensitivity.py"
echo "  !python src/run_multiseed.py"
echo "  !python src/req_ambiguity/utils/zipper.py"
