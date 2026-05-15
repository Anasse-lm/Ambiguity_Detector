# XAI Output Schema

This folder contains `xai_output.schema.json`, which defines the contract between the Explainable AI (XAI) phase and the downstream Large Language Model (LLM) refinement phase.

## Purpose
The trained DeBERTa model detects whether an ambiguity exists. The XAI phase uses Layer Integrated Gradients (LIG) to extract the *exact words* (evidence tokens) that caused the model to flag the ambiguity. 

The `PlaceholderBridge` maps these evidence tokens to structured placeholders defined in `configs/placeholders.yaml`.

This structured data is passed to the LLM. Instead of asking the LLM to blindly "fix" the ambiguity, we pass this JSON which tells the LLM:
1. The original requirement.
2. The exact type of ambiguity detected.
3. The specific words that triggered it.
4. The requested placeholder to use to resolve it.

This significantly reduces LLM hallucinations and ensures the refinement is grounded in the DeBERTa model's predictions.
