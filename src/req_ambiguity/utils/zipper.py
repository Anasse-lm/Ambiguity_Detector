import os
import zipfile
import datetime
from pathlib import Path

def create_artifact_zip(project_root: Path) -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = project_root / "outputs" / f"final_run_{timestamp}.zip"
    
    print(f"Packaging final thesis artifacts into: {zip_path}")
    
    # Generate dynamic README content
    readme_content = f"""# Thesis Final Run Artifacts
Generated: {timestamp}

This archive contains the complete, reproducible training and evaluation pipeline for the DeBERTa-based multi-label ambiguity classifier.

## File Index
- `configs/` : Snapshot of the training configuration (`train.yaml`) used for this run.
- `src/` : Full source code for exact reproducibility.
- `outputs/checkpoints/` : Contains the canonical `best_model.pt` and its metadata.
- `outputs/results/` : All statistical outputs, including the per-label diagnostics, threshold sweeps, hyperparameter sensitivity, and multi-seed aggregation.
- `outputs/figures/` : High-resolution (150-300dpi) plots for the thesis (loss curves, F1 curves, confusion matrices).
- `outputs/seed_*/` : Raw isolation directories for each independent seed run.

## Key Instructions
To load the canonical model for XAI and refinement:
1. Load the state dictionary from `outputs/checkpoints/best_model.pt`
2. Load the optimal decision thresholds from `outputs/results/optimal_thresholds.json`
"""

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Write README
        zipf.writestr("README.md", readme_content)
        
        # Define paths to include
        paths_to_include = [
            "configs",
            "src",
            "requirements.txt",
            "outputs/results",
            "outputs/figures",
            "outputs/reports",
            "outputs/checkpoints/best_model.pt",
            "outputs/checkpoints/best_model_metadata.json"
        ]
        
        # Add seed directories dynamically
        outputs_dir = project_root / "outputs"
        if outputs_dir.exists():
            for seed_dir in outputs_dir.glob("seed_*"):
                if seed_dir.is_dir():
                    paths_to_include.append(f"outputs/{seed_dir.name}")
        
        # Add files
        for relative_path in paths_to_include:
            full_path = project_root / relative_path
            
            if not full_path.exists():
                print(f"Warning: {full_path} not found. Skipping.")
                continue
                
            if full_path.is_file():
                zipf.write(full_path, arcname=relative_path)
            else:
                for root, dirs, files in os.walk(full_path):
                    for file in files:
                        # Exclude cache or temp files
                        if file.endswith(".pyc") or "__pycache__" in root:
                            continue
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, project_root)
                        zipf.write(file_path, arcname=arcname)
                        
    return zip_path

if __name__ == "__main__":
    _ROOT = Path(__file__).resolve().parents[3]
    result = create_artifact_zip(_ROOT)
    print(f"\nFinal artifact zip created successfully at:\n{result.absolute()}")
