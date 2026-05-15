import json
import torch
import transformers
from pathlib import Path
from typing import Any, Mapping

def save_best_checkpoint(
    state_dict: dict[str, torch.Tensor],
    checkpoint_dir: Path,
    metadata: dict[str, Any],
) -> Path:
    """
    Saves the best model state dictionary and metadata.
    Does not save optimizer states to minimize disk space.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "best_model.pt"
    
    # Save the model state
    torch.save(state_dict, best_model_path)
    
    # Enrich metadata with versions
    metadata["torch_version"] = torch.__version__
    metadata["transformers_version"] = transformers.__version__
    
    # Save the metadata
    metadata_path = checkpoint_dir / "best_model_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
        
    return best_model_path
