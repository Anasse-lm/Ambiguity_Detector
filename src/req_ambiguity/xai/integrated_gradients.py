import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from transformers import PreTrainedTokenizer
from captum.attr import LayerIntegratedGradients
import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class AmbiguityExplainer:
    """
    Wraps Captum's Layer Integrated Gradients for token-level
    attribution on the trained DeBERTa classifier.
    """
    def __init__(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizer, device: torch.device, label_cols: List[str]):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.label_cols = label_cols
        
        self.embedding_layer = self.model.encoder.embeddings.word_embeddings
        self.lig = LayerIntegratedGradients(self._forward_func, self.embedding_layer)

        # Ensure cache directory exists
        from req_ambiguity.utils.config import find_project_root
        try:
            self.cache_dir = find_project_root() / "outputs" / "xai" / "cache" / "ig"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            self.cache_dir = Path("outputs/xai/cache/ig")
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _forward_func(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, attention_mask)

    def explain(self, text: str, target_label: str, n_steps: int = 50, story_id: str = None) -> Tuple[List[str], np.ndarray]:
        target_idx = self.label_cols.index(target_label)
        
        cache_path = None
        if story_id is not None:
            safe_label = target_label.replace("/", "_")
            cache_path = self.cache_dir / f"{story_id}_{safe_label}_{n_steps}.json"
            if cache_path.exists():
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    logger.info(f"Cache hit for IG: {story_id} | {target_label} | {n_steps}")
                    return data["tokens"], np.array(data["attributions"])
                except Exception as e:
                    pass

        encoding = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            padding="max_length", max_length=128,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)
        
        attributions = self.lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            target=target_idx,
            additional_forward_args=(attention_mask,),
            n_steps=n_steps,
            return_convergence_delta=False
        )
        
        attributions = attributions.sum(dim=-1).squeeze(0)
        
        norm = torch.linalg.norm(attributions)
        if norm > 0:
            attributions = attributions / norm
            
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        attributions_np = attributions.detach().cpu().numpy()
        
        if cache_path is not None:
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "tokens": tokens,
                        "attributions": attributions_np.tolist()
                    }, f)
            except Exception as e:
                pass
                
        return tokens, attributions_np

    def top_evidence_tokens(self, text: str, target_label: str, top_k: int = 5, story_id: str = None) -> List[Tuple[str, float]]:
        tokens, attributions = self.explain(text, target_label, story_id=story_id)
        
        special_tokens = {
            self.tokenizer.cls_token, self.tokenizer.sep_token,
            self.tokenizer.pad_token, self.tokenizer.unk_token,
        }
        
        scored = []
        for tok, score in zip(tokens, attributions):
            if tok not in special_tokens:
                clean_tok = tok.replace(" ", "")
                if clean_tok:
                    scored.append((clean_tok, float(score)))
                    
        scored.sort(key=lambda x: abs(x[1]), reverse=True)
        return scored[:top_k]
