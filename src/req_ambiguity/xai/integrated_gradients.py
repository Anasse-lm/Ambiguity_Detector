import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from transformers import PreTrainedTokenizer
from captum.attr import LayerIntegratedGradients


class AmbiguityExplainer:
    """
    Wraps Captum's Layer Integrated Gradients for token-level
    attribution on the trained DeBERTa classifier.
    
    Why LAYER Integrated Gradients rather than vanilla IG?
    Vanilla IG computes gradients with respect to input embeddings,
    but transformer inputs are discrete token IDs, not continuous
    embeddings. We cannot meaningfully interpolate between token IDs.
    Layer Integrated Gradients solves this by attributing to the
    EMBEDDING LAYER outputs — these ARE continuous, so the
    interpolation underlying IG is well-defined.
    """
    def __init__(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizer, device: torch.device, label_cols: List[str]):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.label_cols = label_cols
        
        # Identify the embedding layer for attribution.
        # For DeBERTa-v3, this is encoder.embeddings.word_embeddings.
        self.embedding_layer = self.model.encoder.embeddings.word_embeddings
        
        # We can use the model's forward function directly. 
        # We define a thin wrapper just to map kwargs properly if needed, but 
        # Captum passes args positionally. Our model's forward takes (input_ids, attention_mask).
        self.lig = LayerIntegratedGradients(self._forward_func, self.embedding_layer)

    def _forward_func(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward function for Captum. Captum will use the 'target' argument in 
        attribute() to index into the 2D logits tensor returned here.
        """
        return self.model(input_ids, attention_mask)

    def explain(self, text: str, target_label: str, n_steps: int = 50) -> Tuple[List[str], np.ndarray]:
        """
        Computes token-level attributions for the predicted ambiguity
        of a specific label on a single user story.
        """
        target_idx = self.label_cols.index(target_label)
        
        encoding = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            padding="max_length", max_length=128,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # Baseline: same shape as input but filled with pad tokens.
        # This represents "no information" in the input space.
        baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)
        
        attributions = self.lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            target=target_idx,
            additional_forward_args=(attention_mask,),
            n_steps=n_steps,
            return_convergence_delta=False
        )
        
        # Sum across embedding dimensions to get one score per token
        attributions = attributions.sum(dim=-1).squeeze(0)
        
        # Normalize so attributions are comparable
        norm = torch.linalg.norm(attributions)
        if norm > 0:
            attributions = attributions / norm
            
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        attributions = attributions.detach().cpu().numpy()
        
        return tokens, attributions

    def top_evidence_tokens(self, text: str, target_label: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Convenience method: returns just the top-K most influential
        non-special tokens for the predicted ambiguity. These tokens
        are what we will pass to the refinement module as evidence.
        """
        tokens, attributions = self.explain(text, target_label)
        
        # Filter out special tokens and padding
        special_tokens = {
            self.tokenizer.cls_token, self.tokenizer.sep_token,
            self.tokenizer.pad_token, self.tokenizer.unk_token,
        }
        
        # Collect non-special tokens with their scores
        scored = []
        for tok, score in zip(tokens, attributions):
            if tok not in special_tokens:
                # Remove DeBERTa's specific prefix character " " (U+2581) for cleaner prompt tokens
                clean_tok = tok.replace(" ", "")
                if clean_tok: # avoid empty strings
                    scored.append((clean_tok, float(score)))
                    
        # Sort by absolute attribution descending
        scored.sort(key=lambda x: abs(x[1]), reverse=True)
        return scored[:top_k]
