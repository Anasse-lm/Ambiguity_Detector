import re
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class VerificationResult:
    original_text: str
    refined_text: str
    refined_stripped: str
    probs_before: np.ndarray
    probs_after: np.ndarray
    per_label_delta: Dict[str, float]
    aggregate_delta: float
    improved: bool

class Verifier:
    """
    Computes the change in classifier-predicted ambiguity after refinement.
    IMPORTANT: We strip placeholders before verification to ensure the
    classifier is not artificially influenced by tokens it never saw in training.
    """
    def __init__(self, model, tokenizer, device: str, label_cols: List[str]):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.label_cols = label_cols
        
    def _score(self, text: str) -> np.ndarray:
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze(0)
            
        return probs
        
    def verify(self, original_text: str, refined_text: str) -> VerificationResult:
        # Strip placeholders from refined_text
        stripped = re.sub(r'<TBD_[A-Z_]+>', ' ', refined_text)
        stripped = re.sub(r'\s+', ' ', stripped).strip()
        
        probs_before = self._score(original_text)
        probs_after = self._score(stripped)
        
        per_label_delta = {}
        for i, col in enumerate(self.label_cols):
            # A negative delta means the probability DROPPED (ambiguity reduced)
            per_label_delta[col] = float(probs_after[i] - probs_before[i])
            
        aggregate_delta = float(np.mean(list(per_label_delta.values())))
        improved = aggregate_delta < 0
        
        return VerificationResult(
            original_text=original_text,
            refined_text=refined_text,
            refined_stripped=stripped,
            probs_before=probs_before,
            probs_after=probs_after,
            per_label_delta=per_label_delta,
            aggregate_delta=aggregate_delta,
            improved=improved
        )
