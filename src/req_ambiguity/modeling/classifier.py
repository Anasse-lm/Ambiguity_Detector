"""DeBERTa-v3 encoder + linear multi-label head (BCEWithLogitsLoss at train time)."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class DeBERTaAmbiguityClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

    @torch.no_grad()
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        logits = self.forward(input_ids, attention_mask)
        return torch.sigmoid(logits)
