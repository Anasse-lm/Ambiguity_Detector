"""PyTorch Dataset wrapping DeBERTa tokenization for user stories."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class UserStoryDataset(Dataset):
    """
    Multi-label user story dataset.

    Each item returns input_ids, attention_mask, labels (float32 K-vector),
    and raw text for XAI / refinement downstream.
    """

    def __init__(
        self,
        texts: list[str],
        labels: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        if len(texts) != labels.shape[0]:
            raise ValueError("texts and labels length mismatch")
        self._texts = texts
        self._labels = labels
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        encoding = self._tokenizer(
            self._texts[idx],
            max_length=self._max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": self._labels[idx],
            "text": self._texts[idx],
        }

    @classmethod
    def from_dataframe(
        cls,
        df,
        *,
        text_column: str,
        label_cols: list[str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> UserStoryDataset:
        texts = df[text_column].astype(str).tolist()
        labels = torch.tensor(df[label_cols].values, dtype=torch.float32)
        return cls(texts, labels, tokenizer, max_length)
