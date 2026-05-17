from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RefinementRequest:
    prompt_text: str
    model_name: str
    temperature: float
    max_output_tokens: int
    top_p: float

@dataclass
class RefinementResponse:
    text: str
    raw_response: Dict[str, Any]
    backend_used: str

class RefinementBackend(ABC):
    """
    Abstract base class for all LLM refinement backends.
    This pluggable interface allows swapping Gemini for OpenAI or local models
    without changing the orchestration or validation logic.
    """
    @abstractmethod
    def call(self, request: RefinementRequest) -> RefinementResponse:
        pass
