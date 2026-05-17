import os
import time
import logging
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from req_ambiguity.refinement.backends.base import RefinementBackend, RefinementRequest, RefinementResponse

logger = logging.getLogger(__name__)

class GeminiBackend(RefinementBackend):
    """
    Implements the RefinementBackend interface using Google's Gemini API.
    Includes exponential backoff retry logic for transient API failures.
    """
    def __init__(self, max_retries: int = 3, retry_delay_seconds: float = 2.0):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set. Please export it.")
        
        genai.configure(api_key=api_key)
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        
    def call(self, request: RefinementRequest) -> RefinementResponse:
        model = genai.GenerativeModel(request.model_name)
        
        generation_config = genai.types.GenerationConfig(
            temperature=request.temperature,
            top_p=request.top_p,
            max_output_tokens=request.max_output_tokens,
        )
        
        # Disable safety settings which can occasionally false-positive on SE terms
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = model.generate_content(
                    request.prompt_text,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                if not response.parts:
                    reason = str(response.candidates[0].finish_reason) if response.candidates else 'Unknown'
                    raise ValueError(f"Empty response from Gemini. Finish reason: {reason}")
                    
                return RefinementResponse(
                    text=response.text,
                    raw_response={"finish_reason": str(response.candidates[0].finish_reason) if response.candidates else "Unknown"},
                    backend_used="GeminiBackend"
                )
            except Exception as e:
                last_error = e
                logger.warning(f"GeminiBackend call failed (Attempt {attempt}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries:
                    # Exponential backoff
                    sleep_time = self.retry_delay_seconds * (2 ** (attempt - 1))
                    time.sleep(sleep_time)
                    
        raise RuntimeError(f"GeminiBackend failed after {self.max_retries} attempts. Last error: {str(last_error)}")
