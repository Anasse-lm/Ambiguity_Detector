import os
import json
import hashlib
import time
import logging

from req_ambiguity.refinement.backends.base import RefinementBackend, RefinementRequest, RefinementResponse

logger = logging.getLogger(__name__)

class CachedBackend(RefinementBackend):
    """
    Wraps any RefinementBackend with a transparent disk cache.
    Cache key: SHA256 of (model_name, prompt_text, temperature, top_p, max_output_tokens)
    Cache storage: one JSON file per entry under cache_dir
    """
    def __init__(self, backend: RefinementBackend, cache_dir: str, cache_enabled: bool = True):
        self.backend = backend
        self.cache_dir = cache_dir
        self.cache_enabled = cache_enabled
        
        if self.cache_enabled:
            os.makedirs(self.cache_dir, exist_ok=True)
            
    def _hash_request(self, request: RefinementRequest) -> str:
        key_str = f"{request.model_name}|{request.prompt_text}|{request.temperature}|{request.top_p}|{request.max_output_tokens}"
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()
        
    def call(self, request: RefinementRequest) -> RefinementResponse:
        if not self.cache_enabled:
            return self.backend.call(request)
            
        req_hash = self._hash_request(request)
        cache_file = os.path.join(self.cache_dir, f"{req_hash}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                logger.info(f"CACHE HIT {req_hash[:8]}")
                return RefinementResponse(
                    text=cached_data['response']['text'],
                    raw_response=cached_data['response']['raw_response'],
                    backend_used=cached_data['response']['backend_used'] + "_cached"
                )
            except Exception as e:
                logger.warning(f"Failed to read cache file {cache_file}: {str(e)}. Proceeding to backend call.")
                
        # Cache miss
        response = self.backend.call(request)
        
        try:
            cache_entry = {
                "hash": req_hash,
                "timestamp": time.time(),
                "request": {
                    "prompt_text": request.prompt_text,
                    "model_name": request.model_name,
                    "temperature": request.temperature,
                    "max_output_tokens": request.max_output_tokens,
                    "top_p": request.top_p
                },
                "response": {
                    "text": response.text,
                    "raw_response": response.raw_response,
                    "backend_used": response.backend_used
                }
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_entry, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write cache file {cache_file}: {str(e)}")
            
        return response
