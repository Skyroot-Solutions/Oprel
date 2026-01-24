"""
HTTP client for communicating with model backends
"""

from typing import Any, Iterator, Union
import requests

from oprel.client.base import BaseClient
from oprel.core.exceptions import BackendError


class HTTPClient(BaseClient):
    """
    HTTP client for model inference.
    Fallback option when Unix sockets aren't available (Windows).
    """
    
    def __init__(self, port: int):
        self.base_url = f"http://127.0.0.1:{port}"
        self.session = requests.Session()
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[str, Iterator[str]]:
        """
        Generate text via HTTP API.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream response
            **kwargs: Additional parameters
            
        Returns:
            Generated text or token stream
        """
        url = f"{self.base_url}/v1/completions"
        
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            **kwargs,
        }
        
        try:
            if stream:
                return self._stream_response(url, payload)
            else:
                return self._complete_response(url, payload)
        except requests.RequestException as e:
            raise BackendError(f"HTTP request failed: {e}") from e
    
    def _complete_response(self, url: str, payload: dict) -> str:
        """Non-streaming response"""
        response = self.session.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["text"]
    
    def _stream_response(self, url: str, payload: dict) -> Iterator[str]:
        """Streaming response using Server-Sent Events (SSE) format"""
        import json
        
        response = self.session.post(
            url,
            json=payload,
            stream=True,
            timeout=120,
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if not line:
                continue
            
            # Decode bytes to string
            line_str = line.decode('utf-8') if isinstance(line, bytes) else line
            
            # Skip empty lines and comments
            if not line_str.strip() or line_str.startswith(':'):
                continue
            
            # Parse SSE format: "data: {...}" or "data: [DONE]"
            if line_str.startswith('data: '):
                data_str = line_str[6:].strip()
                
                # Handle stream termination
                if data_str == '[DONE]':
                    return
                
                # Skip empty data
                if not data_str:
                    continue
                
                try:
                    data = json.loads(data_str)
                    
                    # Handle different response formats
                    if "choices" in data and len(data["choices"]) > 0:
                        choice = data["choices"][0]
                        # OpenAI format uses "delta" for streaming
                        if "delta" in choice and "content" in choice["delta"]:
                            yield choice["delta"]["content"]
                        # llama.cpp might use "text" directly
                        elif "text" in choice:
                            yield choice["text"]
                        # Some use "content" directly
                        elif "content" in choice:
                            yield choice["content"]
                except json.JSONDecodeError:
                    # Skip malformed JSON (can happen at stream start/end)
                    continue
    
    def close(self) -> None:
        """Close HTTP session"""
        if self.session:
            self.session.close()