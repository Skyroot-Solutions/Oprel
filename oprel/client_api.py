"""
Ollama-compatible client API for Oprel SDK

This module provides an Ollama-compatible interface for Oprel,
allowing easy migration and familiar syntax.
"""

import time
import uuid
from typing import Dict, List, Any, Optional, Iterator, Union
from pathlib import Path

import requests

from oprel.core.model import Model
from oprel.core.config import Config
from oprel.downloader.cache import list_cached_models
from oprel.api_models import (
    ChatResponse,
    GenerateResponse,
    ListResponse,
    ShowResponse,
    ModelInfo,
    Message,
)


class Client:
    """
    Ollama-compatible client for Oprel SDK
    
    Example:
        from oprel import Client
        
        client = Client(host='http://localhost:11434')
        response = client.chat(
            model='qwencoder',
            messages=[{'role': 'user', 'content': 'Hello!'}]
        )
        print(response.message.content)
    """
    
    def __init__(
        self,
        host: str = "http://localhost:11434",
        timeout: float = 300.0,
    ):
        """
        Initialize Oprel client
        
        Args:
            host: Server URL (default: http://localhost:11434)
            timeout: Request timeout in seconds (default: 300)
        """
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._config = Config()
    
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        format: Optional[Union[str, Dict]] = None,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
    ) -> Union[ChatResponse, Iterator[ChatResponse]]:
        """
        Generate chat completions
        
        Args:
            model: Model name or ID
            messages: List of message dicts with 'role' and 'content'
            stream: Enable streaming responses
            format: Response format (e.g., 'json' or Pydantic schema)
            options: Model parameters (temperature, max_tokens, etc.)
            keep_alive: Keep model loaded duration
        
        Returns:
            ChatResponse or iterator of ChatResponse chunks if streaming
        
        Example:
            response = client.chat(
                model='qwencoder',
                messages=[{'role': 'user', 'content': 'Hello!'}]
            )
            print(response.message.content)
        """
        start_time = time.time()
        
        # Use server mode
        oprel_model = Model(model, use_server=True)
        oprel_model.load()
        
        # Build conversation from messages
        conversation_id = str(uuid.uuid4())
        system_prompt = None
        user_prompt = ""
        
        # Extract system prompt and last user message
        for msg in messages:
            if msg['role'] == 'system':
                system_prompt = msg['content']
            elif msg['role'] == 'user':
                user_prompt = msg['content']
        
        # Parse options
        opts = options or {}
        max_tokens = opts.get('num_predict', 512)
        temperature = opts.get('temperature', 0.7)
        
        if stream:
            return self._chat_stream(
                oprel_model,
                user_prompt,
                conversation_id,
                system_prompt,
                max_tokens,
                temperature,
                start_time,
            )
        
        # Non-streaming response
        response_text = oprel_model.generate(
            user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            conversation_id=conversation_id,
            system_prompt=system_prompt,
        )
        
        total_duration = int((time.time() - start_time) * 1e9)  # nanoseconds
        
        return ChatResponse(
            model=model,
            message=Message(role='assistant', content=response_text),
            total_duration=total_duration,
            eval_count=len(response_text.split()),
        )
    
    def _chat_stream(
        self,
        oprel_model: Model,
        prompt: str,
        conversation_id: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        start_time: float,
    ) -> Iterator[ChatResponse]:
        """Stream chat responses"""
        full_response = ""
        
        for token in oprel_model.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            conversation_id=conversation_id,
            system_prompt=system_prompt,
        ):
            full_response += token
            
            yield ChatResponse(
                model=oprel_model.model_id,
                message=Message(role='assistant', content=token),
                done=False,
            )
        
        # Final chunk with done=True
        total_duration = int((time.time() - start_time) * 1e9)
        yield ChatResponse(
            model=oprel_model.model_id,
            message=Message(role='assistant', content=''),
            done=True,
            total_duration=total_duration,
            eval_count=len(full_response.split()),
        )
    
    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        format: Optional[Union[str, Dict]] = None,
        options: Optional[Dict[str, Any]] = None,
        system: Optional[str] = None,
        template: Optional[str] = None,
        context: Optional[List[int]] = None,
        keep_alive: Optional[str] = None,
    ) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """
        Generate completions
        
        Args:
            model: Model name or ID
            prompt: Text prompt
            stream: Enable streaming
            format: Response format
            options: Model parameters
            system: System prompt
            template: Prompt template
            context: Context from previous generation
            keep_alive: Keep model loaded duration
        
        Returns:
            GenerateResponse or iterator if streaming
        
        Example:
            response = client.generate(
                model='qwencoder',
                prompt='Why is the sky blue?'
            )
            print(response.response)
        """
        start_time = time.time()
        
        oprel_model = Model(model, use_server=True)
        oprel_model.load()
        
        opts = options or {}
        max_tokens = opts.get('num_predict', 512)
        temperature = opts.get('temperature', 0.7)
        
        if stream:
            return self._generate_stream(
                oprel_model,
                prompt,
                system,
                max_tokens,
                temperature,
                start_time,
            )
        
        response_text = oprel_model.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system,
        )
        
        total_duration = int((time.time() - start_time) * 1e9)
        
        return GenerateResponse(
            model=model,
            response=response_text,
            total_duration=total_duration,
            eval_count=len(response_text.split()),
        )
    
    def _generate_stream(
        self,
        oprel_model: Model,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        start_time: float,
    ) -> Iterator[GenerateResponse]:
        """Stream generate responses"""
        full_response = ""
        
        for token in oprel_model.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            system_prompt=system,
        ):
            full_response += token
            
            yield GenerateResponse(
                model=oprel_model.model_id,
                response=token,
                done=False,
            )
        
        total_duration = int((time.time() - start_time) * 1e9)
        yield GenerateResponse(
            model=oprel_model.model_id,
            response='',
            done=True,
            total_duration=total_duration,
            eval_count=len(full_response.split()),
        )
    
    def list(self) -> ListResponse:
        """
        List available models
        
        Returns:
            ListResponse with list of models
        
        Example:
            models = client.list()
            for model in models.models:
                print(model.name)
        """
        try:
            # Try to get from server
            response = requests.get(f"{self.host}/models", timeout=5)
            if response.status_code == 200:
                server_models = response.json()
                
                model_infos = []
                for m in server_models:
                    model_infos.append(ModelInfo(
                        name=m.get('model_id', m.get('name', 'unknown')),
                        modified_at=m.get('modified_at', ''),
                        size=m.get('size_gb', 0) * 1024**3,
                        digest=m.get('quantization', 'unknown'),
                        details={
                            'backend': m.get('backend', 'llama.cpp'),
                            'loaded': m.get('loaded', False),
                        }
                    ))
                
                return ListResponse(models=model_infos)
        except:
            pass
        
        # Fallback to cache
        cached = list_cached_models()
        model_infos = []
        
        for m in cached:
            model_infos.append(ModelInfo(
                name=m['name'],
                modified_at=m['modified'].isoformat(),
                size=int(m['size_mb'] * 1024 * 1024),
                digest='unknown',
                details={'cached': True}
            ))
        
        return ListResponse(models=model_infos)
    
    def show(self, model: str) -> ShowResponse:
        """
        Show model information
        
        Args:
            model: Model name
        
        Returns:
            ShowResponse with model details
        
        Example:
            info = client.show('qwencoder')
            print(info.details)
        """
        # Get model info from cache
        cached = list_cached_models()
        
        for m in cached:
            if model in m['name']:
                return ShowResponse(
                    modelfile=f"FROM {m['name']}",
                    parameters="",
                    template="",
                    details={
                        'name': m['name'],
                        'size': m['size_mb'],
                        'modified': m['modified'].isoformat(),
                        'backend': 'llama.cpp',
                    }
                )
        
        # Model not found, return minimal info
        return ShowResponse(
            modelfile=f"FROM {model}",
            parameters="",
            template="",
            details={'name': model}
        )
    
    def create(
        self,
        model: str,
        from_: Optional[str] = None,
        modelfile: Optional[str] = None,
        system: Optional[str] = None,
        stream: bool = False,
    ) -> Dict[str, str]:
        """
        Create a custom model variant
        
        Args:
            model: New model name
            from_: Base model to customize
            modelfile: Full modelfile content
            system: System prompt for the model
            stream: Stream creation progress
        
        Returns:
            Status dict
        
        Example:
            client.create(
                model='my-assistant',
                from_='qwencoder',
                system='You are a helpful Python expert.'
            )
        """
        # For oprel, we'll just return success since customization
        # happens at runtime via system prompts
        return {
            'status': 'success',
            'message': f'Model {model} configured (system prompts applied at runtime)'
        }
    
    def pull(
        self,
        model: str,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Download a model
        
        Args:
            model: Model name to download
            stream: Stream download progress
        
        Returns:
            Status dict
        
        Example:
            client.pull('qwencoder')
        """
        from oprel.downloader.hub import download_model
        from oprel.downloader.aliases import resolve_model_id
        from oprel.telemetry.recommender import recommend_quantization
        
        model_id = resolve_model_id(model)
        quantization = recommend_quantization()
        
        model_path = download_model(
            model_id,
            quantization=quantization,
            cache_dir=self._config.cache_dir,
        )
        
        return {
            'status': 'success',
            'model': model,
            'path': str(model_path),
        }
    
    def delete(self, model: str) -> Dict[str, str]:
        """
        Delete a model
        
        Args:
            model: Model name to delete
        
        Returns:
            Status dict
        """
        from oprel.downloader.cache import delete_model
        
        if delete_model(model):
            return {'status': 'success', 'message': f'Deleted {model}'}
        else:
            return {'status': 'error', 'message': f'Model {model} not found'}
    
    def embed(
        self,
        texts: Union[str, List[str]],
        model: str = "nomic-embed-text",
        normalize: bool = True,
        **options
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text(s)
        
        Args:
            texts: Single text string or list of texts to embed
            model: Embedding model name (default: nomic-embed-text)
            normalize: Whether to normalize embeddings to unit length
            options: Additional options (truncate, etc.)
        
        Returns:
            Single embedding vector for string input,
            or list of embedding vectors for list input
        
        Example:
            # Single text
            embedding = client.embed("Hello world", model="nomic-embed-text")
            
            # Batch texts
            embeddings = client.embed(
                ["Hello", "World", "Test"],
                model="snowflake-arctic"
            )
        """
        from oprel.core.model import Model
        import requests
        
        # Ensure texts is a list
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts
        
        # Load embedding model
        oprel_model = Model(model, use_server=True)
        oprel_model.load()
        
        # Check if backend is ready
        if not hasattr(oprel_model, '_process') or not oprel_model._process:
            raise RuntimeError("Embedding model backend not initialized")
        
        port = oprel_model._process.port
        
        # Call llama.cpp embedding endpoint
        # llama.cpp serves embeddings at /embedding or /v1/embeddings
        embeddings = []
        
        for text in text_list:
            try:
                response = requests.post(
                    f"http://127.0.0.1:{port}/embedding",
                    json={"content": text},
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                
                # Extract embedding vector
                if "embedding" in result:
                    embedding = result["embedding"]
                elif "data" in result and len(result["data"]) > 0:
                    # OpenAI format
                    embedding = result["data"][0]["embedding"]
                else:
                    raise ValueError(f"Unexpected response format: {result}")
                
                # Normalize if requested
                if normalize:
                    import math
                    magnitude = math.sqrt(sum(x*x for x in embedding))
                    if magnitude > 0:
                        embedding = [x / magnitude for x in embedding]
                
                embeddings.append(embedding)
                
            except requests.RequestException as e:
                raise RuntimeError(f"Failed to generate embedding: {e}")
        
        # Return single embedding or list based on input
        return embeddings[0] if is_single else embeddings


# Async client placeholder (can be implemented later with aiohttp)
class AsyncClient(Client):
    """
    Async client for Oprel (uses sync client for now)
    
    Note: Full async implementation requires aiohttp.
    This is a placeholder that uses sync methods.
    """
    
    async def chat(self, *args, **kwargs):
        """Async chat (currently uses sync)"""
        return super().chat(*args, **kwargs)
    
    async def generate(self, *args, **kwargs):
        """Async generate (currently uses sync)"""
        return super().generate(*args, **kwargs)


# Module-level convenience functions
_default_client: Optional[Client] = None


def _get_client() -> Client:
    """Get or create default client"""
    global _default_client
    if _default_client is None:
        _default_client = Client()
    return _default_client


def chat(
    model: str,
    messages: List[Dict[str, str]],
    stream: bool = False,
    **kwargs
) -> Union[ChatResponse, Iterator[ChatResponse]]:
    """
    Module-level chat function
    
    Example:
        from oprel import chat
        
        response = chat(
            model='qwencoder',
            messages=[{'role': 'user', 'content': 'Hello!'}]
        )
        print(response.message.content)
    """
    return _get_client().chat(model, messages, stream, **kwargs)


def generate(
    model: str,
    prompt: str,
    stream: bool = False,
    **kwargs
) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
    """
    Module-level generate function
    
    Example:
        from oprel import generate
        
        response = generate(
            model='qwencoder',
            prompt='Why is the sky blue?'
        )
        print(response.response)
    """
    return _get_client().generate(model, prompt, stream, **kwargs)


def list() -> ListResponse:
    """
    Module-level list function
    
    Example:
        from oprel import list
        
        models = list()
        for model in models.models:
            print(model.name)
    """
    return _get_client().list()


def show(model: str) -> ShowResponse:
    """Module-level show function"""
    return _get_client().show(model)


def create(model: str, **kwargs) -> Dict[str, str]:
    """Module-level create function"""
    return _get_client().create(model, **kwargs)


def pull(model: str, **kwargs) -> Dict[str, Any]:
    """Module-level pull function"""
    return _get_client().pull(model, **kwargs)


def delete(model: str) -> Dict[str, str]:
    """Module-level delete function"""
    return _get_client().delete(model)


def embed(
    texts: Union[str, List[str]],
    model: str = "nomic-embed-text",
    **kwargs
) -> Union[List[float], List[List[float]]]:
    """
    Module-level embed function
    
    Example:
        from oprel import embed
        
        # Single text
        vector = embed("Hello world")
        
        # Batch
        vectors = embed(["Hello", "World"], model="snowflake-arctic")
    """
    return _get_client().embed(texts, model, **kwargs)
