"""
Main user-facing Model API
"""

from pathlib import Path
from typing import Optional, Dict, Any, Iterator
import threading

from oprel.core.config import Config
from oprel.core.exceptions import OprelError, ModelNotFoundError
from oprel.downloader.hub import download_model
from oprel.runtime.process import ModelProcess
from oprel.runtime.monitor import ProcessMonitor
from oprel.client.base import BaseClient
from oprel.client.socket import UnixSocketClient
from oprel.client.http import HTTPClient
from oprel.telemetry.recommender import recommend_quantization
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


class Model:
    """
    Main interface for loading and running local AI models.
    
    Usage:
        >>> from oprel import Model
        >>> model = Model("TheBloke/Llama-2-7B-GGUF")
        >>> response = model.generate("What is Python?")
        >>> print(response)
    """
    
    def __init__(
        self,
        model_id: str,
        quantization: Optional[str] = None,
        max_memory_mb: Optional[int] = None,
        backend: str = "llama.cpp",
        config: Optional[Config] = None,
    ):
        """
        Initialize a model instance.
        
        Args:
            model_id: HuggingFace model ID (e.g., "TheBloke/Llama-2-7B-GGUF")
            quantization: Quantization level (Q4_K_M, Q5_K_M, Q8_0) or None for auto
            max_memory_mb: Maximum memory limit in MB (None for auto)
            backend: Backend engine ("llama.cpp", "vllm", "exllama")
            config: Custom configuration object
        """
        self.model_id = model_id
        self.config = config or Config()
        self.backend_name = backend
        
        # Auto-detect quantization if not specified
        if quantization is None:
            quantization = recommend_quantization()
            logger.info(f"Auto-selected quantization: {quantization}")
        
        self.quantization = quantization
        self.max_memory_mb = max_memory_mb or self.config.default_max_memory_mb
        
        # Runtime state
        self._process: Optional[ModelProcess] = None
        self._monitor: Optional[ProcessMonitor] = None
        self._client: Optional[BaseClient] = None
        self._lock = threading.Lock()
        self._loaded = False
        
    def load(self) -> None:
        """
        Download and load the model into memory.
        Pre-warming step to avoid latency on first generation.
        """
        with self._lock:
            if self._loaded:
                logger.warning("Model already loaded")
                return
            
            # Step 1: Download model if needed
            logger.info(f"Downloading model: {self.model_id}")
            model_path = download_model(
                self.model_id,
                quantization=self.quantization,
                cache_dir=self.config.cache_dir,
            )
            
            # Step 2: Spawn backend process
            logger.info(f"Starting {self.backend_name} backend")
            self._process = ModelProcess(
                model_path=model_path,
                backend=self.backend_name,
                config=self.config,
            )
            self._process.start()
            
            # Step 3: Start health monitor
            self._monitor = ProcessMonitor(
                process=self._process.process,
                max_memory_mb=self.max_memory_mb,
            )
            self._monitor.start()
            
            # Step 4: Initialize client
            # Unix sockets only work on Linux/macOS, not Windows
            import platform
            use_socket = (
                self.config.use_unix_socket 
                and self._process.socket_path 
                and platform.system() != "Windows"
            )
            
            if use_socket:
                self._client = UnixSocketClient(self._process.socket_path)
            else:
                self._client = HTTPClient(self._process.port)
            
            self._loaded = True
            logger.info(f"Model loaded successfully on port {self._process.port}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs: Any,
    ) -> str | Iterator[str]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            stream: Whether to stream tokens incrementally
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text (string if stream=False, iterator if stream=True)
            
        Raises:
            OprelError: If model is not loaded
            MemoryError: If model exceeds memory limit
        """
        if not self._loaded:
            logger.info("Model not loaded, loading now...")
            self.load()
        
        # Check health before generation
        health_error = self._monitor.check_health()
        if health_error:
            raise health_error
        
        # Generate via client
        return self._client.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            **kwargs,
        )
    
    def unload(self) -> None:
        """
        Stop the model process and free resources.
        """
        with self._lock:
            if not self._loaded:
                return
            
            if self._monitor:
                self._monitor.stop()
            
            if self._process:
                self._process.stop()
            
            self._loaded = False
            logger.info("Model unloaded")
    
    def __enter__(self):
        """Context manager support"""
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.unload()
    
    def __del__(self):
        """Cleanup on garbage collection"""
        if self._loaded:
            self.unload()