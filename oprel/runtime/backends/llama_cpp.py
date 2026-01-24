"""
llama.cpp backend implementation
"""

from pathlib import Path
from typing import List

from oprel.core.config import Config
from oprel.runtime.backends.base import BaseBackend
from oprel.telemetry.hardware import get_recommended_threads, detect_gpu


class LlamaCppBackend(BaseBackend):
    """
    Backend implementation for llama.cpp server.
    
    Uses the pre-compiled llama-server binary.
    """
    
    def build_command(self, port: int) -> List[str]:
        """
        Build command for llama-server.
        
        Args:
            port: Server port
            
        Returns:
            Command list
        """
        cmd = [
            str(self.binary_path),
            "--model", str(self.model_path),
            "--host", "127.0.0.1",
            "--port", str(port),
        ]
        
        # CPU threads
        n_threads = self.config.n_threads or get_recommended_threads()
        cmd.extend(["--threads", str(n_threads)])
        
        # GPU layers
        gpu_info = detect_gpu()
        if gpu_info:
            n_gpu_layers = self.config.n_gpu_layers
            if n_gpu_layers != 0:
                cmd.extend(["--n-gpu-layers", str(n_gpu_layers)])
        
        # Context size (default 2048)
        cmd.extend(["--ctx-size", "2048"])
        
        # Batch size (default 512)
        cmd.extend(["--batch-size", "512"])
        
        return cmd
    
    def get_api_format(self) -> str:
        """llama.cpp uses OpenAI-compatible API"""
        return "openai"