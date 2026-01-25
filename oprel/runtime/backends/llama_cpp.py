"""
llama.cpp backend implementation
"""

from pathlib import Path
from typing import List

from oprel.core.config import Config
from oprel.runtime.backends.base import BaseBackend
from oprel.telemetry.hardware import get_recommended_threads, detect_gpu, calculate_gpu_layers
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


class LlamaCppBackend(BaseBackend):
    """
    Backend implementation for llama.cpp server.

    Uses the pre-compiled llama-server binary.
    """

    def build_command(self, port: int) -> List[str]:
        """
        Build command for llama-server with optimal GPU settings.

        Args:
            port: Server port

        Returns:
            Command list
        """
        cmd = [
            str(self.binary_path),
            "--model",
            str(self.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]

        # CPU threads (even with GPU, still need some CPU threads)
        n_threads = self.config.n_threads or get_recommended_threads()
        cmd.extend(["--threads", str(n_threads)])

        # GPU layers - CRITICAL for performance!
        gpu_info = detect_gpu()
        if gpu_info and gpu_info.get("gpu_type") in ("cuda", "metal"):
            # Get model size to calculate layers
            model_size_gb = self.model_path.stat().st_size / (1024**3)
            vram_gb = gpu_info.get("vram_total_gb", 4.0)
            
            n_gpu_layers = self.config.n_gpu_layers
            
            if n_gpu_layers == -1:
                # Auto-calculate optimal layers based on VRAM
                n_gpu_layers = calculate_gpu_layers(vram_gb, model_size_gb)
                logger.info(
                    f"Auto-calculated GPU layers: {n_gpu_layers} "
                    f"(model: {model_size_gb:.1f}GB, VRAM: {vram_gb:.1f}GB)"
                )
            
            if n_gpu_layers > 0:
                cmd.extend(["--n-gpu-layers", str(n_gpu_layers)])
                logger.info(f"GPU acceleration: ENABLED ({n_gpu_layers} layers on {gpu_info['gpu_name']})")
            else:
                logger.warning("GPU detected but n_gpu_layers=0, using CPU only")
        else:
            logger.warning("GPU acceleration: DISABLED (no CUDA/Metal GPU detected)")

        # Context size - match Ollama default (4096)
        ctx_size = getattr(self.config, 'ctx_size', 4096)
        cmd.extend(["--ctx-size", str(ctx_size)])

        # Batch size
        batch_size = getattr(self.config, 'batch_size', 512)
        cmd.extend(["--batch-size", str(batch_size)])

        return cmd

    def get_api_format(self) -> str:
        """llama.cpp uses OpenAI-compatible API"""
        return "openai"
