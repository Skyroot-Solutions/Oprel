"""
Backend selector (M1.22)

Smart backend selection based on hardware capabilities.
Automatically chooses the best backend (vLLM, PyTorch, or llama.cpp) for the user's hardware.

This module is KEY to beating Ollama - we intelligently select the fastest backend
for each hardware configuration instead of using one-size-fits-all.

Backends supported:
- llama.cpp: CPU-only, low VRAM GPUs, hybrid GPU/CPU
- pytorch: Mid-range GPUs (6-16GB VRAM) - 20-30% faster than llama.cpp
- vllm: High-end GPUs (16GB+ VRAM) - for Month 3
"""

import os
from enum import Enum
from typing import Optional, Dict
from pathlib import Path

from oprel.telemetry.profiler import profile_hardware, HardwareProfile
from oprel.telemetry.memory import estimate_max_model_size, check_ram_for_model
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


class BackendType(Enum):
    """Available backend types"""
    LLAMA_CPP = "llama.cpp"
    PYTORCH = "pytorch"
    VLLM = "vllm"  # Future: Month 3


class BackendSelector:
    """
    Intelligent backend selection based on hardware and model requirements.
    
    Selection logic:
    - vLLM: 16GB+ VRAM (Month 3 implementation)
    - PyTorch: 6GB+ VRAM, model fits in VRAM
    - llama.cpp: All other cases (CPU-only, low VRAM, large models)
    
    Advantages over Ollama:
    - Automatic optimal backend selection
    - Graceful fallback chain when backend fails
    - Per-model backend recommendations
    - Hardware profiling with caching for fast decisions
    """
    
    def __init__(self, profile: Optional[HardwareProfile] = None):
        """
        Initialize backend selector.
        
        Args:
            profile: Optional pre-loaded HardwareProfile. If None, will profile automatically.
        """
        self.profile = profile or profile_hardware()
        logger.debug(f"Initialized BackendSelector with profile: {self.profile.gpu_type}")
    
    def select_backend(
        self,
        model_size_gb: Optional[float] = None,
        preferred_backend: Optional[str] = None,
    ) -> str:
        """
        Select the best backend for given model and hardware.
        
        Args:
            model_size_gb: Model file size in GB (optional, for better decision making)
            preferred_backend: User's preferred backend override ("auto", "llama.cpp", "pytorch", "vllm")
            
        Returns:
            Selected backend name: "llama.cpp", "pytorch", or "vllm"
        """
        # Check for manual override via environment variable
        env_backend = os.getenv("OPREL_BACKEND")
        if env_backend and env_backend != "auto":
            logger.info(f"Using backend from OPREL_BACKEND environment variable: {env_backend}")
            return self._validate_backend(env_backend)
        
        # Check for preferred backend parameter
        if preferred_backend and preferred_backend != "auto":
            logger.info(f"Using preferred backend: {preferred_backend}")
            return self._validate_backend(preferred_backend)
        
        # Automatic selection based on hardware
        backend = self._auto_select_backend(model_size_gb)
        
        reason = self._get_selection_reason(backend, model_size_gb)
        logger.info(f"✓ Selected backend: {backend} ({reason})")
        
        return backend
    
    def _auto_select_backend(self, model_size_gb: Optional[float] = None) -> str:
        """
        Automatically select backend based on hardware profile.
        
        Args:
            model_size_gb: Model size in GB
            
        Returns:
            Selected backend name
        """
        # No GPU detected - use llama.cpp
        if not self.profile.has_gpu:
            logger.debug("No GPU detected, selecting llama.cpp for CPU inference")
            return BackendType.LLAMA_CPP.value
        
        vram_gb = self.profile.vram_total_gb or 0.0
        
        # Future: Month 3 - vLLM for high-end GPUs
        # if vram_gb >= 16:
        #     logger.debug(f"High-end GPU ({vram_gb}GB VRAM), selecting vLLM")
        #     return BackendType.VLLM.value
        
        # PyTorch for mid-range GPUs (6GB+ VRAM)
        if vram_gb >= 6:
            # Check if model will fit in VRAM (if size is known)
            if model_size_gb:
                # Model needs ~20% extra for overhead (KV cache, etc.)
                required_vram = model_size_gb * 1.2
                
                if required_vram <= vram_gb * 0.9:  # Leave 10% buffer
                    logger.debug(
                        f"Model ({model_size_gb:.1f}GB) fits in VRAM ({vram_gb:.1f}GB), "
                        f"selecting PyTorch"
                    )
                    return BackendType.PYTORCH.value
                else:
                    logger.debug(
                        f"Model ({model_size_gb:.1f}GB) too large for VRAM ({vram_gb:.1f}GB), "
                        f"using llama.cpp for hybrid CPU/GPU"
                    )
                    return BackendType.LLAMA_CPP.value
            else:
                # Don't know model size, default to PyTorch for mid-range GPU
                logger.debug(f"Mid-range GPU ({vram_gb:.1f}GB VRAM), selecting PyTorch")
                return BackendType.PYTORCH.value
        
        # Low VRAM (<6GB) or unknown - use llama.cpp
        logger.debug(f"Low VRAM GPU ({vram_gb:.1f}GB), selecting llama.cpp")
        return BackendType.LLAMA_CPP.value
    
    def _validate_backend(self, backend: str) -> str:
        """
        Validate and normalize backend name.
        
        Args:
            backend: Backend name to validate
            
        Returns:
            Validated backend name
            
        Raises:
            ValueError: If backend is invalid
        """
        backend = backend.lower().strip()
        
        valid_backends = {
            "llama.cpp": BackendType.LLAMA_CPP.value,
            "llamacpp": BackendType.LLAMA_CPP.value,
            "llama": BackendType.LLAMA_CPP.value,
            "pytorch": BackendType.PYTORCH.value,
            "torch": BackendType.PYTORCH.value,
            "vllm": BackendType.VLLM.value,
        }
        
        if backend in valid_backends:
            return valid_backends[backend]
        
        raise ValueError(
            f"Invalid backend: {backend}. "
            f"Valid options: llama.cpp, pytorch, vllm"
        )
    
    def _get_selection_reason(self, backend: str, model_size_gb: Optional[float] = None) -> str:
        """
        Get human-readable reason for backend selection.
        
        Args:
            backend: Selected backend
            model_size_gb: Model size in GB
            
        Returns:
            Explanation string
        """
        if backend == BackendType.VLLM.value:
            return f"High-end GPU with {self.profile.vram_total_gb:.0f}GB VRAM"
        
        elif backend == BackendType.PYTORCH.value:
            if model_size_gb:
                return f"Model fits in {self.profile.vram_total_gb:.0f}GB VRAM"
            else:
                return f"Mid-range GPU with {self.profile.vram_total_gb:.0f}GB VRAM"
        
        elif backend == BackendType.LLAMA_CPP.value:
            if not self.profile.has_gpu:
                return "CPU-only system"
            elif self.profile.vram_total_gb and self.profile.vram_total_gb < 6:
                return f"Low VRAM ({self.profile.vram_total_gb:.1f}GB)"
            elif model_size_gb and self.profile.vram_total_gb:
                return f"Model too large for {self.profile.vram_total_gb:.0f}GB VRAM"
            else:
                return "Best compatibility"
        
        return "Auto-selected"
    
    def get_fallback_chain(self, preferred_backend: str) -> list[str]:
        """
        Get fallback backend chain if preferred backend fails.
        
        Args:
            preferred_backend: Initially preferred backend
            
        Returns:
            List of backends to try in order
            
        Example:
            >>> selector.get_fallback_chain("pytorch")
            ["pytorch", "llama.cpp"]
        """
        # Always end with llama.cpp as it has broadest compatibility
        if preferred_backend == BackendType.VLLM.value:
            # vLLM -> PyTorch -> llama.cpp
            return [BackendType.VLLM.value, BackendType.PYTORCH.value, BackendType.LLAMA_CPP.value]
        
        elif preferred_backend == BackendType.PYTORCH.value:
            # PyTorch -> llama.cpp
            return [BackendType.PYTORCH.value, BackendType.LLAMA_CPP.value]
        
        else:
            # llama.cpp only (most compatible)
            return [BackendType.LLAMA_CPP.value]
    
    def recommend_backend_for_model(self, model_path: Path) -> Dict[str, any]:
        """
        Get detailed backend recommendation for a specific model file.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Dict with:
                - backend: Recommended backend name
                - reason: Explanation
                - alternatives: List of alternative backends
                - warnings: List of warnings (if any)
        """
        warnings = []
        
        # Get model size
        try:
            model_size_gb = model_path.stat().st_size / (1024 ** 3)
        except Exception as e:
            logger.warning(f"Could not get model size: {e}")
            model_size_gb = None
        
        # Select backend
        backend = self.select_backend(model_size_gb=model_size_gb)
        
        # Get alternatives (fallback chain)
        alternatives = self.get_fallback_chain(backend)
        alternatives.remove(backend)  # Remove selected backend from alternatives
        
        # Check for potential issues
        if model_size_gb:
            # Check RAM
            ram_check = check_ram_for_model(model_size_gb)
            if not ram_check["can_load"]:
                warnings.append(
                    f"Insufficient RAM: Model needs {ram_check['required_gb']:.1f}GB, "
                    f"only {ram_check['available_gb']:.1f}GB available"
                )
            elif ram_check["warning"]:
                warnings.append(ram_check["message"])
            
            # Check VRAM if using GPU backend
            if backend != BackendType.LLAMA_CPP.value and self.profile.vram_total_gb:
                required_vram = model_size_gb * 1.2
                if required_vram > self.profile.vram_total_gb:
                    warnings.append(
                        f"Model ({model_size_gb:.1f}GB) may not fit in VRAM "
                        f"({self.profile.vram_total_gb:.1f}GB). "
                        f"Consider using llama.cpp for hybrid GPU/CPU inference."
                    )
        
        return {
            "backend": backend,
            "reason": self._get_selection_reason(backend, model_size_gb),
            "alternatives": alternatives,
            "warnings": warnings,
        }


def select_backend(
    model_size_gb: Optional[float] = None,
    preferred_backend: Optional[str] = None,
    profile: Optional[HardwareProfile] = None,
) -> str:
    """
    Convenience function to select backend without creating a BackendSelector instance.
    
    Args:
        model_size_gb: Model size in GB (optional)
        preferred_backend: Preferred backend ("auto", "llama.cpp", "pytorch", "vllm")
        profile: Optional HardwareProfile (will profile if not provided)
        
    Returns:
        Selected backend name
        
    Example:
        >>> backend = select_backend(model_size_gb=7.5)
        >>> print(backend)
        "pytorch"
    """
    selector = BackendSelector(profile=profile)
    return selector.select_backend(
        model_size_gb=model_size_gb,
        preferred_backend=preferred_backend,
    )


def get_backend_info(backend: str) -> Dict[str, str]:
    """
    Get information about a backend.
    
    Args:
        backend: Backend name
        
    Returns:
        Dict with backend information
    """
    info = {
        BackendType.LLAMA_CPP.value: {
            "name": "llama.cpp",
            "description": "C++ inference engine with broad compatibility",
            "advantages": [
                "Works on all hardware (CPU, low-end GPU, high-end GPU)",
                "Excellent CPU performance with quantization",
                "Hybrid GPU/CPU inference for large models",
                "Low memory overhead",
            ],
            "best_for": "CPU-only systems, low VRAM GPUs (<6GB), very large models",
        },
        BackendType.PYTORCH.value: {
            "name": "PyTorch",
            "description": "Python ML framework with GPU acceleration",
            "advantages": [
                "20-30% faster than llama.cpp on GPU",
                "Good quantization support (FP16, 8-bit, 4-bit)",
                "torch.compile optimization (15-25% speedup)",
                "Native Python integration",
            ],
            "best_for": "Mid-range GPUs (6-16GB VRAM), models that fit in VRAM",
        },
        BackendType.VLLM.value: {
            "name": "vLLM",
            "description": "High-throughput inference engine with continuous batching",
            "advantages": [
                "10-30x throughput improvement with batching",
                "PagedAttention for efficient memory usage",
                "Best for serving multiple requests",
                "Optimized for production workloads",
            ],
            "best_for": "High-end GPUs (16GB+ VRAM), production serving, batch inference",
        },
    }
    
    return info.get(backend, {
        "name": backend,
        "description": "Unknown backend",
        "advantages": [],
        "best_for": "Unknown",
    })
