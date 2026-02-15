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


class BackendSelector:
    """
    Backend selection logic.
    
    Currently forced to llama.cpp for GGUF models as per user configuration.
    """
    
    def __init__(self, profile: Optional[HardwareProfile] = None):
        self.profile = profile or profile_hardware()
        logger.debug(f"Initialized BackendSelector with profile: {self.profile.gpu_type}")
    
    def select_backend(
        self,
        model_size_gb: Optional[float] = None,
        preferred_backend: Optional[str] = None,
    ) -> str:
        """Always returns llama.cpp"""
        return BackendType.LLAMA_CPP.value
    
    def _auto_select_backend(self, model_size_gb: Optional[float] = None) -> str:
        return BackendType.LLAMA_CPP.value
    
    def _validate_backend(self, backend: str) -> str:
        return BackendType.LLAMA_CPP.value
    
    def _get_selection_reason(self, backend: str, model_size_gb: Optional[float] = None) -> str:
        return "Forced llama.cpp for stability"
    
    def get_fallback_chain(self, preferred_backend: str) -> list[str]:
        return [BackendType.LLAMA_CPP.value]
    
    def recommend_backend_for_model(self, model_path: Path) -> Dict[str, any]:
        return {
            "backend": BackendType.LLAMA_CPP.value,
            "reason": "Forced llama.cpp",
            "alternatives": [],
            "warnings": [],
        }


def select_backend(model_path, vram_gb, model_format='gguf'):
    """Always use llama.cpp for GGUF. It's stable and proven."""
    return 'llama.cpp'


def get_backend_info(backend: str) -> Dict[str, str]:
    """
    Get information about a backend.
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
    }
    
    return info.get(backend, {
        "name": backend,
        "description": "Unknown backend",
        "advantages": [],
        "best_for": "Unknown",
    })
