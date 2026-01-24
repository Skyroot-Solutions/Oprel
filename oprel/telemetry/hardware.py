"""
Hardware detection and telemetry
"""

import platform
from typing import Dict, Any, Optional
import psutil

from oprel.utils.logging import get_logger

logger = get_logger(__name__)


def get_hardware_info() -> Dict[str, Any]:
    """
    Detect system hardware capabilities.

    Returns:
        Dictionary with CPU, RAM, GPU, and OS information
    """
    info = {
        "os": platform.system(),
        "arch": platform.machine(),
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
    }

    # Detect GPU
    gpu_info = detect_gpu()
    if gpu_info:
        info.update(gpu_info)

    return info


def detect_gpu() -> Optional[Dict[str, Any]]:
    """
    Detect available GPU and VRAM.

    Returns:
        GPU info dict or None if no GPU detected
    """
    # Try CUDA (NVIDIA)
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.get_device_properties(0)
            return {
                "gpu_type": "cuda",
                "gpu_name": device.name,
                "vram_total_gb": round(device.total_memory / (1024**3), 2),
            }
    except ImportError:
        pass

    # Try Metal (Apple Silicon)
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import torch

            if torch.backends.mps.is_available():
                # On Apple Silicon, unified memory
                total_ram = psutil.virtual_memory().total
                return {
                    "gpu_type": "metal",
                    "gpu_name": "Apple Silicon (Metal)",
                    "vram_total_gb": round(total_ram / (1024**3), 2),  # Unified memory
                }
        except (ImportError, AttributeError):
            pass

    # No GPU detected
    logger.info("No GPU detected, will use CPU inference")
    return None


def get_recommended_threads() -> int:
    """
    Recommend optimal thread count for CPU inference.

    Returns:
        Number of threads to use
    """
    physical_cores = psutil.cpu_count(logical=False)

    # Use physical cores, not hyperthreads
    # Reserve 1-2 cores for system
    if physical_cores > 4:
        return physical_cores - 2
    elif physical_cores > 2:
        return physical_cores - 1
    else:
        return physical_cores
