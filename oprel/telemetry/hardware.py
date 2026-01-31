"""
Hardware detection and telemetry
"""

import platform
import subprocess
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


def _detect_nvidia_smi() -> Optional[Dict[str, Any]]:
    """
    Detect NVIDIA GPU using nvidia-smi (works without torch).
    
    Returns:
        GPU info dict or None if not available
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
        )
        
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().split('\n')[0]
            parts = line.split(',')
            if len(parts) >= 2:
                name = parts[0].strip()
                memory_mb = float(parts[1].strip())
                memory_gb = memory_mb / 1024
                
                logger.info(f"Detected NVIDIA GPU via nvidia-smi: {name} ({memory_gb:.1f} GB)")
                
                return {
                    "gpu_type": "cuda",
                    "gpu_name": name,
                    "vram_total_gb": round(memory_gb, 2),
                }
    except FileNotFoundError:
        # nvidia-smi not found
        pass
    except subprocess.TimeoutExpired:
        logger.debug("nvidia-smi timed out")
    except Exception as e:
        logger.debug(f"nvidia-smi detection failed: {e}")
    
    return None


def detect_gpu() -> Optional[Dict[str, Any]]:
    """
    Detect available GPU and VRAM.
    Tries multiple methods: torch, nvidia-smi, Metal.

    Returns:
        GPU info dict or None if no GPU detected
    """
    # Method 1: Try PyTorch CUDA (if installed)
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.get_device_properties(0)
            logger.info(f"Detected NVIDIA GPU via PyTorch: {device.name}")
            return {
                "gpu_type": "cuda",
                "gpu_name": device.name,
                "vram_total_gb": round(device.total_memory / (1024**3), 2),
            }
    except ImportError:
        logger.debug("PyTorch not installed, trying nvidia-smi")

    # Method 2: Try nvidia-smi (Windows/Linux without torch)
    nvidia_gpu = _detect_nvidia_smi()
    if nvidia_gpu:
        return nvidia_gpu

    # Method 3: Try Metal (Apple Silicon)
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import torch

            if torch.backends.mps.is_available():
                # On Apple Silicon, unified memory
                total_ram = psutil.virtual_memory().total
                logger.info("Detected Apple Silicon GPU (Metal)")
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


def calculate_gpu_layers(vram_gb: float, model_size_gb: float, reserve_vram_gb: float = 0.5) -> int:
    """
    Calculate optimal number of GPU layers based on available VRAM.
    
    Uses percentage-based approach for better accuracy across different model sizes.
    Target: Use up to 90-95% of VRAM for maximum GPU acceleration.
    
    Args:
        vram_gb: Total GPU VRAM in GB
        model_size_gb: Model file size in GB
        reserve_vram_gb: VRAM to reserve for KV cache and CUDA overhead
        
    Returns:
        Recommended number of GPU layers
    """
    # Reserve VRAM for KV cache and CUDA context
    available_vram = vram_gb - reserve_vram_gb
    
    if available_vram <= 0 or model_size_gb <= 0:
        return 0
    
    # Estimate total layers based on model size
    # 7B models typically have 32-39 layers, 13B has ~40, 70B has ~80
    if model_size_gb < 2:
        estimated_total_layers = 24  # Small models (1-3B)
    elif model_size_gb < 5:
        estimated_total_layers = 35  # 7B Q4/Q5 quantized
    elif model_size_gb < 8:
        estimated_total_layers = 39  # 7B higher quant or small 13B
    elif model_size_gb < 15:
        estimated_total_layers = 45  # 13B models
    else:
        estimated_total_layers = 80  # Large models (70B+)
    
    # Calculate how much of the model fits in VRAM
    # Model weights take ~80-90% of file size when loaded (rest is metadata)
    effective_model_size = model_size_gb * 0.85
    
    # Calculate percentage of model that fits
    vram_ratio = available_vram / effective_model_size
    
    # Convert to layers (with 95% target utilization)
    gpu_layers = int(estimated_total_layers * min(vram_ratio * 0.95, 1.0))
    
    # Ensure reasonable bounds
    gpu_layers = max(0, min(gpu_layers, estimated_total_layers))
    
    logger.debug(
        f"GPU layer calc: VRAM={vram_gb:.1f}GB, model={model_size_gb:.1f}GB, "
        f"available={available_vram:.1f}GB, ratio={vram_ratio:.2f}, layers={gpu_layers}/{estimated_total_layers}"
    )
    
    return gpu_layers
