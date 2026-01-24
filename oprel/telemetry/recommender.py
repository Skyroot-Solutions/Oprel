"""
Automatic quantization and configuration recommendations
"""

from oprel.telemetry.hardware import get_hardware_info
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


# Approximate VRAM requirements (GB) for 7B parameter models
QUANTIZATION_MEMORY = {
    "Q2_K": 3.0,      # 2-bit (lowest quality, smallest size)
    "Q3_K_S": 3.5,    # 3-bit small
    "Q3_K_M": 4.0,    # 3-bit medium
    "Q4_K_S": 4.5,    # 4-bit small
    "Q4_K_M": 5.0,    # 4-bit medium (recommended default)
    "Q5_K_S": 5.5,    # 5-bit small
    "Q5_K_M": 6.0,    # 5-bit medium
    "Q6_K": 7.0,      # 6-bit
    "Q8_0": 8.5,      # 8-bit (highest quality, largest size)
}


def recommend_quantization(model_size_b: int = 7) -> str:
    """
    Recommend optimal quantization based on available memory.
    
    Args:
        model_size_b: Model size in billions of parameters (default: 7B)
        
    Returns:
        Recommended quantization string (e.g., "Q4_K_M")
    """
    hw = get_hardware_info()
    
    # Determine available memory
    if "vram_total_gb" in hw:
        available_gb = hw["vram_total_gb"]
        memory_type = hw["gpu_type"]
    else:
        # No GPU, use RAM (but be conservative)
        available_gb = hw["ram_available_gb"] * 0.5  # Only use 50% of available RAM
        memory_type = "cpu"
    
    logger.info(f"Available memory: {available_gb:.1f}GB ({memory_type})")
    
    # Scale memory requirements by model size (assuming 7B baseline)
    scale_factor = model_size_b / 7.0
    
    # Find best quantization that fits in memory
    # Add 20% safety margin
    available_with_margin = available_gb / 1.2
    
    for quant in reversed(list(QUANTIZATION_MEMORY.keys())):
        required = QUANTIZATION_MEMORY[quant] * scale_factor
        
        if required <= available_with_margin:
            logger.info(
                f"Recommended quantization: {quant} "
                f"(requires {required:.1f}GB, available {available_gb:.1f}GB)"
            )
            return quant
    
    # Fallback to smallest quantization
    logger.warning(
        f"Limited memory ({available_gb:.1f}GB). "
        f"Using smallest quantization Q2_K, but quality may be poor."
    )
    return "Q2_K"


def recommend_n_gpu_layers(model_size_b: int = 7) -> int:
    """
    Recommend how many layers to offload to GPU.
    
    Args:
        model_size_b: Model size in billions of parameters
        
    Returns:
        Number of layers to offload (-1 for all)
    """
    hw = get_hardware_info()
    
    if "vram_total_gb" not in hw:
        return 0  # No GPU, use CPU only
    
    vram_gb = hw["vram_total_gb"]
    
    # Rough estimate: 7B model = 32 layers
    # Each layer ~= 200MB for Q4_K_M
    total_layers = int(model_size_b * 4.5)  # 7B ~= 32 layers
    layer_size_gb = 0.2 * (model_size_b / 7.0)
    
    # Reserve 1GB for activations and overhead
    available_for_layers = max(0, vram_gb - 1.0)
    max_layers = int(available_for_layers / layer_size_gb)
    
    if max_layers >= total_layers:
        logger.info(f"Offloading all {total_layers} layers to GPU")
        return -1  # All layers
    else:
        logger.info(f"Offloading {max_layers}/{total_layers} layers to GPU")
        return max_layers