"""
Automatic quantization and configuration recommendations
"""

from oprel.telemetry.hardware import get_hardware_info
from oprel.core.exceptions import MemoryError as OprelMemoryError
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================
# ACTUAL GGUF memory requirements (7B models, measured)
# ============================================================

QUANTIZATION_MEMORY = {
    "Q2_K": 2.5,      # Lightweight, acceptable for 7B
    "Q3_K_S": 3.0,    # Decent quality
    "Q3_K_M": 3.3,    # Good quality
    "Q4_K_S": 3.8,    # Industry standard (lower bound)
    "Q4_K_M": 4.1,    # ✅ RECOMMENDED DEFAULT (measured: 3.8-4.2GB)
    "Q5_K_S": 4.5,    # High quality
    "Q5_K_M": 4.9,    # Very high quality
    "Q6_K": 5.5,      # Near-lossless
    "Q8_0": 7.2,      # Lossless
}

# Quality-first order (best to acceptable)
PREFERRED_ORDER = [
    "Q5_K_M",  # Best quality that fits most GPUs
    "Q5_K_S",
    "Q4_K_M",  # ✅ Sweet spot (quality + efficiency)
    "Q4_K_S",
]

# Only if user explicitly allows or device is constrained
FALLBACK_LOW = ["Q3_K_M", "Q3_K_S", "Q2_K"]

MIN_REASONABLE_QUANT = "Q4_K_M"


def recommend_quantization(
    model_size_b: int = 7,
    allow_low_quality: bool = False,
) -> str:
    """
    Recommend quantization based on available memory.
    
    Strategy:
    1. Try GPU VRAM first (if available)
    2. Fall back to system RAM
    3. Prefer quality (Q5/Q4), use Q3/Q2 only if needed
    """
    hw = get_hardware_info()

    # ============================================================
    # Determine available memory
    # ============================================================
    
    vram_gb = hw.get("vram_total_gb", 0)
    
    # ✅ FIX: Use total RAM, not available * 0.5
    # Ollama uses total RAM - we should too
    ram_total_gb = hw.get("ram_total_gb", 0)
    
    # Use 70% of total RAM (leave 30% for OS/other apps)
    # This matches Ollama's behavior
    ram_usable = ram_total_gb * 0.7
    
    # Primary memory source
    if vram_gb > 0:
        primary_memory = vram_gb
        primary_type = hw.get("gpu_type", "cuda")
        fallback_memory = ram_usable
        fallback_type = "cpu"
        logger.info(f"GPU VRAM: {vram_gb:.1f}GB ({primary_type})")
        logger.info(f"System RAM: {ram_total_gb:.1f}GB (usable: {ram_usable:.1f}GB)")
    else:
        primary_memory = ram_usable
        primary_type = "cpu"
        fallback_memory = 0
        fallback_type = None
        logger.info(f"System RAM: {ram_total_gb:.1f}GB (usable: {ram_usable:.1f}GB)")

    # ============================================================
    # Scale requirements by model size
    # ============================================================
    
    scale_factor = model_size_b / 7.0

    # ============================================================
    # Try quality-first on primary memory
    # ============================================================
    
    # 20% safety margin (vs Ollama's more aggressive approach)
    available_with_margin = primary_memory * 0.8
    
    for quant in PREFERRED_ORDER:
        required = QUANTIZATION_MEMORY[quant] * scale_factor
        
        if required <= available_with_margin:
            logger.info(
                f"Selected {quant} quantization "
                f"(needs {required:.1f}GB, have {primary_memory:.1f}GB {primary_type})"
            )
            return quant
    
    # ============================================================
    # Try fallback memory (GPU -> RAM)
    # ============================================================
    
    if fallback_memory > 0:
        logger.warning(
            f"GPU VRAM ({vram_gb:.1f}GB) too small for Q4/Q5, "
            f"trying CPU with system RAM ({ram_usable:.1f}GB)"
        )
        
        fallback_with_margin = fallback_memory * 0.8
        
        for quant in PREFERRED_ORDER:
            required = QUANTIZATION_MEMORY[quant] * scale_factor
            
            if required <= fallback_with_margin:
                logger.info(
                    f"Using CPU mode with {quant} "
                    f"(needs {required:.1f}GB, have {ram_usable:.1f}GB RAM)"
                )
                return quant
    
    # ============================================================
    # If still no fit, check if we can use lower quality
    # ============================================================
    
    total_memory = primary_memory + (fallback_memory if fallback_memory else 0)
    
    if not allow_low_quality:
        # ✅ Better error message
        raise OprelMemoryError(
            f"Not enough memory for {model_size_b}B model at Q4_K_M quality.\n"
            f"Available: {primary_memory:.1f}GB {primary_type}"
            + (f" + {fallback_memory:.1f}GB RAM" if fallback_memory else "") + 
            f"\nRequired: {QUANTIZATION_MEMORY['Q4_K_M'] * scale_factor:.1f}GB minimum\n\n"
            f"Solutions:\n"
            f"  1. Use a smaller model:\n"
            f"     oprel run phi3-mini      (2GB model)\n"
            f"     oprel run tinyllama      (1GB model)\n"
            f"  2. Allow lower quality (faster but worse):\n"
            f"     oprel run {model_size_b}b --allow-low-quality\n"
            f"  3. Get more RAM/VRAM\n"
        )
    
    # ============================================================
    # Low quality fallback (Q3/Q2)
    # ============================================================
    
    logger.warning(
        f"⚠️  Using low-quality quantization (Q3/Q2).\n"
        f"   Quality will be degraded. Consider using a smaller model."
    )
    
    # Try Q3/Q2 on primary memory
    for quant in FALLBACK_LOW:
        required = QUANTIZATION_MEMORY[quant] * scale_factor
        
        if required <= available_with_margin:
            logger.warning(f"Using {quant} (needs {required:.1f}GB)")
            return quant
    
    # Try Q3/Q2 on fallback
    if fallback_memory > 0:
        for quant in FALLBACK_LOW:
            required = QUANTIZATION_MEMORY[quant] * scale_factor
            
            if required <= fallback_with_margin:
                logger.warning(f"Using CPU mode with {quant} (needs {required:.1f}GB)")
                return quant
    
    # ============================================================
    # Last resort: Q2_K (will work but poor quality)
    # ============================================================
    
    logger.critical(
        f"⚠️  EXTREME LOW MEMORY MODE ⚠️\n"
        f"   Using Q2_K as absolute last resort.\n"
        f"   Strongly recommend using TinyLlama or Phi-3-mini instead."
    )
    
    return "Q2_K"


def recommend_n_gpu_layers(model_size_b: int = 7) -> int:
    """
    Recommend GPU layer offloading.
    
    Conservative approach: leave headroom for KV cache and activations.
    """
    hw = get_hardware_info()
    
    vram_gb = hw.get("vram_total_gb", 0)
    
    if vram_gb == 0:
        logger.info("No GPU detected, using CPU-only mode")
        return 0
    
    # Estimate transformer layers (rough heuristic)
    # 7B ≈ 32 layers, 13B ≈ 40 layers, 70B ≈ 80 layers
    if model_size_b <= 3:
        total_layers = 28
    elif model_size_b <= 7:
        total_layers = 32
    elif model_size_b <= 13:
        total_layers = 40
    else:
        total_layers = int(model_size_b * 1.2)
    
    # Per-layer VRAM (conservative estimate)
    layer_size_gb = 0.15 * (model_size_b / 7.0)
    
    # Reserve VRAM for overhead (KV cache, CUDA kernels, activations)
    # GTX 1650 4GB: reserve 1GB → 3GB usable
    if vram_gb <= 4:
        reserved = 1.0
    elif vram_gb <= 8:
        reserved = 1.5
    else:
        reserved = 2.0
    
    usable_vram = max(0, vram_gb - reserved)
    
    # Calculate max layers that fit
    max_layers = int(usable_vram / layer_size_gb)
    
    # Safety caps by VRAM tier
    if vram_gb <= 4:
        safe_cap = 16  # GTX 1650: ~half the model
    elif vram_gb <= 6:
        safe_cap = 24
    elif vram_gb <= 8:
        safe_cap = 32
    else:
        safe_cap = total_layers  # Full offload on big GPUs
    
    layers_to_offload = min(max_layers, safe_cap, total_layers)
    
    if layers_to_offload <= 0:
        logger.warning(
            f"GPU VRAM ({vram_gb:.1f}GB) too small for offloading, using CPU only"
        )
        return 0
    
    if layers_to_offload >= total_layers:
        logger.info(f"Offloading all {total_layers} layers to GPU")
        return -1  # -1 means "all layers"
    
    percentage = (layers_to_offload / total_layers) * 100
    logger.info(
        f"Offloading {layers_to_offload}/{total_layers} layers to GPU "
        f"({percentage:.0f}% on GPU, {100-percentage:.0f}% on CPU)"
    )
    
    return layers_to_offload