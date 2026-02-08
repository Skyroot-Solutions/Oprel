"""
Image/Video Model Type Detection

Detects whether a model is a diffusers pipeline or merged checkpoint,
preventing the "black image" bug caused by routing all models through SD/SDXL workflows.
"""

from pathlib import Path
from typing import Literal, Optional
from oprel.utils.logging import get_logger

logger = get_logger(__name__)

ModelFormat = Literal["diffusers", "merged_checkpoint", "gguf", "unknown"]


def detect_image_model_format(model_path: Path) -> ModelFormat:
    """
    Detect image/video model format.
    
    THIS IS CRITICAL: .safetensors is just a container format, NOT a model type!
    
    Args:
        model_path: Path to model directory or file
        
    Returns:
        "diffusers": Diffusers pipeline (model_index.json + subdirs)
        "merged_checkpoint": Single .safetensors file (SD/SDXL merged)
        "gguf": GGUF format (NOT supported for image/video)
        "unknown": Unrecognized format
    """
    if not model_path.exists():
        return "unknown"
    
    # Check if it's a directory (diffusers pipeline)
    if model_path.is_dir():
        # Diffusers pipeline has model_index.json
        if (model_path / "model_index.json").exists():
            logger.info(f"Detected diffusers pipeline: {model_path}")
            return "diffusers"
        
        # Check for nested safetensors (unet/, vae/, text_encoder/)
        has_nested_safetensors = any([
            (model_path / "unet").exists(),
            (model_path / "vae").exists(),
            (model_path / "text_encoder").exists(),
        ])
        
        if has_nested_safetensors:
            logger.info(f"Detected diffusers-style structure: {model_path}")
            return "diffusers"
        
        # Check for merged checkpoint in directory
        safetensors_files = list(model_path.glob("*.safetensors"))
        if safetensors_files:
            # Single .safetensors file in root = merged checkpoint
            logger.info(f"Detected merged checkpoint: {safetensors_files[0].name}")
            return "merged_checkpoint"
    
    # Check if it's a file
    elif model_path.is_file():
        if model_path.suffix == ".safetensors":
            logger.info(f"Detected merged checkpoint file: {model_path.name}")
            return "merged_checkpoint"
        elif model_path.suffix == ".gguf":
            logger.warning(f"GGUF format detected for image model - NOT SUPPORTED")
            return "gguf"
    
    return "unknown"


def is_lora(model_path: Path) -> bool:
    """
    Detect if a model is a LoRA (fine-tuning weights) vs full checkpoint.
    
    LoRA characteristics:
    - Small size: ~100-500MB (vs 4-25GB for full models)
    - Often has "lora" in filename or repo name
    - May have lora-specific metadata
    
    Args:
        model_path: Path to model file or directory
        
    Returns:
        True if LoRA, False if full checkpoint
    """
    # Check filename/path for "lora" keyword
    path_str = str(model_path).lower()
    if "lora" in path_str or "loRA" in str(model_path):
        logger.info(f"Detected LoRA from name: {model_path.name}")
        return True
    
    # Check file size (if it's a file)
    if model_path.is_file() and model_path.suffix == ".safetensors":
        size_mb = model_path.stat().st_size / (1024 * 1024)
        
        # LoRAs are typically < 1GB, full models are > 2GB
        if size_mb < 1000:  # Less than 1GB
            logger.info(f"Detected LoRA from size: {size_mb:.0f}MB")
            return True
        else:
            logger.info(f"Full checkpoint detected: {size_mb:.0f}MB")
            return False
    
    # Check in directory
    if model_path.is_dir():
        safetensors_files = list(model_path.glob("*.safetensors"))
        if safetensors_files:
            return is_lora(safetensors_files[0])
    
    return False



def is_sd_compatible(model_name: str) -> bool:
    """
    Check if model is SD/SDXL compatible.
    
    Only these can use CheckpointLoaderSimple workflow:
    - SD 1.x
    - SD 2.x
    - SDXL
    - SDXL Turbo
    
    Args:
        model_name: Model name or repo ID
        
    Returns:
        True if SD-compatible, False otherwise
    """
    model_lower = model_name.lower()
    
    # SD-compatible models
    sd_keywords = [
        "sd-1", "sd-2", "sd_1", "sd_2",
        "sd15", "sd21",
        "stable-diffusion-v1", "stable-diffusion-v2",
        "sdxl", "sd_xl", "sd-xl",
        "stable-diffusion-xl"
    ]
    
    for keyword in sd_keywords:
        if keyword in model_lower:
            return True
    
    return False


def validate_image_model_format(model_path: Path, model_name: str) -> tuple[bool, Optional[str]]:
    """
    Validate that image model can be loaded.
    
    Args:
        model_path: Path to model
        model_name: Model name/repo ID
        
    Returns:
        (is_valid, error_message)
    """
    format_type = detect_image_model_format(model_path)
    
    # Reject GGUF for image models
    if format_type == "gguf":
        error = (
            f"❌ GGUF format not supported for image generation.\n\n"
            f"Model '{model_name}' is in GGUF format (LLM-only).\n"
            f"Please use .safetensors or diffusers format instead.\n\n"
            f"Supported formats:\n"
            f"  • Diffusers pipeline (model_index.json)\n"
            f"  • Merged checkpoint (.safetensors for SD/SDXL)\n"
        )
        return False, error
    
    # Unknown format
    if format_type == "unknown":
        error = (
            f"❌ Unrecognized model format: {model_name}\n\n"
            f"Expected:\n"
            f"  • Diffusers pipeline (model_index.json + subdirectories)\n"
            f"  • Merged checkpoint (single .safetensors file)\n\n"
            f"Found: {model_path}"
        )
        return False, error
    
    return True, None


def get_comfyui_workflow_type(model_format: ModelFormat, model_name: str) -> str:
    """
    Determine which ComfyUI workflow to use.
    
    CRITICAL: Don't force everything through CheckpointLoaderSimple!
    
    Args:
        model_format: Model format from detect_image_model_format()
        model_name: Model name
        
    Returns:
        "sd_checkpoint": Use CheckpointLoaderSimple (SD/SDXL only)
        "diffusers": Use DiffusersLoader (FLUX, Sana, WAN, etc.)
        "unsupported": Cannot load
    """
    if model_format == "gguf":
        return "unsupported"
    
    if model_format == "diffusers":
        # Diffusers pipelines need DiffusersLoader
        return "diffusers"
    
    if model_format == "merged_checkpoint":
        # Only SD/SDXL merged checkpoints can use CheckpointLoaderSimple
        if is_sd_compatible(model_name):
            return "sd_checkpoint"
        else:
            # Non-SD merged checkpoint (rare, but possible)
            logger.warning(
                f"Merged checkpoint '{model_name}' is not SD-compatible. "
                f"May produce black images if forced through SD workflow."
            )
            return "sd_checkpoint"  # Try anyway, but warn
    
    return "unsupported"
