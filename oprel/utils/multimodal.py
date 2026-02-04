"""
Multimodal utilities for vision, image generation, and video generation.
Handles image/video I/O and format conversions for different model types.
"""

from pathlib import Path
from typing import Optional, List, Union
import base64
from io import BytesIO

from oprel.utils.logging import get_logger

logger = get_logger(__name__)


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image file to base64 string for vision model input.
    Used by VLM models (qwen-vl, llava, moondream, etc.)
    
    Args:
        image_path: Path to image file (jpg, png, webp, etc.)
        
    Returns:
        Base64-encoded string suitable for API requests
    """
    image_file = Path(image_path)
    
    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Read and encode image
    with open(image_file, 'rb') as f:
        image_data = f.read()
    
    encoded = base64.b64encode(image_data).decode('utf-8')
    
    # Detect MIME type from extension
    ext = image_file.suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp',
        '.gif': 'image/gif',
    }
    mime = mime_types.get(ext, 'image/jpeg')
    
    # Return data URL format
    return f"data:{mime};base64,{encoded}"


def save_generated_image(
    image_data: bytes,
    output_path: str,
    format: str = "png"
) -> str:
    """
    Save generated image bytes to file.
    Used by text-to-image models (flux, sana, sdxl, etc.)
    
    Args:
        image_data: Raw image bytes
        output_path: Where to save the image
        format: Image format (png, jpg, webp)
        
    Returns:
        Absolute path to saved image
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure correct extension
    if not output_file.suffix:
        output_file = output_file.with_suffix(f".{format}")
    
    # Save image
    with open(output_file, 'wb') as f:
        f.write(image_data)
    
    logger.info(f"Image saved to: {output_file}")
    return str(output_file.absolute())


def save_generated_video(
    video_data: bytes,
    output_path: str,
    format: str = "mp4"
) -> str:
    """
    Save generated video bytes to file.
    Used by text-to-video models (wan, mochi, cogvideox, etc.)
    
    Args:
        video_data: Raw video bytes
        output_path: Where to save the video
        format: Video format (mp4, webm, gif)
        
    Returns:
        Absolute path to saved video
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure correct extension
    if not output_file.suffix:
        output_file = output_file.with_suffix(f".{format}")
    
    # Save video
    with open(output_file, 'wb') as f:
        f.write(video_data)
    
    logger.info(f"Video saved to: {output_file}")
    return str(output_file.absolute())


def format_vision_prompt(
    text_prompt: str,
    image_paths: List[str],
    model_type: str = "auto"
) -> Union[str, dict]:
    """
    Format prompt for vision models with image inputs.
    Different VLM models expect different formats.
    
    Args:
        text_prompt: Text question/instruction
        image_paths: List of image file paths
        model_type: "qwen-vl", "llava", "moondream", or "auto"
        
    Returns:
        Formatted prompt (string or dict depending on model)
    """
    # Detect model type from image count if auto
    if model_type == "auto":
        if len(image_paths) > 1:
            model_type = "qwen-vl"  # Supports multi-image
        else:
            model_type = "llava"  # Standard VLM
    
    # Encode images
    encoded_images = [encode_image_to_base64(img) for img in image_paths]
    
    # Format based on model type
    if model_type == "qwen-vl":
        # Qwen-VL format: interleaved text and images
        messages = []
        for i, img in enumerate(encoded_images):
            messages.append({
                "type": "image",
                "image": img
            })
        messages.append({
            "type": "text",
            "text": text_prompt
        })
        return {"messages": messages}
    
    elif model_type == "llava":
        # LLaVA format: image + text
        return {
            "prompt": text_prompt,
            "images": encoded_images
        }
    
    elif model_type == "moondream":
        # Moondream simple format
        return {
            "image": encoded_images[0],
            "question": text_prompt
        }
    
    else:
        # Generic format
        return {
            "prompt": text_prompt,
            "images": encoded_images
        }


def get_model_category_from_alias(alias: str) -> Optional[str]:
    """
    Determine model category from alias name.
    Helps CLI auto-detect what kind of input/output to expect.
    
    Args:
        alias: Model alias (e.g., "qwen3-vl-7b", "flux-1-dev")
        
    Returns:
        Category: "vision", "text-to-image", "text-to-video", "text-generation", etc.
    """
    from oprel.downloader.aliases import get_model_category
    
    category = get_model_category(alias)
    if category:
        return category
    
    # Fallback: detect from alias name
    alias_lower = alias.lower()
    
    if any(x in alias_lower for x in ['vl', 'vision', 'llava', 'minicpm-v', 'moondream']):
        return "vision"
    elif any(x in alias_lower for x in ['flux', 'sana', 'sdxl', 'stable-diff', 'pixart']):
        return "text-to-image"
    elif any(x in alias_lower for x in ['wan', 'mochi', 'video', 'cogvideo', 'animatediff']):
        return "text-to-video"
    elif 'embed' in alias_lower:
        return "embeddings"
    else:
        return "text-generation"


def format_multimodal_request(
    prompt: str,
    model_alias: str,
    image_paths: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    **kwargs
) -> dict:
    """
    Auto-format request based on model category.
    Simplifies CLI usage by detecting model type and formatting accordingly.
    
    Args:
        prompt: Text prompt/question
        model_alias: Model alias to use
        image_paths: Input images (for vision models)
        output_path: Where to save generated media
        **kwargs: Additional model-specific parameters
        
    Returns:
        Formatted request dict for API
    """
    category = get_model_category_from_alias(model_alias)
    
    request = {
        "prompt": prompt,
        **kwargs
    }
    
    # Vision models: add image inputs
    if category == "vision":
        if not image_paths:
            raise ValueError("Vision models require --image argument")
        
        # Format with images
        formatted = format_vision_prompt(prompt, image_paths)
        request.update(formatted)
    
    # Image/video generation: add output format
    elif category in ["text-to-image", "text-to-video"]:
        if output_path:
            request["output_path"] = output_path
        else:
            # Auto-generate output filename
            import time
            timestamp = int(time.time())
            ext = "mp4" if category == "text-to-video" else "png"
            request["output_path"] = f"generated_{timestamp}.{ext}"
    
    # Text generation: standard format
    else:
        pass  # Already correct format
    
    return request
