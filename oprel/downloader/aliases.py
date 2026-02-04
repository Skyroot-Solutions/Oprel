"""
Model aliases for user-friendly names to official GGUF models.

Oprel Studio Model Registry (2026 Production)
Source: Official & Verified Community GGUFs (Unsloth, Bartowski, Calcuis)
"""

from typing import Dict, List, Optional
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


OFFICIAL_REPOS = {
       "text-generation": {
        # --- QWEN 3 FAMILY (SOTA 2026) ---
        "qwen3-32b": "Qwen/Qwen3-32B-Instruct-GGUF",       # Flagship Dense
        "qwen3-14b": "Qwen/Qwen3-14B-Instruct-GGUF",       # Best All-Rounder
        "qwen3-8b": "Qwen/Qwen3-8B-Instruct-GGUF",         # Consumer GPU King
        "qwen3-4b": "Qwen/Qwen3-4B-Instruct-GGUF",         #
        "qwen3-3b": "Qwen/Qwen3-3B-Instruct-GGUF",         # Optimized for Mobile
        "qwen3-1.7b": "Qwen/Qwen3-1.7B-Instruct-GGUF",     #
        "qwen3-0.6b": "Qwen/Qwen3-0.6B-Instruct-GGUF",     # IoT / Embedded
        "qwen3-vl-7b": "Qwen/Qwen3-VL-7B-Instruct-GGUF",   # Added Alternative for missing 4B

        # --- QWEN 2.5 FAMILY (Reliable Legacy) ---
        "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",

        # --- GEMMA 3 FAMILY (Google) ---
        "gemma3-27b": "google/gemma-3-27b-it-GGUF",        #
        "gemma3-12b": "google/gemma-3-12b-it-GGUF",        #
        "gemma3-4b": "google/gemma-3-4b-it-GGUF",          #
        "gemma3-1b": "google/gemma-3-1b-it-GGUF",          #

        # --- GEMMA 2 FAMILY ---
        "gemma2-27b": "google/gemma-2-27b-it-GGUF",
        "gemma2-9b": "google/gemma-2-9b-it-GGUF",
        "gemma2-2b": "google/gemma-2-2b-it-GGUF",

        # --- LLAMA 3.x FAMILY ---
        "llama3.3-70b": "unsloth/Llama-3.3-70B-Instruct-GGUF", # Replaced phantom 8B with real 70B
        "llama3.3-8b": "unsloth/Llama-3.3-8B-Instruct-GGUF",
        "llama3.1-8b": "unsloth/Meta-Llama-3.1-8B-Instruct-GGUF",
        "llama3.2-3b": "unsloth/Llama-3.2-3B-Instruct-GGUF",
        "llama3.2-1b": "unsloth/Llama-3.2-1B-Instruct-GGUF",
        
        # --- MISTRAL & OTHERS ---
        "mistral-small-3": "bartowski/Mistral-Small-24B-Instruct-2501-GGUF",
        "mistral-nemo": "bartowski/Mistral-Nemo-Instruct-2407-GGUF",
        "gpt-oss-20b": "unsloth/gpt-oss-20b-GGUF",
    },

    "coding": {
        # --- QWEN 3 CODER (2026 SOTA) ---
        "qwen3-coder-32b": "Qwen/Qwen3-Coder-32B-Instruct-GGUF",
        "qwen3-coder-14b": "Qwen/Qwen3-Coder-14B-Instruct-GGUF",
        "qwen3-coder-8b": "Qwen/Qwen3-Coder-8B-Instruct-GGUF",
        "qwen3-coder-4b": "Qwen/Qwen3-Coder-4B-Instruct-GGUF",
        "qwen3-coder-1.7b": "Qwen/Qwen3-Coder-1.7B-Instruct-GGUF",

        # --- QWEN 2.5 CODER ---
        "qwen2.5-coder-14b": "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
        "qwen2.5-coder-7b": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "qwen2.5-coder-3b": "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
        "qwen2.5-coder-1.5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",

        # --- SPECIALISTS ---
        "phi-4-14b": "microsoft/Phi-4-GGUF",
        "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct-GGUF",
        "deepseek-coder-v2-16b": "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
        "hermes-3-8b": "NousResearch/Hermes-3-Llama-3.1-8B-GGUF",
    },

    "reasoning": {
        "deepseek-r1-14b": "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        "deepseek-r1-8b": "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF",
        "deepseek-r1-7b": "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        "deepseek-r1-1.5b": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
        "qwen3-reasoning-7b": "Qwen/Qwen3-7B-Reasoning-GGUF",
    },


    "text-to-video": {
        # --- WAN FAMILY (Best in Class) ---
        "wan2.1-14b": "calcuis/wan2-gguf",               # Verified GGUF
        "wan2.1-1.3b": "calcuis/wan-1.3b-gguf",          # Verified GGUF
        "wan2.2-5b": "Wan-AI/Wan2.2-TI2V-5B",            # Official Space

        # --- OTHER VIDEO ARCHITECTURES ---
        "mochi-1-10b": "Genmo/mochi-1-preview-GGUF",     #
        "ltx-video-2b": "Lightricks/LTX-Video-2B-v0.9-GGUF",
        "cogvideox-5b": "THUDM/CogVideoX-5b-GGUF",
        "hunyuan-video-distill": "calcuis/hunyuan-video-gguf",
        "animatediff-lightning": "calcuis/animatediff-lightning-gguf",
        "svd-xt-1.1": "calcuis/svd-xt-1.1-gguf",
        "latte-1b": "calcuis/latte-1b-gguf",
    },

    "text-to-image": {
        "flux-1-dev": "city96/FLUX.1-dev-gguf",          # Corrected from gguf-org
        "flux-1-schnell": "City96/FLUX.1-schnell-gguf",  # Verified
        "sana-1.6b": "city96/Sana-1600M-1024px-GGUF",    # Alternative to Base
        "sana-600m": "Efficient-Large-Model/Sana_600M_1024px", # Base model
        "sdxl-turbo": "vitink/sdxl-turbo-gguf",
        "lumina-next": "Alpha-VLLM/Lumina-Next-SFT-GGUF",
        "stable-cascade": "City96/Stable-Cascade-GGUF",
        "playground-v2.5": "City96/Playground-v2.5-1024px-aesthetic-GGUF",
        "auraflow-0.2": "fal/AuraFlow-v0.2-GGUF",
        "pixart-sigma": "City96/PixArt-Sigma-xl-2-1024-MS-GGUF",
    },

    "vision": {
        # --- QWEN VL (Best OCR/Vision) ---
        "qwen3-vl-32b": "Qwen/Qwen3-VL-32B-Instruct-GGUF",
        "qwen3-vl-14b": "Qwen/Qwen3-VL-14B-Instruct-GGUF",
        "qwen3-vl-7b": "Qwen/Qwen3-VL-7B-Instruct-GGUF",
        "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct-GGUF",
        "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct-GGUF",

        # --- OTHERS ---
        "llama3.2-vision-11b": "unsloth/Llama-3.2-11B-Vision-Instruct-GGUF",
        "minicpm-v-2.6": "openbmb/MiniCPM-V-2.6-gguf",     # Best Mobile VLM
        "moondream2": "vikhyatk/moondream2-GGUF",          # Tiny (1.8B)
        "yi-vl-6b": "mmnga/Yi-VL-6B-gguf",
        "llava-v1.6-vicuna-7b": "cjpais/llava-v1.6-vicuna-7b-gguf",
        "internvl2-8b": "OpenGVLab/InternVL2-8B-GGUF",
    },

    "embeddings": {
        "nomic-embed-text": "nomic-ai/nomic-embed-text-v1.5-GGUF",
        "bge-m3": "audo/bge-m3-GGUF",
        "snowflake-arctic": "ChristianAzinn/snowflake-arctic-embed-m-gguf",
        "all-minilm-l6-v2": "faldor/all-MiniLM-L6-v2-gguf",
        "e5-small": "intfloat/e5-small-v2-gguf",
    }
}

# Flatten all models into a single dict for backward compatibility
MODEL_ALIASES: Dict[str, str] = {}
for category, models in OFFICIAL_REPOS.items():
    MODEL_ALIASES.update(models)

# Category metadata with descriptions and icons
CATEGORY_INFO = {
    "text-generation": {
        "name": "Text Generation",
        "icon": "📝",
        "description": "General-purpose chat and instruction-following models",
    },
    "coding": {
        "name": "Coding",
        "icon": "👨‍💻",
        "description": "Code generation and software development models",
    },
    "reasoning": {
        "name": "Reasoning",
        "icon": "🧠",
        "description": "Advanced reasoning and chain-of-thought models",
    },
    "text-to-video": {
        "name": "Text-to-Video",
        "icon": "🎥",
        "description": "Video generation from text prompts",
    },
    "text-to-image": {
        "name": "Text-to-Image",
        "icon": "🎨",
        "description": "Image generation from text prompts",
    },
    "vision": {
        "name": "Vision/Multimodal",
        "icon": "👁️",
        "description": "Vision-language models for image understanding",
    },
    "embeddings": {
        "name": "Embeddings",
        "icon": "📚",
        "description": "Text embeddings for RAG and semantic search",
    },
}


def resolve_model_id(model_id: str) -> str:
    """
    Resolve user-friendly names to official GGUF model IDs.
    
    Args:
        model_id: User input (e.g., "llama3", "qwencoder")
        
    Returns:
        HuggingFace GGUF model ID
    """
    # Direct match
    if model_id in MODEL_ALIASES:
        resolved = MODEL_ALIASES[model_id]
        logger.info(f"Resolved '{model_id}' -> '{resolved}'")
        return resolved
    
    # Already a GGUF path
    if "/" in model_id and "gguf" in model_id.lower():
        return model_id
    
    # Fuzzy match (case-insensitive)
    model_lower = model_id.lower()
    for alias, gguf_id in MODEL_ALIASES.items():
        if model_lower == alias.lower():
            logger.info(f"Resolved '{model_id}' -> '{gguf_id}'")
            return gguf_id
    
    # No match - return original
    return model_id


def list_available_aliases() -> Dict[str, str]:
    """Get all available model aliases."""
    return MODEL_ALIASES.copy()


def list_models_by_category(category: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """
    Get models organized by category.
    
    Args:
        category: Optional category filter (e.g., "coding", "vision")
        
    Returns:
        Dictionary of {category: {alias: repo_id}} or single category if filtered
    """
    if category:
        # Validate category
        if category not in OFFICIAL_REPOS:
            valid_cats = ", ".join(OFFICIAL_REPOS.keys())
            raise ValueError(f"Invalid category '{category}'. Valid: {valid_cats}")
        return {category: OFFICIAL_REPOS[category]}
    
    return OFFICIAL_REPOS.copy()


def get_model_category(alias: str) -> Optional[str]:
    """
    Find which category a model alias belongs to.
    
    Args:
        alias: Model alias
        
    Returns:
        Category name or None if not found
    """
    for category, models in OFFICIAL_REPOS.items():
        if alias in models:
            return category
    return None


def get_categories() -> List[str]:
    """Get list of all available categories."""
    return list(OFFICIAL_REPOS.keys())


def get_category_info(category: str) -> Dict[str, str]:
    """Get metadata about a category."""
    return CATEGORY_INFO.get(category, {
        "name": category,
        "icon": "📦",
        "description": f"{category} models"
    })


def search_aliases(query: str, category: Optional[str] = None) -> List[str]:
    """
    Search for model aliases matching a query.
    
    Args:
        query: Search term
        category: Optional category filter
        
    Returns:
        Sorted list of matching aliases
    """
    query_lower = query.lower()
    
    # Get models to search
    if category:
        if category not in OFFICIAL_REPOS:
            return []
        search_dict = OFFICIAL_REPOS[category]
    else:
        search_dict = MODEL_ALIASES
    
    return sorted([
        alias for alias in search_dict.keys()
        if query_lower in alias.lower()
    ])


def get_model_category(alias: str) -> Optional[str]:
    """
    Get the category of a model alias.
    Used by CLI to determine model type (vision, image gen, video gen, etc.)
    
    Args:
        alias: Model alias (e.g., "qwen3-vl-7b", "flux-1-dev")
        
    Returns:
        Category name or None if not found
    """
    for category, models in OFFICIAL_REPOS.items():
        if alias in models:
            return category
    return None

