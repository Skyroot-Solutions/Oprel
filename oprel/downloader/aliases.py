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
        "qwen3-235b": "Qwen/Qwen3-235B-GGUF",     # Flagship Dense
        "qwen3-32b": "Qwen/Qwen3-32B-GGUF",       # Flagship Dense
        "qwen3-14b": "Qwen/Qwen3-14B-GGUF",       # Best All-Rounder
        "qwen3-8b": "Qwen/Qwen3-8B-GGUF",         # Consumer GPU King
        "qwen3-4b": "Qwen/Qwen3-4B-GGUF",         #
        "qwen3-3b": "Qwen/Qwen3-3B-GGUF",         # Optimized for Mobile
        "qwen3-1.7b": "Qwen/Qwen3-1.7B-GGUF",     #
        "qwen3-0.6b": "Qwen/Qwen3-0.6B-GGUF",     # IoT / Embedded

        # --- QWEN 2.5 FAMILY (Reliable Legacy) ---
        "qwen2.5-72b": "Qwen/Qwen2.5-72B-GGUF",     # Flagship Dense
        "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",

        # --- GEMMA 3 FAMILY (Google) ---
        "gemma3-27b": "unsloth/gemma-3-27b-it-GGUF",        #
        "gemma3-12b": "unsloth/gemma-3-12b-it-GGUF",        #
        "gemma3-4b": "unsloth/gemma-3-4b-it-GGUF",          #
        "gemma3-1b": "unsloth/gemma-3-1b-it-GGUF",          #

        # --- GEMMA 2 FAMILY ---
        "gemma2-27b": "lmstudio-community/gemma-2-27b-it-GGUF",
        "gemma2-9b": "lmstudio-community/gemma-2-9b-it-GGUF",
        "gemma2-2b": "lmstudio-community/gemma-2-2b-it-GGUF",

        # --- LLAMA 3.x FAMILY ---
        "llama3.3-70b": "unsloth/Llama-3.3-70B-Instruct-GGUF", # Replaced phantom 8B with real 70B
        "llama3.3-8b": "unsloth/Llama-3.3-8B-Instruct-GGUF",
        "llama3.1-8b": "unsloth/Meta-Llama-3.1-8B-Instruct-GGUF",
        "llama3.2-3b": "unsloth/Llama-3.2-3B-Instruct-GGUF",
        "llama3.2-1b": "unsloth/Llama-3.2-1B-Instruct-GGUF",
        
        # --- MISTRAL & OTHERS ---
        "mistral-3-3b" : "mistralai/Ministral-3-3B-Reasoning-2512-GGUF",
        "mistral-3-8b" : "mistralai/Ministral-3-8B-Instruct-2512-GGUF",
        "mistral-3-14b" : "mistralai/Ministral-3-14B-Instruct-2512-GGUF",
        "devstral-2-24b": "unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF",

        # --- GPT & OTHERS ---
        "gpt-oss-20b": "unsloth/gpt-oss-20b-GGUF",
        "gpt-oss-120b": "unsloth/gpt-oss-120b-GGUF",
    },

    "coding": {
        # --- QWEN 3 CODER (2026 SOTA) ---
        "qwen3-coder-next-80b" :"Qwen/Qwen3-Coder-Next-GGUF",
        "qwen3-coder-31b":"unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",

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

    "vision": {
        # --- QWEN VL (Best OCR/Vision) ---
        "qwen3-vl-32b": "Qwen/Qwen3-VL-32B-Instruct-GGUF",
        "qwen3-vl-32b": "Qwen/Qwen3-VL-32B-Thinking-GGUF",
        "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "qwen3-vl-4b": "Qwen/Qwen3-VL-4B-Thinking-GGUF",
        "qwen3-vl-4b": "Qwen/Qwen3-VL-4B-Instruct-GGUF",
        "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct-GGUF",
        "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Thinking-GGUF",
        "qwen2.5-vl-7b": "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
        "qwen2.5-vl-3b": "unsloth/Qwen2.5-VL-3B-Instruct-GGUF",

        # --- Google Gemma 3 VL ---
        "gemma3-vl-12b": "google/gemma-3-12b-it-qat-q4_0-gguf",
        "gemma3-vl-4b": "google/gemma-3-4b-it-qat-q4_0-gguf",

        # --- OTHERS ---
        "LFM2-vl-3b":"LiquidAI/LFM2-VL-3B-GGUF",
        "LFM2.5-vl-1.6b":"LiquidAI/LFM2.5-VL-1.6B-GGUF",
        "LFM2-VL-450M":"LiquidAI/LFM2-VL-450M-GGUF",
        "Deepseek-OCR-3B":"NexaAI/DeepSeek-OCR-GGUF",
    },

    "embeddings": {
        "embeddinggemma-0.3b":"unsloth/embeddinggemma-300m-GGUF",
        "nomic-embed-text": "nomic-ai/nomic-embed-text-v1.5-GGUF",
        "bge-m3": "audo/bge-m3-GGUF",
        "snowflake-arctic": "ChristianAzinn/snowflake-arctic-embed-m-gguf",
        "all-minilm-l6-v2": "faldor/all-MiniLM-L6-v2-gguf",
        "e5-small": "intfloat/e5-small-v2-gguf",
    },
    
    # Image/Video models (Safetensors - managed by ComfyUI)
    "text-to-image": {
        # FLUX models (12B-23B - best quality)
        "flux-1-dev": "black-forest-labs/FLUX.1-dev",
        "flux-1-schnell": "black-forest-labs/FLUX.1-schnell",
        
        # Sana models (2.7B-4B - efficient, high quality)
        "sana-1.6b": "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
        "sana-2.7b": "Efficient-Large-Model/Sana_2700M_1024px_diffusers",
        "sana-4b": "Efficient-Large-Model/Sana_4000M_1024px_diffusers",
        
        # PixArt models (600M-2.5B - fast, good quality)
        "pixart-sigma": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",  # 2.5B
        "pixart-alpha": "PixArt-alpha/PixArt-XL-2-1024-MS",  # 600M
        
        # AuraFlow (6.8B - open source FLUX alternative)
        "auraflow": "fal/AuraFlow-v0.3",
        
        # SDXL models (3.5B-7B - fast, widely compatible)
        "sdxl-turbo": "stabilityai/sdxl-turbo",
        "sdxl-base": "stabilityai/stable-diffusion-xl-base-1.0",
        "sdxl-lightning": "ByteDance/SDXL-Lightning",  # 4-step fast
        
        # SD 1.5 (860M - lightweight, fast)
        "sd-1.5": "runwayml/stable-diffusion-v1-5",
        "sd-1.5-turbo": "stabilityai/sd-turbo",
        
        # Playground v2.5 (2.5B - aesthetic quality)
        "playground-v2.5": "playgroundai/playground-v2.5-1024px-aesthetic",
        
        # Kandinsky (3.3B - unique style)
        "kandinsky-3": "kandinsky-community/kandinsky-3",
        
        # Würstchen (1B+30B - two-stage, efficient)
        "wurstchen": "warp-ai/wuerstchen",
    },
    
    # "text-to-video": {
    #     "animatediff-motion": "guoyww/animatediff:mm_sd_v15_v2.ckpt",
        
    #     "svd": "stabilityai/stable-video-diffusion-img2vid:svd.safetensors",
    #     "svd-xt": "stabilityai/stable-video-diffusion-img2vid-xt:svd_xt.safetensors",
    # }
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
        "description": "General-purpose chat and instruction-following models (GGUF)",
        "backend": "llama.cpp",
    },
    "coding": {
        "name": "Coding",
        "icon": "👨‍💻",
        "description": "Code generation and software development models (GGUF)",
        "backend": "llama.cpp",
    },
    "reasoning": {
        "name": "Reasoning",
        "icon": "🧠",
        "description": "Advanced reasoning and chain-of-thought models (GGUF)",
        "backend": "llama.cpp",
    },
    "vision": {
        "name": "Vision/Multimodal",
        "icon": "👁️",
        "description": "Vision-language models for image understanding (GGUF + mmproj)",
        "backend": "llama.cpp",
    },
    "embeddings": {
        "name": "Embeddings",
        "icon": "📚",
        "description": "Text embeddings for RAG and semantic search (GGUF)",
        "backend": "llama.cpp",
    },
    "text-to-image": {
        "name": "Text-to-Image",
        "icon": "🎨",
        "description": "Image generation from text prompts (Safetensors)",
        "backend": "comfyui",
    },
    # "text-to-video": {
    #     "name": "Text-to-Video",
    #     "icon": "🎥",
    #     "description": "Video generation from text prompts (Safetensors)",
    #     "backend": "comfyui",
    # },
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
        "description": f"{category} models",
        "backend": "unknown"
    })


def get_model_backend(alias: str) -> str:
    """
    Get the backend engine required for a model.
    
    Args:
        alias: Model alias
        
    Returns:
        Backend name ("llama.cpp" or "comfyui")
    """
    category = get_model_category(alias)
    if category:
        info = get_category_info(category)
        return info.get("backend", "llama.cpp")
    return "llama.cpp"  # Default to llama.cpp


def is_comfyui_model(alias: str) -> bool:
    """Check if model requires ComfyUI backend."""
    return get_model_backend(alias) == "comfyui"


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

