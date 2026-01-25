"""
Model aliases for user-friendly names to official GGUF models.

Uses current, actively maintained GGUF sources (not TheBloke).
"""

from oprel.utils.logging import get_logger

logger = get_logger(__name__)

# Official model mappings to GGUF
MODEL_ALIASES = {
    # === Meta Llama Family ===
    "llama3.3": "bartowski/Llama-3.3-70B-Instruct-GGUF",
    "llama3.3-70b": "bartowski/Llama-3.3-70B-Instruct-GGUF",
    "llama3.2": "bartowski/Llama-3.2-3B-Instruct-GGUF",
    "llama3.2-1b": "bartowski/Llama-3.2-1B-Instruct-GGUF",
    "llama3.2-3b": "bartowski/Llama-3.2-3B-Instruct-GGUF",
    "llama3.1": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    "llama3.1-8b": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    "llama3.1-70b": "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
    "llama3": "bartowski/Meta-Llama-3-8B-Instruct-GGUF",
    "llama3-8b": "bartowski/Meta-Llama-3-8B-Instruct-GGUF",
    "llama3-70b": "bartowski/Meta-Llama-3-70B-Instruct-GGUF",
    "llama2": "lmstudio-community/Meta-Llama-2-7B-Chat-GGUF",
    "llama2-7b": "lmstudio-community/Meta-Llama-2-7B-Chat-GGUF",
    "llama2-13b": "lmstudio-community/Meta-Llama-2-13B-Chat-GGUF",
    
    # === Google Gemma Family ===
    "gemma2": "bartowski/gemma-2-9b-it-GGUF",
    "gemma2-2b": "bartowski/gemma-2-2b-it-GGUF",
    "gemma2-9b": "bartowski/gemma-2-9b-it-GGUF",
    "gemma2-27b": "bartowski/gemma-2-27b-it-GGUF",
    "gemma": "lmstudio-community/gemma-2b-it-GGUF",
    "gemma-2b": "lmstudio-community/gemma-2b-it-GGUF",
    "gemma-7b": "lmstudio-community/gemma-7b-it-GGUF",
    
    # === Mistral AI Family ===
    "mistral": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
    "mistral-7b": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
    "mixtral": "bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF",
    "mixtral-8x7b": "bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF",
    
    # === Alibaba Qwen Family ===
    "qwen2.5": "bartowski/Qwen2.5-7B-Instruct-GGUF",
    "qwen2.5-7b": "bartowski/Qwen2.5-7B-Instruct-GGUF",
    "qwen2.5-14b": "bartowski/Qwen2.5-14B-Instruct-GGUF",
    "qwen2.5-32b": "bartowski/Qwen2.5-32B-Instruct-GGUF",
    "qwencoder": "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
    "qwencoder-7b": "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
    "qwencoder-32b": "bartowski/Qwen2.5-Coder-32B-Instruct-GGUF",
    
    # === Microsoft Phi Family ===
    "phi3": "bartowski/Phi-3-mini-4k-instruct-GGUF",
    "phi3-mini": "bartowski/Phi-3-mini-4k-instruct-GGUF",
    "phi3.5": "bartowski/Phi-3.5-mini-instruct-GGUF",
    "phi3.5-mini": "bartowski/Phi-3.5-mini-instruct-GGUF",
    
    # === DeepSeek ===
    "deepseek": "bartowski/DeepSeek-V2.5-GGUF",
    "deepseek-coder": "bartowski/DeepSeek-Coder-V2-Instruct-GGUF",
    
    # === Yi (01.AI) ===
    "yi": "bartowski/Yi-1.5-9B-Chat-GGUF",
    "yi-9b": "bartowski/Yi-1.5-9B-Chat-GGUF",
    "yi-34b": "bartowski/Yi-1.5-34B-Chat-GGUF",
    
    # === SmolLM (Tiny) ===
    "smollm": "bartowski/SmolLM-1.7B-Instruct-GGUF",
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


def list_available_aliases() -> dict:
    """Get all available model aliases."""
    return MODEL_ALIASES.copy()


def search_aliases(query: str) -> list:
    """Search for model aliases matching a query."""
    query_lower = query.lower()
    return sorted([
        alias for alias in MODEL_ALIASES.keys()
        if query_lower in alias.lower()
    ])
