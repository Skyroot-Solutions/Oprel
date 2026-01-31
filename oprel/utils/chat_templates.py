"""
Chat template formatting for different model types
"""

from oprel.utils.logging import get_logger

logger = get_logger(__name__)


def detect_model_type(model_id: str) -> str:
    """
    Detect model type from model ID.
    
    Args:
        model_id: HuggingFace model ID or alias
        
    Returns:
        Model type: "qwen", "llama3", "llama2", "gemma", "mistral", or "unknown"
    """
    model_lower = model_id.lower()
    
    if "qwen" in model_lower:
        return "qwen"
    elif "llama-3" in model_lower or "llama3" in model_lower:
        return "llama3"
    elif "llama-2" in model_lower or "llama2" in model_lower:
        return "llama2"
    elif "gemma" in model_lower:
        return "gemma"
    elif "mistral" in model_lower or "mixtral" in model_lower:
        return "mistral"
    elif "phi" in model_lower:
        return "phi"
    else:
        return "unknown"


def format_chat_prompt(
    model_id: str,
    user_message: str,
    system_prompt: str = None,
    conversation_history: list = None,
) -> str:
    """
    Format a prompt with the correct chat template for the model.
    
    Args:
        model_id: Model identifier
        user_message: User's message
        system_prompt: Optional system instruction
        conversation_history: Optional list of previous messages
        
    Returns:
        Properly formatted prompt string
    """
    model_type = detect_model_type(model_id)
    
    if model_type == "qwen":
        return format_qwen_prompt(user_message, system_prompt, conversation_history)
    elif model_type == "llama3":
        return format_llama3_prompt(user_message, system_prompt, conversation_history)
    elif model_type == "llama2":
        return format_llama2_prompt(user_message, system_prompt, conversation_history)
    elif model_type == "gemma":
        return format_gemma_prompt(user_message, system_prompt, conversation_history)
    elif model_type == "mistral":
        return format_mistral_prompt(user_message, system_prompt, conversation_history)
    elif model_type == "phi":
        return format_phi_prompt(user_message, system_prompt, conversation_history)
    else:
        # Unknown model - return raw prompt with warning
        logger.warning(f"Unknown model type for '{model_id}', using raw prompt")
        return user_message


def format_qwen_prompt(
    user_message: str,
    system_prompt: str = None,
    conversation_history: list = None,
) -> str:
    """Format prompt for Qwen models (ChatML format)"""
    
    # Default system prompt for Qwen
    if system_prompt is None:
        system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    
    # Start with system message
    formatted = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    
    # Add conversation history if provided
    if conversation_history:
        for msg in conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    
    # Add current user message
    formatted += f"<|im_start|>user\n{user_message}<|im_end|>\n"
    formatted += "<|im_start|>assistant\n"
    
    return formatted


def format_llama3_prompt(
    user_message: str,
    system_prompt: str = None,
    conversation_history: list = None,
) -> str:
    """Format prompt for Llama 3 models"""
    
    formatted = "<|begin_of_text|>"
    
    # Add system prompt if provided
    if system_prompt:
        formatted += f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
    
    # Add conversation history
    if conversation_history:
        for msg in conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    
    # Add current message
    formatted += f"<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>"
    formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    return formatted


def format_llama2_prompt(
    user_message: str,
    system_prompt: str = None,
    conversation_history: list = None,
) -> str:
    """Format prompt for Llama 2 models"""
    
    formatted = ""
    
    # Add system prompt
    if system_prompt:
        formatted += f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
    
    # Add conversation history + current message
    if conversation_history:
        for msg in conversation_history:
            if msg.get("role") == "user":
                formatted += f"[INST] {msg.get('content')} [/INST] "
            elif msg.get("role") == "assistant":
                formatted += f"{msg.get('content')} </s>"
    
    # Add current message
    formatted += f"[INST] {user_message} [/INST] "
    
    return formatted


def format_gemma_prompt(
    user_message: str,
    system_prompt: str = None,
    conversation_history: list = None,
) -> str:
    """Format prompt for Gemma models"""
    
    formatted = "<bos>"
    
    # Add system as first user message if provided
    if system_prompt:
        formatted += f"<start_of_turn>user\n{system_prompt}<end_of_turn>\n"
        formatted += "<start_of_turn>model\nUnderstood.<end_of_turn>\n"
    
    # Add conversation history
    if conversation_history:
        for msg in conversation_history:
            role = "user" if msg.get("role") == "user" else "model"
            content = msg.get("content", "")
            formatted += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
    
    # Add current message
    formatted += f"<start_of_turn>user\n{user_message}<end_of_turn>\n"
    formatted += "<start_of_turn>model\n"
    
    return formatted


def format_mistral_prompt(
    user_message: str,
    system_prompt: str = None,
    conversation_history: list = None,
) -> str:
    """Format prompt for Mistral models"""
    
    formatted = ""
    
    # Mistral puts system in first user message
    if system_prompt:
        formatted += f"[INST] {system_prompt}\n\n"
    
    # Add conversation history
    if conversation_history:
        for msg in conversation_history:
            if msg.get("role") == "user":
                formatted += f"{msg.get('content')} [/INST]"
            elif msg.get("role") == "assistant":
                formatted += f" {msg.get('content')}</s>[INST] "
    
    # Add current message
    if system_prompt and not conversation_history:
        formatted += f"{user_message} [/INST]"
    else:
        formatted += f"[INST] {user_message} [/INST]"
    
    return formatted


def format_phi_prompt(
    user_message: str,
    system_prompt: str = None,
    conversation_history: list = None,
) -> str:
    """Format prompt for Phi models (ChatML-like)"""
    
    formatted = ""
    
    if system_prompt:
        formatted += f"<|system|>\n{system_prompt}<|end|>\n"
    
    if conversation_history:
        for msg in conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"<|{role}|>\n{content}<|end|>\n"
    
    formatted += f"<|user|>\n{user_message}<|end|>\n"
    formatted += "<|assistant|>\n"
    
    return formatted