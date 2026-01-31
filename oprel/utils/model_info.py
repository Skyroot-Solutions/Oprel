"""
Model information extraction from GGUF files.

Provides utilities to read model metadata, context length, architecture,
and other details directly from GGUF model files.
"""

import struct
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from oprel.utils.logging import get_logger

logger = get_logger(__name__)

# GGUF format constants
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3

# GGUF value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12


def read_gguf_string(f) -> str:
    """Read a GGUF string (length-prefixed)."""
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8', errors='replace')


def read_gguf_value(f, value_type: int) -> Any:
    """Read a GGUF value based on its type."""
    if value_type == GGUF_TYPE_UINT8:
        return struct.unpack('<B', f.read(1))[0]
    elif value_type == GGUF_TYPE_INT8:
        return struct.unpack('<b', f.read(1))[0]
    elif value_type == GGUF_TYPE_UINT16:
        return struct.unpack('<H', f.read(2))[0]
    elif value_type == GGUF_TYPE_INT16:
        return struct.unpack('<h', f.read(2))[0]
    elif value_type == GGUF_TYPE_UINT32:
        return struct.unpack('<I', f.read(4))[0]
    elif value_type == GGUF_TYPE_INT32:
        return struct.unpack('<i', f.read(4))[0]
    elif value_type == GGUF_TYPE_FLOAT32:
        return struct.unpack('<f', f.read(4))[0]
    elif value_type == GGUF_TYPE_BOOL:
        return struct.unpack('<B', f.read(1))[0] != 0
    elif value_type == GGUF_TYPE_STRING:
        return read_gguf_string(f)
    elif value_type == GGUF_TYPE_UINT64:
        return struct.unpack('<Q', f.read(8))[0]
    elif value_type == GGUF_TYPE_INT64:
        return struct.unpack('<q', f.read(8))[0]
    elif value_type == GGUF_TYPE_FLOAT64:
        return struct.unpack('<d', f.read(8))[0]
    elif value_type == GGUF_TYPE_ARRAY:
        array_type = struct.unpack('<I', f.read(4))[0]
        array_len = struct.unpack('<Q', f.read(8))[0]
        return [read_gguf_value(f, array_type) for _ in range(array_len)]
    else:
        raise ValueError(f"Unknown GGUF type: {value_type}")


def read_gguf_metadata(model_path: Path) -> Dict[str, Any]:
    """
    Read metadata from a GGUF model file.
    
    Args:
        model_path: Path to the GGUF file
        
    Returns:
        Dictionary of metadata key-value pairs
    """
    metadata = {}
    
    try:
        with open(model_path, 'rb') as f:
            # Read header
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != GGUF_MAGIC:
                logger.warning(f"Not a valid GGUF file: {model_path}")
                return {}
            
            version = struct.unpack('<I', f.read(4))[0]
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
            
            metadata['_gguf_version'] = version
            metadata['_tensor_count'] = tensor_count
            
            # Read metadata key-value pairs
            for _ in range(metadata_kv_count):
                key = read_gguf_string(f)
                value_type = struct.unpack('<I', f.read(4))[0]
                value = read_gguf_value(f, value_type)
                metadata[key] = value
                
    except Exception as e:
        logger.warning(f"Error reading GGUF metadata: {e}")
        
    return metadata


def get_model_context_length(model_path: Path) -> int:
    """
    Get the context length from a GGUF model file.
    
    Args:
        model_path: Path to the GGUF file
        
    Returns:
        Context length in tokens, or default 4096 if not found
    """
    metadata = read_gguf_metadata(model_path)
    
    # Try different metadata keys for context length
    ctx_keys = [
        'llama.context_length',
        'qwen2.context_length', 
        'mistral.context_length',
        'gemma.context_length',
        'phi3.context_length',
        'general.context_length',
    ]
    
    for key in ctx_keys:
        if key in metadata:
            ctx_len = metadata[key]
            logger.info(f"Model context length from {key}: {ctx_len}")
            return ctx_len
    
    # Fallback: check for n_ctx_train
    if 'n_ctx_train' in metadata:
        return metadata['n_ctx_train']
    
    logger.warning(f"Could not determine context length for {model_path}, using default 4096")
    return 4096


def get_model_architecture(model_path: Path) -> str:
    """
    Get the model architecture from a GGUF file.
    
    Args:
        model_path: Path to the GGUF file
        
    Returns:
        Architecture string (e.g., 'llama', 'qwen2', 'mistral')
    """
    metadata = read_gguf_metadata(model_path)
    
    arch = metadata.get('general.architecture', 'unknown')
    logger.info(f"Model architecture: {arch}")
    return arch


def get_model_info(model_path: Path) -> Dict[str, Any]:
    """
    Get comprehensive model information from a GGUF file.
    
    Args:
        model_path: Path to the GGUF file
        
    Returns:
        Dictionary with model details
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    metadata = read_gguf_metadata(model_path)
    file_size_gb = model_path.stat().st_size / (1024**3)
    
    # Extract key information
    info = {
        'file_path': str(model_path),
        'file_name': model_path.name,
        'file_size_gb': round(file_size_gb, 2),
        'architecture': metadata.get('general.architecture', 'unknown'),
        'name': metadata.get('general.name', model_path.stem),
        'quantization': metadata.get('general.file_type', 'unknown'),
        'vocab_size': None,
        'context_length': get_model_context_length(model_path),
        'embedding_length': None,
        'block_count': None,
        'head_count': None,
        'head_count_kv': None,
    }
    
    # Architecture-specific metadata extraction
    arch = info['architecture']
    arch_prefix = f"{arch}."
    
    for key, value in metadata.items():
        if key.startswith(arch_prefix):
            short_key = key[len(arch_prefix):]
            if short_key == 'context_length':
                info['context_length'] = value
            elif short_key == 'embedding_length':
                info['embedding_length'] = value
            elif short_key == 'block_count':
                info['block_count'] = value
            elif short_key == 'attention.head_count':
                info['head_count'] = value
            elif short_key == 'attention.head_count_kv':
                info['head_count_kv'] = value
    
    # Estimate parameters from file size and quantization
    info['estimated_parameters'] = estimate_parameters(file_size_gb, info.get('quantization'))
    
    # Include raw metadata for debugging
    info['_raw_metadata'] = {k: v for k, v in metadata.items() 
                            if not k.startswith('tokenizer.ggml.')}
    
    return info


def estimate_parameters(file_size_gb: float, quantization: Any) -> str:
    """
    Estimate model parameters from file size and quantization.
    
    Args:
        file_size_gb: Model file size in GB
        quantization: Quantization type/level
        
    Returns:
        Estimated parameters as string (e.g., "7B", "13B")
    """
    # Approximate bits per parameter for common quantizations
    # These are rough estimates
    quant_bpp = {
        'Q2_K': 2.5,
        'Q3_K_S': 3.0,
        'Q3_K_M': 3.5,
        'Q3_K_L': 3.9,
        'Q4_0': 4.5,
        'Q4_K_S': 4.5,
        'Q4_K_M': 4.8,
        'Q5_0': 5.5,
        'Q5_K_S': 5.5,
        'Q5_K_M': 5.7,
        'Q6_K': 6.5,
        'Q8_0': 8.5,
        'F16': 16.0,
        'F32': 32.0,
    }
    
    # Default to Q4_K_M if unknown
    bpp = 4.8
    
    if isinstance(quantization, int):
        # GGUF file type codes
        file_type_map = {
            0: 32.0,   # F32
            1: 16.0,   # F16
            2: 4.5,    # Q4_0
            3: 4.0,    # Q4_1
            7: 8.5,    # Q8_0
            15: 4.8,   # Q4_K_M
            17: 5.7,   # Q5_K_M
        }
        bpp = file_type_map.get(quantization, 4.8)
    elif isinstance(quantization, str):
        for q, b in quant_bpp.items():
            if q in quantization.upper():
                bpp = b
                break
    
    # Calculate: file_size_bytes = params * bits_per_param / 8
    file_size_bits = file_size_gb * 1024**3 * 8
    estimated_params = file_size_bits / bpp
    
    # Convert to human-readable format
    if estimated_params >= 1e12:
        return f"{estimated_params/1e12:.1f}T"
    elif estimated_params >= 1e9:
        return f"{estimated_params/1e9:.1f}B"
    elif estimated_params >= 1e6:
        return f"{estimated_params/1e6:.1f}M"
    else:
        return f"{estimated_params:.0f}"


def display_model_info(model_path: str, show_raw: bool = False):
    """
    Display formatted model information.
    
    Args:
        model_path: Path to the GGUF model file
        show_raw: Whether to show raw metadata
    """
    info = get_model_info(Path(model_path))
    
    print(f"\n{'='*60}")
    print(f"📋 Model Information: {info['file_name']}")
    print(f"{'='*60}")
    print(f"📁 Path: {info['file_path']}")
    print(f"📏 File Size: {info['file_size_gb']:.2f} GB")
    print(f"🏗️  Architecture: {info['architecture']}")
    print(f"📛 Name: {info['name']}")
    print(f"🔢 Estimated Parameters: {info['estimated_parameters']}")
    print(f"📝 Context Length: {info['context_length']:,} tokens")
    
    if info['embedding_length']:
        print(f"📐 Embedding Length: {info['embedding_length']}")
    if info['block_count']:
        print(f"🧱 Layers (Blocks): {info['block_count']}")
    if info['head_count']:
        print(f"👁️  Attention Heads: {info['head_count']}")
    if info['head_count_kv']:
        print(f"🔑 KV Heads: {info['head_count_kv']}")
    
    # Memory estimation
    print(f"\n💾 Memory Estimates:")
    ctx = info['context_length']
    kv_heads = info['head_count_kv'] or 4
    layers = info['block_count'] or 32
    embed = info['embedding_length'] or 4096
    
    # KV cache size estimation: 2 * layers * kv_heads * head_dim * ctx * 2 bytes (fp16)
    head_dim = embed // (info['head_count'] or 32)
    kv_cache_mb = (2 * layers * kv_heads * head_dim * ctx * 2) / (1024**2)
    print(f"   KV Cache (full ctx): {kv_cache_mb:.0f} MB")
    print(f"   Model Weights: {info['file_size_gb']*1024:.0f} MB")
    print(f"   Total (estimated): {kv_cache_mb + info['file_size_gb']*1024:.0f} MB")
    
    if show_raw:
        print(f"\n📋 Raw Metadata:")
        for key, value in sorted(info['_raw_metadata'].items()):
            if not isinstance(value, (list, bytes)) or len(str(value)) < 100:
                print(f"   {key}: {value}")


def compare_with_ollama_memory(model_path: str) -> Dict[str, Any]:
    """
    Compare Oprel's memory usage approach with Ollama's.
    
    Args:
        model_path: Path to the GGUF model file
        
    Returns:
        Comparison dict with memory estimates
    """
    info = get_model_info(Path(model_path))
    
    ctx = info['context_length']
    file_size_mb = info['file_size_gb'] * 1024
    layers = info['block_count'] or 32
    kv_heads = info['head_count_kv'] or 4
    embed = info['embedding_length'] or 4096
    head_dim = embed // (info['head_count'] or 32)
    
    # Ollama typically loads full context KV cache upfront
    ollama_kv_cache = (2 * layers * kv_heads * head_dim * ctx * 2) / (1024**2)
    ollama_total = file_size_mb + ollama_kv_cache + 500  # +500MB overhead
    
    # Oprel uses dynamic/lazy KV allocation (start with smaller ctx)
    oprel_initial_ctx = min(ctx, 4096)  # Start with 4K context
    oprel_kv_cache = (2 * layers * kv_heads * head_dim * oprel_initial_ctx * 2) / (1024**2)
    oprel_total = file_size_mb + oprel_kv_cache + 200  # Less overhead
    
    savings_mb = ollama_total - oprel_total
    savings_pct = (savings_mb / ollama_total) * 100
    
    return {
        'model_name': info['name'],
        'model_size_mb': file_size_mb,
        'max_context': ctx,
        'ollama_estimate_mb': round(ollama_total),
        'oprel_estimate_mb': round(oprel_total),
        'savings_mb': round(savings_mb),
        'savings_percent': round(savings_pct, 1),
        'explanation': f"""
Memory Comparison for {info['name']}:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model Weights:     {file_size_mb:,.0f} MB

Ollama Approach:
  - Pre-allocates full {ctx:,} token KV cache
  - KV Cache: {ollama_kv_cache:,.0f} MB
  - Overhead: ~500 MB (Go runtime, HTTP server)
  - Total: {ollama_total:,.0f} MB

Oprel Approach:  
  - Starts with {oprel_initial_ctx:,} token context
  - KV Cache: {oprel_kv_cache:,.0f} MB  
  - Overhead: ~200 MB (Python, FastAPI)
  - Total: {oprel_total:,.0f} MB

💡 Savings: {savings_mb:,.0f} MB ({savings_pct:.1f}%)
"""
    }
