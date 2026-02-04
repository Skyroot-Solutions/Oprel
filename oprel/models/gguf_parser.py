"""
GGUF File Parser - Extract metadata from GGUF model files.

This module provides fast parsing of GGUF format model files to extract
key metadata needed for model configuration and optimization.
"""

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import os


# GGUF format constants
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3

# GGUF metadata type codes
GGUF_TYPE_U8 = 0
GGUF_TYPE_I8 = 1
GGUF_TYPE_U16 = 2
GGUF_TYPE_I16 = 3
GGUF_TYPE_U32 = 4
GGUF_TYPE_I32 = 5
GGUF_TYPE_F32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_U64 = 10
GGUF_TYPE_I64 = 11
GGUF_TYPE_F64 = 12


@dataclass
class GGUFMetadata:
    """
    Metadata extracted from a GGUF model file.
    
    This contains key parameters needed for model configuration,
    memory estimation, and layer calculation.
    """
    # Model identification
    architecture: str  # Model architecture (e.g., "llama", "mistral", "phi")
    model_name: str  # Full model name
    quantization: str  # Quantization type (e.g., "Q4_K_M", "F16")
    
    # Model dimensions
    num_layers: int  # Transformer layers (typically 32, 40, 60, 80)
    embedding_dim: int  # Hidden/embedding dimension
    num_attention_heads: int  # Number of attention heads
    num_kv_heads: int  # Number of key-value heads (for GQA)
    context_length: int  # Maximum context window
    vocab_size: int  # Vocabulary size
    
    # File info
    file_size_bytes: int  # Total file size
    file_path: str  # Path to the file
    
    # Additional metadata
    block_count: int  # Same as num_layers in most architectures
    
    def __str__(self) -> str:
        return (
            f"GGUFMetadata({self.architecture}, {self.model_name}, {self.quantization}, "
            f"{self.num_layers} layers, {self.context_length} ctx, "
            f"{self.file_size_bytes / (1024**3):.1f}GB)"
        )


# Metadata key mappings (architecture-specific keys to generic names)
METADATA_KEYS = {
    "general.architecture": "architecture",
    "general.name": "model_name",
    "general.file_type": "quantization",
    
    # Generic architecture-agnostic keys
    "<arch>.block_count": "num_layers",
    "<arch>.embedding_length": "embedding_dim",
    "<arch>.attention.head_count": "num_attention_heads",
    "<arch>.attention.head_count_kv": "num_kv_heads",
    "<arch>.context_length": "context_length",
    "<arch>.vocab_size": "vocab_size",
}


def parse_gguf_fast(file_path: str) -> GGUFMetadata:
    """
    Fast parse of GGUF file to extract essential metadata.
    
    This reads only the header section of the GGUF file to extract
    metadata, without loading tensor data.
    
    Args:
        file_path: Path to GGUF file
        
    Returns:
        GGUFMetadata object with model information
        
    Raises:
        ValueError: If file is not valid GGUF format
        FileNotFoundError: If file doesn't exist
        
    Example:
        >>> meta = parse_gguf_fast("llama-2-7b-q4_k_m.gguf")
        >>> print(f"Layers: {meta.num_layers}, Context: {meta.context_length}")
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"GGUF file not found: {file_path}")
    
    file_size = file_path.stat().st_size
    
    with open(file_path, 'rb') as f:
        # Read GGUF header
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF file: wrong magic number 0x{magic:08x}")
        
        version = struct.unpack('<I', f.read(4))[0]
        if version != GGUF_VERSION:
            raise ValueError(f"Unsupported GGUF version: {version}")
        
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
        
        # Parse metadata key-value pairs
        metadata = {}
        for _ in range(metadata_kv_count):
            # Read key
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8')
            
            # Read value type
            value_type = struct.unpack('<I', f.read(4))[0]
            
            # Read value based on type
            value = _read_metadata_value(f, value_type)
            metadata[key] = value
    
    # Extract architecture
    arch = metadata.get("general.architecture", "unknown")
    
    # Helper to get metadata with architecture substitution
    def get_meta(key_pattern: str, default=None):
        if "<arch>" in key_pattern:
            key = key_pattern.replace("<arch>", arch)
        else:
            key = key_pattern
        return metadata.get(key, default)
    
    # Extract key fields with fallbacks
    num_layers = get_meta(f"{arch}.block_count", 32)
    embedding_dim = get_meta(f"{arch}.embedding_length", 4096)
    num_attention_heads = get_meta(f"{arch}.attention.head_count", 32)
    num_kv_heads = get_meta(f"{arch}.attention.head_count_kv", num_attention_heads)
    context_length = get_meta(f"{arch}.context_length", 2048)
    vocab_size = get_meta(f"{arch}.vocab_size", 32000)
    
    # Quantization from file_type
    file_type = metadata.get("general.file_type", 0)
    quantization = _file_type_to_quant(file_type)
    
    return GGUFMetadata(
        architecture=arch,
        model_name=metadata.get("general.name", file_path.stem),
        quantization=quantization,
        num_layers=num_layers,
        embedding_dim=embedding_dim,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        context_length=context_length,
        vocab_size=vocab_size,
        file_size_bytes=file_size,
        file_path=str(file_path),
        block_count=num_layers,  # Same as num_layers in most architectures
    )


def _read_metadata_value(f, value_type: int):
    """Read a metadata value based on its type."""
    if value_type == GGUF_TYPE_U8:
        return struct.unpack('<B', f.read(1))[0]
    elif value_type == GGUF_TYPE_I8:
        return struct.unpack('<b', f.read(1))[0]
    elif value_type == GGUF_TYPE_U16:
        return struct.unpack('<H', f.read(2))[0]
    elif value_type == GGUF_TYPE_I16:
        return struct.unpack('<h', f.read(2))[0]
    elif value_type == GGUF_TYPE_U32:
        return struct.unpack('<I', f.read(4))[0]
    elif value_type == GGUF_TYPE_I32:
        return struct.unpack('<i', f.read(4))[0]
    elif value_type == GGUF_TYPE_F32:
        return struct.unpack('<f', f.read(4))[0]
    elif value_type == GGUF_TYPE_U64:
        return struct.unpack('<Q', f.read(8))[0]
    elif value_type == GGUF_TYPE_I64:
        return struct.unpack('<q', f.read(8))[0]
    elif value_type == GGUF_TYPE_F64:
        return struct.unpack('<d', f.read(8))[0]
    elif value_type == GGUF_TYPE_BOOL:
        return struct.unpack('<?', f.read(1))[0]
    elif value_type == GGUF_TYPE_STRING:
        str_len = struct.unpack('<Q', f.read(8))[0]
        return f.read(str_len).decode('utf-8')
    elif value_type == GGUF_TYPE_ARRAY:
        # Read array type and length
        array_type = struct.unpack('<I', f.read(4))[0]
        array_len = struct.unpack('<Q', f.read(8))[0]
        # Read array elements
        return [_read_metadata_value(f, array_type) for _ in range(array_len)]
    else:
        raise ValueError(f"Unknown metadata type: {value_type}")


def _file_type_to_quant(file_type: int) -> str:
    """Convert GGUF file_type number to quantization string."""
    quant_map = {
        0: "F32",
        1: "F16",
        2: "Q4_0",
        3: "Q4_1",
        6: "Q5_0",
        7: "Q5_1",
        8: "Q8_0",
        9: "Q8_1",
        10: "Q2_K",
        11: "Q3_K_S",
        12: "Q3_K_M",
        13: "Q3_K_L",
        14: "Q4_K_S",
        15: "Q4_K_M",
        16: "Q5_K_S",
        17: "Q5_K_M",
        18: "Q6_K",
        19: "Q8_K",
        20: "IQ2_XXS",
        21: "IQ2_XS",
        22: "IQ3_XXS",
        23: "IQ1_S",
        24: "IQ4_NL",
        25: "IQ3_S",
        26: "IQ2_S",
        27: "IQ4_XS",
        28: "IQ1_M",
    }
    return quant_map.get(file_type, f"UNKNOWN_{file_type}")
