"""
Model utilities and parsers for the Oprel SDK.
"""

from .gguf_parser import GGUFMetadata, parse_gguf_fast

__all__ = ["GGUFMetadata", "parse_gguf_fast"]
