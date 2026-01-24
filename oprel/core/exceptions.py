"""
Custom exceptions for Oprel SDK
"""


class OprelError(Exception):
    """Base exception for all Oprel errors"""
    pass


class ModelNotFoundError(OprelError):
    """Raised when a model cannot be found or downloaded"""
    pass


class MemoryError(OprelError):
    """
    Raised when model exceeds memory limit.
    Unlike system MemoryError, this is caught gracefully.
    """
    pass


class BackendError(OprelError):
    """Raised when backend process fails to start or crashes"""
    pass


class BinaryNotFoundError(OprelError):
    """Raised when required binary is missing and cannot be downloaded"""
    pass


class UnsupportedPlatformError(OprelError):
    """Raised when running on unsupported OS/architecture"""
    pass


class InvalidQuantizationError(OprelError):
    """Raised when requested quantization is not available"""
    pass