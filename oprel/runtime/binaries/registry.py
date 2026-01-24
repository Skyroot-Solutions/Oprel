"""
Registry of pre-compiled binary URLs
"""

# Binary registry: Maps (backend, version) -> platform URLs
# Platform format: "System-Machine" (e.g., "Darwin-arm64", "Linux-x86_64", "Windows-AMD64")
# archive_type indicates how the download should be handled: "zip", "tar.gz", or "exe"
BINARY_REGISTRY = {
    "llama.cpp": {
        "b7822": {  # llama.cpp build 7822 (latest as of Jan 2026)
            "Darwin-arm64": {
                "url": "https://github.com/ggml-org/llama.cpp/releases/download/b7822/llama-b7822-bin-macos-arm64.tar.gz",
                "archive_type": "tar.gz",
                "binary_name": "llama-server",
            },
            "Darwin-x86_64": {
                "url": "https://github.com/ggml-org/llama.cpp/releases/download/b7822/llama-b7822-bin-macos-x64.tar.gz",
                "archive_type": "tar.gz",
                "binary_name": "llama-server",
            },
            "Linux-x86_64": {
                "url": "https://github.com/ggml-org/llama.cpp/releases/download/b7822/llama-b7822-bin-ubuntu-x64.tar.gz",
                "archive_type": "tar.gz",
                "binary_name": "llama-server",
            },
            "Windows-AMD64": {
                "url": "https://github.com/ggml-org/llama.cpp/releases/download/b7822/llama-b7822-bin-win-cpu-x64.zip",
                "archive_type": "zip",
                "binary_name": "llama-server.exe",
            },
            "Windows-ARM64": {
                "url": "https://github.com/ggml-org/llama.cpp/releases/download/b7822/llama-b7822-bin-win-cpu-arm64.zip",
                "archive_type": "zip",
                "binary_name": "llama-server.exe",
            },
        },
        # Alias to most recent stable version
        "latest": "b7822",
    },
    # Future backends
    # "vllm": {...},
    # "exllama": {...},
}


def get_binary_info(backend: str, version: str, platform_key: str) -> dict | None:
    """
    Get binary info for a specific backend, version, and platform.
    
    Args:
        backend: Backend name (e.g., "llama.cpp")
        version: Version string (e.g., "b7822" or "latest")
        platform_key: Platform string (e.g., "Windows-AMD64")
        
    Returns:
        Dict with url, archive_type, binary_name, or None if not found
    """
    backend_info = BINARY_REGISTRY.get(backend, {})
    if not backend_info:
        return None
    
    version_info = backend_info.get(version)
    
    # Handle "latest" alias
    if isinstance(version_info, str):
        version_info = backend_info.get(version_info)
    
    if not version_info:
        return None
    
    return version_info.get(platform_key)


def get_supported_platforms(backend: str, version: str) -> list[str]:
    """
    Get list of supported platforms for a backend version.
    
    Args:
        backend: Backend name
        version: Version string
        
    Returns:
        List of platform strings (e.g., ["Darwin-arm64", "Linux-x86_64"])
    """
    backend_info = BINARY_REGISTRY.get(backend, {})
    if not backend_info:
        return []
    
    version_info = backend_info.get(version)
    
    # Handle "latest" alias
    if isinstance(version_info, str):
        version_info = backend_info.get(version_info)
    
    if not version_info or isinstance(version_info, str):
        return []
    
    return list(version_info.keys())