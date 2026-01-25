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
                "gpu_type": "metal",
            },
            "Darwin-x86_64": {
                "url": "https://github.com/ggml-org/llama.cpp/releases/download/b7822/llama-b7822-bin-macos-x64.tar.gz",
                "archive_type": "tar.gz",
                "binary_name": "llama-server",
                "gpu_type": "cpu",
            },
            "Linux-x86_64": {
                "url": "https://github.com/ggml-org/llama.cpp/releases/download/b7822/llama-b7822-bin-ubuntu-x64.tar.gz",
                "archive_type": "tar.gz",
                "binary_name": "llama-server",
                "gpu_type": "cpu",
            },
            "Linux-x86_64-vulkan": {
                "url": "https://github.com/ggml-org/llama.cpp/releases/download/b7822/llama-b7822-bin-ubuntu-vulkan-x64.tar.gz",
                "archive_type": "tar.gz",
                "binary_name": "llama-server",
                "gpu_type": "vulkan",
            },
            "Windows-AMD64": {
                "url": "https://github.com/ggml-org/llama.cpp/releases/download/b7822/llama-b7822-bin-win-cpu-x64.zip",
                "archive_type": "zip",
                "binary_name": "llama-server.exe",
                "gpu_type": "cpu",
            },
            "Windows-AMD64-cuda": {
                # CUDA 12.4 binary for Windows x64
                "url": "https://github.com/ggml-org/llama.cpp/releases/download/b7822/llama-b7822-bin-win-cuda-12.4-x64.zip",
                "dll_url": "https://github.com/ggml-org/llama.cpp/releases/download/b7822/cudart-llama-bin-win-cuda-12.4-x64.zip",
                "archive_type": "zip",
                "binary_name": "llama-server.exe",
                "gpu_type": "cuda",
            },
            "Windows-AMD64-vulkan": {
                "url": "https://github.com/ggml-org/llama.cpp/releases/download/b7822/llama-b7822-bin-win-vulkan-x64.zip",
                "archive_type": "zip",
                "binary_name": "llama-server.exe",
                "gpu_type": "vulkan",
            },
            "Windows-ARM64": {
                "url": "https://github.com/ggml-org/llama.cpp/releases/download/b7822/llama-b7822-bin-win-cpu-arm64.zip",
                "archive_type": "zip",
                "binary_name": "llama-server.exe",
                "gpu_type": "cpu",
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


def get_optimal_platform_key(backend: str, version: str, base_platform: str, has_cuda: bool) -> str:
    """
    Get the optimal platform key based on available GPU.
    
    Args:
        backend: Backend name
        version: Version string
        base_platform: Base platform (e.g., "Windows-AMD64")
        has_cuda: Whether CUDA GPU is available
        
    Returns:
        Platform key to use (e.g., "Windows-AMD64-cuda" or "Windows-AMD64")
    """
    if has_cuda:
        cuda_key = f"{base_platform}-cuda"
        # Check if CUDA version exists
        backend_info = BINARY_REGISTRY.get(backend, {})
        version_info = backend_info.get(version)
        if isinstance(version_info, str):
            version_info = backend_info.get(version_info)
        
        if version_info and cuda_key in version_info:
            return cuda_key
    
    return base_platform


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
