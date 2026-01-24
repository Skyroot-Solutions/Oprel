"""
Platform-specific utilities and detection
"""

import platform
import sys
from typing import Literal

from oprel.utils.logging import get_logger

logger = get_logger(__name__)


PlatformType = Literal["Darwin", "Linux", "Windows"]
ArchType = Literal["x86_64", "arm64", "AMD64", "aarch64"]


def get_platform() -> PlatformType:
    """
    Get the current operating system.

    Returns:
        "Darwin" (macOS), "Linux", or "Windows"
    """
    return platform.system()  # type: ignore


def get_architecture() -> ArchType:
    """
    Get the CPU architecture.

    Returns:
        "x86_64", "arm64", "AMD64", or "aarch64"
    """
    return platform.machine()  # type: ignore


def get_platform_key() -> str:
    """
    Get platform key for binary lookups.

    Returns:
        String like "Darwin-arm64" or "Linux-x86_64"
    """
    return f"{get_platform()}-{get_architecture()}"


def is_macos() -> bool:
    """Check if running on macOS"""
    return get_platform() == "Darwin"


def is_linux() -> bool:
    """Check if running on Linux"""
    return get_platform() == "Linux"


def is_windows() -> bool:
    """Check if running on Windows"""
    return get_platform() == "Windows"


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (M1/M2/M3)"""
    return is_macos() and get_architecture() == "arm64"


def is_64bit() -> bool:
    """Check if running on 64-bit architecture"""
    return sys.maxsize > 2**32


def supports_unix_sockets() -> bool:
    """Check if platform supports Unix domain sockets"""
    return is_macos() or is_linux()


def get_shell_executable() -> str:
    """
    Get the preferred shell executable for subprocess.

    Returns:
        Path to shell ("bash", "sh", "cmd.exe", etc.)
    """
    if is_windows():
        return "cmd.exe"
    else:
        return "/bin/bash"


def get_num_cores() -> dict:
    """
    Get CPU core information.

    Returns:
        Dict with physical_cores and logical_cores
    """
    import psutil

    return {
        "physical_cores": psutil.cpu_count(logical=False) or 1,
        "logical_cores": psutil.cpu_count(logical=True) or 1,
    }
