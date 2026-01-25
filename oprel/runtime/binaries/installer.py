"""
Binary installation and management
"""

import platform
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from oprel.core.exceptions import BinaryNotFoundError, UnsupportedPlatformError
from oprel.runtime.binaries.registry import get_binary_info, get_supported_platforms, get_optimal_platform_key
from oprel.telemetry.hardware import detect_gpu
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


def ensure_binary(
    backend: str,
    version: str,
    binary_dir: Path,
    force_download: bool = False,
) -> Path:
    """
    Ensure the required binary is installed.
    Automatically selects CUDA version if GPU is available.

    Args:
        backend: Backend name ("llama.cpp", "vllm", etc.)
        version: Binary version (e.g., "b7822" or "latest")
        binary_dir: Directory to store binaries
        force_download: Re-download even if exists

    Returns:
        Path to binary executable

    Raises:
        UnsupportedPlatformError: If platform not supported
        BinaryNotFoundError: If download fails
    """
    # Detect platform
    system = platform.system()
    machine = platform.machine()
    base_platform_key = f"{system}-{machine}"
    
    # Check if CUDA is available - if so, prefer CUDA binary
    gpu_info = detect_gpu()
    has_cuda = gpu_info is not None and gpu_info.get("gpu_type") == "cuda"
    
    # Get optimal platform key (adds -cuda suffix if CUDA available and supported)
    platform_key = get_optimal_platform_key(backend, version, base_platform_key, has_cuda)
    
    if platform_key != base_platform_key:
        logger.info(f"CUDA GPU detected! Using GPU-accelerated binary: {platform_key}")

    # Get binary info from registry
    binary_info = get_binary_info(backend, version, platform_key)
    
    # Fallback to base platform if CUDA version not found
    if not binary_info and platform_key != base_platform_key:
        logger.warning(f"CUDA binary not available for {platform_key}, falling back to CPU")
        platform_key = base_platform_key
        binary_info = get_binary_info(backend, version, platform_key)

    if not binary_info:
        available = get_supported_platforms(backend, version)
        if not available:
            raise BinaryNotFoundError(f"No binary found for {backend} version {version}")
        raise UnsupportedPlatformError(
            f"Platform {platform_key} not supported. Available: {available}"
        )

    url = binary_info["url"]
    archive_type = binary_info["archive_type"]
    binary_name = binary_info["binary_name"]
    gpu_type = binary_info.get("gpu_type", "cpu")

    # Use different directory for CUDA vs CPU binaries to avoid conflicts
    if gpu_type == "cuda":
        actual_binary_dir = binary_dir / "cuda"
    else:
        actual_binary_dir = binary_dir / "cpu"
    
    binary_path = actual_binary_dir / binary_name

    # Check if already exists with required shared libraries
    if binary_path.exists() and not force_download:
        # Check for CUDA-specific libraries if this is a CUDA binary
        if gpu_type == "cuda":
            cuda_dll = list(actual_binary_dir.glob("*cuda*.dll")) + list(actual_binary_dir.glob("*cuda*.so*"))
            if not cuda_dll:
                logger.info(f"CUDA binary exists but CUDA libraries missing, re-downloading...")
            else:
                logger.info(f"CUDA binary already exists: {binary_path}")
                return binary_path
        elif system == "Linux":
            # Check for any .so files in the binary directory
            so_files = list(actual_binary_dir.glob("*.so*"))
            if not so_files:
                logger.info(f"Binary exists but shared libraries missing, re-downloading...")
            else:
                logger.info(f"Binary already exists: {binary_path}")
                return binary_path
        else:
            logger.info(f"Binary already exists: {binary_path}")
            return binary_path

    # Download and extract binary
    logger.info(f"Downloading {backend} ({gpu_type.upper()}) binary from {url}")
    actual_binary_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download main binary
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{archive_type}") as tmp:
            tmp_path = Path(tmp.name)

        logger.info(f"Downloading to temp file: {tmp_path}")
        urllib.request.urlretrieve(url, tmp_path)

        # Extract based on archive type
        if archive_type == "zip":
            _extract_zip(tmp_path, actual_binary_dir, binary_name)
        elif archive_type == "tar.gz":
            _extract_tarball(tmp_path, actual_binary_dir, binary_name)
        elif archive_type == "exe":
            # Direct executable, just move it
            shutil.move(tmp_path, binary_path)
        else:
            raise BinaryNotFoundError(f"Unknown archive type: {archive_type}")

        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()
            
        # Check for separate DLL download (Windows CUDA)
        dll_url = binary_info.get("dll_url")
        if dll_url:
            logger.info(f"Downloading required DLLs from {dll_url}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_dll:
                tmp_dll_path = Path(tmp_dll.name)
            
            urllib.request.urlretrieve(dll_url, tmp_dll_path)
            # DLL zip usually has libs in specific folder, extract flat
            _extract_zip(tmp_dll_path, actual_binary_dir, "non-existent-file-to-force-extract-all")
            
            if tmp_dll_path.exists():
                tmp_dll_path.unlink()

        # Make executable on Unix
        if system != "Windows":
            binary_path.chmod(0o755)

        if not binary_path.exists():
            raise BinaryNotFoundError(
                f"Binary {binary_name} not found after extraction. "
                "The archive structure may have changed."
            )

        logger.info(f"Binary installed: {binary_path} ({gpu_type.upper()})")
        return binary_path

    except Exception as e:
        # Clean up on failure
        if "tmp_path" in locals() and tmp_path.exists():
            tmp_path.unlink()
        raise BinaryNotFoundError(f"Failed to download/extract binary: {e}") from e


def _extract_zip(zip_path: Path, output_dir: Path, binary_name: str) -> None:
    """Extract binary from zip archive."""
    logger.info(f"Extracting zip archive: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Extract all files first
        zf.extractall(output_dir)

        # Find and move the binary to the root of output_dir
        binary_found = False
        for name in zf.namelist():
            if name.endswith(binary_name):
                logger.info(f"Found binary in archive: {name}")
                extracted = output_dir / name
                target = output_dir / binary_name

                if extracted != target and extracted.exists():
                    if target.exists():
                        target.unlink()
                    shutil.move(str(extracted), str(target))
                binary_found = True
                break

        if not binary_found:
            logger.warning(f"Binary {binary_name} not found in archive")
            logger.info(f"Extracted contents: {list(zf.namelist())[:10]}...")

        # Copy all shared libraries (.so, .dll) to the output_dir root
        # This ensures they're found alongside the binary at runtime
        for name in zf.namelist():
            if name.endswith(".so") or ".so." in name or name.endswith(".dll"):
                src = output_dir / name
                if src.exists() and src.is_file():
                    dst = output_dir / src.name
                    if src != dst:
                        if dst.exists():
                            dst.unlink()
                        shutil.copy2(str(src), str(dst))
                        logger.debug(f"Copied library: {src.name}")


def _extract_tarball(tar_path: Path, output_dir: Path, binary_name: str) -> None:
    """Extract binary from tar.gz archive."""
    logger.info(f"Extracting tarball: {tar_path}")

    with tarfile.open(tar_path, "r:gz") as tf:
        # Extract all files first
        tf.extractall(output_dir)

        # Find and move the binary to the root of output_dir
        binary_found = False
        for member in tf.getmembers():
            if member.name.endswith(binary_name):
                logger.info(f"Found binary in archive: {member.name}")
                extracted = output_dir / member.name
                target = output_dir / binary_name

                if extracted != target and extracted.exists():
                    if target.exists():
                        target.unlink()
                    shutil.move(str(extracted), str(target))
                binary_found = True
                break

        if not binary_found:
            logger.warning(f"Binary {binary_name} not found in archive")
            members = tf.getnames()
            logger.info(f"Extracted contents: {members[:10]}...")

        # Copy all shared libraries (.so) to the output_dir root
        for member in tf.getmembers():
            if member.name.endswith(".so") or ".so." in member.name:
                src = output_dir / member.name
                if src.exists() and src.is_file():
                    dst = output_dir / src.name
                    if src != dst:
                        if dst.exists():
                            dst.unlink()
                        shutil.copy2(str(src), str(dst))
                        logger.debug(f"Copied library: {src.name}")
