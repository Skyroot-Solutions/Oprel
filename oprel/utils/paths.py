"""
Path resolution and validation utilities
"""

import os
from pathlib import Path
from typing import Optional

from oprel.utils.logging import get_logger

logger = get_logger(__name__)


def expand_path(path: str | Path) -> Path:
    """
    Expand environment variables and user home in path.

    Args:
        path: Path with potential ~, $VAR, or ${VAR} references

    Returns:
        Fully resolved absolute path

    Examples:
        >>> expand_path("~/models")
        Path("/home/user/models")
        >>> expand_path("$HOME/.cache/oprel")
        Path("/home/user/.cache/oprel")
    """
    path_str = str(path)

    # Expand environment variables
    path_str = os.path.expandvars(path_str)

    # Expand user home directory
    path_obj = Path(path_str).expanduser()

    # Return absolute path
    return path_obj.resolve()


def ensure_dir(path: Path, create: bool = True) -> Path:
    """
    Ensure a directory exists, optionally creating it.

    Args:
        path: Directory path
        create: If True, create directory if it doesn't exist

    Returns:
        The directory path

    Raises:
        FileNotFoundError: If directory doesn't exist and create=False
    """
    if not path.exists():
        if create:
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {path}")
        else:
            raise FileNotFoundError(f"Directory does not exist: {path}")

    elif not path.is_dir():
        raise NotADirectoryError(f"Path exists but is not a directory: {path}")

    return path


def validate_file_exists(path: Path, extensions: Optional[list[str]] = None) -> Path:
    """
    Validate that a file exists and optionally has correct extension.

    Args:
        path: File path to validate
        extensions: Allowed extensions (e.g., [".gguf", ".bin"])

    Returns:
        The validated path

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If extension is invalid
    """
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if extensions:
        if path.suffix.lower() not in [ext.lower() for ext in extensions]:
            raise ValueError(
                f"Invalid file extension: {path.suffix}. " f"Expected one of: {extensions}"
            )

    return path


def get_file_size_mb(path: Path) -> float:
    """
    Get file size in megabytes.

    Args:
        path: File path

    Returns:
        Size in MB (rounded to 2 decimals)
    """
    if not path.exists():
        return 0.0

    size_bytes = path.stat().st_size
    return round(size_bytes / (1024 * 1024), 2)


def get_available_disk_space_gb(path: Path) -> float:
    """
    Get available disk space at path location.

    Args:
        path: Path to check (file or directory)

    Returns:
        Available space in GB
    """
    stat = os.statvfs(path.parent if path.is_file() else path)

    # Available space = fragment size * available fragments
    available_bytes = stat.f_bavail * stat.f_frsize
    return round(available_bytes / (1024**3), 2)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing invalid characters.

    Args:
        filename: Potentially unsafe filename

    Returns:
        Safe filename with invalid chars replaced

    Examples:
        >>> sanitize_filename("my/model:v1.0")
        "my_model_v1.0"
    """
    # Characters not allowed in filenames
    invalid_chars = '<>:"/\\|?*'

    safe_name = filename
    for char in invalid_chars:
        safe_name = safe_name.replace(char, "_")

    # Remove leading/trailing dots and spaces
    safe_name = safe_name.strip(". ")

    # Ensure not empty
    if not safe_name:
        safe_name = "unnamed"

    return safe_name


def get_relative_path(path: Path, base: Path) -> Path:
    """
    Get path relative to base directory.

    Args:
        path: Full path
        base: Base directory

    Returns:
        Relative path

    Examples:
        >>> get_relative_path(Path("/home/user/models/llama.gguf"), Path("/home/user"))
        Path("models/llama.gguf")
    """
    try:
        return path.relative_to(base)
    except ValueError:
        # If not relative, return absolute
        return path


def find_files_by_pattern(
    directory: Path,
    pattern: str,
    recursive: bool = True,
    max_depth: Optional[int] = None,
) -> list[Path]:
    """
    Find files matching a glob pattern.

    Args:
        directory: Directory to search
        pattern: Glob pattern (e.g., "*.gguf", "**/*.bin")
        recursive: If True, search subdirectories
        max_depth: Maximum directory depth (None = unlimited)

    Returns:
        List of matching file paths
    """
    if not directory.exists():
        return []

    if recursive and max_depth is None:
        # Use rglob for unlimited recursive search
        return list(directory.rglob(pattern))

    elif recursive and max_depth:
        # Manual depth-limited recursive search
        matches = []

        def search(current_dir: Path, depth: int):
            if depth > max_depth:
                return

            # Search current level
            matches.extend(current_dir.glob(pattern))

            # Recurse into subdirectories
            for subdir in current_dir.iterdir():
                if subdir.is_dir():
                    search(subdir, depth + 1)

        search(directory, 0)
        return matches

    else:
        # Non-recursive search
        return list(directory.glob(pattern))


def is_safe_path(path: Path, allowed_base: Path) -> bool:
    """
    Check if a path is safe (within allowed base directory).
    Prevents directory traversal attacks.

    Args:
        path: Path to validate
        allowed_base: Base directory that path must be under

    Returns:
        True if path is safe, False otherwise

    Examples:
        >>> is_safe_path(Path("/cache/models/llama.gguf"), Path("/cache"))
        True
        >>> is_safe_path(Path("/etc/passwd"), Path("/cache"))
        False
    """
    try:
        # Resolve both paths to absolute
        abs_path = path.resolve()
        abs_base = allowed_base.resolve()

        # Check if path is relative to base
        abs_path.relative_to(abs_base)
        return True
    except ValueError:
        return False
