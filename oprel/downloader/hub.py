"""
HuggingFace Hub integration for model downloads
"""

from pathlib import Path
from typing import Optional
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import HfHubHTTPError

from oprel.core.exceptions import ModelNotFoundError, InvalidQuantizationError
from oprel.downloader.cache import get_cache_path
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


def download_model(
    model_id: str,
    quantization: str = "Q4_K_M",
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> Path:
    """
    Download a model from HuggingFace Hub.

    Args:
        model_id: Repository ID (e.g., "TheBloke/Llama-2-7B-GGUF")
        quantization: Quantization level (Q4_K_M, Q5_K_M, Q8_0, etc.)
        cache_dir: Custom cache directory
        force_download: Skip cache and re-download

    Returns:
        Path to downloaded model file

    Raises:
        ModelNotFoundError: If model or quantization not found
    """
    cache_dir = cache_dir or get_cache_path()

    try:
        # List available files in the repo
        logger.info(f"Searching for {quantization} version of {model_id}")
        files = list_repo_files(model_id)

        # Find matching GGUF file
        matching_files = [
            f for f in files if f.endswith(".gguf") and quantization.lower() in f.lower()
        ]

        if not matching_files:
            available = [f for f in files if f.endswith(".gguf")]
            raise InvalidQuantizationError(
                f"No {quantization} quantization found. Available: {available}"
            )

        # Use the first match (usually there's only one)
        filename = matching_files[0]

        # Try to get from local cache first
        try:
            model_path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                cache_dir=str(cache_dir),
                local_files_only=True,
            )
            logger.info(f"Using cached model: {model_path}")
            return Path(model_path)
        except Exception:
            # Not in cache, need to download
            pass

        logger.info(f"Downloading: {filename}")

        # Download with resume support
        model_path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            cache_dir=str(cache_dir),
            resume_download=True,
            force_download=force_download,
        )

        logger.info(f"Model downloaded to: {model_path}")
        return Path(model_path)

    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            raise ModelNotFoundError(f"Model not found: {model_id}") from e
        raise ModelNotFoundError(f"Failed to download model: {e}") from e
    except Exception as e:
        raise ModelNotFoundError(f"Unexpected error downloading model: {e}") from e


def list_available_quantizations(model_id: str) -> list[str]:
    """
    List all available quantization levels for a model.

    Args:
        model_id: HuggingFace model repository ID

    Returns:
        List of quantization strings (e.g., ["Q4_K_M", "Q5_K_M", "Q8_0"])
    """
    try:
        files = list_repo_files(model_id)
        gguf_files = [f for f in files if f.endswith(".gguf")]

        # Extract quantization from filenames
        # Example: "llama-2-7b.Q4_K_M.gguf" -> "Q4_K_M"
        quantizations = []
        for f in gguf_files:
            parts = f.replace(".gguf", "").split(".")
            if len(parts) >= 2:
                quantizations.append(parts[-1].upper())

        return sorted(set(quantizations))

    except Exception as e:
        logger.warning(f"Could not list quantizations for {model_id}: {e}")
        return []
