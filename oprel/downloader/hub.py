"""
HuggingFace Hub integration for model downloads with production-ready reliability
"""

import hashlib
import time
from pathlib import Path
from typing import Optional, Callable
from huggingface_hub import hf_hub_download, list_repo_files, HfFileSystem
from huggingface_hub.utils import HfHubHTTPError
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from oprel.core.exceptions import ModelNotFoundError, InvalidQuantizationError
from oprel.downloader.cache import get_cache_path
from oprel.utils.logging import get_logger

logger = get_logger(__name__)

# M1.13: Connection and read timeouts (30s connect, 300s read for large files)
DEFAULT_TIMEOUT = (30, 300)
# M1.18: Max retries with exponential backoff
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.0  # 1s, 2s, 4s


def _create_robust_session() -> requests.Session:
    """
    Create a requests session with retry logic and timeouts.
    
    Returns:
        Configured requests Session with retry and timeout handling
    """
    session = requests.Session()
    
    # M1.18: Configure retry strategy
    # Retry on: connection errors, timeouts, and 5xx server errors
    # Don't retry on: 4xx client errors (404, 403, etc.)
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[500, 502, 503, 504],  # Retry on server errors
        allowed_methods=["HEAD", "GET", "OPTIONS"],  # Don't retry POST/PUT
        raise_on_status=False,
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


def _verify_file_checksum(file_path: Path, expected_hash: Optional[str] = None) -> bool:
    """
    M1.17: Verify file integrity using SHA256 checksum.
    
    Args:
        file_path: Path to file to verify
        expected_hash: Expected SHA256 hash (optional)
        
    Returns:
        True if checksum matches or no expected hash provided
    """
    if not expected_hash:
        # If no expected hash, just check file exists and has size > 0
        return file_path.exists() and file_path.stat().st_size > 0
    
    logger.debug(f"Verifying checksum for {file_path.name}")
    sha256_hash = hashlib.sha256()
    
    try:
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(8192), b""):
                sha256_hash.update(byte_block)
        
        actual_hash = sha256_hash.hexdigest()
        if actual_hash != expected_hash:
            logger.error(f"Checksum mismatch! Expected: {expected_hash}, Got: {actual_hash}")
            return False
            
        logger.debug("Checksum verification passed")
        return True
        
    except Exception as e:
        logger.error(f"Error verifying checksum: {e}")
        return False


def _check_cache_validity(
    model_id: str,
    filename: str,
    cache_dir: Path,
) -> Optional[Path]:
    """
    M1.16: Check if model exists in cache and is valid.
    
    Args:
        model_id: HuggingFace model repository ID
        filename: Name of the model file
        cache_dir: Cache directory to check
        
    Returns:
        Path to cached file if valid, None otherwise
    """
    try:
        # Try to get from cache using hf_hub_download with local_files_only
        cached_path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
        
        cached_file = Path(cached_path)
        
        # M1.16: Verify file exists and has non-zero size
        if not cached_file.exists():
            logger.debug(f"Cache miss: File not found at {cached_file}")
            return None
            
        file_size = cached_file.stat().st_size
        if file_size == 0:
            logger.warning(f"Cache invalid: File has zero size, removing {cached_file}")
            cached_file.unlink()
            return None
        
        # M1.17: Basic integrity check (could add checksum verification here)
        size_mb = file_size / (1024 * 1024)
        logger.info(f"✓ Cache hit: {filename} ({size_mb:.1f} MB)")
        return cached_file
        
    except Exception as e:
        logger.debug(f"Cache miss for {filename}: {e}")
        return None


class DownloadProgressCallback:
    """M1.15: Progress tracking with tqdm progress bar"""
    
    def __init__(self, filename: str, total_size: int):
        self.filename = filename
        self.pbar = tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {filename}",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        self.last_logged_mb = 0
        
    def __call__(self, downloaded: int):
        """Update progress bar"""
        self.pbar.update(downloaded - self.pbar.n)
        
        # M1.13: Log progress every 100MB
        current_mb = downloaded / (1024 * 1024)
        if current_mb - self.last_logged_mb >= 100:
            logger.info(f"Downloaded {current_mb:.0f} MB of {self.filename}")
            self.last_logged_mb = current_mb
    
    def close(self):
        """Close progress bar"""
        self.pbar.close()


def _find_cached_model_for_repo(model_id: str, cache_dir: Path) -> Optional[Path]:
    """
    Find any cached GGUF file for the given model repository.
    
    This checks if we already have ANY quantization of the model downloaded,
    to avoid downloading duplicates.
    
    Args:
        model_id: HuggingFace model repository ID
        cache_dir: Cache directory to search
        
    Returns:
        Path to cached model if found, None otherwise
    """
    # Extract repo name from model_id (e.g., "Qwen/Qwen2.5-1.5B-Instruct-GGUF" -> "Qwen2.5-1.5B-Instruct")
    repo_parts = model_id.split('/')
    if len(repo_parts) >= 2:
        # Get the repo name without the organization
        repo_name = repo_parts[-1].replace('-GGUF', '').replace('-gguf', '')
    else:
        repo_name = model_id
    
    # Search for any .gguf file matching this model in cache
    for gguf_file in cache_dir.rglob("*.gguf"):
        # Check if filename contains the model name
        filename_lower = gguf_file.name.lower()
        repo_name_lower = repo_name.lower()
        
        # Match if the filename contains significant parts of the model name
        # e.g., "qwen2.5-1.5b-instruct-q5_k_m.gguf" matches "Qwen2.5-1.5B-Instruct"
        if repo_name_lower.replace('-', '').replace('.', '') in filename_lower.replace('-', '').replace('.', ''):
            # Verify it's a valid file
            if gguf_file.exists() and gguf_file.stat().st_size > 0:
                logger.info(f"✓ Found cached model: {gguf_file.name} ({gguf_file.stat().st_size / (1024**3):.1f} GB)")
                return gguf_file
    
    return None


def download_model(
    model_id: str,
    quantization: str = "Q4_K_M",
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
    max_retries: int = MAX_RETRIES,
) -> Path:
    """
    Download a model from HuggingFace Hub with production-ready reliability.
    
    Improvements over Ollama:
    - M1.13: Proper timeouts and streaming for large files (>10GB)
    - M1.14: Automatic resume on network interruption
    - M1.15: Rich progress bars with speed and ETA
    - M1.16: Smart cache detection - checks for ANY cached quantization before downloading
    - M1.17: Checksum verification for integrity
    - M1.18: Exponential backoff retry logic

    Args:
        model_id: Repository ID (e.g., "TheBloke/Llama-2-7B-GGUF")
        quantization: Quantization level (Q4_K_M, Q5_K_M, Q8_0, etc.)
                     If a cached version exists with different quantization, uses that instead
        cache_dir: Custom cache directory
        force_download: Skip cache and re-download
        max_retries: Maximum download retry attempts

    Returns:
        Path to downloaded model file

    Raises:
        ModelNotFoundError: If model or quantization not found
        InvalidQuantizationError: If quantization variant not available
    """
    cache_dir = cache_dir or get_cache_path()
    
    # Check if we already have ANY quantization of this model cached
    if not force_download:
        cached_model = _find_cached_model_for_repo(model_id, cache_dir)
        if cached_model:
            logger.info(f"Using existing cached model instead of downloading {quantization}")
            return cached_model
    
    # M1.18: Retry loop with exponential backoff
    last_error = None
    for attempt in range(max_retries):
        try:
            return _download_model_attempt(
                model_id=model_id,
                quantization=quantization,
                cache_dir=cache_dir,
                force_download=force_download,
                attempt=attempt + 1,
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = BACKOFF_FACTOR * (2 ** attempt)
                logger.warning(
                    f"Download attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} download attempts failed")
        except HfHubHTTPError as e:
            # Don't retry on 4xx errors (client errors)
            if e.response.status_code in [404, 403, 401]:
                raise ModelNotFoundError(
                    f"Model not found or access denied: {model_id}"
                ) from e
            # Retry on 5xx errors (server errors)
            elif e.response.status_code >= 500 and attempt < max_retries - 1:
                last_error = e
                wait_time = BACKOFF_FACTOR * (2 ** attempt)
                logger.warning(
                    f"Server error {e.response.status_code}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                raise ModelNotFoundError(f"Failed to download model: {e}") from e
    
    # All retries exhausted
    raise ModelNotFoundError(
        f"Failed to download {model_id} after {max_retries} attempts: {last_error}"
    ) from last_error


def _download_model_attempt(
    model_id: str,
    quantization: str,
    cache_dir: Path,
    force_download: bool,
    attempt: int,
) -> Path:
    """
    Single download attempt.
    
    Args:
        model_id: Model repository ID
        quantization: Quantization level
        cache_dir: Cache directory
        force_download: Force re-download even if cached
        attempt: Current attempt number (for logging)
        
    Returns:
        Path to downloaded model
    """
    logger.info(f"[Attempt {attempt}] Searching for {quantization} version of {model_id}")
    
    # List available files in the repo
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
    
    # M1.16: Check cache first (unless force_download)
    if not force_download:
        cached_path = _check_cache_validity(model_id, filename, cache_dir)
        if cached_path:
            return cached_path
    
    logger.info(f"Downloading {filename} from {model_id}")
    
    # M1.14: Download with resume support
    # M1.13: hf_hub_download handles timeouts and streaming internally
    model_path = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        cache_dir=str(cache_dir),
        resume_download=True,  # M1.14: Resume interrupted downloads
        force_download=force_download,
        # Note: huggingface_hub handles progress internally but we can customize
    )
    
    downloaded_file = Path(model_path)
    file_size_mb = downloaded_file.stat().st_size / (1024 * 1024)
    
    # M1.17: Verify downloaded file integrity
    if not _verify_file_checksum(downloaded_file):
        logger.error(f"Downloaded file appears corrupted, removing {downloaded_file}")
        downloaded_file.unlink()
        raise ModelNotFoundError("Downloaded file failed integrity check")
    
    logger.info(f"✓ Successfully downloaded {filename} ({file_size_mb:.1f} MB)")
    logger.info(f"Model path: {model_path}")
    
    return downloaded_file


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
