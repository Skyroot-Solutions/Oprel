"""
ComfyUI installation and management.

Similar to llama.cpp binary downloader, this manages ComfyUI as an embedded engine.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from oprel.utils.logging import get_logger
from oprel.core.config import Config

logger = get_logger(__name__)


def get_comfyui_dir() -> Path:
    """Get ComfyUI installation directory."""
    config = Config()
    return Path(config.cache_dir) / "comfyui"


def is_comfyui_installed() -> bool:
    """Check if ComfyUI is installed."""
    comfyui_dir = get_comfyui_dir()
    main_py = comfyui_dir / "main.py"
    return main_py.exists()


def install_comfyui(force: bool = False) -> Path:
    """
    Install ComfyUI to cache directory.
    
    Args:
        force: Force reinstall even if already installed
        
    Returns:
        Path to ComfyUI directory
    """
    comfyui_dir = get_comfyui_dir()
    
    if is_comfyui_installed() and not force:
        logger.info(f"ComfyUI already installed at {comfyui_dir}")
        return comfyui_dir
    
    logger.info("Installing ComfyUI...")
    
    # Create directory
    comfyui_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if PyTorch has CUDA support
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        logger.info(f"PyTorch CUDA available: {has_cuda}")
        
        if not has_cuda:
            logger.warning("PyTorch doesn't have CUDA support. Installing PyTorch with CUDA...")
            logger.info("This may take a few minutes...")
            
            # Uninstall CPU-only PyTorch
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"],
                capture_output=True
            )
            
            # Install PyTorch with CUDA 12.1
            subprocess.run(
                [sys.executable, "-m", "pip", "install", 
                 "torch", "torchvision", "torchaudio",
                 "--index-url", "https://download.pytorch.org/whl/cu121"],
                check=True
            )
            logger.info("✓ PyTorch with CUDA installed")
    except ImportError:
        logger.info("PyTorch not found, will be installed with ComfyUI dependencies")
    
    # Clone ComfyUI repository
    logger.info("Cloning ComfyUI repository...")
    try:
        subprocess.run(
            ["git", "clone", "https://github.com/comfyanonymous/ComfyUI", str(comfyui_dir)],
            check=True,
            capture_output=True
        )
    except subprocess.CalledProcessError as e:
        # If directory exists, try to update
        if (comfyui_dir / ".git").exists():
            logger.info("Updating existing ComfyUI installation...")
            subprocess.run(
                ["git", "pull"],
                cwd=str(comfyui_dir),
                check=True
            )
        else:
            raise RuntimeError(f"Failed to clone ComfyUI: {e.stderr.decode()}")
    
    # Install dependencies
    logger.info("Installing ComfyUI dependencies...")
    requirements_file = comfyui_dir / "requirements.txt"
    
    if requirements_file.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            check=True
        )
    
    # Ensure PyTorch has CUDA after installation
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("⚠ PyTorch still doesn't have CUDA. Installing CUDA-enabled version...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--force-reinstall",
                 "torch", "torchvision", "torchaudio",
                 "--index-url", "https://download.pytorch.org/whl/cu121"],
                check=True
            )
    except Exception as e:
        logger.error(f"Failed to verify PyTorch CUDA: {e}")
    
    logger.info(f"✓ ComfyUI installed to {comfyui_dir}")
    return comfyui_dir


def get_comfyui_models_dir(model_type: str = "checkpoints") -> Path:
    """
    Get ComfyUI models directory.
    
    Args:
        model_type: Type of model (checkpoints, loras, vaes, etc.)
        
    Returns:
        Path to models directory
    """
    comfyui_dir = get_comfyui_dir()
    models_dir = comfyui_dir / "models" / model_type
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def download_checkpoint(
    repo_id: str,
    filename: str,
    model_type: str = "checkpoints"
) -> Path:
    """
    Download a ComfyUI checkpoint from HuggingFace.
    
    Args:
        repo_id: HuggingFace repo ID (e.g., "black-forest-labs/FLUX.1-schnell")
        filename: Model filename (e.g., "flux1-schnell.safetensors")
        model_type: Model type directory
        
    Returns:
        Path to downloaded model
    """
    from huggingface_hub import hf_hub_download
    
    models_dir = get_comfyui_models_dir(model_type)
    output_path = models_dir / filename
    
    if output_path.exists():
        logger.info(f"Model already exists: {filename}")
        return output_path
    
    logger.info(f"Downloading {filename} from {repo_id}...")
    
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(models_dir),
        local_dir_use_symlinks=False
    )
    
    logger.info(f"✓ Downloaded to {output_path}")
    return Path(downloaded_path)


def list_installed_checkpoints() -> list[str]:
    """List all installed checkpoint models."""
    checkpoints_dir = get_comfyui_models_dir("checkpoints")
    
    if not checkpoints_dir.exists():
        return []
    
    models = []
    for file in checkpoints_dir.glob("*.safetensors"):
        models.append(file.name)
    for file in checkpoints_dir.glob("*.ckpt"):
        models.append(file.name)
    
    return sorted(models)


def ensure_comfyui_ready() -> Path:
    """
    Ensure ComfyUI is installed and ready.
    
    Returns:
        Path to ComfyUI directory
    """
    if not is_comfyui_installed():
        logger.info("ComfyUI not found, installing...")
        return install_comfyui()
    
    return get_comfyui_dir()
