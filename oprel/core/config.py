"""
Global configuration management
"""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class Config(BaseModel):
    """
    Global configuration for Oprel SDK.

    Can be customized per-model or set globally.
    """

    # Paths
    cache_dir: Path = Field(default_factory=lambda: Path.home() / ".cache" / "oprel" / "models")
    binary_dir: Path = Field(default_factory=lambda: Path.home() / ".cache" / "oprel" / "bin")

    # Memory limits
    default_max_memory_mb: int = Field(
        default=8192, description="Default max memory per model in MB"
    )

    # Performance
    use_unix_socket: bool = Field(
        default=True, description="Use Unix sockets instead of HTTP (Linux/Mac only)"
    )
    n_threads: Optional[int] = Field(default=None, description="CPU threads (None for auto-detect)")
    n_gpu_layers: int = Field(default=-1, description="GPU layers to offload (-1 for auto)")
    ctx_size: int = Field(default=4096, description="Context size in tokens")
    batch_size: int = Field(default=512, description="Batch size for processing")

    # Networking
    default_port_range: tuple[int, int] = Field(
        default=(54321, 54420), description="Port range for HTTP servers"
    )

    # Monitoring
    health_check_interval_sec: float = Field(
        default=1.0, description="How often to check process health"
    )

    # Logging
    log_level: str = Field(
        default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )

    # Binary management
    auto_install_binaries: bool = Field(
        default=True, description="Automatically download backend binaries"
    )
    binary_version: str = Field(default="b7822", description="llama.cpp binary version to use")

    class Config:
        arbitrary_types_allowed = True

    def ensure_dirs(self) -> None:
        """Create necessary directories if they don't exist"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.binary_dir.mkdir(parents=True, exist_ok=True)
