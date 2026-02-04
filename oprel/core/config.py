"""
Global configuration management

Production-ready configuration for Oprel SDK.
Includes all settings for Week 1 features.
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
    
    # Memory Optimization (key differentiator from Ollama)
    kv_cache_type: str = Field(
        default="f16", 
        description="KV cache precision: f16 (default), q8_0 (50% savings), q4_0 (75% savings)"
    )
    flash_attention: bool = Field(
        default=True, 
        description="Use Flash Attention for memory efficiency and speed"
    )
    mmap: bool = Field(
        default=True, 
        description="Use memory-mapped model loading for faster startup"
    )

    # Networking
    default_port_range: tuple[int, int] = Field(
        default=(54321, 54420), description="Port range for HTTP servers"
    )

    # Timeouts (M1.11 - Production-ready timeouts)
    connect_timeout_sec: float = Field(
        default=10.0, description="Connection timeout in seconds"
    )
    read_timeout_sec: float = Field(
        default=300.0, description="Read timeout for non-streaming requests (CPU can be slow)"
    )
    stream_timeout_sec: float = Field(
        default=120.0, description="Per-chunk timeout for streaming responses"
    )
    startup_timeout_sec: int = Field(
        default=60, description="Maximum seconds to wait for backend startup"
    )

    # Monitoring
    health_check_interval_sec: float = Field(
        default=1.0, description="How often to check process health"
    )
    health_check_timeout_sec: float = Field(
        default=5.0, description="Timeout for health check requests"
    )

    # Auto-restart settings (M1.8)
    auto_restart: bool = Field(
        default=False, description="Automatically restart backend on crash"
    )
    max_restarts: int = Field(
        default=3, description="Maximum restart attempts before giving up"
    )
    restart_delay_sec: float = Field(
        default=2.0, description="Delay between restart attempts"
    )

    # VRAM Monitoring (M1.4)
    vram_monitor_enabled: bool = Field(
        default=True, description="Enable VRAM monitoring during model loading"
    )
    vram_warning_threshold: float = Field(
        default=85.0, description="VRAM utilization % to trigger warning"
    )
    vram_critical_threshold: float = Field(
        default=95.0, description="VRAM utilization % to trigger critical alert"
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



