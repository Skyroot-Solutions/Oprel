"""
Subprocess management for model backends
"""

import subprocess
import socket
import time
from pathlib import Path
from typing import Optional

from oprel.core.config import Config
from oprel.core.exceptions import BackendError
from oprel.runtime.backends.base import BaseBackend
from oprel.runtime.backends.llama_cpp import LlamaCppBackend
from oprel.runtime.binaries.installer import ensure_binary
from oprel.telemetry.hardware import detect_gpu
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


class ModelProcess:
    """
    Manages a model backend subprocess.
    Handles startup, health checks, and graceful shutdown.
    """

    def __init__(
        self,
        model_path: Path,
        backend: str = "llama.cpp",
        config: Optional[Config] = None,
    ):
        self.model_path = model_path
        self.backend_name = backend
        self.config = config or Config()

        # Runtime state
        self.process: Optional[subprocess.Popen] = None
        self.port: Optional[int] = None
        self.socket_path: Optional[Path] = None
        self._backend: Optional[BaseBackend] = None

    def _log_model_info(self) -> None:
        """Log minimal model information."""
        try:
            model_size_bytes = self.model_path.stat().st_size
            model_size_gb = model_size_bytes / (1024**3)
            logger.info(f"Loading model: {self.model_path.name} ({model_size_gb:.2f} GB)")
        except Exception as e:
            logger.debug(f"Could not log model info: {e}")

    def start(self) -> None:
        """
        Start the model backend process.

        Raises:
            BackendError: If process fails to start
        """
        # Ensure backend binary is installed
        binary_path = ensure_binary(
            backend=self.backend_name,
            version=self.config.binary_version,
            binary_dir=self.config.binary_dir,
        )

        # Select backend implementation
        if self.backend_name == "llama.cpp":
            self._backend = LlamaCppBackend(
                binary_path=binary_path,
                model_path=self.model_path,
                config=self.config,
            )
        else:
            raise BackendError(f"Unsupported backend: {self.backend_name}")

        # Log model information (like Ollama)
        self._log_model_info()

        # Find available port
        self.port = self._find_free_port()

        # Build command
        cmd = self._backend.build_command(port=self.port)
        logger.debug(f"Starting process: {' '.join(cmd)}")

        # Spawn process
        try:
            # Set up environment with library path for Linux
            import os
            import platform

            env = os.environ.copy()
            if platform.system() == "Linux":
                # Add binary directory to LD_LIBRARY_PATH so shared libs are found
                binary_dir = str(self.config.binary_dir)
                existing_path = env.get("LD_LIBRARY_PATH", "")
                if existing_path:
                    env["LD_LIBRARY_PATH"] = f"{binary_dir}:{existing_path}"
                else:
                    env["LD_LIBRARY_PATH"] = binary_dir
                logger.debug(f"Set LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}")

            # Windows: Hide the console window completely
            creation_flags = 0
            if platform.system() == "Windows":
                DETACHED_PROCESS = 0x00000008
                CREATE_NEW_PROCESS_GROUP = 0x00000200
                creation_flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                env=env,
                creationflags=creation_flags,
            )
        except Exception as e:
            raise BackendError(f"Failed to start process: {e}") from e

        # Wait for server to be ready
        if not self._wait_for_ready(timeout=60):
            self.stop()
            raise BackendError("Process failed to start within timeout")

        # Note: llama-server only supports HTTP, not Unix sockets
        # Keep socket_path as None to ensure HTTPClient is used
        self.socket_path = None

        logger.debug(f"Backend process started (PID: {self.process.pid}, Port: {self.port})")

    def is_running(self) -> bool:
        """Check if the backend process is still running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def stop(self) -> None:
        """Gracefully stop the process and all child processes"""
        if self.process:
            logger.debug(f"Stopping backend process (PID: {self.process.pid})")
            
            try:
                # Try to kill child processes first (if any)
                import psutil
                try:
                    parent = psutil.Process(self.process.pid)
                    children = parent.children(recursive=True)
                    
                    # Terminate children first
                    for child in children:
                        try:
                            child.terminate()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    
                    # Wait for children to terminate
                    _, alive = psutil.wait_procs(children, timeout=3)
                    
                    # Kill any that didn't terminate
                    for child in alive:
                        try:
                            child.kill()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Process might already be gone
                    pass
            except ImportError:
                # psutil not available, just terminate the main process
                pass
            
            # Now terminate the main process
            self.process.terminate()

            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Process did not terminate, killing...")
                self.process.kill()
                self.process.wait()

            self.process = None

    def _find_free_port(self) -> int:
        """Find an available port in the configured range"""
        start, end = self.config.default_port_range

        for port in range(start, end):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", port))
                    return port
            except OSError:
                continue

        raise BackendError(f"No free ports in range {start}-{end}")

    def _wait_for_ready(self, timeout: int = 60) -> bool:
        """
        Wait for the backend server to be ready.

        Args:
            timeout: Maximum seconds to wait

        Returns:
            True if ready, False if timeout
        """
        import requests

        start = time.time()
        health_url = f"http://127.0.0.1:{self.port}/health"

        logger.debug(f"Waiting for model to be ready (timeout: {timeout}s)...")

        while time.time() - start < timeout:
            # Check if process crashed
            if self.process and self.process.poll() is not None:
                logger.error(f"Process exited with code {self.process.returncode}")
                logger.error("Check that the model file is valid and llama-server is compatible")
                return False

            try:
                # Try to hit the health endpoint
                response = requests.get(health_url, timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    # llama-server returns {"status": "ok"} when ready
                    if data.get("status") == "ok":
                        logger.debug("Model backend ready")
                        return True
                    else:
                        # Model still loading
                        logger.debug(f"Health check response: {data}")
            except requests.exceptions.ConnectionError:
                # Server not listening yet
                pass
            except requests.exceptions.Timeout:
                # Server slow to respond
                pass
            except Exception as e:
                logger.debug(f"Health check error: {e}")

            time.sleep(1.0)

        logger.error(f"Server failed to become ready within {timeout}s")
        return False
