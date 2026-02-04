"""
Runtime package for Oprel SDK

Contains:
- process.py: Subprocess management with production features
- cuda_errors.py: CUDA error code translation
- monitor.py: Process health monitoring
- backends/: Backend implementations (llama.cpp, etc.)
- binaries/: Binary management and installation
"""

from oprel.runtime.process import ModelProcess, kill_all_oprel_processes
from oprel.runtime.cuda_errors import (
    CudaErrorHandler,
    translate_exit_code,
    translate_cuda_error,
    get_gpu_troubleshooting_tips,
)
from oprel.runtime.monitor import ProcessMonitor

__all__ = [
    "ModelProcess",
    "kill_all_oprel_processes",
    "CudaErrorHandler",
    "translate_exit_code",
    "translate_cuda_error",
    "get_gpu_troubleshooting_tips",
    "ProcessMonitor",
]