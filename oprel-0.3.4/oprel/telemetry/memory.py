"""
RAM detection and monitoring (M1.20)

Production-ready system memory management to prevent OOM crashes.
Beats Ollama by providing proactive memory monitoring and smart model size estimation.
"""

import psutil
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from pathlib import Path

from oprel.utils.logging import get_logger

logger = get_logger(__name__)

# Reserve memory for OS and other processes (in GB)
# This is conservative to prevent system freezing
OS_RESERVED_GB = 2.0

# Memory warning thresholds
MEMORY_WARNING_THRESHOLD = 0.85  # Warn at 85% usage
MEMORY_CRITICAL_THRESHOLD = 0.95  # Critical at 95% usage


@dataclass
class RAMSnapshot:
    """Point-in-time system RAM snapshot"""
    
    total_gb: float
    available_gb: float
    used_gb: float
    percent_used: float
    
    def is_critical(self) -> bool:
        """Returns True if RAM usage is critically high (>95%)"""
        return self.percent_used >= MEMORY_CRITICAL_THRESHOLD * 100
    
    def is_warning(self) -> bool:
        """Returns True if RAM usage is high (>85%)"""
        return self.percent_used >= MEMORY_WARNING_THRESHOLD * 100


def get_available_ram() -> float:
    """
    Get currently available system RAM in GB.
    
    Returns:
        Available RAM in gigabytes
    """
    try:
        mem = psutil.virtual_memory()
        # Use 'available' which is more accurate than 'free'
        # 'available' accounts for memory that can be freed if needed
        available_gb = mem.available / (1024 ** 3)
        
        logger.debug(f"Available RAM: {available_gb:.2f} GB")
        return available_gb
        
    except Exception as e:
        logger.error(f"Failed to get available RAM: {e}")
        # Return conservative estimate if detection fails
        return 4.0


def get_total_ram() -> float:
    """
    Get total system RAM in GB.
    
    Returns:
        Total RAM in gigabytes
    """
    try:
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024 ** 3)
        
        logger.debug(f"Total RAM: {total_gb:.2f} GB")
        return total_gb
        
    except Exception as e:
        logger.error(f"Failed to get total RAM: {e}")
        # Return conservative estimate if detection fails
        return 8.0


def get_ram_snapshot() -> RAMSnapshot:
    """
    Get detailed snapshot of current RAM usage.
    
    Returns:
        RAMSnapshot with current memory state
    """
    try:
        mem = psutil.virtual_memory()
        
        snapshot = RAMSnapshot(
            total_gb=mem.total / (1024 ** 3),
            available_gb=mem.available / (1024 ** 3),
            used_gb=mem.used / (1024 ** 3),
            percent_used=mem.percent,
        )
        
        return snapshot
        
    except Exception as e:
        logger.error(f"Failed to get RAM snapshot: {e}")
        # Return conservative fallback
        return RAMSnapshot(
            total_gb=8.0,
            available_gb=2.0,
            used_gb=6.0,
            percent_used=75.0,
        )


def estimate_max_model_size() -> float:
    """
    Estimate maximum model size that can be safely loaded into RAM.
    
    This accounts for:
    - OS reserved memory (2GB)
    - Model overhead (context cache, activations): ~20%
    - Safety margin to prevent OOM
    
    Returns:
        Estimated max model size in GB
        
    Example:
        16GB RAM -> ~10GB usable for models after reserves and overhead
    """
    try:
        total_ram_gb = get_total_ram()
        
        # Reserve memory for OS and other processes
        usable_ram_gb = total_ram_gb - OS_RESERVED_GB
        
        # Account for model overhead (KV cache, activations, etc.)
        # Models typically need ~20% extra memory beyond file size
        max_model_size_gb = usable_ram_gb * 0.8
        
        # Ensure we have at least 1GB minimum
        max_model_size_gb = max(1.0, max_model_size_gb)
        
        logger.debug(
            f"Estimated max model size: {max_model_size_gb:.1f}GB "
            f"(Total RAM: {total_ram_gb:.1f}GB, Reserved: {OS_RESERVED_GB}GB)"
        )
        
        return max_model_size_gb
        
    except Exception as e:
        logger.error(f"Failed to estimate max model size: {e}")
        # Return conservative minimum
        return 2.0


def check_ram_for_model(model_size_gb: float) -> Dict[str, any]:
    """
    Check if there's enough RAM to load a model safely.
    
    Args:
        model_size_gb: Model file size in gigabytes
        
    Returns:
        Dict with:
            - can_load: bool - Whether model can be loaded
            - warning: bool - Whether loading might be risky
            - available_gb: float - Currently available RAM
            - required_gb: float - RAM needed for model
            - message: str - Human-readable explanation
            
    Example:
        >>> check_ram_for_model(7.5)
        {
            'can_load': True,
            'warning': False,
            'available_gb': 12.3,
            'required_gb': 9.0,
            'message': 'Sufficient RAM available'
        }
    """
    try:
        # Get current RAM state
        available_gb = get_available_ram()
        
        # Model needs ~20% extra for overhead (KV cache, activations)
        required_gb = model_size_gb * 1.2
        
        # Can load if we have enough available RAM
        can_load = available_gb >= required_gb
        
        # Warning if we're cutting it close (within 2GB of limit)
        warning = can_load and (available_gb - required_gb < 2.0)
        
        if not can_load:
            message = (
                f"Insufficient RAM: Need {required_gb:.1f}GB, "
                f"only {available_gb:.1f}GB available. "
                f"Consider closing other applications or using a smaller model."
            )
        elif warning:
            message = (
                f"RAM is adequate but limited: {available_gb:.1f}GB available "
                f"for {required_gb:.1f}GB required. "
                f"Monitor memory usage during loading."
            )
        else:
            message = f"Sufficient RAM: {available_gb:.1f}GB available for {required_gb:.1f}GB required."
        
        return {
            "can_load": can_load,
            "warning": warning,
            "available_gb": available_gb,
            "required_gb": required_gb,
            "message": message,
        }
        
    except Exception as e:
        logger.error(f"Failed to check RAM for model: {e}")
        # Conservative fallback - assume we can try but warn
        return {
            "can_load": True,
            "warning": True,
            "available_gb": 0.0,
            "required_gb": model_size_gb * 1.2,
            "message": f"Could not verify RAM availability: {e}",
        }


def monitor_ram_usage() -> None:
    """
    Monitor current RAM usage and log warnings if usage is high.
    
    Call this periodically during model loading to detect memory pressure.
    """
    try:
        snapshot = get_ram_snapshot()
        
        if snapshot.is_critical():
            logger.error(
                f"CRITICAL: RAM usage at {snapshot.percent_used:.1f}%! "
                f"Used: {snapshot.used_gb:.1f}GB / {snapshot.total_gb:.1f}GB. "
                f"Risk of OOM crash!"
            )
        elif snapshot.is_warning():
            logger.warning(
                f"WARNING: High RAM usage at {snapshot.percent_used:.1f}%. "
                f"Used: {snapshot.used_gb:.1f}GB / {snapshot.total_gb:.1f}GB"
            )
        else:
            logger.debug(
                f"RAM usage: {snapshot.percent_used:.1f}% "
                f"({snapshot.used_gb:.1f}GB / {snapshot.total_gb:.1f}GB)"
            )
            
    except Exception as e:
        logger.error(f"Failed to monitor RAM usage: {e}")


def get_model_size_from_file(model_path: Path) -> float:
    """
    Get model file size in GB.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Model size in gigabytes
    """
    try:
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return 0.0
        
        size_bytes = model_path.stat().st_size
        size_gb = size_bytes / (1024 ** 3)
        
        logger.debug(f"Model size: {size_gb:.2f}GB ({model_path.name})")
        return size_gb
        
    except Exception as e:
        logger.error(f"Failed to get model size: {e}")
        return 0.0


def recommend_quantization_for_ram(available_ram_gb: float) -> Tuple[str, str]:
    """
    Recommend model quantization level based on available RAM.
    
    Args:
        available_ram_gb: Available system RAM in GB
        
    Returns:
        Tuple of (quantization_level, explanation)
        
    Example:
        >>> recommend_quantization_for_ram(16.0)
        ('Q8_0', 'You have enough RAM for high-quality 8-bit quantization')
    """
    if available_ram_gb >= 32:
        return (
            "Q8_0",
            "You have plenty of RAM. Q8_0 provides near-FP16 quality with minimal compression."
        )
    elif available_ram_gb >= 16:
        return (
            "Q5_K_M",
            "You have enough RAM for good quality. Q5_K_M balances quality and size well."
        )
    elif available_ram_gb >= 8:
        return (
            "Q4_K_M",
            "Limited RAM detected. Q4_K_M provides good quality at reasonable size."
        )
    else:
        return (
            "Q2_K",
            "Very limited RAM. Q2_K is highly compressed but quality may suffer. "
            "Consider using a smaller model (1B-3B parameters)."
        )


def print_ram_report() -> None:
    """
    Print a detailed report of system RAM to console.
    Useful for `oprel doctor` command.
    """
    try:
        snapshot = get_ram_snapshot()
        max_model_size = estimate_max_model_size()
        quant, quant_msg = recommend_quantization_for_ram(snapshot.available_gb)
        
        print("\n" + "="*60)
        print("SYSTEM MEMORY REPORT")
        print("="*60)
        print(f"Total RAM:           {snapshot.total_gb:.2f} GB")
        print(f"Used RAM:            {snapshot.used_gb:.2f} GB ({snapshot.percent_used:.1f}%)")
        print(f"Available RAM:       {snapshot.available_gb:.2f} GB")
        print(f"OS Reserved:         {OS_RESERVED_GB:.2f} GB")
        print(f"\nMax Model Size:      {max_model_size:.1f} GB")
        print(f"Recommended Quant:   {quant}")
        print(f"Reason:              {quant_msg}")
        print("="*60 + "\n")
        
        # Show status
        if snapshot.is_critical():
            print("⚠️  WARNING: RAM usage is critically high!")
            print("   Close other applications before loading models.")
        elif snapshot.is_warning():
            print("⚠️  CAUTION: RAM usage is elevated.")
            print("   Monitor memory during model loading.")
        else:
            print("✓ RAM status: Healthy")
        
        print()
        
    except Exception as e:
        logger.error(f"Failed to print RAM report: {e}")
        print(f"\nError generating RAM report: {e}\n")


# Backwards compatibility alias
def get_available_memory() -> float:
    """Alias for get_available_ram() for backward compatibility"""
    return get_available_ram()
