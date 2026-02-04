"""
Memory pressure monitoring - auto-unload models when RAM/VRAM low.
Prevents system crashes by proactive cleanup.
"""

import time
import threading
from typing import Optional, Callable
import psutil

from oprel.telemetry.hardware import get_vram_info
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


class MemoryPressureMonitor:
    """
    Monitor system memory pressure and trigger auto-cleanup.
    Prevents OOM crashes by unloading models before critical threshold.
    """
    
    def __init__(
        self,
        ram_warning_pct: float = 0.85,
        ram_critical_pct: float = 0.95,
        vram_warning_pct: float = 0.90,
        vram_critical_pct: float = 0.95,
        check_interval_sec: float = 2.0,
        on_pressure: Optional[Callable] = None
    ):
        """
        Initialize memory pressure monitor.
        
        Args:
            ram_warning_pct: RAM usage warning threshold
            ram_critical_pct: RAM usage critical threshold (trigger cleanup)
            vram_warning_pct: VRAM usage warning threshold
            vram_critical_pct: VRAM usage critical threshold
            check_interval_sec: Monitoring frequency
            on_pressure: Callback when memory pressure detected
        """
        self.ram_warning = ram_warning_pct
        self.ram_critical = ram_critical_pct
        self.vram_warning = vram_warning_pct
        self.vram_critical = vram_critical_pct
        self.check_interval = check_interval_sec
        self.on_pressure = on_pressure
        
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None
        self._warned_ram = False
        self._warned_vram = False
    
    def start(self):
        """Start background monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.debug("Memory pressure monitoring started")
    
    def stop(self):
        """Stop monitoring"""
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.debug("Memory pressure monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitoring:
            try:
                self._check_memory_pressure()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Memory pressure monitoring error: {e}")
                time.sleep(5.0)
    
    def _check_memory_pressure(self):
        """Check RAM and VRAM pressure"""
        # Check RAM
        ram = psutil.virtual_memory()
        ram_pct = ram.percent / 100.0
        
        if ram_pct >= self.ram_critical:
            if not self._warned_ram:
                logger.error(
                    f"CRITICAL RAM pressure: {ram_pct*100:.0f}% used "
                    f"({ram.used/(1024**3):.1f}/{ram.total/(1024**3):.1f}GB)"
                )
                self._warned_ram = True
                
                if self.on_pressure:
                    self.on_pressure("ram", "critical", ram_pct)
        
        elif ram_pct >= self.ram_warning:
            if not self._warned_ram:
                logger.warning(
                    f"High RAM usage: {ram_pct*100:.0f}% "
                    f"({ram.used/(1024**3):.1f}/{ram.total/(1024**3):.1f}GB)"
                )
                self._warned_ram = True
                
                if self.on_pressure:
                    self.on_pressure("ram", "warning", ram_pct)
        
        elif ram_pct < self.ram_warning * 0.9:
            self._warned_ram = False
        
        # Check VRAM
        vram_info = get_vram_info()
        if vram_info:
            total_gb = vram_info.get("vram_total_gb", 0)
            used_gb = vram_info.get("vram_used_gb", 0)
            vram_pct = (used_gb / total_gb) if total_gb > 0 else 0
            
            if vram_pct >= self.vram_critical:
                if not self._warned_vram:
                    logger.error(
                        f"CRITICAL VRAM pressure: {vram_pct*100:.0f}% used "
                        f"({used_gb:.1f}/{total_gb:.1f}GB)"
                    )
                    self._warned_vram = True
                    
                    if self.on_pressure:
                        self.on_pressure("vram", "critical", vram_pct)
            
            elif vram_pct >= self.vram_warning:
                if not self._warned_vram:
                    logger.warning(
                        f"High VRAM usage: {vram_pct*100:.0f}% "
                        f"({used_gb:.1f}/{total_gb:.1f}GB)"
                    )
                    self._warned_vram = True
                    
                    if self.on_pressure:
                        self.on_pressure("vram", "warning", vram_pct)
            
            elif vram_pct < self.vram_warning * 0.9:
                self._warned_vram = False
    
    def __enter__(self):
        """Context manager support"""
        self.start()
        return self
    
    def __exit__(self, *args):
        """Context manager cleanup"""
        self.stop()


def create_auto_cleanup_monitor():
    """
    Create memory pressure monitor with auto-cleanup callback.
    Automatically unloads idle models when memory critical.
    """
    def on_pressure(mem_type: str, level: str, usage_pct: float):
        if level == "critical":
            logger.warning(
                f"{mem_type.upper()} critical ({usage_pct*100:.0f}%). "
                f"Consider unloading models or using smaller quantization."
            )
            # TODO: Integrate with model manager to auto-unload idle models
    
    return MemoryPressureMonitor(on_pressure=on_pressure)
