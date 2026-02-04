"""
VRAM monitoring during inference - prevents OOM crashes.
Proactive warnings before memory exhaustion.
"""

import time
import threading
from typing import Optional, Callable
from dataclasses import dataclass

from oprel.telemetry.hardware import get_vram_info
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VRAMSnapshot:
    """VRAM usage snapshot"""
    timestamp: float
    total_gb: float
    used_gb: float
    free_gb: float
    usage_percent: float


class VRAMMonitor:
    """
    Real-time VRAM monitoring with configurable thresholds.
    Warns before OOM and can trigger auto-actions.
    """
    
    def __init__(
        self,
        warn_threshold: float = 0.90,  # Warn at 90% usage
        critical_threshold: float = 0.95,  # Critical at 95%
        check_interval_sec: float = 0.5,
        on_warning: Optional[Callable] = None,
        on_critical: Optional[Callable] = None
    ):
        """
        Initialize VRAM monitor.
        
        Args:
            warn_threshold: Warning threshold (0-1)
            critical_threshold: Critical threshold (0-1)
            check_interval_sec: Monitoring frequency
            on_warning: Callback for warning state
            on_critical: Callback for critical state
        """
        self.warn_threshold = warn_threshold
        self.critical_threshold = critical_threshold
        self.check_interval = check_interval_sec
        self.on_warning = on_warning
        self.on_critical = on_critical
        
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None
        self._warned = False
        self._critical_warned = False
    
    def start(self):
        """Start background monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._warned = False
        self._critical_warned = False
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.debug("VRAM monitoring started")
    
    def stop(self):
        """Stop monitoring"""
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.debug("VRAM monitoring stopped")
    
    def get_snapshot(self) -> Optional[VRAMSnapshot]:
        """Get current VRAM snapshot"""
        vram_info = get_vram_info()
        if not vram_info:
            return None
        
        total_gb = vram_info.get("vram_total_gb", 0)
        used_gb = vram_info.get("vram_used_gb", 0)
        free_gb = total_gb - used_gb
        usage_pct = (used_gb / total_gb) if total_gb > 0 else 0
        
        return VRAMSnapshot(
            timestamp=time.time(),
            total_gb=total_gb,
            used_gb=used_gb,
            free_gb=free_gb,
            usage_percent=usage_pct
        )
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitoring:
            try:
                snapshot = self.get_snapshot()
                if snapshot:
                    self._check_thresholds(snapshot)
                
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"VRAM monitoring error: {e}")
                time.sleep(1.0)
    
    def _check_thresholds(self, snapshot: VRAMSnapshot):
        """Check if usage exceeds thresholds"""
        usage = snapshot.usage_percent
        
        # Critical threshold
        if usage >= self.critical_threshold:
            if not self._critical_warned:
                logger.error(
                    f"CRITICAL: VRAM at {usage*100:.1f}% "
                    f"({snapshot.used_gb:.1f}/{snapshot.total_gb:.1f}GB)"
                )
                self._critical_warned = True
                
                if self.on_critical:
                    try:
                        self.on_critical(snapshot)
                    except Exception as e:
                        logger.error(f"Critical callback error: {e}")
        
        # Warning threshold
        elif usage >= self.warn_threshold:
            if not self._warned:
                logger.warning(
                    f"WARNING: VRAM at {usage*100:.1f}% "
                    f"({snapshot.used_gb:.1f}/{snapshot.total_gb:.1f}GB)"
                )
                self._warned = True
                
                if self.on_warning:
                    try:
                        self.on_warning(snapshot)
                    except Exception as e:
                        logger.error(f"Warning callback error: {e}")
        
        # Reset warnings if usage drops
        elif usage < self.warn_threshold * 0.95:
            self._warned = False
            self._critical_warned = False
    
    def __enter__(self):
        """Context manager support"""
        self.start()
        return self
    
    def __exit__(self, *args):
        """Context manager cleanup"""
        self.stop()


def monitor_during_inference(
    warn_threshold: float = 0.90,
    critical_threshold: float = 0.95,
    auto_reduce_context: bool = True
):
    """
    Context manager: Monitor VRAM during inference.
    
    Example:
        >>> with monitor_during_inference():
        ...     model.generate(prompt)
    """
    def on_warning(snapshot):
        logger.warning(
            f"High VRAM usage. Consider reducing context length or batch size."
        )
    
    def on_critical(snapshot):
        logger.error(
            f"VRAM critical! OOM crash imminent. "
            f"Free: {snapshot.free_gb:.2f}GB"
        )
        if auto_reduce_context:
            # TODO: Signal to reduce context
            pass
    
    return VRAMMonitor(
        warn_threshold=warn_threshold,
        critical_threshold=critical_threshold,
        on_warning=on_warning,
        on_critical=on_critical
    )
