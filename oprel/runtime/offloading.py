"""
Hybrid GPU/CPU offloading - run large models on small GPUs.
Beats Ollama by enabling 13B models on 4GB GPUs.
"""

from typing import Optional
from dataclasses import dataclass

from oprel.runtime.layer_calculator import calculate_optimal_gpu_layers
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OffloadStrategy:
    """Hybrid offloading configuration"""
    gpu_layers: int
    cpu_layers: int
    total_layers: int
    gpu_percentage: float
    expected_speed_tokps: float
    vram_usage_gb: float
    ram_usage_gb: float
    recommendation: str
    
    def __str__(self) -> str:
        return (
            f"Hybrid: {self.gpu_layers}/{self.total_layers} layers on GPU "
            f"({self.gpu_percentage:.0f}%), ~{self.expected_speed_tokps:.1f} tok/s"
        )


class HybridOffloadCalculator:
    """
    Calculate optimal GPU/CPU layer split for models that don't fit.
    Maximizes GPU utilization while keeping CPU fallback performant.
    """
    
    # Performance estimates (tokens/sec per layer)
    GPU_SPEED_PER_LAYER = 3.0  # GPU processes ~3 tok/s per layer
    CPU_SPEED_PER_LAYER = 0.3  # CPU processes ~0.3 tok/s per layer
    
    def __init__(self):
        pass
    
    def calculate(
        self,
        model_size_gb: float,
        total_layers: int,
        available_vram_gb: float,
        available_ram_gb: float,
        context_length: int = 2048,
        hidden_dim: int = 4096,
        num_kv_heads: Optional[int] = None,
        num_attention_heads: Optional[int] = None
    ) -> OffloadStrategy:
        """
        Calculate optimal hybrid offloading strategy.
        
        Args:
            model_size_gb: Model size
            total_layers: Total transformer layers
            available_vram_gb: Available VRAM
            available_ram_gb: Available RAM
            context_length: Target context
            hidden_dim, num_kv_heads, num_attention_heads: Model dims
            
        Returns:
            OffloadStrategy with recommended split
        """
        # Calculate GPU layer capacity
        layer_calc = calculate_optimal_gpu_layers(
            model_size_gb=model_size_gb,
            total_layers=total_layers,
            available_vram_gb=available_vram_gb,
            context_length=context_length,
            hidden_dim=hidden_dim,
            num_kv_heads=num_kv_heads,
            num_attention_heads=num_attention_heads
        )
        
        gpu_layers = layer_calc.gpu_layers
        cpu_layers = total_layers - gpu_layers
        gpu_pct = layer_calc.gpu_percentage
        
        # Estimate performance
        # Model: GPU layers are fast, CPU layers are slow
        # Weighted average by layer count
        if total_layers > 0:
            gpu_speed = (gpu_layers / total_layers) * self.GPU_SPEED_PER_LAYER * total_layers
            cpu_speed = (cpu_layers / total_layers) * self.CPU_SPEED_PER_LAYER * total_layers
            total_speed = gpu_speed + cpu_speed
        else:
            total_speed = 0
        
        # RAM usage estimate (CPU layers)
        weight_per_layer = model_size_gb / total_layers if total_layers > 0 else 0
        ram_usage = cpu_layers * weight_per_layer
        
        # Generate recommendation
        if gpu_layers == 0:
            rec = f"CPU only - model too large for GPU. Expect ~{total_speed:.1f} tok/s"
        elif gpu_layers == total_layers:
            rec = f"Full GPU - all layers fit. Expect ~{total_speed:.1f} tok/s"
        elif gpu_pct >= 70:
            rec = f"Mostly GPU ({gpu_pct:.0f}%) - good performance. Expect ~{total_speed:.1f} tok/s"
        elif gpu_pct >= 40:
            rec = f"Balanced hybrid ({gpu_pct:.0f}% GPU). Expect ~{total_speed:.1f} tok/s"
        else:
            rec = f"Mostly CPU ({100-gpu_pct:.0f}% CPU) - slower. Expect ~{total_speed:.1f} tok/s"
        
        # Warn if OOM risk on CPU (RAM)
        if ram_usage > available_ram_gb * 0.8:
            rec += f" WARNING: {ram_usage:.1f}GB RAM needed, only {available_ram_gb:.1f}GB available"
        
        logger.info(
            f"Hybrid offload: {gpu_layers} GPU + {cpu_layers} CPU layers. "
            f"Expected: ~{total_speed:.1f} tok/s"
        )
        
        return OffloadStrategy(
            gpu_layers=gpu_layers,
            cpu_layers=cpu_layers,
            total_layers=total_layers,
            gpu_percentage=gpu_pct,
            expected_speed_tokps=total_speed,
            vram_usage_gb=layer_calc.model_vram_gb + layer_calc.kv_cache_vram_gb,
            ram_usage_gb=ram_usage,
            recommendation=rec
        )
    
    def calculate_from_metadata(
        self,
        gguf_metadata,
        available_vram_gb: float,
        available_ram_gb: float,
        context_length: Optional[int] = None
    ) -> OffloadStrategy:
        """
        Convenience: Calculate from GGUF metadata.
        
        Args:
            gguf_metadata: GGUFMetadata object
            available_vram_gb: Available VRAM
            available_ram_gb: Available RAM
            context_length: Override context (uses metadata if None)
        """
        ctx_len = context_length or gguf_metadata.context_length
        
        return self.calculate(
            model_size_gb=gguf_metadata.file_size_bytes / (1024 ** 3),
            total_layers=gguf_metadata.num_layers,
            available_vram_gb=available_vram_gb,
            available_ram_gb=available_ram_gb,
            context_length=ctx_len,
            hidden_dim=gguf_metadata.embedding_dim,
            num_kv_heads=gguf_metadata.num_kv_heads,
            num_attention_heads=gguf_metadata.num_attention_heads
        )


def recommend_offload_strategy(
    model_size_gb: float,
    total_layers: int,
    vram_gb: float,
    ram_gb: float
) -> str:
    """
    Quick recommendation: GPU-only, hybrid, or CPU-only.
    
    Returns:
        "gpu", "hybrid", or "cpu"
    """
    calculator = HybridOffloadCalculator()
    
    # Simple estimation without full metadata
    strategy = calculator.calculate(
        model_size_gb=model_size_gb,
        total_layers=total_layers,
        available_vram_gb=vram_gb,
        available_ram_gb=ram_gb
    )
    
    if strategy.gpu_layers == total_layers:
        return "gpu"
    elif strategy.gpu_layers > 0:
        return "hybrid"
    else:
        return "cpu"
