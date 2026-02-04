"""
CPU model recommendations based on hardware tier.
Helps users choose optimal models for CPU-only inference.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import psutil

from oprel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelRecommendation:
    """Model recommendation for CPU"""
    model_alias: str
    size_gb: float
    expected_tokps: str
    quality: str  # "excellent", "good", "acceptable"
    use_case: str


class CPUModelRecommender:
    """
    Recommend models based on CPU tier.
    Ensures users don't try to run 30B models on 4-core CPUs.
    """
    
    # CPU tier definitions
    CPU_TIERS = {
        "high_end": {
            "min_cores": 8,
            "examples": "i7-12700K, Ryzen 7 5800X, Apple M1 Pro",
            "recommended_models": [
                ("qwen3-8b", 4.5, "5-10 tok/s", "excellent"),
                ("llama3.3-8b", 4.3, "5-8 tok/s", "excellent"),
                ("qwen3-coder-8b", 4.5, "5-10 tok/s", "excellent"),
                ("phi-4-14b", 8.0, "3-5 tok/s", "good"),
                ("qwen3-7b-reasoning", 4.0, "5-8 tok/s", "excellent"),
            ],
            "avoid": "13B+ models (too slow, <2 tok/s)"
        },
        "mid_range": {
            "min_cores": 4,
            "examples": "i5-10400, Ryzen 5 3600, Apple M1",
            "recommended_models": [
                ("qwen3-4b", 2.5, "8-12 tok/s", "excellent"),
                ("qwen3-3b", 1.8, "10-15 tok/s", "excellent"),
                ("llama3.2-3b", 2.0, "8-12 tok/s", "good"),
                ("phi-3.5-mini", 2.4, "8-12 tok/s", "excellent"),
                ("qwen3-coder-4b", 2.5, "8-12 tok/s", "excellent"),
            ],
            "avoid": "7B+ models (slow, 3-5 tok/s)"
        },
        "low_end": {
            "min_cores": 2,
            "examples": "i3, Ryzen 3, older laptops",
            "recommended_models": [
                ("qwen3-1.7b", 1.0, "5-10 tok/s", "good"),
                ("llama3.2-1b", 0.8, "8-12 tok/s", "acceptable"),
                ("qwen3-0.6b", 0.4, "15-20 tok/s", "acceptable"),
            ],
            "avoid": "3B+ models (very slow, <3 tok/s)"
        }
    }
    
    def __init__(self):
        self.cpu_count_physical = psutil.cpu_count(logical=False) or psutil.cpu_count()
        self.ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        self.tier = self._detect_tier()
    
    def _detect_tier(self) -> str:
        """Detect CPU tier based on core count"""
        if self.cpu_count_physical >= 8:
            return "high_end"
        elif self.cpu_count_physical >= 4:
            return "mid_range"
        else:
            return "low_end"
    
    def get_recommendations(
        self,
        use_case: Optional[str] = None
    ) -> List[ModelRecommendation]:
        """
        Get model recommendations for current CPU.
        
        Args:
            use_case: Filter by use case ("coding", "chat", "reasoning")
            
        Returns:
            List of recommended models
        """
        tier_info = self.CPU_TIERS[self.tier]
        models = []
        
        for alias, size_gb, speed, quality in tier_info["recommended_models"]:
            # Filter by use case
            if use_case:
                if use_case == "coding" and "coder" not in alias:
                    continue
                elif use_case == "reasoning" and "reasoning" not in alias:
                    continue
            
            # Check RAM capacity
            if self.ram_gb < size_gb * 1.5:
                continue  # Not enough RAM
            
            # Determine use case
            if "coder" in alias:
                model_use_case = "coding"
            elif "reasoning" in alias:
                model_use_case = "reasoning"
            else:
                model_use_case = "chat"
            
            models.append(ModelRecommendation(
                model_alias=alias,
                size_gb=size_gb,
                expected_tokps=speed,
                quality=quality,
                use_case=model_use_case
            ))
        
        return models
    
    def print_recommendations(self):
        """Print user-friendly recommendations"""
        tier_info = self.CPU_TIERS[self.tier]
        
        print(f"\n{'='*60}")
        print(f"CPU Tier: {self.tier.upper().replace('_', ' ')}")
        print(f"Cores: {self.cpu_count_physical} physical, RAM: {self.ram_gb:.1f}GB")
        print(f"Examples: {tier_info['examples']}")
        print(f"{'='*60}\n")
        
        recommendations = self.get_recommendations()
        
        if recommendations:
            print("Recommended Models:")
            for rec in recommendations:
                print(f"  • {rec.model_alias:20} ({rec.size_gb:.1f}GB) "
                      f"- {rec.expected_tokps:12} [{rec.quality}]")
            print()
        
        print(f"⚠️  Avoid: {tier_info['avoid']}")
        print(f"\nUsage: oprel run {recommendations[0].model_alias if recommendations else 'model-name'} \"your prompt\"")
        print("       oprel list-models --category coding")
    
    def should_warn_about_model(self, model_size_gb: float) -> Optional[str]:
        """
        Check if model is too large for CPU tier.
        
        Returns:
            Warning message if model is too large, None otherwise
        """
        if self.tier == "low_end" and model_size_gb > 2.0:
            return (
                f"Warning: {model_size_gb:.1f}GB model may be very slow on this CPU. "
                f"Expect <3 tok/s. Consider smaller models (1-2GB)."
            )
        elif self.tier == "mid_range" and model_size_gb > 5.0:
            return (
                f"Warning: {model_size_gb:.1f}GB model will be slow on this CPU. "
                f"Expect 3-5 tok/s. Consider 3-4GB models for better performance."
            )
        elif self.tier == "high_end" and model_size_gb > 10.0:
            return (
                f"Warning: {model_size_gb:.1f}GB model will be slow on CPU. "
                f"Expect <2 tok/s. Consider GPU or smaller models."
            )
        
        return None


def show_cpu_recommendations():
    """CLI command: Show CPU model recommendations"""
    recommender = CPUModelRecommender()
    recommender.print_recommendations()


def check_model_for_cpu(model_size_gb: float) -> Optional[str]:
    """
    Check if model is suitable for current CPU.
    
    Returns:
        Warning message or None
    """
    recommender = CPUModelRecommender()
    return recommender.should_warn_about_model(model_size_gb)
