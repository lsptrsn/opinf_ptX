"""Optimization algorithm configuration."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class OptimizationConfig:
    """Configuration for optimal control optimization.

    Attributes:
        alpha_u: Control smoothness penalty weight (higher = smoother)
        slack_penalty: Soft constraint penalty (0 = hard constraints)
        max_time_steps: Maximum number of time steps (None = use all)
        enable_caching: Enable CNN callback caching for speedup
        cache_size: Number of cached CNN evaluations
        hotspot_sample_rate: Sample every Nth time point for constraints (1 = all)
    """
    alpha_u: float = 100
    slack_penalty: float = 1e5
    max_time_steps: Optional[int] = None
    enable_caching: bool = True
    cache_size: int = 1000
    hotspot_sample_rate: int = 10  # Check every 10th time point
    time_sampling_stride: int = 10

    def __post_init__(self):
        """Validate configuration."""
        assert self.alpha_u >= 0, "alpha_u must be non-negative"
        assert self.slack_penalty >= 0, "slack_penalty must be non-negative"
        if self.max_time_steps is not None:
            assert self.max_time_steps > 0, "max_time_steps must be positive"
