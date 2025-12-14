"""
Algorithm configuration parameters.

Modify these values to tune routing behavior.
"""

from pydantic import BaseModel, Field


class AlgorithmConfig(BaseModel):
    """Configuration for routing algorithms."""

    # ==========================================================================
    # Bandwidth scaling
    # ==========================================================================
    # CSV đã có bandwidth_capacity tính sẵn theo công thức:
    #   bandwidth_capacity = (width_m / 3.5) × 600 × highway_factor
    #   Đơn vị: xe/giờ
    #
    # bandwidth_scale là hệ số nhân để điều chỉnh:
    #   B_effective = bandwidth_capacity × bandwidth_scale / time_window
    #
    # Ví dụ với time_window = 1 giờ:
    #   - scale=1.0: dùng nguyên capacity (600 xe/làn/giờ)
    #   - scale=0.5: giảm 50% (giờ cao điểm, tắc đường)
    #   - scale=0.01: chỉ cho ~6 xe/làn trong time window
    bandwidth_scale: float = Field(
        default=1.0,
        description="Hệ số scale cho bandwidth. B = bandwidth_capacity × scale",
    )

    # ==========================================================================
    # Penalty routing parameters
    # ==========================================================================
    # Penalty = exp(beta * max(load / B - 1, 0))
    # When load <= B: penalty = 0
    # When load > B: penalty grows exponentially with overflow
    penalty_beta: float = Field(
        default=1.0,
        description="Penalty exponent (beta). Higher = stronger exponential penalty for overflow.",
    )

    # Number of alternative paths to compute per vehicle
    k_paths: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of candidate paths per vehicle for penalty routing.",
    )

    # Large penalty for k-shortest path computation (to find diverse routes)
    k_shortest_penalty: float = Field(
        default=1000.0,
        description="Penalty added to edges after finding each path in k-shortest.",
    )


# Global config instance - can be modified at runtime or loaded from file
_config = AlgorithmConfig()


def get_config() -> AlgorithmConfig:
    """Get current algorithm configuration."""
    return _config


def update_config(**kwargs) -> AlgorithmConfig:
    """Update configuration parameters."""
    global _config
    _config = AlgorithmConfig(**{**_config.model_dump(), **kwargs})
    return _config


def reset_config() -> AlgorithmConfig:
    """Reset configuration to defaults."""
    global _config
    _config = AlgorithmConfig()
    return _config
