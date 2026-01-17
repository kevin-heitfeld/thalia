"""
Base Configuration Classes.

This module provides the fundamental BaseConfig class with device, dtype, and seed.

Component-specific configs (NeuralComponentConfig, PathwayConfig, LearningComponentConfig)
have been moved to core/component_config.py to break the CONFIG ↔ REGIONS circular import.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class BaseConfig:
    """Base configuration with common fields for all components.

    This provides standard fields that appear in almost every config:
    - device: Hardware device (cpu/cuda)
    - dtype: Tensor data type
    - seed: Random seed for reproducibility
    """

    device: str = "cpu"
    """Device to run on: 'cpu', 'cuda', 'cuda:0', etc."""

    dtype: str = "float32"
    """Data type for tensors: 'float32', 'float64', 'float16'"""

    seed: Optional[int] = None
    """Random seed for reproducibility. None = no seeding."""

    def __post_init__(self):
        """Convert torch.device to string if needed."""
        if isinstance(self.device, torch.device):
            self.device = str(self.device)

    def get_torch_device(self) -> torch.device:
        """Get PyTorch device object."""
        return torch.device(self.device)

    def get_torch_dtype(self) -> torch.dtype:
        """Get PyTorch dtype object."""
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int32": torch.int32,
            "int64": torch.int64,
            "bool": torch.bool,
        }
        if self.dtype not in dtype_map:
            raise ValueError(
                f"Unknown dtype '{self.dtype}'. " f"Choose from: {list(dtype_map.keys())}"
            )
        return dtype_map[self.dtype]


# =============================================================================
# COMPONENT CONFIGS MOVED TO core/component_config.py
# =============================================================================
# The following classes have been moved to break CONFIG ↔ REGIONS circular:
# - NeuralComponentConfig
# - LearningComponentConfig
# - PathwayConfig
#
# They are re-exported from config/__init__.py for backward compatibility.
# Import them from thalia.config or thalia.core.component_config
# =============================================================================


__all__ = [
    "BaseConfig",
]
