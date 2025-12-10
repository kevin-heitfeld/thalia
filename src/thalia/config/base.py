"""
Base Configuration Classes.

This module provides base configuration classes with common fields to reduce
duplication across the codebase. All specific configs should inherit from these.

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
                f"Unknown dtype '{self.dtype}'. "
                f"Choose from: {list(dtype_map.keys())}"
            )
        return dtype_map[self.dtype]


@dataclass
class NeuralComponentConfig(BaseConfig):
    """Base config for neural components (neurons, layers, regions).
    
    Extends BaseConfig with neural-specific parameters:
    - n_neurons: Number of neurons
    - dt_ms: Simulation timestep (should match GlobalConfig.dt_ms)
    """
    
    n_neurons: int = 100
    """Number of neurons in the component."""
    
    dt_ms: float = 1.0
    """Simulation timestep in milliseconds. Set from GlobalConfig.dt_ms by Brain."""


@dataclass
class LearningComponentConfig(BaseConfig):
    """Base config for learning components.
    
    Extends BaseConfig with learning-specific parameters:
    - learning_rate: Base learning rate
    - enabled: Whether learning is enabled
    """
    
    learning_rate: float = 0.01
    """Base learning rate."""
    
    enabled: bool = True
    """Whether this learning component is enabled."""


@dataclass  
class RegionConfigBase(NeuralComponentConfig):
    """Base config for brain regions.
    
    Extends NeuralComponentConfig with region-specific parameters:
    - n_input: Input dimension
    - n_output: Output dimension
    - learn: Whether learning is enabled
    """
    
    n_input: int = 128
    """Input dimension to the region."""
    
    n_output: int = 64
    """Output dimension from the region."""
    
    learn: bool = True
    """Whether learning is enabled in this region."""


__all__ = [
    "BaseConfig",
    "NeuralComponentConfig",
    "LearningComponentConfig",
    "RegionConfigBase",
]
