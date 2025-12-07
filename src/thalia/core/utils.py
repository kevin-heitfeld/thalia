"""
Core Utilities for THALIA.

This module provides common utility functions used across the codebase
to reduce code duplication and ensure consistency.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from typing import Union, Tuple
import torch


def ensure_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor has a batch dimension.
    
    If the tensor is 1D (shape [N]), adds a batch dimension to make it 2D
    (shape [1, N]). If already 2D or higher, returns unchanged.
    
    This is a common pattern throughout the codebase where functions need
    to handle both batched and unbatched inputs consistently.
    
    Args:
        tensor: Input tensor of any shape
        
    Returns:
        Tensor with at least 2 dimensions
        
    Example:
        >>> x = torch.randn(100)  # Shape: [100]
        >>> x = ensure_batch_dim(x)  # Shape: [1, 100]
        
        >>> y = torch.randn(32, 100)  # Shape: [32, 100]
        >>> y = ensure_batch_dim(y)  # Shape: [32, 100] (unchanged)
    """
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    return tensor


def ensure_batch_dims(*tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Ensure multiple tensors have batch dimensions.
    
    Convenience function to process multiple tensors at once.
    
    Args:
        *tensors: Variable number of input tensors
        
    Returns:
        Tuple of tensors with batch dimensions ensured
        
    Example:
        >>> input_spikes, output_spikes = ensure_batch_dims(input_spikes, output_spikes)
    """
    return tuple(ensure_batch_dim(t) for t in tensors)


def remove_batch_dim(tensor: torch.Tensor, had_batch: bool) -> torch.Tensor:
    """Remove batch dimension if it was added by ensure_batch_dim.
    
    Args:
        tensor: Tensor that may have had batch dim added
        had_batch: Whether the original tensor had a batch dimension
        
    Returns:
        Tensor with batch dim removed if had_batch is False
        
    Example:
        >>> x = torch.randn(100)
        >>> had_batch = x.dim() > 1
        >>> x = ensure_batch_dim(x)
        >>> # ... process x ...
        >>> x = remove_batch_dim(x, had_batch)  # Back to original shape
    """
    if not had_batch and tensor.dim() > 1:
        return tensor.squeeze(0)
    return tensor


def ensure_1d(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is 1D by squeezing or averaging batch dimension.
    
    For 2D tensors with batch_size=1, squeezes to 1D.
    For 2D tensors with batch_size>1, averages across batch dimension.
    Already 1D tensors are returned unchanged.
    
    This is useful for operations like torch.outer() that require 1D inputs.
    
    Args:
        tensor: Input tensor (1D or 2D)
        
    Returns:
        1D tensor
        
    Example:
        >>> x = torch.randn(1, 100)  # Shape: [1, 100]
        >>> x = ensure_1d(x)  # Shape: [100]
        
        >>> y = torch.randn(32, 100)  # Shape: [32, 100]  
        >>> y = ensure_1d(y)  # Shape: [100] (averaged across batch)
    """
    if tensor.dim() == 2:
        return tensor.squeeze(0) if tensor.shape[0] == 1 else tensor.mean(dim=0)
    return tensor


def clamp_weights(
    weights: torch.Tensor,
    w_min: float = 0.0,
    w_max: float = 1.0,
    inplace: bool = True,
) -> torch.Tensor:
    """Clamp weight tensor to valid range.
    
    Standard pattern for enforcing weight bounds after learning updates.
    Operates in-place by default for efficiency.
    
    Args:
        weights: Weight tensor to clamp
        w_min: Minimum weight value (default: 0.0)
        w_max: Maximum weight value (default: 1.0)
        inplace: If True, modify weights in place (default: True)
        
    Returns:
        Clamped weight tensor
        
    Example:
        >>> clamp_weights(self.weights, cfg.w_min, cfg.w_max)
        >>> # Or with .data for nn.Parameter:
        >>> clamp_weights(self.weights.data, cfg.w_min, cfg.w_max)
    """
    if inplace:
        return weights.clamp_(w_min, w_max)
    return weights.clamp(w_min, w_max)


def apply_soft_bounds(
    dw: torch.Tensor,
    weights: torch.Tensor,
    w_min: float = 0.0,
    w_max: float = 1.0,
) -> torch.Tensor:
    """Apply soft bounds to weight updates.
    
    Scales weight updates by headroom/footroom to prevent hard saturation:
    - Positive updates are scaled by (w_max - weights) / w_max
    - Negative updates are scaled by (weights - w_min) / w_max
    
    This provides smooth approach to bounds rather than abrupt clamping.
    
    Args:
        dw: Weight update tensor
        weights: Current weights
        w_min: Minimum weight value
        w_max: Maximum weight value
        
    Returns:
        Scaled weight update tensor
        
    Example:
        >>> dw = compute_stdp_update(...)
        >>> dw = apply_soft_bounds(dw, self.weights, cfg.w_min, cfg.w_max)
        >>> self.weights += dw
    """
    w_range = w_max - w_min + 1e-8
    headroom = (w_max - weights) / w_range
    footroom = (weights - w_min) / w_range
    return torch.where(
        dw > 0,
        dw * headroom.clamp(0, 1),
        dw * footroom.clamp(0, 1),
    )


def cosine_similarity_safe(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-8,
    dim: int = -1,
) -> torch.Tensor:
    """Compute cosine similarity with safe epsilon handling.
    
    Provides consistent epsilon handling for numerical stability.
    Works with both 1D vectors and batched tensors.
    
    Args:
        a: First tensor
        b: Second tensor  
        eps: Small constant for numerical stability (default: 1e-8)
        dim: Dimension along which to compute similarity (default: -1)
        
    Returns:
        Cosine similarity value(s)
        
    Example:
        >>> sim = cosine_similarity_safe(pattern1, pattern2)
        >>> # For batched: sim has shape [batch_size]
    """
    # Handle 1D vectors
    if a.dim() == 1 and b.dim() == 1:
        norm_a = a.norm() + eps
        norm_b = b.norm() + eps
        return (a @ b) / (norm_a * norm_b)
    
    # Handle higher dimensional tensors
    a_norm = a / (a.norm(dim=dim, keepdim=True) + eps)
    b_norm = b / (b.norm(dim=dim, keepdim=True) + eps)
    return (a_norm * b_norm).sum(dim=dim)


def zeros_like_config(
    *dims: int,
    device: Union[str, torch.device],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a zero tensor with dimensions and device from config.
    
    Args:
        *dims: Tensor dimensions
        device: Device for the tensor
        dtype: Data type (default: float32)
        
    Returns:
        Zero tensor with specified shape on specified device
        
    Example:
        >>> trace = zeros_like_config(n_output, n_input, device=self.device)
    """
    return torch.zeros(*dims, device=torch.device(device), dtype=dtype)


def ones_like_config(
    *dims: int,
    device: Union[str, torch.device],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a ones tensor with dimensions and device from config.
    
    Args:
        *dims: Tensor dimensions
        device: Device for the tensor
        dtype: Data type (default: float32)
        
    Returns:
        Ones tensor with specified shape on specified device
    """
    return torch.ones(*dims, device=torch.device(device), dtype=dtype)


def assert_single_instance(batch_size: int, context: str = "This component") -> None:
    """Assert that batch_size is 1, enforcing THALIA's single-instance architecture.
    
    THALIA models a single continuous brain state processing a temporal stream.
    Unlike ML training frameworks that batch independent samples for efficiency,
    THALIA maintains continuous temporal dynamics (membrane potentials, synaptic
    traces, adaptation state, working memory) that cannot be meaningfully batched.
    
    For parallel evaluation (e.g., in RL training), instantiate multiple instances
    rather than using batch_size > 1.
    
    Args:
        batch_size: The batch size to validate
        context: Description of where this check occurs (for error message)
        
    Raises:
        ValueError: If batch_size != 1
        
    Example:
        >>> def reset_state(self, batch_size: int = 1) -> None:
        ...     assert_single_instance(batch_size, "LayeredCortex")
        ...     # Continue with state initialization
    """
    if batch_size != 1:
        raise ValueError(
            f"{context} only supports batch_size=1, got {batch_size}. "
            "THALIA models a single continuous brain with temporal dynamics "
            "(membrane potentials, synaptic traces, adaptation state, working memory). "
            "For parallel simulations, create multiple instances."
        )


__all__ = [
    "ensure_batch_dim",
    "ensure_batch_dims", 
    "remove_batch_dim",
    "ensure_1d",
    "clamp_weights",
    "apply_soft_bounds",
    "cosine_similarity_safe",
    "zeros_like_config",
    "ones_like_config",
    "assert_single_instance",
]
