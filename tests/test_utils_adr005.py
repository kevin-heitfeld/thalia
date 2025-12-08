"""
Utilities for ADR-005 compliant test fixtures.

ADR-005 specifies that Thalia uses a single-brain architecture with NO batch dimension.
All tensors are 1D: [n_neurons], not 2D: [batch=1, n_neurons].

This module provides utilities to create ADR-005 compliant test inputs.
"""

import torch


def make_input_1d(n_input: int, device: str = 'cpu') -> torch.Tensor:
    """
    Create 1D input tensor (ADR-005 compliant).
    
    Args:
        n_input: Number of input neurons
        device: Device to create tensor on
        
    Returns:
        1D tensor of shape [n_input]
    """
    return torch.randn(n_input, device=device)


def make_spikes_1d(n_neurons: int, sparsity: float = 0.1, device: str = 'cpu') -> torch.Tensor:
    """
    Create 1D binary spike tensor (ADR-005 compliant).
    
    Args:
        n_neurons: Number of neurons
        sparsity: Fraction of neurons that spike (0-1)
        device: Device to create tensor on
        
    Returns:
        1D binary tensor of shape [n_neurons]
    """
    spikes = torch.zeros(n_neurons, device=device)
    n_active = int(n_neurons * sparsity)
    if n_active > 0:
        indices = torch.randperm(n_neurons)[:n_active]
        spikes[indices] = 1.0
    return spikes


def make_pattern_1d(n_neurons: int, pattern_indices: list, device: str = 'cpu') -> torch.Tensor:
    """
    Create 1D spike pattern with specific active neurons.
    
    Args:
        n_neurons: Total number of neurons
        pattern_indices: Indices of neurons that should spike
        device: Device to create tensor on
        
    Returns:
        1D binary tensor of shape [n_neurons]
    """
    spikes = torch.zeros(n_neurons, device=device)
    spikes[pattern_indices] = 1.0
    return spikes


def make_zero_input_1d(n_input: int, device: str = 'cpu') -> torch.Tensor:
    """
    Create 1D zero input (ADR-005 compliant).
    
    Args:
        n_input: Number of input neurons
        device: Device to create tensor on
        
    Returns:
        1D tensor of zeros, shape [n_input]
    """
    return torch.zeros(n_input, device=device)


def make_strong_input_1d(n_input: int, scale: float = 10.0, device: str = 'cpu') -> torch.Tensor:
    """
    Create 1D strong input to trigger spikes (ADR-005 compliant).
    
    Args:
        n_input: Number of input neurons
        scale: Scaling factor for input strength
        device: Device to create tensor on
        
    Returns:
        1D tensor of shape [n_input]
    """
    return torch.randn(n_input, device=device) * scale


def convert_2d_to_1d(tensor_2d: torch.Tensor) -> torch.Tensor:
    """
    Convert 2D tensor [1, n] to 1D tensor [n] for ADR-005 compliance.
    
    Args:
        tensor_2d: 2D tensor of shape [1, n] or [batch, n]
        
    Returns:
        1D tensor of shape [n]
        
    Raises:
        ValueError: If batch size > 1
    """
    if tensor_2d.dim() == 1:
        return tensor_2d  # Already 1D
    elif tensor_2d.dim() == 2:
        if tensor_2d.shape[0] != 1:
            raise ValueError(
                f"ADR-005: Cannot convert batch size {tensor_2d.shape[0]} > 1 to 1D. "
                "Thalia uses single-brain architecture."
            )
        return tensor_2d.squeeze(0)
    else:
        raise ValueError(
            f"Expected 1D or 2D tensor, got {tensor_2d.dim()}D"
        )


def assert_1d_tensor(tensor: torch.Tensor, name: str = "tensor"):
    """
    Assert that tensor is 1D (ADR-005 compliant).
    
    Args:
        tensor: Tensor to check
        name: Name of tensor for error messages
        
    Raises:
        AssertionError: If tensor is not 1D
    """
    assert tensor.dim() == 1, (
        f"{name} must be 1D per ADR-005, got shape {tensor.shape}. "
        f"Use make_input_1d() or convert_2d_to_1d() to fix."
    )


def assert_binary_spikes(spikes: torch.Tensor):
    """
    Assert that spike tensor contains only 0s and 1s.
    
    Args:
        spikes: Spike tensor to check
        
    Raises:
        AssertionError: If spikes contain non-binary values
    """
    unique_values = torch.unique(spikes)
    assert torch.all((unique_values == 0) | (unique_values == 1)), (
        f"Spikes must be binary (0 or 1), got values: {unique_values.tolist()}"
    )


# Common test patterns
def get_test_patterns() -> dict:
    """
    Get common test patterns for unit tests.
    
    Returns:
        Dict with pattern generators for common test scenarios
    """
    return {
        'sparse': lambda n: make_spikes_1d(n, sparsity=0.1),
        'dense': lambda n: make_spikes_1d(n, sparsity=0.5),
        'zero': lambda n: make_zero_input_1d(n),
        'strong': lambda n: make_strong_input_1d(n, scale=10.0),
        'weak': lambda n: make_input_1d(n) * 0.1,
    }
