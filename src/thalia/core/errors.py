"""
Custom exception classes and validation utilities for Thalia.

This module provides:
1. Hierarchical exception classes for different error categories
2. Validation utilities that enforce biological plausibility constraints
3. Consistent error message formatting across components

Exception Hierarchy:
====================
ThaliaError (base)
├── ComponentError - Errors in brain regions or pathways
├── ConfigurationError - Invalid configuration parameters
├── BiologicalPlausibilityError - Violations of biological constraints
├── CheckpointError - Errors in saving/loading state
└── IntegrationError - Errors in brain-wide coordination

Usage Examples:
===============
    # Raise component error
    raise ComponentError("Hippocampus", "DG pattern separation failed")

    # Validate spike tensor
    validate_spike_tensor(spikes, name="ca3_output")

    # Check device consistency
    validate_device_consistency({"weights": weights, "bias": bias}, device)

Design Philosophy:
==================
- Specific exception types enable targeted error handling
- Error messages reference ADRs for educational value
- Validation utilities prevent constraint violations early
- Consistent formatting improves debugging experience

Author: Thalia Project
Date: December 12, 2025
"""

from __future__ import annotations

from typing import Dict

import torch


# =============================================================================
# Exception Hierarchy
# =============================================================================

class ThaliaError(Exception):
    """Base exception for all Thalia-specific errors.

    All custom exceptions in Thalia inherit from this class, enabling
    code to catch Thalia errors specifically:

        try:
            brain.forward(input_spikes)
        except ThaliaError as e:
            # Handle Thalia-specific errors
            logger.error(f"Thalia error: {e}")
        except Exception as e:
            # Handle other errors
            logger.error(f"Unexpected error: {e}")
    """


class ComponentError(ThaliaError):
    """Error in brain component (region or pathway).

    Raised when a region or pathway encounters an error during processing,
    learning, or state management.

    Args:
        component_name: Name of the component (e.g., "Hippocampus", "V1→V2 pathway")
        message: Description of the error

    Example:
        raise ComponentError("Striatum", "D1 pathway has zero weights")
    """

    def __init__(self, component_name: str, message: str):
        super().__init__(f"[{component_name}] {message}")
        self.component_name = component_name


class ConfigurationError(ThaliaError):
    """Invalid configuration parameters.

    Raised when configuration values are out of valid range or incompatible
    with each other.

    Example:
        raise ConfigurationError("tau_mem must be positive, got -10.0")
    """


class BiologicalPlausibilityError(ThaliaError):
    """Operation violates biological plausibility constraints.

    Raised when code attempts operations that violate documented biological
    constraints (ADR-004, ADR-005, ADR-006, ADR-007).

    Error messages reference relevant ADRs for educational value.

    Example:
        raise BiologicalPlausibilityError(
            "Spikes must be bool dtype (ADR-004), got float32"
        )
    """


class CheckpointError(ThaliaError):
    """Error in checkpoint save/load operations.

    Raised when checkpoint format is invalid, version incompatible, or
    deserialization fails.

    Example:
        raise CheckpointError("Checkpoint version 2.0 not compatible with 1.0 code")
    """


class IntegrationError(ThaliaError):
    """Error in brain-wide coordination or integration.

    Raised when components fail to integrate properly (pathway connections,
    neuromodulator broadcasts, growth coordination).

    Example:
        raise IntegrationError("Pathway V1→V2 dimensions incompatible: 256→512 expected 256→256")
    """


# =============================================================================
# Validation Utilities
# =============================================================================

def validate_spike_tensor(
    spikes: torch.Tensor,
    name: str = "spikes",
    expected_dim: int = 1,
) -> None:
    """Validate that spike tensor meets biological constraints (ADR-004, ADR-005).

    Checks:
    1. Dtype is torch.bool (ADR-004: binary spikes)
    2. Dimensionality matches expected (ADR-005: no batch dimension = 1D)

    Args:
        spikes: Spike tensor to validate
        name: Name for error messages (e.g., "ca3_output", "motor_spikes")
        expected_dim: Expected tensor dimensionality (default: 1 per ADR-005)

    Raises:
        BiologicalPlausibilityError: If constraints violated

    Example:
        >>> validate_spike_tensor(ca3_spikes, name="ca3_output")
        >>> validate_spike_tensor(weight_matrix, name="dg_ca3_weights", expected_dim=2)
    """
    if spikes.dtype != torch.bool:
        raise BiologicalPlausibilityError(
            f"{name} must be bool dtype (ADR-004: binary spikes), got {spikes.dtype}"
        )

    if spikes.dim() != expected_dim:
        raise BiologicalPlausibilityError(
            f"{name} must be {expected_dim}D, got shape {spikes.shape}. "
            f"Note: ADR-005 prohibits batch dimensions (use 1D for spike vectors)."
        )


def validate_device_consistency(
    tensors: Dict[str, torch.Tensor],
    expected_device: torch.device,
) -> None:
    """Validate that all tensors are on the expected device.

    Helps catch device mismatch errors early, which are common when mixing
    CPU and CUDA tensors.

    Args:
        tensors: Dictionary of tensor_name → tensor
        expected_device: Expected device for all tensors

    Raises:
        ConfigurationError: If any tensor on wrong device

    Example:
        >>> validate_device_consistency(
        ...     {"weights": self.weights, "bias": self.bias},
        ...     self.device
        ... )
    """
    mismatches = []
    for name, tensor in tensors.items():
        if tensor.device != expected_device:
            mismatches.append(f"{name}: {tensor.device}")

    if mismatches:
        raise ConfigurationError(
            f"Device mismatch. Expected {expected_device}, but found:\n" +
            "\n".join(f"  - {m}" for m in mismatches)
        )


def validate_weight_matrix(
    weights: torch.Tensor,
    n_output: int,
    n_input: int,
    name: str = "weights",
) -> None:
    """Validate weight matrix dimensions and properties.

    Checks:
    1. Shape is [n_output, n_input]
    2. Contains no NaN or Inf values
    3. Is 2D tensor

    Args:
        weights: Weight matrix to validate
        n_output: Expected number of output neurons
        n_input: Expected number of input neurons
        name: Name for error messages

    Raises:
        ConfigurationError: If validation fails

    Example:
        >>> validate_weight_matrix(self.ca3_dg_weights, self.n_ca3, self.n_dg)
    """
    # Check dimensionality
    if weights.dim() != 2:
        raise ConfigurationError(
            f"{name} must be 2D, got shape {weights.shape}"
        )

    # Check shape
    expected_shape = (n_output, n_input)
    if weights.shape != expected_shape:
        raise ConfigurationError(
            f"{name} shape mismatch. Expected {expected_shape}, got {weights.shape}"
        )

    # Check for invalid values
    if torch.isnan(weights).any():
        raise ConfigurationError(f"{name} contains NaN values")

    if torch.isinf(weights).any():
        raise ConfigurationError(f"{name} contains Inf values")


def validate_positive(
    value: float,
    name: str,
    allow_zero: bool = False,
) -> None:
    """Validate that a value is positive.

    Useful for time constants, learning rates, and other parameters that
    must be positive.

    Args:
        value: Value to check
        name: Parameter name for error messages
        allow_zero: Whether zero is acceptable (default: False)

    Raises:
        ConfigurationError: If value not positive

    Example:
        >>> validate_positive(tau_mem, "tau_mem")
        >>> validate_positive(learning_rate, "learning_rate", allow_zero=True)
    """
    if allow_zero:
        if value < 0:
            raise ConfigurationError(f"{name} must be non-negative, got {value}")
    else:
        if value <= 0:
            raise ConfigurationError(f"{name} must be positive, got {value}")


def validate_probability(
    value: float,
    name: str,
) -> None:
    """Validate that a value is in [0, 1] range.

    Args:
        value: Probability value to check
        name: Parameter name for error messages

    Raises:
        ConfigurationError: If not in valid range

    Example:
        >>> validate_probability(spike_probability, "spike_probability")
    """
    if not 0 <= value <= 1:
        raise ConfigurationError(
            f"{name} must be in [0, 1] range, got {value}"
        )


def validate_temporal_causality(
    current_time: int,
    reference_time: int,
    context: str = "operation",
) -> None:
    """Validate that operation respects temporal causality.

    Ensures that operations don't access future information, which would
    violate biological plausibility.

    Args:
        current_time: Current timestep
        reference_time: Referenced timestep
        context: Description of operation for error message

    Raises:
        BiologicalPlausibilityError: If accessing future information

    Example:
        >>> validate_temporal_causality(
        ...     current_time=10,
        ...     reference_time=5,
        ...     context="STDP window calculation"
        ... )
    """
    if reference_time > current_time:
        raise BiologicalPlausibilityError(
            f"{context} violates temporal causality: "
            f"cannot access t={reference_time} from t={current_time}"
        )


__all__ = [
    # Exception classes
    "ThaliaError",
    "ComponentError",
    "ConfigurationError",
    "BiologicalPlausibilityError",
    "CheckpointError",
    "IntegrationError",
    # Validation utilities
    "validate_spike_tensor",
    "validate_device_consistency",
    "validate_weight_matrix",
    "validate_positive",
    "validate_probability",
    "validate_temporal_causality",
]
