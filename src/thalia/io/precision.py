"""
Mixed Precision Support - FP16/FP32 conversion for checkpoints.

Provides utilities for converting tensors between FP32 and FP16 to reduce
checkpoint file size. Typically saves ~50% storage for weight-dominated checkpoints.

Usage:
    # Save with FP16
    BrainCheckpoint.save(brain, path, precision_policy='fp16')

    # Custom policy
    policy = PrecisionPolicy(
        weights='fp16',       # Large weight matrices
        biases='fp32',        # Keep biases in FP32
        membrane='fp32',      # Critical neuron state
        traces='fp16',        # Eligibility traces can be FP16
    )
    BrainCheckpoint.save(brain, path, precision_policy=policy)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Union

import torch


@dataclass
class PrecisionPolicy:
    """Policy for mixed precision checkpoint saving.

    Attributes:
        weights: Precision for weight matrices ('fp32', 'fp16', 'auto')
        biases: Precision for bias vectors ('fp32', 'fp16')
        membrane: Precision for membrane potentials ('fp32', 'fp16')
        traces: Precision for eligibility/STDP traces ('fp32', 'fp16')
        thresholds: Precision for adaptive thresholds ('fp32', 'fp16')
        conductances: Precision for synaptic conductances ('fp32', 'fp16')
        default: Default precision for unspecified tensors ('fp32', 'fp16')
    """

    weights: str = "fp32"
    biases: str = "fp32"
    membrane: str = "fp32"
    traces: str = "fp32"
    thresholds: str = "fp32"
    conductances: str = "fp32"
    default: str = "fp32"

    def __post_init__(self):
        """Validate precision strings."""
        valid = {"fp32", "fp16", "auto"}
        for field, value in self.__dict__.items():
            if value not in valid:
                raise ValueError(
                    f"Invalid precision '{value}' for field '{field}'. Must be one of {valid}"
                )


# Predefined policies
PRECISION_POLICIES = {
    "fp32": PrecisionPolicy(
        weights="fp32",
        biases="fp32",
        membrane="fp32",
        traces="fp32",
        thresholds="fp32",
        conductances="fp32",
        default="fp32",
    ),
    "fp16": PrecisionPolicy(
        weights="fp16",
        biases="fp32",  # Keep biases in FP32 for stability
        membrane="fp32",  # Keep membrane in FP32 for accuracy
        traces="fp16",  # Traces can be FP16
        thresholds="fp32",  # Keep thresholds in FP32
        conductances="fp16",  # Conductances can be FP16
        default="fp32",
    ),
    "mixed": PrecisionPolicy(
        weights="auto",  # Auto-detect based on size
        biases="fp32",
        membrane="fp32",
        traces="fp16",
        thresholds="fp32",
        conductances="fp16",
        default="fp32",
    ),
}


def get_precision_policy(policy: Union[str, PrecisionPolicy, None]) -> PrecisionPolicy:
    """Get precision policy from string name or return existing policy.

    Args:
        policy: Policy name ('fp32', 'fp16', 'mixed'), PrecisionPolicy instance, or None

    Returns:
        PrecisionPolicy instance

    Raises:
        ValueError: If policy name is invalid
    """
    if policy is None:
        return PRECISION_POLICIES["fp32"]
    elif isinstance(policy, str):
        if policy not in PRECISION_POLICIES:
            raise ValueError(
                f"Unknown precision policy '{policy}'. "
                f"Valid options: {list(PRECISION_POLICIES.keys())}"
            )
        return PRECISION_POLICIES[policy]
    elif isinstance(policy, PrecisionPolicy):
        return policy
    else:
        raise TypeError(f"Expected str or PrecisionPolicy, got {type(policy)}")


def determine_tensor_precision(
    tensor_name: str,
    tensor: torch.Tensor,
    policy: PrecisionPolicy,
) -> torch.dtype:
    """Determine target precision for a tensor based on policy.

    Args:
        tensor_name: Name of the tensor (e.g., 'w_ff', 'membrane', 'eligibility')
        tensor: The tensor itself
        policy: Precision policy to apply

    Returns:
        Target dtype (torch.float32 or torch.float16)
    """
    # Map tensor names to policy fields
    name_lower = tensor_name.lower()

    # Check for specific patterns
    if "weight" in name_lower or "w_" in name_lower:
        precision = policy.weights
    elif "bias" in name_lower or "b_" in name_lower:
        precision = policy.biases
    elif "membrane" in name_lower or "v_mem" in name_lower:
        precision = policy.membrane
    elif "trace" in name_lower or "eligibility" in name_lower or "stdp" in name_lower:
        precision = policy.traces
    elif "threshold" in name_lower or "v_thresh" in name_lower or "theta" in name_lower:
        precision = policy.thresholds
    elif "conductance" in name_lower or "g_" in name_lower:
        precision = policy.conductances
    else:
        precision = policy.default

    # Handle 'auto' precision
    if precision == "auto":
        # Use FP16 for large tensors (>1MB), FP32 for small ones
        size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
        precision = "fp16" if size_mb > 1.0 else "fp32"

    return torch.float16 if precision == "fp16" else torch.float32


def convert_tensor_precision(
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Convert tensor to target precision.

    Args:
        tensor: Input tensor
        target_dtype: Target dtype (torch.float32 or torch.float16)

    Returns:
        Tensor in target precision
    """
    if tensor.dtype == target_dtype:
        return tensor

    # Only convert floating point tensors
    if tensor.dtype not in (torch.float32, torch.float64, torch.float16):
        return tensor

    return tensor.to(target_dtype)


def apply_precision_policy_to_state(
    state: Dict[str, Any],
    policy: Union[str, PrecisionPolicy, None],
    in_place: bool = False,
) -> Dict[str, Any]:
    """Apply precision policy to entire state dict.

    Args:
        state: State dict (nested structure with tensors)
        policy: Precision policy to apply
        in_place: Whether to modify state in-place or create copy

    Returns:
        State dict with tensors converted to appropriate precision
    """
    policy = get_precision_policy(policy)

    if not in_place:
        state = copy.deepcopy(state)

    _apply_precision_recursive(state, policy, parent_key="")

    return state


def _apply_precision_recursive(
    obj: Any,
    policy: PrecisionPolicy,
    parent_key: str = "",
) -> None:
    """Recursively apply precision policy to nested structures.

    Modifies obj in-place.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{parent_key}.{key}" if parent_key else key

            if isinstance(value, torch.Tensor):
                # Convert tensor
                target_dtype = determine_tensor_precision(full_key, value, policy)
                obj[key] = convert_tensor_precision(value, target_dtype)
            elif isinstance(value, (dict, list)):
                # Recurse into nested structures
                _apply_precision_recursive(value, policy, full_key)

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            full_key = f"{parent_key}[{i}]"

            if isinstance(item, torch.Tensor):
                target_dtype = determine_tensor_precision(full_key, item, policy)
                obj[i] = convert_tensor_precision(item, target_dtype)
            elif isinstance(item, (dict, list)):
                _apply_precision_recursive(item, policy, full_key)


def restore_precision_to_fp32(
    state: Dict[str, Any],
    in_place: bool = False,
) -> Dict[str, Any]:
    """Restore all tensors to FP32 precision.

    Args:
        state: State dict (nested structure with tensors)
        in_place: Whether to modify state in-place or create copy

    Returns:
        State dict with all floating point tensors in FP32
    """
    if not in_place:
        state = copy.deepcopy(state)

    _restore_fp32_recursive(state)

    return state


def _restore_fp32_recursive(obj: Any) -> None:
    """Recursively restore tensors to FP32.

    Modifies obj in-place.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, torch.Tensor):
                if value.dtype in (torch.float16, torch.float64):
                    obj[key] = value.to(torch.float32)
            elif isinstance(value, (dict, list)):
                _restore_fp32_recursive(value)

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, torch.Tensor):
                if item.dtype in (torch.float16, torch.float64):
                    obj[i] = item.to(torch.float32)
            elif isinstance(item, (dict, list)):
                _restore_fp32_recursive(item)


def get_precision_statistics(state: Dict[str, Any]) -> Dict[str, Any]:
    """Compute statistics about tensor precision in state dict.

    Args:
        state: State dict (nested structure with tensors)

    Returns:
        Statistics dict with counts and sizes by precision
    """
    stats: Dict[str, Dict[str, float]] = {
        "fp32": {"count": 0.0, "bytes": 0.0},
        "fp16": {"count": 0.0, "bytes": 0.0},
        "fp64": {"count": 0.0, "bytes": 0.0},
        "int": {"count": 0.0, "bytes": 0.0},
        "other": {"count": 0.0, "bytes": 0.0},
    }

    def _count_recursive(obj: Any) -> None:
        if isinstance(obj, dict):
            for value in obj.values():
                _count_recursive(value)
        elif isinstance(obj, list):
            for item in obj:
                _count_recursive(item)
        elif isinstance(obj, torch.Tensor):
            num_bytes = obj.numel() * obj.element_size()

            if obj.dtype == torch.float32:
                stats["fp32"]["count"] += 1
                stats["fp32"]["bytes"] += num_bytes
            elif obj.dtype == torch.float16:
                stats["fp16"]["count"] += 1
                stats["fp16"]["bytes"] += num_bytes
            elif obj.dtype == torch.float64:
                stats["fp64"]["count"] += 1
                stats["fp64"]["bytes"] += num_bytes
            elif obj.dtype in (torch.int32, torch.int64, torch.int8, torch.int16):
                stats["int"]["count"] += 1
                stats["int"]["bytes"] += num_bytes
            else:
                stats["other"]["count"] += 1
                stats["other"]["bytes"] += num_bytes

    _count_recursive(state)

    # Add human-readable sizes
    total_bytes = sum(s["bytes"] for s in stats.values())
    for key in stats:
        stats[key]["mb"] = stats[key]["bytes"] / (1024 * 1024)
        if total_bytes > 0:
            stats[key]["percent"] = 100.0 * stats[key]["bytes"] / total_bytes
        else:
            stats[key]["percent"] = 0.0

    total_mb_value = float(total_bytes / (1024 * 1024))
    stats["total_mb"] = total_mb_value  # type: ignore[assignment]

    return stats
