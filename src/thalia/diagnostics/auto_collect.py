"""
Automatic Diagnostic Collection Utilities.

This module provides decorators and utilities for standardizing diagnostic
collection across the codebase, reducing boilerplate in get_diagnostics() methods.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

import torch


def auto_diagnostics(
    weights: Optional[Union[str, List[str]]] = None,
    spikes: Optional[Union[str, List[str]]] = None,
    traces: Optional[Union[str, List[str]]] = None,
    scalars: Optional[Union[str, List[str]]] = None,
    state_attrs: Optional[Union[str, List[str]]] = None,
):
    """Decorator to automatically collect common diagnostics.

    This decorator wraps get_diagnostics() methods and automatically collects
    standard metrics from specified attributes. The wrapped method can still
    add custom diagnostics by returning a dict that will be merged.

    Args:
        weights: Attribute name(s) containing weight tensors to analyze
        spikes: Attribute name(s) containing spike tensors to analyze
        traces: Attribute name(s) containing trace tensors (eligibility, NMDA, etc.)
        scalars: Attribute name(s) for scalar values to include
        state_attrs: Attribute name(s) from self.state to include

    Example:
        >>> @auto_diagnostics(
        >>>     weights=['d1_weights', 'd2_weights'],
        >>>     spikes=['d1_spikes', 'd2_spikes'],
        >>>     scalars=['dopamine', 'exploration_prob'],
        >>> )
        >>> def get_diagnostics(self) -> Dict[str, Any]:
        >>>     # Auto-collected: weight stats, spike stats, scalars
        >>>     # Only need to add custom metrics
        >>>     return {
        >>>         "net_weights": (self.d1_weights - self.d2_weights).mean().item(),
        >>>     }

    Returns:
        Decorated function that auto-collects diagnostics
    """
    # Normalize inputs to lists
    weights_list = _normalize_to_list(weights)
    spikes_list = _normalize_to_list(spikes)
    traces_list = _normalize_to_list(traces)
    scalars_list = _normalize_to_list(scalars)
    state_attrs_list = _normalize_to_list(state_attrs)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> Dict[str, Any]:
            # Start with auto-collected diagnostics
            diag = {}

            # Check if object has auto_collect_diagnostics method (from mixin)
            if hasattr(self, 'auto_collect_diagnostics'):
                # Build input dicts for auto_collect_diagnostics
                weight_dict = {}
                spike_dict = {}
                trace_dict = {}
                scalar_dict = {}

                # Collect weights
                for attr_name in weights_list:
                    if hasattr(self, attr_name):
                        tensor = getattr(self, attr_name)
                        if tensor is not None:
                            weight_dict[attr_name] = tensor

                # Collect spikes
                for attr_name in spikes_list:
                    if hasattr(self, attr_name):
                        tensor = getattr(self, attr_name)
                        if tensor is not None:
                            spike_dict[attr_name] = tensor
                    # Also check in state
                    elif hasattr(self, 'state') and hasattr(self.state, attr_name):
                        tensor = getattr(self.state, attr_name)
                        if tensor is not None:
                            spike_dict[attr_name] = tensor

                # Collect traces
                for attr_name in traces_list:
                    if hasattr(self, attr_name):
                        tensor = getattr(self, attr_name)
                        if tensor is not None:
                            trace_dict[attr_name] = tensor

                # Collect scalars
                for attr_name in scalars_list:
                    if hasattr(self, attr_name):
                        value = getattr(self, attr_name)
                        scalar_dict[attr_name] = value

                # Collect state attributes
                if hasattr(self, 'state'):
                    for attr_name in state_attrs_list:
                        if hasattr(self.state, attr_name):
                            value = getattr(self.state, attr_name)
                            scalar_dict[f"state_{attr_name}"] = value

                # Auto-collect using mixin
                diag.update(self.auto_collect_diagnostics(
                    weights=weight_dict if weight_dict else None,
                    spikes=spike_dict if spike_dict else None,
                    traces=trace_dict if trace_dict else None,
                    scalars=scalar_dict if scalar_dict else None,
                ))

            else:
                # Fallback: manual collection using basic methods
                diag.update(_manual_collect(
                    self,
                    weights_list,
                    spikes_list,
                    traces_list,
                    scalars_list,
                    state_attrs_list,
                ))

            # Call original function and merge results
            custom_diag = func(self, *args, **kwargs)
            if custom_diag:
                diag.update(custom_diag)

            return diag

        return wrapper
    return decorator


def _normalize_to_list(value: Optional[Union[str, List[str]]]) -> List[str]:
    """Normalize input to list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _manual_collect(
    obj: Any,
    weights: List[str],
    spikes: List[str],
    traces: List[str],
    scalars: List[str],
    state_attrs: List[str],
) -> Dict[str, Any]:
    """Manual diagnostic collection for objects without mixin."""
    diag = {}

    # Collect weights
    for attr_name in weights:
        if hasattr(obj, attr_name):
            tensor = getattr(obj, attr_name)
            if tensor is not None and isinstance(tensor, torch.Tensor):
                diag.update({
                    f"{attr_name}_mean": tensor.mean().item(),
                    f"{attr_name}_std": tensor.std().item(),
                    f"{attr_name}_min": tensor.min().item(),
                    f"{attr_name}_max": tensor.max().item(),
                })

    # Collect spikes
    for attr_name in spikes:
        tensor = None
        if hasattr(obj, attr_name):
            tensor = getattr(obj, attr_name)
        elif hasattr(obj, 'state') and hasattr(obj.state, attr_name):
            tensor = getattr(obj.state, attr_name)

        if tensor is not None and isinstance(tensor, torch.Tensor):
            diag[f"{attr_name}_active_count"] = int(tensor.sum().item())
            diag[f"{attr_name}_rate"] = tensor.float().mean().item()

    # Collect traces
    for attr_name in traces:
        if hasattr(obj, attr_name):
            tensor = getattr(obj, attr_name)
            if tensor is not None and isinstance(tensor, torch.Tensor):
                diag.update({
                    f"{attr_name}_mean": tensor.mean().item(),
                    f"{attr_name}_max": tensor.max().item(),
                })

    # Collect scalars
    for attr_name in scalars:
        if hasattr(obj, attr_name):
            value = getattr(obj, attr_name)
            if value is not None:
                diag[attr_name] = float(value) if not isinstance(value, (list, dict)) else value

    # Collect state attributes
    if hasattr(obj, 'state'):
        for attr_name in state_attrs:
            if hasattr(obj.state, attr_name):
                value = getattr(obj.state, attr_name)
                if value is not None:
                    diag[f"state_{attr_name}"] = float(value) if not isinstance(value, (list, dict, torch.Tensor)) else value

    return diag


__all__ = [
    'auto_diagnostics',
]
