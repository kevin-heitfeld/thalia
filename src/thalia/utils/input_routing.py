"""
Standardized Multi-Port Input Routing.

This module provides a unified interface for handling multi-port inputs across
brain regions, eliminating the need for ad-hoc isinstance() checks and manual
dict.get() chains.

**Design Goals**:
1. **Type Safety**: Handle both Dict[str, Tensor] and Tensor inputs uniformly
2. **Alias Support**: Map multiple port names to canonical names (e.g., "input" → "sensory")
3. **Default Values**: Provide fallback tensors when optional ports are missing
4. **Error Messages**: Clear feedback when required ports are missing

**Usage Pattern**:

```python
# In region forward() method:
from thalia.utils.input_routing import InputRouter

def forward(self, inputs, **kwargs):
    # Define port mapping with aliases
    routed = InputRouter.route(
        inputs,
        port_mapping={
            "sensory": ["sensory", "input", "default"],  # canonical: [aliases...]
            "feedback": ["l6_feedback", "feedback"],
        },
        defaults={
            "feedback": torch.zeros(self.n_trn, device=self.device),  # optional port
        },
        required=["sensory"],  # must be present (after alias resolution)
    )

    sensory_input = routed["sensory"]
    feedback_input = routed["feedback"]
```

**Backward Compatibility**:
- Single tensor input: Automatically wrapped as {"default": tensor}
- No breaking changes to existing code

**Architecture Pattern**:
This replaces the repeated pattern in regions:

```python
# OLD (repeated in 6+ regions):
if isinstance(inputs, torch.Tensor):
    input_spikes = inputs
else:
    input_spikes = inputs.get("sensory", inputs.get("input", inputs.get("default")))
    feedback = inputs.get("l6_feedback", None)
    if input_spikes is None:
        raise ValueError(...)

# NEW (standardized):
routed = InputRouter.route(
    inputs,
    port_mapping={"sensory": ["sensory", "input", "default"], "feedback": ["l6_feedback"]},
    defaults={"feedback": None},
    required=["sensory"],
)
input_spikes = routed["sensory"]
feedback = routed["feedback"]
```

References:
- Architecture Review 2025-12-20, Tier 2, Recommendation 2.2
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch


class InputRouter:
    """Standardized multi-port input routing with alias resolution."""

    @staticmethod
    def route(
        inputs: Union[Dict[str, torch.Tensor], torch.Tensor],
        port_mapping: Dict[str, List[str]],
        defaults: Optional[Dict[str, Optional[torch.Tensor]]] = None,
        required: Optional[List[str]] = None,
        component_name: str = "Component",
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Route inputs to canonical port names with alias resolution.

        Args:
            inputs: Either a dict of named inputs or a single tensor (backward compat)
            port_mapping: Dict mapping canonical port names to lists of aliases.
                         Example: {"sensory": ["sensory", "input", "default"]}
                         First alias found in inputs dict is used.
            defaults: Optional dict of default values for ports when not found.
                     Use None for optional ports with no default tensor.
            required: List of canonical port names that must be present (after defaults).
                     If a required port is missing and has no default, raises ValueError.
            component_name: Name of component for error messages (e.g., "ThalamicRelay")

        Returns:
            Dict mapping canonical port names to tensors (or None for optional missing ports)

        Raises:
            ValueError: If a required port is missing and has no default

        Example:
            >>> routed = InputRouter.route(
            ...     {"input": tensor1, "l6_feedback": tensor2},
            ...     port_mapping={
            ...         "sensory": ["sensory", "input", "default"],
            ...         "feedback": ["l6_feedback", "feedback"],
            ...     },
            ...     defaults={"feedback": torch.zeros(100)},
            ...     required=["sensory"],
            ... )
            >>> sensory = routed["sensory"]  # tensor1 (via "input" alias)
            >>> feedback = routed["feedback"]  # tensor2
        """
        # Convert single tensor to dict (backward compatibility)
        if isinstance(inputs, torch.Tensor):
            inputs = {"default": inputs}

        # Resolve aliases and build output dict
        routed: Dict[str, Optional[torch.Tensor]] = {}
        defaults = defaults or {}
        required = required or []

        for canonical_name, aliases in port_mapping.items():
            # Try each alias in order
            resolved_tensor = None
            for alias in aliases:
                if alias in inputs:
                    resolved_tensor = inputs[alias]
                    break

            # If not found, try default
            if resolved_tensor is None:
                if canonical_name in defaults:
                    resolved_tensor = defaults[canonical_name]
                elif canonical_name in required:
                    # Required port missing with no default
                    available_keys = list(inputs.keys())
                    raise ValueError(
                        f"{component_name}.forward: Required port '{canonical_name}' not found. "
                        f"Tried aliases: {aliases}. Available keys: {available_keys}"
                    )

            routed[canonical_name] = resolved_tensor

        return routed

    @staticmethod
    def concatenate_sources(
        inputs: Union[Dict[str, torch.Tensor], torch.Tensor],
        component_name: str = "Component",
        n_input: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Concatenate all source inputs into single tensor.

        This is a common pattern for regions that accept multiple sources but
        treat them uniformly (e.g., striatum, cerebellum).

        Supports zero-input execution for clock-driven architecture:
        when empty dict provided, returns zeros of size n_input.

        Args:
            inputs: Either dict of named inputs or single tensor (backward compat)
            component_name: Name of component for error messages
            n_input: Expected input size (required for empty dict handling)
            device: Device for zero tensor creation (required for empty dict)

        Returns:
            Concatenated 1D tensor of all inputs, or zeros if empty

        Example:
            >>> input_spikes = InputRouter.concatenate_sources(
            ...     {"cortex": cortex_spikes, "hippocampus": hippo_spikes}
            ... )
            >>> # Zero-input execution (clock-driven)
            >>> input_spikes = InputRouter.concatenate_sources(
            ...     {}, n_input=128, device=device
            ... )
        """
        if isinstance(inputs, torch.Tensor):
            return inputs
        elif isinstance(inputs, dict):
            if not inputs:
                # Empty input: return zeros for clock-driven execution
                if n_input is None or device is None:
                    raise ValueError(
                        f"{component_name}.forward: Empty input dict requires n_input and device "
                        f"for zero-input execution (clock-driven architecture)"
                    )
                return torch.zeros(n_input, dtype=torch.bool, device=device)
            return torch.cat(list(inputs.values()), dim=0)
        else:
            raise TypeError(
                f"{component_name}.forward: inputs must be Tensor or Dict[str, Tensor], "
                f"got {type(inputs)}"
            )
