"""
Pathway State Management - Base Protocol and Implementations.

This module provides the PathwayState protocol for standardized state serialization
across axonal pathways and synaptic connections. Parallel to RegionState for neural regions.

Design Principles:
==================
1. Protocol-based (no inheritance) - avoids multiple inheritance complexity
2. Versioned serialization - enables checkpoint migration
3. Device-aware - tensors move to correct device on load
4. Growth-compatible - state adapts when pathways grow

Key Components:
===============
- PathwayState: Protocol defining to_dict/from_dict/reset interface
- AxonalProjectionState: State for delay buffers in spike routing pathways
- Future: SynapticConnectionState for weight matrices (if needed)

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, ClassVar, Tuple

import torch


class PathwayState(ABC):
    """Protocol for pathway state serialization (parallel to RegionState).

    Pathways (AxonalProjection, etc.) implement this protocol to enable:
    1. Checkpoint save/load with delay buffer preservation
    2. State migration across versions
    3. Growth compatibility (buffer expansion)
    4. Device management

    Unlike RegionState, PathwayState focuses on:
    - Delay buffers (in-flight spikes)
    - Routing state (not weights - those are at dendrites)
    - Transmission dynamics (STP if enabled at pathway level)

    Example:
        @dataclass
        class MyPathwayState(PathwayState):
            STATE_VERSION: ClassVar[int] = 1

            delay_buffers: Dict[str, Tuple[torch.Tensor, int]]

            def to_dict(self) -> Dict[str, Any]:
                return {
                    "version": self.STATE_VERSION,
                    "delay_buffers": {
                        k: {"buffer": buf.cpu(), "ptr": ptr}
                        for k, (buf, ptr) in self.delay_buffers.items()
                    }
                }

            @classmethod
            def from_dict(cls, data, device):
                buffers = {
                    k: (v["buffer"].to(device), v["ptr"])
                    for k, v in data["delay_buffers"].items()
                }
                return cls(delay_buffers=buffers)

            def reset(self) -> None:
                for buf, _ in self.delay_buffers.values():
                    buf.zero_()

    See Also:
        docs/patterns/state-management.md - State management patterns
        docs/design/state-management-refactoring-plan.md - Full refactoring plan
    """

    # Subclasses should define this
    STATE_VERSION: ClassVar[int] = 1

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize pathway state to dictionary.

        Must include:
        - "version": int - STATE_VERSION for migration support
        - All state fields in serializable format
        - Tensors should be moved to CPU

        Returns:
            Dictionary with all state data
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any], device: Optional[torch.device] = None) -> "PathwayState":
        """Deserialize pathway state from dictionary with automatic migration.

        Should:
        1. Check version field
        2. Apply migrations if needed
        3. Move tensors to specified device
        4. Construct and return instance

        Args:
            data: Serialized state dictionary
            device: Target device for tensors (default: CPU)

        Returns:
            New instance with loaded state
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset pathway state to initial values.

        Should:
        - Zero tensors (buffers, traces)
        - Reset scalars to defaults
        - Preserve tensor shapes/devices
        """
        pass


@dataclass
class AxonalProjectionState(PathwayState):
    """State for AxonalProjection with delay buffers.

    Stores in-flight spikes for each source in the multi-source projection.
    Each source has an independent circular delay buffer that must be preserved
    across checkpoints to maintain temporal dynamics.

    Biological Justification:
    =========================
    Axonal conduction delays (1-100 m/s) mean spikes take 1-20ms to propagate.
    During this time, spikes are "in flight" in axons. Without state serialization,
    these spikes are lost across checkpoint boundaries, breaking temporal dynamics
    (e.g., D1/D2 opponent pathway timing in striatum).

    State Components:
    =================
    - delay_buffers: Dict mapping source_key → (buffer, ptr, max_delay, size)
      - buffer: torch.Tensor [max_delay+1, size] - circular buffer of spikes
      - ptr: int - current write position (0 to max_delay)
      - max_delay: int - buffer depth in timesteps
      - size: int - number of neurons in source

    Growth Compatibility:
    =====================
    When source region grows, buffer expands while preserving in-flight spikes.
    See CircularDelayBuffer.grow() for implementation.

    Example:
        # Create state
        state = AxonalProjectionState(delay_buffers={
            "cortex:l5": (
                torch.zeros(6, 128),  # 5ms delay + 1
                2,  # current pointer
                5,  # max_delay
                128  # n_neurons
            )
        })

        # Save
        data = state.to_dict()

        # Load
        restored = AxonalProjectionState.from_dict(data, device="cuda")
    """

    STATE_VERSION: ClassVar[int] = 1

    # Dict mapping source_key → (buffer, ptr, max_delay, size)
    delay_buffers: Dict[str, Tuple[torch.Tensor, int, int, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize delay buffers to dictionary.

        Returns:
            {
                "version": 1,
                "delay_buffers": {
                    "source_key": {
                        "buffer": Tensor[max_delay+1, size],
                        "ptr": int,
                        "max_delay": int,
                        "size": int
                    },
                    ...
                }
            }
        """
        return {
            "version": self.STATE_VERSION,
            "delay_buffers": {
                key: {
                    "buffer": buf.cpu(),
                    "ptr": ptr,
                    "max_delay": max_delay,
                    "size": size,
                }
                for key, (buf, ptr, max_delay, size) in self.delay_buffers.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: Optional[torch.device] = None) -> "AxonalProjectionState":
        """Deserialize delay buffers from dictionary.

        Args:
            data: Serialized state dictionary
            device: Target device for tensors (default: CPU)

        Returns:
            AxonalProjectionState with restored buffers
        """
        if device is None:
            device = torch.device("cpu")

        # Check version (future: add migrations)
        version = data.get("version", 1)
        if version != cls.STATE_VERSION:
            # Future: apply migrations here
            pass

        # Restore delay buffers
        delay_buffers = {}
        for key, buf_data in data["delay_buffers"].items():
            delay_buffers[key] = (
                buf_data["buffer"].to(device),
                buf_data["ptr"],
                buf_data["max_delay"],
                buf_data["size"],
            )

        return cls(delay_buffers=delay_buffers)

    def reset(self) -> None:
        """Reset all delay buffers to zero (clear in-flight spikes)."""
        for buf, _, _, _ in self.delay_buffers.values():
            buf.zero_()


# Utility functions for state management

def save_pathway_state(pathway: Any) -> Dict[str, Any]:
    """Save pathway state using its get_state() method.

    Args:
        pathway: Pathway object with get_state() method

    Returns:
        Serialized state dictionary
    """
    if hasattr(pathway, "get_state"):
        state = pathway.get_state()
        return state.to_dict()
    else:
        raise AttributeError(f"Pathway {type(pathway).__name__} does not implement get_state()")


def load_pathway_state(pathway: Any, data: Dict[str, Any]) -> None:
    """Load pathway state using its load_state() method.

    Args:
        pathway: Pathway object with load_state() method
        data: Serialized state dictionary
    """
    if hasattr(pathway, "load_state"):
        state_class = type(pathway.get_state())
        device = getattr(pathway, "device", torch.device("cpu"))
        state = state_class.from_dict(data, device)
        pathway.load_state(state)
    else:
        raise AttributeError(f"Pathway {type(pathway).__name__} does not implement load_state()")
