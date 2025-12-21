"""
Region State Protocol and Base Implementation.

Defines the protocol for region state management to enable:
- Consistent state serialization across all brain regions
- Checkpoint save/load for long-term training
- State reset and device transfer
- Versioned state format for backward compatibility

Design Philosophy:
==================
- **Protocol-based**: No inheritance complexity, clean separation of concerns
- **Version management**: STATE_VERSION field for migration support
- **Device-aware**: Explicit device handling for CPU/CUDA transfer
- **Type-safe**: Dataclasses with clear field documentation

Architecture Pattern:
=====================
1. **RegionState (Protocol)**: Abstract interface for state operations
   - to_dict() → Dict[str, Any]: Serialize state to dictionary
   - from_dict(data, device) → Self: Deserialize from dictionary
   - reset() → None: Reset state to initial conditions

2. **Concrete implementations**: Dataclass per region type
   - Example: PrefrontalState, HippocampusState, LayeredCortexState
   - Each contains region-specific state fields (spikes, traces, STP, etc.)
   - STATE_VERSION constant for backward compatibility

3. **Utility functions**:
   - save_region_state(state, path): Save to checkpoint file
   - load_region_state(StateClass, path, device): Load from checkpoint
   - transfer_state(state, device): Transfer to different device

Usage Example:
==============
```python
# Create region and capture state
region = Hippocampus(config)
state = region.get_state()  # Returns HippocampusState instance

# Save to checkpoint
save_region_state(state, "checkpoint_hippocampus_epoch100.pt")

# Load and restore
loaded_state = load_region_state(HippocampusState, "checkpoint_hippocampus_epoch100.pt", device="cuda")
region.load_state(loaded_state)

# Device transfer
cpu_state = transfer_state(gpu_state, device="cpu")
```

Migration Strategy:
===================
Phase 1 (this file): Foundation
- RegionState protocol
- Utility functions
- Type hints and validation

Phase 2: Migrate existing states (one at a time)
- Phase 2.1: PrefrontalState (simple, no STP)
- Phase 2.2: ThalamicRelayState (with STP)
- Phase 2.3: HippocampusState (with STP)
- Phase 2.4: LayeredCortexState (complex, with STP)

Phase 3: Complex regions
- Phase 3.1: CerebellumState (with STP)
- Phase 3.2: StriatumState (D1/D2, eligibility, STP)

See: docs/design/state-management-refactoring-plan.md

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Type, TypeVar
from pathlib import Path

import torch


# Type variable for state classes
TRegionState = TypeVar("TRegionState", bound="RegionState")


class RegionState(ABC):
    """Protocol for neural region state management.

    All region state classes should implement this protocol for consistent
    state serialization, checkpointing, and device transfer.

    Design: Protocol-based (not ABC with inheritance) to avoid multiple
    inheritance complexity. Each region implements this interface via
    duck typing.
    """

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary for checkpointing.

        Returns:
            Dictionary with all state fields, suitable for torch.save()
            Must include 'state_version' field for backward compatibility

        Example:
            {
                'state_version': 1,
                'spikes': tensor_or_none,
                'membrane': tensor_or_none,
                'traces': {...},
                'stp_state': {...} if STP enabled else None,
            }
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls: Type[TRegionState], data: Dict[str, Any], device: str) -> TRegionState:
        """Deserialize state from dictionary.

        Args:
            data: Dictionary from to_dict(), loaded from checkpoint
            device: Target device ('cpu', 'cuda', 'cuda:0', etc.)

        Returns:
            New state instance with data loaded to specified device

        Note:
            Should handle version migration if state_version differs
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset state to initial conditions.

        Clears:
        - Spike history
        - Membrane potentials
        - Traces (eligibility, STDP, etc.)
        - STP state (if present)

        Preserves:
        - Synaptic weights (NOT part of state, part of parameters)
        - Configuration
        """
        pass


@dataclass
class BaseRegionState:
    """Base implementation with common fields for most regions.

    Provides default implementations for common state elements.
    Regions can inherit from this or implement RegionState directly.

    Common fields:
    - spikes: Recent spike history
    - membrane: Membrane potentials
    - STATE_VERSION: Version number for migration
    """

    STATE_VERSION: int = 1
    """State format version for backward compatibility."""

    spikes: Optional[torch.Tensor] = None
    """Recent spike output [n_neurons] (bool or float)."""

    membrane: Optional[torch.Tensor] = None
    """Current membrane potentials [n_neurons] (float)."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize common state fields."""
        return {
            'state_version': self.STATE_VERSION,
            'spikes': self.spikes,
            'membrane': self.membrane,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: str) -> "BaseRegionState":
        """Deserialize common state fields."""
        version = data.get('state_version', 1)
        if version != cls.STATE_VERSION:
            # Future: Handle version migration
            pass

        # Transfer tensors to target device
        spikes = data.get('spikes')
        if spikes is not None and isinstance(spikes, torch.Tensor):
            spikes = spikes.to(device)

        membrane = data.get('membrane')
        if membrane is not None and isinstance(membrane, torch.Tensor):
            membrane = membrane.to(device)

        return cls(
            spikes=spikes,
            membrane=membrane,
        )

    def reset(self) -> None:
        """Reset common state fields."""
        self.spikes = None
        self.membrane = None


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def save_region_state(state: RegionState, path: str | Path) -> None:
    """Save region state to checkpoint file.

    Args:
        state: Region state instance (implements RegionState protocol)
        path: Path to checkpoint file (will create parent dirs)

    Example:
        >>> state = region.get_state()
        >>> save_region_state(state, "checkpoints/hippocampus_epoch100.pt")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = state.to_dict()
    torch.save(state_dict, path)


def load_region_state(
    state_class: Type[TRegionState],
    path: str | Path,
    device: str = "cpu",
) -> TRegionState:
    """Load region state from checkpoint file.

    Args:
        state_class: State class to instantiate (e.g., HippocampusState)
        path: Path to checkpoint file
        device: Target device for loaded tensors

    Returns:
        Loaded state instance on specified device

    Example:
        >>> state = load_region_state(HippocampusState, "checkpoint.pt", device="cuda")
        >>> region.load_state(state)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state_dict = torch.load(path, map_location=device)
    return state_class.from_dict(state_dict, device=device)


def transfer_state(state: TRegionState, device: str) -> TRegionState:
    """Transfer state to different device (CPU ↔ CUDA).

    Args:
        state: Source state instance
        device: Target device ('cpu', 'cuda', 'cuda:0', etc.)

    Returns:
        New state instance on target device

    Example:
        >>> gpu_state = region.get_state()  # On CUDA
        >>> cpu_state = transfer_state(gpu_state, device="cpu")
    """
    state_dict = state.to_dict()
    return type(state).from_dict(state_dict, device=device)


def get_state_version(state_dict: Dict[str, Any]) -> int:
    """Extract state version from serialized state.

    Args:
        state_dict: Dictionary from to_dict() or torch.load()

    Returns:
        Version number (default 1 if not present)

    Used for:
        - Version checking before deserialization
        - Migration decision logic
    """
    return state_dict.get('state_version', 1)


# =====================================================================
# TYPE VALIDATION
# =====================================================================

def validate_state_protocol(state_class: Type) -> bool:
    """Check if class implements RegionState protocol correctly.

    Args:
        state_class: Class to validate

    Returns:
        True if implements protocol, False otherwise

    Checks:
        - Has to_dict() method
        - Has from_dict() classmethod
        - Has reset() method

    Example:
        >>> assert validate_state_protocol(HippocampusState)
    """
    required_methods = ['to_dict', 'from_dict', 'reset']

    for method in required_methods:
        if not hasattr(state_class, method):
            return False

    # Check from_dict is classmethod
    if not isinstance(getattr(state_class, 'from_dict', None), classmethod):
        # In Python 3.11+, classmethods are callable, so check differently
        from_dict_attr = getattr(state_class, 'from_dict', None)
        if from_dict_attr is None or not callable(from_dict_attr):
            return False

    return True
