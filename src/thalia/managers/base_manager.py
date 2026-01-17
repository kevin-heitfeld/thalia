"""
Base Manager Class for Thalia Components

Provides standardized initialization pattern for all manager classes.
Managers are helper objects that encapsulate specific responsibilities
within brain regions (learning, homeostasis, exploration, etc.).

Design Philosophy:
==================
Managers follow a consistent pattern:
1. Accept a config object (manager-specific configuration)
2. Accept a context object (shared resources like device, dimensions)
3. Initialize state in a standard way
4. Provide reset_state() for trial boundaries

This standardization makes managers:
- Easy to understand and maintain
- Testable in isolation
- Composable within regions
- Extensible without breaking existing code

Usage Example:
==============
    from thalia.managers.base_manager import BaseManager, ManagerContext

    class MyManagerConfig:
        learning_rate: float = 0.01
        tau_ms: float = 20.0

    class MyManager(BaseManager[MyManagerConfig]):
        def __init__(self, config: MyManagerConfig, context: ManagerContext):
            super().__init__(config, context)
            # Manager-specific initialization
            self.weights = torch.zeros(context.n_output, context.n_input, device=context.device)

        def process(self, inputs):
            # Manager logic here
            pass
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Optional, TypeVar

import torch


@dataclass
class ManagerContext:
    """Shared context for manager initialization.

    This object contains resources commonly needed by managers:
    - Device for tensor allocation
    - Dimension information (input/output sizes)
    - Timestep (dt_ms) for temporal dynamics

    Benefits:
    - Single object to pass instead of many parameters
    - Extensible: Can add fields without breaking existing managers
    - Type-safe: Ensures all managers have consistent context

    Attributes:
        device: PyTorch device (cpu/cuda)
        n_input: Number of input neurons (optional)
        n_output: Number of output neurons (optional)
        dt_ms: Simulation timestep in milliseconds
        metadata: Additional context-specific data
    """

    device: torch.device
    n_input: Optional[int] = None
    n_output: Optional[int] = None
    dt_ms: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


TConfig = TypeVar("TConfig")


class BaseManager(ABC, Generic[TConfig]):
    """Abstract base class for all manager components.

    Managers are helper objects that encapsulate specific responsibilities
    within brain regions. Examples:
    - LearningManager: Handles weight updates (eligibility × dopamine)
    - ExplorationManager: Handles UCB tracking and adaptive exploration
    - PlasticityManager: Handles STDP learning for hippocampus

    All managers follow a standard pattern:
    1. Accept config (manager-specific settings)
    2. Accept context (shared resources)
    3. Provide reset_state() for trial boundaries
    4. Provide get_diagnostics() for monitoring

    This standardization makes managers easy to understand, test, and extend.
    """

    def __init__(self, config: TConfig, context: ManagerContext):
        """Initialize manager with config and context.

        Args:
            config: Manager-specific configuration
            context: Shared context (device, dimensions, dt_ms)
        """
        self.config = config
        self.context = context
        self.device = context.device

    def reset_state(self) -> None:
        """Reset manager state for new trial/episode.

        Override this method to clear any accumulated state.
        Called at trial boundaries or when resetting the simulation.

        Default implementation does nothing (stateless manager).
        """
        pass

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information for monitoring.

        Override this method to provide manager-specific metrics.
        Used for logging, debugging, and visualization.

        Returns:
            Dictionary of diagnostic metrics
        """
        return {}

    def to(self, device: torch.device) -> BaseManager:
        """Move manager tensors to specified device.

        Override this method if manager owns tensors.
        Default implementation updates device reference.

        Args:
            device: Target device

        Returns:
            Self (for chaining)
        """
        self.device = device
        self.context.device = device
        return self
