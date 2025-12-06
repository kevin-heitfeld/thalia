"""
Event-Driven Region Base Classes.

This module provides the base adapter class that wraps existing brain regions 
to work with the event-driven simulation framework.

Adapters handle:
1. Event translation: Convert events to region-specific inputs
2. Membrane decay: Apply decay between events (no wasted computation)
3. Output routing: Create events with appropriate delays
4. State tracking: Track last update time for decay calculation

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn

from ..event_system import (
    Event,
    EventType,
    SpikePayload,
    ThetaPayload,
    DopaminePayload,
    Connection,
    RegionInterface,
    get_axonal_delay,
)


@dataclass
class EventRegionConfig:
    """Configuration for event-driven region wrapper."""

    name: str  # Unique region name
    output_targets: List[str]  # Where to send output spikes
    membrane_tau_ms: float = 20.0  # Membrane time constant for decay
    device: str = "cpu"


class EventDrivenRegionBase(RegionInterface, nn.Module):
    """Base class for event-driven region adapters.

    Handles common functionality:
    - Time tracking for membrane decay
    - Theta phase tracking
    - Connection management
    - State monitoring

    Subclasses implement region-specific processing.
    """

    def __init__(self, config: EventRegionConfig):
        nn.Module.__init__(self)
        self._name = config.name
        self._output_targets = config.output_targets
        self._membrane_tau = config.membrane_tau_ms
        self._device = torch.device(config.device)

        # Time tracking
        self._last_update_time: float = 0.0
        self._current_time: float = 0.0

        # Theta state (updated by theta events)
        self._theta_phase: float = 0.0
        self._encoding_strength: float = 0.5
        self._retrieval_strength: float = 0.5

        # Dopamine state
        self._dopamine_level: float = 0.0

        # Build connections
        self._connections = [
            Connection(
                source=self._name,
                target=target,
                delay_ms=get_axonal_delay(self._name, target),
            )
            for target in self._output_targets
        ]

    @property
    def name(self) -> str:
        return self._name

    @property
    @abstractmethod
    def impl(self) -> Any:
        """Return the underlying brain region implementation.
        
        This provides a consistent interface to access the wrapped region
        regardless of which adapter type is being used.
        """
        ...

    def get_connections(self) -> List[Connection]:
        return self._connections

    def process_event(self, event: Event) -> List[Event]:
        """Process an incoming event and return output events.

        Handles common event types (theta, dopamine) and delegates
        spike processing to subclass.
        """
        self._current_time = event.time

        # Apply membrane decay since last update
        dt = event.time - self._last_update_time
        if dt > 0:
            self._apply_decay(dt)
        self._last_update_time = event.time

        # Handle event by type
        if event.event_type == EventType.THETA:
            return self._handle_theta(event)
        elif event.event_type == EventType.DOPAMINE:
            return self._handle_dopamine(event)
        elif event.event_type in (EventType.SPIKE, EventType.SENSORY):
            return self._handle_spikes(event)
        else:
            return []

    def _handle_theta(self, event: Event) -> List[Event]:
        """Update theta state from theta event."""
        if isinstance(event.payload, ThetaPayload):
            self._theta_phase = event.payload.phase
            self._encoding_strength = event.payload.encoding_strength
            self._retrieval_strength = event.payload.retrieval_strength
        return []  # Theta updates don't produce output events

    def _handle_dopamine(self, event: Event) -> List[Event]:
        """Update dopamine state from dopamine event."""
        if isinstance(event.payload, DopaminePayload):
            self._dopamine_level = event.payload.level
            # Trigger learning updates if needed
            self._on_dopamine(event.payload)
        return []  # Dopamine updates typically don't produce output events

    def _handle_spikes(self, event: Event) -> List[Event]:
        """Process incoming spikes - to be implemented by subclass."""
        if isinstance(event.payload, SpikePayload):
            output_spikes = self._process_spikes(
                event.payload.spikes,
                event.source,
            )

            # Create output events for each connection
            if output_spikes is not None and output_spikes.sum() > 0:
                return self._create_output_events(output_spikes)

        return []

    def _create_output_events(self, spikes: torch.Tensor) -> List[Event]:
        """Create output spike events for all connections."""
        events = []
        for conn in self._connections:
            event = Event(
                time=self._current_time + conn.delay_ms,
                event_type=EventType.SPIKE,
                source=self._name,
                target=conn.target,
                payload=SpikePayload(spikes=spikes.clone()),
            )
            events.append(event)
        return events

    # Abstract methods for subclasses
    def _apply_decay(self, dt_ms: float) -> None:
        """Apply membrane decay for dt milliseconds.

        Subclasses should implement exponential decay:
        membrane *= exp(-dt / tau)
        """
        pass

    def _process_spikes(
        self,
        input_spikes: torch.Tensor,
        source: str,
    ) -> Optional[torch.Tensor]:
        """Process incoming spikes and return output spikes.

        Subclasses implement region-specific spike processing.

        Args:
            input_spikes: Binary spike tensor from source region
            source: Name of source region (for routing)

        Returns:
            Output spike tensor, or None if no output
        """
        raise NotImplementedError

    def _on_dopamine(self, payload: DopaminePayload) -> None:
        """Handle dopamine signal - override for learning."""
        pass

    def get_state(self) -> Dict[str, Any]:
        """Return current state for monitoring."""
        return {
            "name": self._name,
            "current_time": self._current_time,
            "theta_phase": self._theta_phase,
            "encoding_strength": self._encoding_strength,
            "retrieval_strength": self._retrieval_strength,
            "dopamine": self._dopamine_level,
        }

    def reset(self) -> None:
        """Reset to initial state."""
        self._last_update_time = 0.0
        self._current_time = 0.0
        self._theta_phase = 0.0
        self._encoding_strength = 0.5
        self._retrieval_strength = 0.5
        self._dopamine_level = 0.0


class SimpleLIFRegion(EventDrivenRegionBase):
    """Simple LIF neuron population for testing the event system.

    This is a minimal implementation to verify the event-driven
    architecture works correctly before adapting the full regions.
    """

    def __init__(
        self,
        config: EventRegionConfig,
        n_neurons: int,
        n_inputs: int,
    ):
        super().__init__(config)

        self.n_neurons = n_neurons
        self.n_inputs = n_inputs

        # Neuron state
        self.membrane = torch.zeros(n_neurons, device=self._device)
        self.threshold = torch.ones(n_neurons, device=self._device)

        # Weights
        self.weights = nn.Parameter(
            torch.randn(n_neurons, n_inputs, device=self._device) * 0.1
        )

    def _apply_decay(self, dt_ms: float) -> None:
        """Apply exponential membrane decay."""
        decay = math.exp(-dt_ms / self._membrane_tau)
        self.membrane *= decay

    def _process_spikes(
        self,
        input_spikes: torch.Tensor,
        source: str,
    ) -> Optional[torch.Tensor]:
        """Process input spikes through LIF neurons."""
        # Flatten input if needed
        if input_spikes.dim() > 1:
            input_spikes = input_spikes.squeeze()

        # Compute input current
        if input_spikes.shape[0] == self.n_inputs:
            current = torch.matmul(self.weights, input_spikes.float())
        else:
            # Input size mismatch - skip or adapt
            return None

        # Update membrane
        self.membrane += current

        # Check for spikes
        spikes = (self.membrane >= self.threshold).float()

        # Reset spiked neurons
        self.membrane = torch.where(
            spikes > 0,
            torch.zeros_like(self.membrane),
            self.membrane,
        )

        return spikes

    def get_state(self) -> Dict[str, Any]:
        """Return current state."""
        state = super().get_state()
        state.update(
            {
                "membrane_mean": self.membrane.mean().item(),
                "membrane_max": self.membrane.max().item(),
            }
        )
        return state

    def reset(self) -> None:
        """Reset neuron state."""
        super().reset()
        self.membrane.zero_()
