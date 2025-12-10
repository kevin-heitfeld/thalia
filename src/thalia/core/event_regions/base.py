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
    accumulation_window_ms: float = 5.0  # Time window for buffering multi-source inputs
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

        # Input buffering for multi-source regions
        self._accumulation_window: float = config.accumulation_window_ms
        self._input_buffers: Dict[str, Optional[torch.Tensor]] = {}
        self._input_sizes: Dict[str, int] = {}
        self._last_input_times: Dict[str, float] = {}

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

            # Set dopamine on underlying region state if available
            if hasattr(self, 'impl') and hasattr(self.impl, 'state'):
                if self.impl.state is not None:
                    self.impl.state.dopamine = event.payload.level

            # Trigger region-specific learning updates
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

    # ===== Input Buffering Methods =====
    # Common functionality for regions that receive multi-source input

    def configure_input_sources(self, **source_sizes: int) -> None:
        """Configure expected input sources and their sizes.

        Example:
            region.configure_input_sources(cortex=512, hippocampus=256, pfc=128)

        Args:
            **source_sizes: Named arguments mapping source names to input sizes
        """
        self._input_sizes = source_sizes
        for source in source_sizes:
            self._input_buffers[source] = None
            self._last_input_times[source] = -1000.0

    def _buffer_input(self, source: str, spikes: torch.Tensor) -> None:
        """Store input spikes in buffer for given source.

        Args:
            source: Name of the source region
            spikes: Spike tensor to buffer
        """
        self._input_buffers[source] = spikes
        self._last_input_times[source] = self._current_time

    def _clear_input_buffers(self) -> None:
        """Clear all input buffers after processing."""
        for source in self._input_buffers:
            self._input_buffers[source] = None

    def _is_source_timed_out(self, source: str) -> bool:
        """Check if a source has timed out waiting for input.

        Args:
            source: Name of the source to check

        Returns:
            True if the source hasn't sent input within accumulation window
        """
        if source not in self._last_input_times:
            return True
        time_since_input = self._current_time - self._last_input_times[source]
        return time_since_input > self._accumulation_window

    def _build_combined_input(
        self,
        source_order: List[str],
        require_sources: Optional[List[str]] = None,
    ) -> Optional[torch.Tensor]:
        """Build combined input from buffers in specified order.

        Args:
            source_order: List of source names in the order to concatenate
            require_sources: List of sources that must be present (not timed out).
                           If None, only the first source is required.

        Returns:
            Combined tensor of shape [sum(input_sizes)], or None if not ready
        """
        if require_sources is None:
            require_sources = [source_order[0]] if source_order else []

        # Check if required sources are available
        for source in require_sources:
            if source not in self._input_buffers or self._input_buffers[source] is None:
                # Required source not available
                return None

        # Determine device from available buffers (ADR-005: no batch dimension)
        device = None
        for source in source_order:
            buf = self._input_buffers.get(source)
            if buf is not None:
                device = buf.device
                break

        if device is None:
            return None

        # Build parts, using zeros for missing/timed-out sources (ADR-005: 1D tensors)
        parts = []
        for source in source_order:
            buf = self._input_buffers.get(source)
            if buf is not None:
                parts.append(buf)
            elif source in self._input_sizes and self._input_sizes[source] > 0:
                # Use zeros for timed-out or missing optional sources
                parts.append(
                    torch.zeros(self._input_sizes[source], device=device)
                )

        if not parts:
            return None

        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    # ===== End Input Buffering Methods =====

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

    def reset_state(self) -> None:
        """Reset to initial state."""
        self._last_update_time = 0.0
        self._current_time = 0.0
        self._theta_phase = 0.0
        self._encoding_strength = 0.5
        self._retrieval_strength = 0.5
        self._dopamine_level = 0.0
