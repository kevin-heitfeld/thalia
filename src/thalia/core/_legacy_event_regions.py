"""
Event-Driven Region Adapters.

This module provides adapters that wrap existing brain regions to work
with the event-driven simulation framework. These adapters handle:

1. Event translation: Convert events to region-specific inputs
2. Membrane decay: Apply decay between events (no wasted computation)
3. Output routing: Create events with appropriate delays
4. State tracking: Track last update time for decay calculation

The adapters allow gradual migration from the sequential BrainSystem
to the parallel event-driven architecture.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn

from .event_system import (
    Event, EventType, SpikePayload, ThetaPayload, DopaminePayload,
    Connection, RegionInterface, get_axonal_delay,
)


@dataclass
class EventRegionConfig:
    """Configuration for event-driven region wrapper."""
    name: str                            # Unique region name
    output_targets: List[str]            # Where to send output spikes
    membrane_tau_ms: float = 20.0        # Membrane time constant for decay
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
        state.update({
            "membrane_mean": self.membrane.mean().item(),
            "membrane_max": self.membrane.max().item(),
        })
        return state

    def reset(self) -> None:
        """Reset neuron state."""
        super().reset()
        self.membrane.zero_()


# ============================================================================
# Wrapped Brain Region Adapters
# ============================================================================

class EventDrivenCortex(EventDrivenRegionBase):
    """Event-driven wrapper for LayeredCortex.

    Adapts the existing LayeredCortex to work with the event-driven
    simulation framework. Handles:
    - Layer-specific input routing
    - Membrane decay between events
    - Dual output: L2/3 → cortical targets, L5 → subcortical targets
    - Top-down projection from PFC

    Architecture:
        Sensory Input → L4 → L2/3 → L5
                              ↓      ↓
                          (hippocampus, pfc)  (striatum)
                              
        PFC → top-down projection → L2/3
    """

    def __init__(
        self,
        config: EventRegionConfig,
        cortex: Any,  # LayeredCortex instance
        pfc_size: int = 0,  # Size of PFC output for top-down projection
    ):
        super().__init__(config)
        self.cortex = cortex
        self._pfc_size = pfc_size

        # Track pending top-down modulation
        self._pending_top_down: Optional[torch.Tensor] = None

        # Accumulated input (for handling multiple sources)
        self._accumulated_input: Optional[torch.Tensor] = None
        
        # Top-down projection from PFC to L2/3
        # Only create if PFC size is provided
        self._top_down_projection: Optional[torch.nn.Linear] = None
        if pfc_size > 0 and hasattr(cortex, 'l23_size'):
            self._top_down_projection = torch.nn.Linear(
                pfc_size, cortex.l23_size, bias=False
            )
            # Initialize with small weights (modulatory, not driving)
            torch.nn.init.normal_(
                self._top_down_projection.weight, 
                mean=0.0, 
                std=0.1 / pfc_size**0.5
            )

    def _apply_decay(self, dt_ms: float) -> None:
        """Apply decay to cortex neurons.

        Directly decay the membrane potentials of the LIF neurons
        in each cortical layer.
        """
        import math
        decay_factor = math.exp(-dt_ms / self._membrane_tau)

        # Decay each layer's neurons
        for layer_name in ['l4_neurons', 'l23_neurons', 'l5_neurons']:
            neurons = getattr(self.cortex, layer_name, None)
            if neurons is not None and hasattr(neurons, 'membrane'):
                if neurons.membrane is not None:
                    neurons.membrane *= decay_factor

        # Also decay the recurrent activity trace
        if hasattr(self.cortex, 'state') and self.cortex.state is not None:
            if self.cortex.state.l23_recurrent_activity is not None:
                self.cortex.state.l23_recurrent_activity *= decay_factor

    def _process_spikes(
        self,
        input_spikes: torch.Tensor,
        source: str,
    ) -> Optional[torch.Tensor]:
        """Process input through cortex layers."""
        # Ensure batch dimension
        if input_spikes.dim() == 1:
            input_spikes = input_spikes.unsqueeze(0)

        # Handle top-down input from PFC
        if source == "pfc":
            # Project PFC spikes to L2/3 size if projection exists
            if self._top_down_projection is not None:
                projected = self._top_down_projection(input_spikes.float())
                # Convert to modulatory signal (between 0 and 1)
                self._pending_top_down = torch.sigmoid(projected)
            else:
                # No projection - skip top-down (sizes don't match)
                self._pending_top_down = None
            return None  # Top-down alone doesn't drive output

        # Forward through cortex with current theta modulation
        output = self.cortex.forward(
            input_spikes,
            encoding_mod=self._encoding_strength,
            retrieval_mod=self._retrieval_strength,
            top_down=self._pending_top_down,
        )

        # Clear pending top-down after use
        self._pending_top_down = None

        # Output is typically L5 activity (for subcortical targets)
        # But we might want L2/3 for cortical targets
        return output.squeeze()

    def _create_output_events(self, spikes: torch.Tensor) -> List[Event]:
        """Create layer-specific output events.

        L5 output → subcortical targets (striatum)
        L2/3 output → cortical targets (hippocampus, pfc)
        """
        events = []

        # Get layer-specific spikes if available
        l23_spikes = None
        l5_spikes = None

        if hasattr(self.cortex, 'state') and self.cortex.state is not None:
            l23_spikes = self.cortex.state.l23_spikes
            l5_spikes = self.cortex.state.l5_spikes

        # If we don't have separate layer outputs, use the combined output
        if l23_spikes is None or l5_spikes is None:
            # Fall back to base implementation
            return super()._create_output_events(spikes)

        # Create events for each connection with appropriate layer routing
        for conn in self._connections:
            target = conn.target

            # Choose appropriate layer output for target
            if target in ["striatum", "motor"]:
                # Subcortical targets get L5 output
                output_spikes = l5_spikes
                source_layer = "L5"
            else:
                # Cortical targets (hippocampus, pfc) get L2/3 output
                output_spikes = l23_spikes
                source_layer = "L23"

            if output_spikes is not None and output_spikes.sum() > 0:
                event = Event(
                    time=self._current_time + conn.delay_ms,
                    event_type=EventType.SPIKE,
                    source=self._name,
                    target=target,
                    payload=SpikePayload(
                        spikes=output_spikes.squeeze().clone(),
                        source_layer=source_layer,
                    ),
                )
                events.append(event)

        return events

    def get_state(self) -> Dict[str, Any]:
        """Return cortex state."""
        state = super().get_state()
        # Add cortex-specific diagnostics
        if hasattr(self.cortex, 'get_diagnostics'):
            state["cortex"] = self.cortex.get_diagnostics()
        return state


class EventDrivenHippocampus(EventDrivenRegionBase):
    """Event-driven wrapper for TrisynapticHippocampus.

    Adapts the hippocampus for event-driven simulation. Handles:
    - Phase determination from theta (ENCODE/DELAY/RETRIEVE)
    - EC direct input pathway (raw sensory for comparison)
    - STP on mossy fibers (facilitating)

    Architecture:
        Cortex (EC L2) → DG → CA3 → CA1 → Output
                         ↑    ↑
                 EC L3 direct path
    """

    def __init__(
        self,
        config: EventRegionConfig,
        hippocampus: Any,  # TrisynapticHippocampus instance
    ):
        super().__init__(config)
        self.hippocampus = hippocampus

        # Track EC direct input (from sensory, bypasses cortex)
        self._ec_direct_input: Optional[torch.Tensor] = None

        # Import TrialPhase for phase determination
        from thalia.regions.hippocampus import TrialPhase
        self._TrialPhase = TrialPhase

    def _get_trial_phase(self) -> Any:
        """Determine trial phase from theta modulation.

        Encoding strength high → ENCODE phase
        Retrieval strength high → RETRIEVE phase
        Neither dominant → DELAY phase
        """
        if self._encoding_strength > 0.6:
            return self._TrialPhase.ENCODE
        elif self._retrieval_strength > 0.6:
            return self._TrialPhase.RETRIEVE
        else:
            return self._TrialPhase.DELAY

    def _apply_decay(self, dt_ms: float) -> None:
        """Apply decay to hippocampal neurons."""
        import math
        decay_factor = math.exp(-dt_ms / self._membrane_tau)

        # Decay neurons in each subregion
        for layer_name in ['dg_neurons', 'ca3_neurons', 'ca1_neurons']:
            neurons = getattr(self.hippocampus, layer_name, None)
            if neurons is not None and hasattr(neurons, 'membrane'):
                if neurons.membrane is not None:
                    neurons.membrane *= decay_factor

        # Decay NMDA trace (slower time constant)
        if hasattr(self.hippocampus, 'state') and self.hippocampus.state is not None:
            if self.hippocampus.state.nmda_trace is not None:
                nmda_decay = math.exp(-dt_ms / 100.0)  # ~100ms NMDA time constant
                self.hippocampus.state.nmda_trace *= nmda_decay

    def _handle_spikes(self, event: Event) -> List[Event]:
        """Override to handle EC direct input specially."""
        if isinstance(event.payload, SpikePayload):
            # Check if this is EC direct input (from sensory or special pathway)
            if event.source == "sensory_direct" or event.payload.source_layer == "EC_L3":
                self._ec_direct_input = event.payload.spikes
                return []  # Don't process yet, wait for main input

            # Process main input
            output_spikes = self._process_spikes(
                event.payload.spikes,
                event.source,
            )

            # Clear EC direct input after use
            self._ec_direct_input = None

            if output_spikes is not None and output_spikes.sum() > 0:
                return self._create_output_events(output_spikes)

        return []

    def _process_spikes(
        self,
        input_spikes: torch.Tensor,
        source: str,
    ) -> Optional[torch.Tensor]:
        """Process input through hippocampal circuit."""
        # Ensure batch dimension
        if input_spikes.dim() == 1:
            input_spikes = input_spikes.unsqueeze(0)

        # Determine phase from theta
        phase = self._get_trial_phase()

        # Forward through hippocampus
        output = self.hippocampus.forward(
            input_spikes,
            phase=phase,
            encoding_mod=self._encoding_strength,
            retrieval_mod=self._retrieval_strength,
            dt=1.0,  # Event-driven doesn't use fixed dt
            ec_direct_input=self._ec_direct_input,
        )

        return output.squeeze()

    def new_trial(self) -> None:
        """Signal new trial to hippocampus."""
        if hasattr(self.hippocampus, 'new_trial'):
            self.hippocampus.new_trial()
        self._ec_direct_input = None

    def get_state(self) -> Dict[str, Any]:
        """Return hippocampus state."""
        state = super().get_state()
        state["trial_phase"] = self._get_trial_phase().name
        if hasattr(self.hippocampus, 'get_diagnostics'):
            state["hippocampus"] = self.hippocampus.get_diagnostics()
        return state


class EventDrivenPFC(EventDrivenRegionBase):
    """Event-driven wrapper for Prefrontal Cortex.

    Handles working memory maintenance and dopamine-gated learning.

    Key features:
    - Working memory: Maintained via recurrent activity in the PFC
    - Dopamine gating: DA controls what enters working memory
    - Top-down control: Sends modulatory signals to cortex
    - Input buffering: Accumulates inputs from cortex and hippocampus

    The PFC.forward() method handles:
    - input_spikes: Concatenated input from cortex (L2/3) and hippocampus
    - encoding_mod/retrieval_mod: Theta modulation
    - dopamine_signal: For gating working memory updates
    
    Input Buffering:
    ================
    PFC receives inputs from both cortex and hippocampus on separate pathways.
    These are buffered and concatenated before processing. This allows for
    event-driven operation where cortex and hippocampus spikes arrive at
    different times.
    """

    def __init__(
        self,
        config: EventRegionConfig,
        pfc: Any,  # Prefrontal instance
        cortex_input_size: int = 0,
        hippocampus_input_size: int = 0,
    ):
        super().__init__(config)
        self.pfc = pfc
        
        # Input sizes for buffering (set via configure_inputs)
        self._cortex_input_size = cortex_input_size
        self._hippocampus_input_size = hippocampus_input_size
        
        # Input buffers - accumulate spikes from different sources
        self._cortex_buffer: Optional[torch.Tensor] = None
        self._hippocampus_buffer: Optional[torch.Tensor] = None
        
        # Time window for accumulating inputs (ms)
        self._accumulation_window: float = 5.0
        self._last_cortex_time: float = -1000.0
        self._last_hippocampus_time: float = -1000.0

        # Accumulate dopamine signal for next forward pass
        self._pending_dopamine_signal: float = 0.0
    
    def configure_inputs(
        self, 
        cortex_input_size: int, 
        hippocampus_input_size: int,
    ) -> None:
        """Configure input sizes for buffering.
        
        Call this after construction if sizes weren't provided.
        """
        self._cortex_input_size = cortex_input_size
        self._hippocampus_input_size = hippocampus_input_size
    
    def _clear_buffers(self) -> None:
        """Clear input buffers."""
        self._cortex_buffer = None
        self._hippocampus_buffer = None

    def _apply_decay(self, dt_ms: float) -> None:
        """Apply decay to PFC neurons."""
        import math
        decay_factor = math.exp(-dt_ms / self._membrane_tau)

        # Decay PFC neurons
        if hasattr(self.pfc, 'neurons') and hasattr(self.pfc.neurons, 'membrane'):
            if self.pfc.neurons.membrane is not None:
                self.pfc.neurons.membrane *= decay_factor

        # Dopamine decays via update() call with 0 signal
        # The update() method handles decay internally
        if hasattr(self.pfc, 'dopamine_system'):
            self.pfc.dopamine_system.update(0.0, dt_ms)

    def _on_dopamine(self, payload: DopaminePayload) -> None:
        """Handle dopamine signal for PFC.

        DA level gets passed to PFC.forward() which uses it to:
        - Gate what enters working memory (high DA = update WM)
        - Modulate learning (via dopamine-gated STDP)
        """
        # Store dopamine signal for next forward pass
        self._pending_dopamine_signal = payload.level

    def _process_spikes(
        self,
        input_spikes: torch.Tensor,
        source: str,
    ) -> Optional[torch.Tensor]:
        """Buffer input spikes and process when we have both inputs.
        
        PFC requires concatenated input from cortex and hippocampus.
        This method buffers inputs and processes when:
        1. We have input from both sources, OR
        2. One input times out (process with zeros for missing source)
        """
        if input_spikes.dim() == 1:
            input_spikes = input_spikes.unsqueeze(0)
        
        # Buffer input based on source
        if source == "cortex":
            self._cortex_buffer = input_spikes
            self._last_cortex_time = self._current_time
        elif source == "hippocampus":
            self._hippocampus_buffer = input_spikes
            self._last_hippocampus_time = self._current_time
        else:
            # Unknown source - try to process directly if sizes match
            return self._forward_pfc(input_spikes)
        
        # Check if we should process now
        # Process if we have both inputs, or if input sizes weren't configured
        if self._cortex_input_size == 0 and self._hippocampus_input_size == 0:
            # Sizes not configured - just pass through (legacy mode)
            return self._forward_pfc(input_spikes)
        
        # Build combined input
        combined = self._build_combined_input()
        if combined is not None:
            result = self._forward_pfc(combined)
            self._clear_buffers()
            return result
        
        # Don't have complete input yet - return None (no output)
        return None
    
    def _build_combined_input(self) -> Optional[torch.Tensor]:
        """Build combined input from buffers if ready."""
        # If we have both buffers, combine them
        if self._cortex_buffer is not None and self._hippocampus_buffer is not None:
            return torch.cat([self._cortex_buffer, self._hippocampus_buffer], dim=-1)
        
        # If only cortex arrived and hippocampus timed out, use zeros
        if self._cortex_buffer is not None:
            time_since_hippocampus = self._current_time - self._last_hippocampus_time
            if time_since_hippocampus > self._accumulation_window:
                # Hippocampus timed out - use zeros
                batch_size = self._cortex_buffer.shape[0]
                zeros = torch.zeros(
                    batch_size, self._hippocampus_input_size,
                    device=self._cortex_buffer.device,
                )
                return torch.cat([self._cortex_buffer, zeros], dim=-1)
        
        # If only hippocampus arrived and cortex timed out, use zeros
        if self._hippocampus_buffer is not None:
            time_since_cortex = self._current_time - self._last_cortex_time
            if time_since_cortex > self._accumulation_window:
                # Cortex timed out - use zeros  
                batch_size = self._hippocampus_buffer.shape[0]
                zeros = torch.zeros(
                    batch_size, self._cortex_input_size,
                    device=self._hippocampus_buffer.device,
                )
                return torch.cat([zeros, self._hippocampus_buffer], dim=-1)
        
        # Not ready yet
        return None
    
    def _forward_pfc(self, combined_input: torch.Tensor) -> torch.Tensor:
        """Forward combined input through PFC."""
        # Forward through PFC with theta modulation and dopamine
        output = self.pfc.forward(
            combined_input,
            dt=1.0,  # Event-driven doesn't use fixed dt
            encoding_mod=self._encoding_strength,
            retrieval_mod=self._retrieval_strength,
            dopamine_signal=self._pending_dopamine_signal,
        )

        # Clear pending dopamine after use
        self._pending_dopamine_signal = 0.0

        return output.squeeze()

    def learn(
        self,
        input_spikes: torch.Tensor,
        output_spikes: torch.Tensor,
        reward: float = 0.0,
    ) -> None:
        """Apply dopamine-gated STDP learning."""
        if hasattr(self.pfc, 'learn'):
            self.pfc.learn(
                input_spikes=input_spikes,
                output_spikes=output_spikes,
                reward=reward,
            )

    def get_state(self) -> Dict[str, Any]:
        """Return PFC state."""
        state = super().get_state()
        if hasattr(self.pfc, 'state') and self.pfc.state is not None:
            state["wm_active"] = self.pfc.state.working_memory is not None
            if self.pfc.state.working_memory is not None:
                state["wm_mean"] = float(self.pfc.state.working_memory.mean())
            state["gate_value"] = float(self.pfc.dopamine_system.get_gate()) if hasattr(self.pfc, 'dopamine_system') else 0.0
        return state


class EventDrivenStriatum(EventDrivenRegionBase):
    """Event-driven wrapper for Striatum.

    Handles action selection and reinforcement learning.

    Key features:
    - D1/D2 pathway balance for action selection
    - Dopamine-modulated learning (three-factor rule)
    - RPE-based updates via dopamine system
    - Input buffering: Accumulates inputs from cortex, hippocampus, and PFC

    The Striatum.forward() method handles:
    - input_spikes: Concatenated input from cortex (L5) + hippocampus + PFC
    - encoding_mod/retrieval_mod: Theta modulation
    - explore: Whether to use exploration
    
    Input Buffering:
    ================
    Striatum receives inputs from cortex L5, hippocampus, and PFC on separate pathways.
    These are buffered and concatenated before processing. 
    Order: [cortex_l5 | hippocampus | pfc]
    """

    def __init__(
        self,
        config: EventRegionConfig,
        striatum: Any,  # Striatum instance
        cortex_input_size: int = 0,
        hippocampus_input_size: int = 0,
        pfc_input_size: int = 0,
    ):
        super().__init__(config)
        self.striatum = striatum
        
        # Input sizes for buffering
        self._cortex_input_size = cortex_input_size
        self._hippocampus_input_size = hippocampus_input_size
        self._pfc_input_size = pfc_input_size
        
        # Input buffers - accumulate spikes from different sources
        self._cortex_buffer: Optional[torch.Tensor] = None
        self._hippocampus_buffer: Optional[torch.Tensor] = None
        self._pfc_buffer: Optional[torch.Tensor] = None
        
        # Time window for accumulating inputs (ms)
        self._accumulation_window: float = 5.0
        self._last_cortex_time: float = -1000.0
        self._last_hippocampus_time: float = -1000.0
        self._last_pfc_time: float = -1000.0

        # Track recent activity for learning
        self._recent_input: Optional[torch.Tensor] = None
        self._recent_output: Optional[torch.Tensor] = None
        self._selected_action: Optional[int] = None
    
    def configure_inputs(
        self,
        cortex_input_size: int,
        hippocampus_input_size: int,
        pfc_input_size: int,
    ) -> None:
        """Configure input sizes for buffering."""
        self._cortex_input_size = cortex_input_size
        self._hippocampus_input_size = hippocampus_input_size
        self._pfc_input_size = pfc_input_size
    
    def _clear_buffers(self) -> None:
        """Clear input buffers."""
        self._cortex_buffer = None
        self._hippocampus_buffer = None
        self._pfc_buffer = None

    def _apply_decay(self, dt_ms: float) -> None:
        """Apply decay to striatal neurons."""
        import math
        decay_factor = math.exp(-dt_ms / self._membrane_tau)

        # Decay D1 and D2 pathway neurons
        if hasattr(self.striatum, 'd1_neurons') and self.striatum.d1_neurons.membrane is not None:
            self.striatum.d1_neurons.membrane *= decay_factor
        if hasattr(self.striatum, 'd2_neurons') and self.striatum.d2_neurons.membrane is not None:
            self.striatum.d2_neurons.membrane *= decay_factor

    def _on_dopamine(self, payload: DopaminePayload) -> None:
        """Handle dopamine for reinforcement learning.

        Dopamine drives learning via the three-factor rule:
        - Positive DA (burst) → strengthen eligible synapses (D1 LTP, D2 LTD)
        - Negative DA (dip) → weaken eligible synapses (D1 LTD, D2 LTP)

        The actual learning is applied via striatum.learn() method.
        """
        if self._recent_input is not None and self._recent_output is not None:
            # Compute reward from dopamine level
            reward = payload.level
            correct = payload.is_burst  # Burst indicates correct/rewarded

            if hasattr(self.striatum, 'learn'):
                self.striatum.learn(
                    input_spikes=self._recent_input,
                    output_spikes=self._recent_output,
                    correct=correct,
                    reward=reward,
                )

    def _process_spikes(
        self,
        input_spikes: torch.Tensor,
        source: str,
    ) -> Optional[torch.Tensor]:
        """Buffer input spikes and process when we have all inputs.
        
        Striatum requires concatenated input from cortex L5, hippocampus, and PFC.
        """
        if input_spikes.dim() == 1:
            input_spikes = input_spikes.unsqueeze(0)
        
        # Buffer input based on source
        if source == "cortex":
            self._cortex_buffer = input_spikes
            self._last_cortex_time = self._current_time
        elif source == "hippocampus":
            self._hippocampus_buffer = input_spikes
            self._last_hippocampus_time = self._current_time
        elif source == "pfc":
            self._pfc_buffer = input_spikes
            self._last_pfc_time = self._current_time
        else:
            # Unknown source - try to process directly if sizes match
            return self._forward_striatum(input_spikes)
        
        # Check if we should process now
        # If sizes not configured, pass through (legacy mode)
        total_expected = (
            self._cortex_input_size + 
            self._hippocampus_input_size + 
            self._pfc_input_size
        )
        if total_expected == 0:
            # Sizes not configured - just pass through
            return self._forward_striatum(input_spikes)
        
        # Build combined input
        combined = self._build_combined_input()
        if combined is not None:
            result = self._forward_striatum(combined)
            self._clear_buffers()
            return result
        
        # Don't have complete input yet
        return None
    
    def _build_combined_input(self) -> Optional[torch.Tensor]:
        """Build combined input from buffers if ready.
        
        Combines in order: [cortex_l5 | hippocampus | pfc]
        Uses zeros for missing components if they've timed out.
        """
        device = None
        batch_size = 1
        
        # Determine device and batch size from any available buffer
        for buf in [self._cortex_buffer, self._hippocampus_buffer, self._pfc_buffer]:
            if buf is not None:
                device = buf.device
                batch_size = buf.shape[0]
                break
        
        if device is None:
            return None
        
        # Check if we have at least cortex (the primary input)
        if self._cortex_buffer is None:
            return None
        
        # Build components, using zeros for missing ones if timed out
        parts = []
        
        # Cortex L5
        parts.append(self._cortex_buffer)
        
        # Hippocampus
        if self._hippocampus_buffer is not None:
            parts.append(self._hippocampus_buffer)
        elif self._hippocampus_input_size > 0:
            # Use zeros if timed out or size is configured
            parts.append(torch.zeros(
                batch_size, self._hippocampus_input_size, device=device
            ))
        
        # PFC  
        if self._pfc_buffer is not None:
            parts.append(self._pfc_buffer)
        elif self._pfc_input_size > 0:
            # Use zeros if timed out or size is configured
            parts.append(torch.zeros(
                batch_size, self._pfc_input_size, device=device
            ))
        
        return torch.cat(parts, dim=-1)
    
    def _forward_striatum(self, combined_input: torch.Tensor) -> torch.Tensor:
        """Forward combined input through striatum."""
        # Store for learning
        self._recent_input = combined_input.clone()
        
        # Forward through striatum
        output = self.striatum.forward(
            combined_input,
            dt=1.0,
            encoding_mod=self._encoding_strength,
            retrieval_mod=self._retrieval_strength,
            explore=True,  # Enable exploration in event-driven mode
        )

        # Store output for learning
        self._recent_output = output.clone()

        # Track selected action
        if hasattr(self.striatum, 'get_selected_action'):
            self._selected_action = self.striatum.get_selected_action()
        elif output is not None:
            self._selected_action = int(output.argmax().item())

        return output.squeeze()

    def get_selected_action(self) -> Optional[int]:
        """Get the currently selected action."""
        return self._selected_action

    def get_state(self) -> Dict[str, Any]:
        """Return striatum state."""
        state = super().get_state()
        state["selected_action"] = self._selected_action

        if hasattr(self.striatum, 'get_diagnostics'):
            state["striatum"] = self.striatum.get_diagnostics()

        return state


class EventDrivenCerebellum(EventDrivenRegionBase):
    """Event-driven wrapper for Cerebellum.

    Handles motor refinement through error-corrective learning.

    Key features:
    - Receives motor commands from striatum
    - Learns via climbing fiber error signals
    - Refines motor output through supervised learning

    The Cerebellum.forward() method handles:
    - input_spikes: Motor commands from striatum
    - encoding_mod/retrieval_mod: Theta modulation

    Learning requires explicit error signals via learn() method.
    """

    def __init__(
        self,
        config: EventRegionConfig,
        cerebellum: Any,  # Cerebellum instance
    ):
        super().__init__(config)
        self.cerebellum = cerebellum

        # Track recent activity for learning
        self._recent_input: Optional[torch.Tensor] = None
        self._recent_output: Optional[torch.Tensor] = None
        self._pending_error: Optional[torch.Tensor] = None

    def _apply_decay(self, dt_ms: float) -> None:
        """Apply decay to cerebellar neurons."""
        decay_factor = math.exp(-dt_ms / self._membrane_tau)

        if hasattr(self.cerebellum, 'neurons') and self.cerebellum.neurons.membrane is not None:
            self.cerebellum.neurons.membrane *= decay_factor

    def _on_dopamine(self, payload: DopaminePayload) -> None:
        """Handle dopamine signals.

        In the cerebellum, dopamine modulates climbing fiber sensitivity.
        We use the dopamine signal as a proxy for error feedback.
        """
        # Convert dopamine to error signal for learning
        if self._recent_input is not None and self._recent_output is not None:
            # Use dopamine as teaching signal
            # Positive dopamine = correct, negative = error
            error_magnitude = abs(payload.level)
            if error_magnitude > 0.1 and hasattr(self.cerebellum, 'learn'):
                # Create target based on dopamine direction
                # Positive: reinforce current output
                # Negative: suppress current output
                if payload.is_burst:
                    # Reward - reinforce what we did
                    target = self._recent_output.clone()
                elif payload.is_dip:
                    # Error - suppress what we did
                    target = torch.zeros_like(self._recent_output)
                else:
                    # Neutral - no learning
                    return

                self.cerebellum.learn(
                    input_spikes=self._recent_input,
                    output_spikes=self._recent_output,
                    target=target,
                )

    def _process_spikes(
        self,
        input_spikes: torch.Tensor,
        source: str,
    ) -> Optional[torch.Tensor]:
        """Process motor command spikes through cerebellum."""
        if input_spikes.dim() == 1:
            input_spikes = input_spikes.unsqueeze(0)

        # Store for learning
        self._recent_input = input_spikes.clone()

        # Forward through cerebellum
        output = self.cerebellum.forward(
            input_spikes,
            dt=1.0,
            encoding_mod=self._encoding_strength,
            retrieval_mod=self._retrieval_strength,
        )

        # Store output for learning
        self._recent_output = output.clone()

        return output.squeeze()

    def learn_with_error(
        self,
        target: torch.Tensor,
    ) -> Dict[str, Any]:
        """Apply error-corrective learning with explicit target.

        This is the primary learning interface for the cerebellum.
        Call this when the correct output is known (e.g., from sensory feedback).

        Args:
            target: Target output pattern

        Returns:
            Learning metrics
        """
        if self._recent_input is None or self._recent_output is None:
            return {"error": "No recent activity to learn from"}

        return self.cerebellum.learn(
            input_spikes=self._recent_input,
            output_spikes=self._recent_output,
            target=target,
        )

    def get_state(self) -> Dict[str, Any]:
        """Return cerebellum state."""
        state = super().get_state()

        if hasattr(self.cerebellum, 'climbing_fiber'):
            state["error"] = self.cerebellum.climbing_fiber.error.clone()

        if hasattr(self.cerebellum, 'get_diagnostics'):
            state["cerebellum"] = self.cerebellum.get_diagnostics()

        return state


# ============================================================================
# Factory function for creating event-driven brain system
# ============================================================================

def create_event_driven_brain(
    n_input: int,
    n_output: int,
    hidden_size: int = 256,
    device: str = "cpu",
) -> Dict[str, EventDrivenRegionBase]:
    """Create a complete event-driven brain with all regions.

    This is a convenience function that creates and wires together
    all the brain regions for event-driven simulation.

    Args:
        n_input: Size of sensory input
        n_output: Number of output actions
        hidden_size: Size of hidden layers
        device: Torch device

    Returns:
        Dict mapping region names to EventDrivenRegion instances
    """
    from thalia.regions.cortex import LayeredCortex, LayeredCortexConfig
    from thalia.regions.hippocampus import TrisynapticHippocampus, TrisynapticConfig
    from thalia.regions.prefrontal import Prefrontal, PrefrontalConfig
    from thalia.regions.striatum import Striatum, StriatumConfig

    regions = {}

    # Create base region instances
    cortex_config = LayeredCortexConfig(
        n_input=n_input,
        n_output=hidden_size,
        device=device,
    )
    cortex = LayeredCortex(cortex_config)

    hippo_config = TrisynapticConfig(
        n_input=hidden_size,  # Receives from cortex
        n_output=hidden_size,
        device=device,
    )
    hippocampus = TrisynapticHippocampus(hippo_config)

    pfc_config = PrefrontalConfig(
        n_input=hidden_size,
        n_output=hidden_size,
        device=device,
    )
    pfc = Prefrontal(pfc_config)

    striatum_config = StriatumConfig(
        n_input=hidden_size,
        n_output=n_output,
        device=device,
    )
    striatum = Striatum(striatum_config)

    # Wrap in event-driven adapters
    regions["cortex"] = EventDrivenCortex(
        EventRegionConfig(
            name="cortex",
            output_targets=["hippocampus", "pfc", "striatum"],
            device=device,
        ),
        cortex,
    )

    regions["hippocampus"] = EventDrivenHippocampus(
        EventRegionConfig(
            name="hippocampus",
            output_targets=["pfc", "cortex"],
            device=device,
        ),
        hippocampus,
    )

    regions["pfc"] = EventDrivenPFC(
        EventRegionConfig(
            name="pfc",
            output_targets=["cortex", "striatum", "hippocampus"],
            device=device,
        ),
        pfc,
    )

    regions["striatum"] = EventDrivenStriatum(
        EventRegionConfig(
            name="striatum",
            output_targets=["motor"],
            device=device,
        ),
        striatum,
    )

    return regions
