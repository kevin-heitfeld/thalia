"""
Axonal Projection - Pure spike routing without synaptic weights.

This module implements axonal projections that transmit spikes between regions
with realistic axonal delays. Unlike traditional pathways, AxonalProjection has:
- NO synaptic weights (synapses belong to target regions)
- NO learning rules
- NO neurons
- ONLY spike routing and axonal conduction delays

Architecture v2.0: Explicit separation of axons (transmission) from synapses (integration).

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch

from thalia.core.base.component_config import NeuralComponentConfig

if TYPE_CHECKING:
    from thalia.core.neural_region import NeuralRegion
from thalia.core.pathway_state import AxonalProjectionState
from thalia.core.protocols.component import RoutingComponent
from thalia.managers.component_registry import register_pathway
from thalia.typing import SourceOutputs
from thalia.utils.delay_buffer import CircularDelayBuffer


@dataclass
class SourceSpec:
    """Specification for an axonal source with per-target delay variation.

    Supports realistic axonal branching where collaterals to different targets
    have different conduction velocities (myelination, distance, diameter).

    Attributes:
        region_name: Name of source region
        port: Optional output port (e.g., 'l23', 'l5', 'ca1')
        size: Number of axons from this source
        delay_ms: Default axonal conduction delay in milliseconds
        target_delays: Optional dict mapping target names to specific delays
            Example: {"striatum": 5.0, "thalamus": 10.0}
            If not specified, uses delay_ms for all targets
    """

    region_name: str
    port: Optional[str] = None
    size: int = 0
    delay_ms: float = 2.0
    target_delays: Optional[Dict[str, float]] = None

    def compound_key(self) -> str:
        """Get compound key for this source (region:port or just region).

        Used to distinguish multiple ports from the same region
        (e.g., "cortex:l6a" vs "cortex:l6b").
        """
        if self.port:
            return f"{self.region_name}:{self.port}"
        return self.region_name

    def get_delay_for_target(self, target_name: str) -> float:
        """Get axonal delay to a specific target.

        Args:
            target_name: Name of target region

        Returns:
            Delay in milliseconds (target-specific or default)
        """
        if self.target_delays and target_name in self.target_delays:
            return self.target_delays[target_name]
        return self.delay_ms


@register_pathway(
    "axonal",
    aliases=["axonal_projection", "pure_axon"],
    description="Pure axonal transmission with delays, no synaptic weights (v2.0)",
    version="2.0",
)
class AxonalProjection(RoutingComponent):
    """Pure axonal transmission between brain regions.

    Represents the axons connecting regions, NOT the synapses at terminals.
    Biologically accurate: axons transmit spikes with delays, synapses integrate
    them with weights (and synapses are located at the TARGET region).

    Key Principles:
    1. NO weights - synapses belong to target region's dendrites
    2. NO learning - learning happens at synapses, not axons
    3. NO neurons - axons are transmission lines, not computational units
    4. Concatenation - multi-source projections concatenate spikes
    5. Delays - handled by EventScheduler in event-driven execution

    Example:
        # Single source
        projection = AxonalProjection(
            sources=[("cortex", "l5", 128, 2.0)],
            device="cpu"
        )

        # Multi-source (concatenates cortex L5 + hippocampus + pfc)
        projection = AxonalProjection(
            sources=[
                ("cortex", "l5", 128, 2.0),
                ("hippocampus", None, 64, 3.0),
                ("pfc", None, 32, 2.0),
            ],
            device="cpu"
        )
        # Total output: 128 + 64 + 32 = 224 axons

    Args:
        sources: List of (region_name, port, size, delay_ms) tuples
        device: Torch device for computation
        dt_ms: Simulation timestep in milliseconds
        config: Optional configuration dict
    """

    def __init__(
        self,
        sources: List[
            Union[
                Tuple[str, Optional[str], int, float],
                Tuple[str, Optional[str], int, float, Dict[str, float]],
            ]
        ],
        device: str = "cpu",
        dt_ms: float = 1.0,
        config: Optional[Union[NeuralComponentConfig, Dict[str, Any]]] = None,
        target_name: Optional[str] = None,
    ):
        """Initialize axonal projection with optional per-target delays.

        Args:
            sources: List of source specifications, each can be:
                - (region_name, port, size, delay_ms): Single delay for all targets
                - (region_name, port, size, delay_ms, target_delays): Per-target delays
                  where target_delays is Dict[target_name, delay_ms]
            device: Torch device
            dt_ms: Simulation timestep
            config: Optional configuration
            target_name: Name of target region (for per-target delay selection)
        """
        # Create minimal config for RoutingComponent
        if config is None:
            config = SimpleNamespace(device=device)  # type: ignore[assignment]

        # RoutingComponent.__init__ sets self._device as read-only property
        super().__init__(config)

        self.dt_ms = dt_ms
        self.target_name = target_name

        # Parse sources into SourceSpec objects
        self.sources: List[SourceSpec] = []
        total_size = 0

        for source_tuple in sources:
            if len(source_tuple) == 5:
                # New: (region_name, port, size, delay_ms, target_delays)
                region_name, port, size, delay_ms, target_delays = source_tuple
            elif len(source_tuple) == 4:
                # Standard: (region_name, port, size, delay_ms)
                region_name, port, size, delay_ms = source_tuple
                target_delays = None
            elif len(source_tuple) == 3:
                # Backward compatibility: (region_name, port, size)
                region_name, port, size = source_tuple
                delay_ms = 2.0  # Default
                target_delays = None
            else:
                raise ValueError(f"Invalid source tuple: {source_tuple}")

            spec = SourceSpec(
                region_name=region_name,
                port=port,
                size=size,
                delay_ms=delay_ms,
                target_delays=target_delays,
            )
            self.sources.append(spec)
            total_size += size

        # Output is concatenation of all sources
        self.n_input = 0  # Axons don't have inputs (they ARE the input)
        self.n_output = total_size

        # Update config
        self.config.n_input = self.n_input
        self.config.n_output = self.n_output

        # Create delay buffers for each source
        # Use target-specific delay if available, otherwise default delay
        self._delay_buffers: Dict[str, CircularDelayBuffer] = {}
        for spec in self.sources:
            # Get appropriate delay for this target
            effective_delay = (
                spec.get_delay_for_target(target_name) if target_name else spec.delay_ms
            )
            delay_steps = int(effective_delay / self.dt_ms)
            source_key = spec.compound_key()
            self._delay_buffers[source_key] = CircularDelayBuffer(
                max_delay=delay_steps,
                size=spec.size,
                device=device,
                dtype=torch.bool,  # Spikes are binary
            )

    def forward(self, source_outputs: SourceOutputs) -> SourceOutputs:
        """Route spikes from sources with axonal delays.

        Biologically accurate: Returns dict so target regions can route inputs
        to different neuron populations (e.g., thalamus sensory→relay, L6→TRN).

        Port-Based Routing (Phase 3.2):
        When SourceSpec includes a port (e.g., "l6a", "l5"), the pathway will:
        1. Look up the source region object in source_outputs
        2. Call region.get_port_output(port) to get port-specific spikes
        3. Route those spikes with the specified axonal delay

        This enables biologically-accurate routing where different cortical layers
        project to different targets (e.g., L6a→TRN, L6b→relay).

        Per-Target Delays:
        When SourceSpec includes target_delays, the appropriate delay is used
        automatically based on target_name specified during initialization.
        This models realistic axonal branching where collaterals to different
        targets have different conduction velocities.

        Delays are implemented via circular buffers internally. Each timestep:
        1. Write current spikes to buffer
        2. Read delayed spikes from buffer (using target-specific delay)
        3. Advance buffer pointer

        Args:
            source_outputs: Dict mapping source keys to spike tensors OR region objects.
                - Tensor mode (legacy): {"cortex:l6a": tensor, "hippocampus": tensor}
                - Region mode (port-aware): {"cortex": region_obj, "hippocampus": region_obj}
                Keys can be compound ("cortex:l6a") or simple ("cortex")

        Returns:
            Dict mapping source keys to delayed spike tensors (NOT concatenated)

        Example (Port-Based Routing):
            # Pathway has sources=[("cortex", "l6a", 32, 2.0), ("cortex", "l5", 128, 2.0)]
            region_outputs = {"cortex": cortex_region}  # Pass region object
            delayed = projection.forward(region_outputs)
            # Returns: {"cortex:l6a": [32], "cortex:l5": [128]}
            # Each tensor from different cortex layer

        Example (Legacy Tensor Mode):
            tensor_outputs = {
                "cortex:l5": cortex_l5_spikes,    # [128]
                "hippocampus": hipp_spikes,       # [64]
            }
            delayed = projection.forward(tensor_outputs)
            # Returns: {"cortex:l5": [128], "hippocampus": [64]}
        """
        delayed_outputs = {}

        for source_spec in self.sources:
            # Get compound key (e.g., "cortex:l6a" or just "cortex")
            source_key = source_spec.compound_key()
            buffer = self._delay_buffers[source_key]
            # Use target-specific delay if available
            effective_delay = (
                source_spec.get_delay_for_target(self.target_name)
                if self.target_name
                else source_spec.delay_ms
            )
            delay_steps = int(effective_delay / self.dt_ms)

            # Get current spikes from source
            # Support both tensor mode (legacy) and region mode (port-aware)
            if not isinstance(source_outputs, dict):
                # Invalid input
                spikes = torch.zeros(source_spec.size, dtype=torch.bool, device=self.device)
            elif source_key in source_outputs:
                # Tensor mode: compound key already in dict ("cortex:l6a")
                source_value = source_outputs[source_key]
                if isinstance(source_value, torch.Tensor):
                    spikes = source_value
                else:
                    # Region object with compound key - extract port
                    spikes = source_value.get_port_output(source_spec.port)
            elif source_spec.region_name in source_outputs:
                # Region mode: lookup by region name, then get port output
                source_value = source_outputs[source_spec.region_name]
                if isinstance(source_value, torch.Tensor):
                    # Tensor without port - use directly
                    spikes = source_value
                elif hasattr(source_value, 'get_port_output'):
                    # Region object - get port-specific output
                    spikes = source_value.get_port_output(source_spec.port)
                else:
                    # Unknown type
                    spikes = torch.zeros(source_spec.size, dtype=torch.bool, device=self.device)
            else:
                # Source not firing this timestep, use zeros
                spikes = torch.zeros(source_spec.size, dtype=torch.bool, device=self.device)

            # Validate size (only if we got actual spikes)
            if spikes is not None and spikes.numel() > 0:
                if spikes.shape[0] != source_spec.size:
                    raise ValueError(
                        f"Size mismatch for {source_key}: "
                        f"expected {source_spec.size}, got {spikes.shape[0]}"
                    )

            # Write current spikes to buffer
            buffer.write(spikes)

            # Read delayed spikes (from delay_steps timesteps ago)
            delayed_spikes = buffer.read(delay_steps)

            # Store in dict preserving source identity
            delayed_outputs[source_key] = delayed_spikes

            # Advance buffer for next timestep
            buffer.advance()

        return delayed_outputs

    def grow_source(self, source_name: str, new_size: int) -> None:
        """Grow axonal projection for a source that expanded.

        When a source region grows output neurons, the axonal projection must
        grow to accommodate more axons. This only updates routing - no weights
        to resize (synapses are at target region).

        Args:
            source_name: Name of source region that grew
            new_size: New size of that source

        Example:
            # Cortex grew from 128 to 148 neurons
            cortex.grow_output(20)
            projection.grow_source("cortex", new_size=148)
            # Now projection routes 148 + 64 + 32 = 244 axons
        """
        # Find source spec
        source_idx = None
        for idx, spec in enumerate(self.sources):
            if spec.region_name == source_name:
                source_idx = idx
                break

        if source_idx is None:
            raise ValueError(f"Source '{source_name}' not found in projection")

        old_size = self.sources[source_idx].size
        size_delta = new_size - old_size

        if size_delta == 0:
            return  # No change

        # Update source spec
        self.sources[source_idx].size = new_size

        # Update total output size
        self.n_output += size_delta
        self.config.n_output = self.n_output

        # Grow delay buffer for this source
        source_key = self.sources[source_idx].compound_key()
        if source_key in self._delay_buffers:
            self._delay_buffers[source_key].grow(new_size)

    def reset_state(self) -> None:
        """Reset all delay buffers to zeros."""
        for buffer in self._delay_buffers.values():
            buffer.reset()

    # =================================================================
    # GROWTH API (RoutingComponent-specific)
    # =================================================================

    def grow_input(
        self,
        n_new: int,
        initialization: str = "sparse_random",
        sparsity: float = 0.1,
    ) -> None:
        """Routing components don't have input dimension.

        Note: Axons grow via grow_source() instead.
        """

    def grow_output(
        self,
        n_new: int,
        initialization: str = "sparse_random",
        sparsity: float = 0.1,
    ) -> None:
        """Axons don't grow output independently - use grow_source().

        Raises:
            NotImplementedError: Use grow_source(source_name, new_size) instead
        """
        raise NotImplementedError(
            "AxonalProjection doesn't support grow_output(). "
            "Use grow_source(source_name, new_size) instead."
        )

    def get_capacity_metrics(self) -> Dict[str, Any]:
        """Routing components don't have capacity metrics."""
        return {
            "n_output": self.n_output,
            "n_sources": len(self.sources),
            "utilization": 1.0,  # Always fully utilized (routing)
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get routing diagnostics."""
        return {
            "n_output": self.n_output,
            "n_sources": len(self.sources),
            "sources": [
                {
                    "name": spec.region_name,
                    "port": spec.port,
                    "size": spec.size,
                    "delay_ms": spec.delay_ms,
                }
                for spec in self.sources
            ],
        }

    def check_health(self) -> Any:
        """Routing components are always healthy (no learning to fail)."""
        from thalia.diagnostics.health_monitor import HealthReport

        return HealthReport(
            is_healthy=True,
            overall_severity=0.0,
            issues=[],
            summary="Routing component is healthy",
            metrics=self.get_diagnostics(),
        )

    def set_oscillator_phases(
        self,
        phases: Dict[str, float],
        signals: Dict[str, float] | None = None,
        theta_slot: int = 0,
        coupled_amplitudes: Dict[str, float] | None = None,
    ) -> None:
        """Routing components don't use oscillators."""

    # =================================================================
    # CHECKPOINTING (PathwayState Protocol)
    # =================================================================

    def get_full_state(self) -> Dict[str, Any]:
        """Get full state (alias for get_state for compatibility)."""
        state = self.get_state()
        return state.to_dict()

    def load_full_state(self, state: Dict[str, Any]) -> None:
        """Load full state (alias for load_state for compatibility)."""
        state_obj = AxonalProjectionState.from_dict(state, device=self.device)
        self.load_state(state_obj)

    def get_state(self) -> AxonalProjectionState:
        """Get state for checkpointing using PathwayState protocol.

        Returns:
            AxonalProjectionState with delay buffer data
        """
        # Extract delay buffer state
        delay_buffers = {}
        for key, buffer in self._delay_buffers.items():
            delay_buffers[key] = (
                buffer.buffer.clone(),  # [max_delay+1, size]
                buffer.ptr,
                buffer.max_delay,
                buffer.size,
            )

        return AxonalProjectionState(delay_buffers=delay_buffers)

    def load_state(self, state: AxonalProjectionState) -> None:
        """Load state from checkpoint using PathwayState protocol.

        Args:
            state: AxonalProjectionState object
        """
        # Load delay buffer states
        for key, (buf, ptr, max_delay, size) in state.delay_buffers.items():
            if key in self._delay_buffers:
                # Update existing buffer
                self._delay_buffers[key].buffer = buf.to(self.device)
                self._delay_buffers[key].ptr = ptr
                # Verify consistency
                assert (
                    self._delay_buffers[key].max_delay == max_delay
                ), f"Delay mismatch for {key}: {self._delay_buffers[key].max_delay} != {max_delay}"
                assert (
                    self._delay_buffers[key].size == size
                ), f"Size mismatch for {key}: {self._delay_buffers[key].size} != {size}"

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters when brain timestep changes.

        Resizes delay buffers to accommodate new timestep while preserving
        spike history. Delays are specified in milliseconds (fixed), but the
        number of steps changes with dt:
            delay_steps = delay_ms / dt_ms

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        old_dt_ms = self.dt_ms
        self.dt_ms = dt_ms

        # Resize each delay buffer
        for spec in self.sources:
            source_key = spec.compound_key()
            if source_key not in self._delay_buffers:
                continue

            # Get appropriate delay for this target
            effective_delay = (
                spec.get_delay_for_target(self.target_name) if self.target_name else spec.delay_ms
            )

            # Resize buffer
            buffer = self._delay_buffers[source_key]
            buffer.resize_for_new_dt(
                new_dt_ms=dt_ms,
                delay_ms=effective_delay,
                old_dt_ms=old_dt_ms,
            )

    def __repr__(self) -> str:
        """Human-readable representation."""
        source_strs = []
        for spec in self.sources:
            port_str = f"[{spec.port}]" if spec.port else ""
            if spec.target_delays and self.target_name:
                # Show target-specific delay
                delay = spec.get_delay_for_target(self.target_name)
                source_strs.append(
                    f"{spec.region_name}{port_str}({spec.size}, {delay}ms→{self.target_name})"
                )
            else:
                source_strs.append(f"{spec.region_name}{port_str}({spec.size}, {spec.delay_ms}ms)")

        sources = " + ".join(source_strs)
        target_str = f" → {self.target_name}" if self.target_name else ""
        return f"AxonalProjection({sources}{target_str}: {self.n_output} axons)"
