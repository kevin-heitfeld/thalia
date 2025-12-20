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

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

import torch

from thalia.core.protocols.component import RoutingComponent
from thalia.managers.component_registry import register_pathway
from thalia.core.base.component_config import NeuralComponentConfig


@dataclass
class SourceSpec:
    """Specification for an axonal source.

    Attributes:
        region_name: Name of source region
        port: Optional output port (e.g., 'l23', 'l5', 'ca1')
        size: Number of axons from this source
        delay_ms: Axonal conduction delay in milliseconds
    """
    region_name: str
    port: Optional[str] = None
    size: int = 0
    delay_ms: float = 2.0


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
    5. Delays - realistic axonal conduction delays per source

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
        sources: List[Tuple[str, Optional[str], int, float]],
        device: str = "cpu",
        dt_ms: float = 1.0,
        config: Optional[Union[NeuralComponentConfig, Dict[str, Any]]] = None,
    ):
        # Create minimal config for RoutingComponent
        from types import SimpleNamespace
        if config is None:
            config = SimpleNamespace(device=device)

        # RoutingComponent.__init__ sets self._device as read-only property
        super().__init__(config)

        self.dt_ms = dt_ms

        # Parse sources into SourceSpec objects
        self.sources: List[SourceSpec] = []
        total_size = 0

        for source_tuple in sources:
            if len(source_tuple) == 4:
                region_name, port, size, delay_ms = source_tuple
            elif len(source_tuple) == 3:
                # Backward compatibility: (region_name, port, size)
                region_name, port, size = source_tuple
                delay_ms = 2.0  # Default
            else:
                raise ValueError(f"Invalid source tuple: {source_tuple}")

            spec = SourceSpec(
                region_name=region_name,
                port=port,
                size=size,
                delay_ms=delay_ms,
            )
            self.sources.append(spec)
            total_size += size

        # Output is concatenation of all sources
        self.n_input = 0  # Axons don't have inputs (they ARE the input)
        self.n_output = total_size

        # Update config
        self.config.n_input = self.n_input
        self.config.n_output = self.n_output

        # Delay buffers for each source (circular buffers)
        self._delay_buffers: Dict[str, torch.Tensor] = {}
        self._buffer_positions: Dict[str, int] = {}

        self._init_delay_buffers()

    def _init_delay_buffers(self) -> None:
        """Initialize circular delay buffers for axonal conduction delays."""
        for source_spec in self.sources:
            # Buffer size = delay / dt_ms (minimum 1)
            buffer_size = max(1, int(source_spec.delay_ms / self.dt_ms))

            # Create circular buffer [buffer_size, n_axons]
            self._delay_buffers[source_spec.region_name] = torch.zeros(
                buffer_size, source_spec.size,
                dtype=torch.bool,
                device=self.device
            )
            self._buffer_positions[source_spec.region_name] = 0

    def forward(
        self,
        source_outputs: Dict[str, torch.Tensor],
        apply_delays: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Route spikes from sources preserving source identity.

        Biologically accurate: Returns dict so target regions can route inputs
        to different neuron populations (e.g., thalamus sensory→relay, L6→TRN).

        Args:
            source_outputs: Dict mapping region names to spike tensors
            apply_delays: Whether to apply axonal delays (False for testing)

        Returns:
            Dict mapping source names to delayed spike tensors (NOT concatenated)

        Example:
            outputs = {
                "cortex": cortex_l5_spikes,  # [128]
                "hippocampus": hipp_spikes,   # [64]
                "pfc": pfc_spikes,            # [32]
            }
            delayed = projection.forward(outputs)
            # Returns: {"cortex": [128], "hippocampus": [64], "pfc": [32]}

            # Target regions can concatenate if needed:
            concatenated = torch.cat([delayed["cortex"], delayed["hippocampus"]])
        """
        delayed_outputs = {}

        for source_spec in self.sources:
            # Get spikes from source
            # Check if source_outputs is a dict first (avoid Tensor.__contains__ error)
            if not isinstance(source_outputs, dict) or source_spec.region_name not in source_outputs:
                # Source not firing this timestep, use zeros
                spikes = torch.zeros(source_spec.size, dtype=torch.bool, device=self.device)
            else:
                spikes = source_outputs[source_spec.region_name]

                # Extract port if specified (e.g., cortex['l5'])
                if source_spec.port is not None and isinstance(spikes, dict):
                    spikes = spikes[source_spec.port]

                # Validate size
                if spikes.shape[0] != source_spec.size:
                    raise ValueError(
                        f"Size mismatch for {source_spec.region_name}: "
                        f"expected {source_spec.size}, got {spikes.shape[0]}"
                    )

            # Apply axonal delay
            if apply_delays:
                delayed_spikes = self._apply_delay(source_spec.region_name, spikes)
            else:
                delayed_spikes = spikes

            # Store in dict preserving source identity
            delayed_outputs[source_spec.region_name] = delayed_spikes

        return delayed_outputs

    def _apply_delay(self, source_name: str, spikes: torch.Tensor) -> torch.Tensor:
        """Apply axonal conduction delay using circular buffer.

        Args:
            source_name: Name of source region
            spikes: Current spike tensor [n_axons]

        Returns:
            Delayed spike tensor [n_axons]
        """
        buffer = self._delay_buffers[source_name]
        pos = self._buffer_positions[source_name]

        # Get delayed spikes from current position
        delayed_spikes = buffer[pos].clone()

        # Write new spikes to buffer
        buffer[pos] = spikes

        # Advance position (circular)
        self._buffer_positions[source_name] = (pos + 1) % buffer.shape[0]

        return delayed_spikes

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

        # Resize delay buffer for this source
        old_buffer = self._delay_buffers[source_name]
        buffer_size = old_buffer.shape[0]

        # Create new buffer with expanded size
        new_buffer = torch.zeros(
            buffer_size, new_size,
            dtype=torch.bool,
            device=self.device
        )

        # Copy old data (preserve existing axons)
        new_buffer[:, :old_size] = old_buffer

        self._delay_buffers[source_name] = new_buffer

    def reset_state(self) -> None:
        """Reset delay buffers to zero (clear axonal transmission history)."""
        for source_name in self._delay_buffers.keys():
            self._delay_buffers[source_name].zero_()
            self._buffer_positions[source_name] = 0

    # =================================================================
    # GROWTH API (RoutingComponent-specific)
    # =================================================================

    def grow_input(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
    ) -> None:
        """Routing components don't have input dimension.

        Note: Axons grow via grow_source() instead.
        """
        pass

    def grow_output(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
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
        pass

    # =================================================================
    # CHECKPOINTING
    # =================================================================

    def get_full_state(self) -> Dict[str, Any]:
        """Get full state (alias for get_state for compatibility)."""
        return self.get_state()

    def load_full_state(self, state: Dict[str, Any]) -> None:
        """Load full state (alias for load_state for compatibility)."""
        self.load_state(state)

    def get_state(self) -> Dict[str, Any]:
        """Get state for checkpointing.

        Returns:
            Dict with delay buffers and positions
        """
        return {
            "delay_buffers": {
                name: buffer.cpu()
                for name, buffer in self._delay_buffers.items()
            },
            "buffer_positions": self._buffer_positions.copy(),
            "sources": [
                {
                    "region_name": spec.region_name,
                    "port": spec.port,
                    "size": spec.size,
                    "delay_ms": spec.delay_ms,
                }
                for spec in self.sources
            ],
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from checkpoint.

        Args:
            state: State dict from get_state()
        """
        # Load delay buffers
        for name, buffer in state["delay_buffers"].items():
            self._delay_buffers[name] = buffer.to(self.device)

        # Load buffer positions
        self._buffer_positions = state["buffer_positions"].copy()

    def __repr__(self) -> str:
        """Human-readable representation."""
        source_strs = []
        for spec in self.sources:
            port_str = f"[{spec.port}]" if spec.port else ""
            source_strs.append(f"{spec.region_name}{port_str}({spec.size}, {spec.delay_ms}ms)")

        sources = " + ".join(source_strs)
        return f"AxonalProjection({sources} → {self.n_output} axons)"
