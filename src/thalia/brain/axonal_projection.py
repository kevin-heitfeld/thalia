"""
Axonal Projection - Pure spike routing without synaptic weights.

This module implements axonal projections that transmit spikes between regions
with realistic axonal delays. AxonalProjection has:
- NO synaptic weights (synapses belong to target regions)
- NO learning rules
- NO neurons
- ONLY spike routing and axonal conduction delays

Explicit separation of axons (transmission) from synapses (integration).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn as nn

from thalia.constants import DEFAULT_DT_MS
from thalia.typing import (
    BrainSpikesDict,
    SpikesSourceKey,
    RegionSpikesDict,
    compound_key,
)
from thalia.utils import (
    CircularDelayBuffer,
    validate_spike_tensor,
    validate_spike_tensors,
)


@dataclass
class AxonalProjectionSourceSpec:
    """Specification for an axonal source."""

    region_name: str
    port: str
    size: int
    delay_ms: float

    def compound_key(self) -> SpikesSourceKey:
        """Get compound key for this source (region:port)."""
        return compound_key(self.region_name, self.port)


class AxonalProjection(nn.Module):
    """Pure axonal transmission between brain regions.

    Key Principles:
    1. NO weights - synapses belong to target region's dendrites
    2. NO learning - learning happens at synapses, not axons
    3. NO neurons - axons are transmission lines, not computational units
    4. Concatenation - multi-source projections concatenate spikes
    5. Delays - handled by EventScheduler in event-driven execution
    """

    @property
    def device(self) -> torch.device:
        """Device where tensors are located."""
        return torch.device(self._device)

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        sources: List[AxonalProjectionSourceSpec],
        dt_ms: float = DEFAULT_DT_MS,
        device: str = "cpu",
    ):
        """Initialize axonal projection."""
        super().__init__()

        self.sources = sources
        self._device = device
        self.dt_ms = dt_ms

        # Create delay buffers for each source
        # Use target-specific delay if available, otherwise default delay
        self.delay_buffers: Dict[str, CircularDelayBuffer] = {}
        for spec in self.sources:
            delay_steps = int(spec.delay_ms / self.dt_ms)
            source_key = spec.compound_key()
            self.delay_buffers[source_key] = CircularDelayBuffer(
                max_delay=delay_steps,
                size=spec.size,
                device=device,
                dtype=torch.bool,  # Spikes are binary
            )

        # Ensure all parameters are on correct device
        self.to(self.device)

    # =========================================================================
    # SPIKE ROUTING
    # =========================================================================

    def read_delayed_outputs(self) -> BrainSpikesDict:
        """Read delayed outputs from buffers WITHOUT writing or advancing."""
        delayed_outputs: BrainSpikesDict = {}

        for source_spec in self.sources:
            source_key = source_spec.compound_key()
            buffer = self.delay_buffers[source_key]

            delay_steps = int(source_spec.delay_ms / self.dt_ms)

            # Read delayed spikes (from delay_steps timesteps ago)
            # This does NOT advance the buffer
            delayed_spikes = buffer.read(delay_steps)

            # Store in RegionSpikesDict format
            if source_spec.region_name not in delayed_outputs:
                delayed_outputs[source_spec.region_name] = {}

            port_key = source_spec.port if source_spec.port else source_spec.region_name
            delayed_outputs[source_spec.region_name][port_key] = delayed_spikes

        return delayed_outputs

    def write_and_advance(self, source_outputs: BrainSpikesDict) -> None:
        """Write current outputs to buffers and advance pointers."""
        for _region_name, spikes in source_outputs.items():
            validate_spike_tensors(spikes, context="AxonalProjection.write_and_advance")

        for source_spec in self.sources:
            source_key = source_spec.compound_key()
            buffer = self.delay_buffers[source_key]

            # Extract spikes from RegionSpikesDict
            spikes = None
            if source_spec.region_name in source_outputs:
                region_ports: RegionSpikesDict = source_outputs[source_spec.region_name]

                if not source_spec.port:
                    raise ValueError(
                        f"Port must be specified for source '{source_spec.region_name}' in AxonalProjection"
                    )

                if source_spec.port in region_ports:
                    spikes = region_ports[source_spec.port]

            if spikes is None:
                # No output from this source, write zeros
                spikes = torch.zeros(source_spec.size, dtype=torch.bool, device=self.device)

            # Validate spike tensor
            validate_spike_tensor(spikes)

            if spikes.shape[0] != source_spec.size:
                raise ValueError(
                    f"Size mismatch for {source_spec.region_name}:{source_spec.port}: "
                    f"expected {source_spec.size}, got {spikes.shape[0]}"
                )

            # Write current spikes to buffer
            buffer.write(spikes)

            # Advance buffer for next timestep
            buffer.advance()

    # =========================================================================
    # TEMPORAL PARAMETER MANAGEMENT
    # =========================================================================

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters when brain timestep changes.

        Resizes delay buffers to accommodate new timestep while preserving
        spike history. Delays are specified in milliseconds (fixed), but the
        number of steps changes with dt:
            delay_steps = delay_ms / dt_ms
        """
        old_dt_ms = self.dt_ms
        self.dt_ms = dt_ms

        # Resize each delay buffer
        for spec in self.sources:
            source_key = spec.compound_key()
            if source_key not in self.delay_buffers:
                continue

            # Resize buffer
            buffer = self.delay_buffers[source_key]
            buffer.resize_for_new_dt(
                new_dt_ms=dt_ms,
                delay_ms=spec.delay_ms,
                old_dt_ms=old_dt_ms,
            )

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get routing diagnostics."""
        return {
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
