"""Module defining the AxonalTract class for pure axonal transmission between brain regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from thalia.typing import (
    BrainOutput,
    PopulationName,
    RegionName,
    RegionOutput
)
from thalia.utils import (
    CircularDelayBuffer,
    HeterogeneousDelayBuffer,
    validate_spike_tensor,
    validate_spike_tensors,
)


@dataclass
class AxonalTractSourceSpec:
    """Specification for an axonal source."""

    region_name: RegionName
    population: PopulationName
    size: int
    delay_ms: float
    delay_std_ms: float = 0.0  # Standard deviation for heterogeneous delays (0 = uniform)


class AxonalTract(nn.Module):
    """Pure axonal transmission between brain regions."""

    @property
    def device(self) -> torch.device:
        """Device where tensors are located."""
        return torch.device(self._device)

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, source_specs: List[AxonalTractSourceSpec], dt_ms: float, device: str):
        """Initialize axonal tract."""
        super().__init__()

        self.source_specs = source_specs
        self.dt_ms = dt_ms
        self._device = device

        # Create delay buffers for each source
        # Use heterogeneous delays if delay_std_ms > 0, otherwise uniform delays
        self.delay_buffers: Dict[Tuple[RegionName, PopulationName], CircularDelayBuffer | HeterogeneousDelayBuffer] = {}
        for spec in self.source_specs:
            source_key = (spec.region_name, spec.population)

            if spec.delay_std_ms > 0:
                # Heterogeneous delays: sample per-neuron delays from Gaussian
                mean_delay_steps = spec.delay_ms / self.dt_ms
                std_delay_steps = spec.delay_std_ms / self.dt_ms

                # Sample delays (clamp to reasonable range: 0.5*mean to 3*mean)
                delays_steps = torch.randn(spec.size) * std_delay_steps + mean_delay_steps
                delays_steps = torch.clamp(
                    delays_steps,
                    min=max(0, mean_delay_steps * 0.5),
                    max=mean_delay_steps * 3.0
                ).long()

                self.delay_buffers[source_key] = HeterogeneousDelayBuffer(
                    delays=delays_steps,
                    size=spec.size,
                    device=device,
                    dtype=torch.bool,
                )
            else:
                # Uniform delay: all neurons have same delay
                delay_steps = int(spec.delay_ms / self.dt_ms)
                self.delay_buffers[source_key] = CircularDelayBuffer(
                    max_delay=delay_steps,
                    size=spec.size,
                    device=device,
                    dtype=torch.bool,
                )

        # Ensure all parameters are on correct device
        self.to(self.device)

    # =========================================================================
    # SPIKE ROUTING
    # =========================================================================

    def read_delayed_outputs(self) -> BrainOutput:
        """Read delayed outputs from buffers WITHOUT writing or advancing."""
        delayed_outputs: BrainOutput = {}

        for source_spec in self.source_specs:
            source_key = (source_spec.region_name, source_spec.population)
            buffer = self.delay_buffers[source_key]

            # Read delayed spikes
            # HeterogeneousDelayBuffer: uses per-neuron delays
            # CircularDelayBuffer: uses uniform delay
            if isinstance(buffer, HeterogeneousDelayBuffer):
                delayed_spikes = buffer.read_heterogeneous()
            else:
                delay_steps = int(source_spec.delay_ms / self.dt_ms)
                delayed_spikes = buffer.read(delay_steps)

            # Store in output dict under source region and population
            if source_spec.region_name not in delayed_outputs:
                delayed_outputs[source_spec.region_name] = {}

            delayed_outputs[source_spec.region_name][source_spec.population] = delayed_spikes

        return delayed_outputs

    def write_and_advance(self, source_outputs: BrainOutput) -> None:
        """Write current outputs to buffers and advance pointers."""
        for _region_name, spikes in source_outputs.items():
            validate_spike_tensors(spikes, context="AxonalTract.write_and_advance")

        for source_spec in self.source_specs:
            source_key = (source_spec.region_name, source_spec.population)
            buffer = self.delay_buffers[source_key]

            # Extract spikes from RegionOutput
            spikes = None
            if source_spec.region_name in source_outputs:
                population_outputs: RegionOutput = source_outputs[source_spec.region_name]
                if source_spec.population in population_outputs:
                    spikes = population_outputs[source_spec.population]

            if spikes is not None:
                validate_spike_tensor(spikes)

                if spikes.shape[0] != source_spec.size:
                    raise ValueError(
                        f"Size mismatch for {source_spec.region_name}:{source_spec.population}: "
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
        for spec in self.source_specs:
            source_key = (spec.region_name, spec.population)
            assert source_key in self.delay_buffers, f"Source key '{source_key}' not found in delay buffers."

            buffer = self.delay_buffers[source_key]
            buffer.resize_for_new_dt(
                new_dt_ms=dt_ms,
                delay_ms=spec.delay_ms,
                old_dt_ms=old_dt_ms,
            )
