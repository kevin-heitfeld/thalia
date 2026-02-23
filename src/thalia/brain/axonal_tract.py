"""Module defining the AxonalTract class for pure axonal transmission between brain regions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from thalia.typing import (
    BrainOutput,
    SynapseId,
    SynapticInput,
)
from thalia.utils import (
    CircularDelayBuffer,
    HeterogeneousDelayBuffer,
    validate_spike_tensor,
)


@dataclass
class AxonalTractSourceSpec:
    """Specification for an axonal source.

    ``synapse_id`` is the single source of truth for routing metadata (source
    region/population, target region/population, and polarity).  The ``size``,
    ``delay_ms``, and ``delay_std_ms`` fields carry the physical axon properties
    that are not encoded in ``SynapseId``.
    """

    synapse_id: SynapseId  # full routing key: source, target, and is_inhibitory
    size: int
    delay_ms: float
    delay_std_ms: float  # Standard deviation for heterogeneous delays (0 = uniform)


class AxonalTract(nn.Module):
    """Pure axonal transmission for a single source→target synapse.

    Each ``AxonalTract`` manages the delay buffer for exactly **one**
    :class:`AxonalTractSourceSpec` (one source population → one target
    population).  The :class:`BrainBuilder` creates one tract per
    :class:`ConnectionSpec`, keyed by its :class:`SynapseId`.
    """

    @property
    def device(self) -> torch.device:
        """Device where tensors are located."""
        return torch.device(self._device)

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, spec: AxonalTractSourceSpec, dt_ms: float, device: str):
        """Initialize an axonal tract for a single source/target pair.

        Args:
            spec: Source specification (synapse_id, size, delay_ms, delay_std_ms).
            dt_ms: Simulation timestep in milliseconds.
            device: PyTorch device string (e.g. ``"cpu"`` or ``"cuda:0"``).
        """
        super().__init__()

        self.spec = spec
        self.dt_ms = dt_ms
        self._device = device

        # Create the delay buffer for this source.
        # Use heterogeneous delays when delay_std_ms > 0, uniform otherwise.
        if spec.delay_std_ms > 0:
            # Heterogeneous delays: sample per-neuron delays from a Gaussian.
            mean_steps = spec.delay_ms / self.dt_ms
            std_steps = spec.delay_std_ms / self.dt_ms
            delays_steps = torch.randn(spec.size) * std_steps + mean_steps
            delays_steps = torch.clamp(
                delays_steps,
                min=max(0.0, mean_steps * 0.5),
                max=mean_steps * 3.0,
            ).long()
            self._delay_buffer: nn.Module = HeterogeneousDelayBuffer(
                delays=delays_steps,
                size=spec.size,
                device=device,
                dtype=torch.bool,
            )
        else:
            delay_steps = max(1, int(spec.delay_ms / self.dt_ms))
            self._delay_buffer = CircularDelayBuffer(
                max_delay=delay_steps,
                size=spec.size,
                device=device,
                dtype=torch.bool,
            )

        self.to(self.device)

    # =========================================================================
    # SPIKE ROUTING
    # =========================================================================

    def read_delayed_outputs(self) -> SynapticInput:
        """Read the delayed output for this tract without writing or advancing.

        Returns:
            A one-entry :class:`SynapticInput` dict keyed by ``spec.synapse_id``.
        """
        if isinstance(self._delay_buffer, HeterogeneousDelayBuffer):
            delayed_spikes = self._delay_buffer.read_heterogeneous()
        else:
            delay_steps = max(1, int(self.spec.delay_ms / self.dt_ms))
            delayed_spikes = self._delay_buffer.read(delay_steps)  # type: ignore[attr-defined]

        return {self.spec.synapse_id: delayed_spikes}

    def write_and_advance(self, source_outputs: BrainOutput) -> None:
        """Write the current source spikes to the buffer and advance the pointer.

        Args:
            source_outputs: Full brain output dict from the current timestep.
                            The tract extracts its source region/population.
        """
        sid = self.spec.synapse_id
        population_outputs = source_outputs.get(sid.source_region, {})
        spikes = population_outputs.get(sid.source_population)

        if spikes is not None:
            validate_spike_tensor(spikes)
            if spikes.shape[0] != self.spec.size:
                raise ValueError(
                    f"AxonalTract size mismatch for {sid}: "
                    f"expected {self.spec.size} neurons, got {spikes.shape[0]}."
                )
            self._delay_buffer.write(spikes)  # type: ignore[attr-defined]

        self._delay_buffer.advance()  # type: ignore[attr-defined]

    # =========================================================================
    # TEMPORAL PARAMETER MANAGEMENT
    # =========================================================================

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Resize the delay buffer for a new simulation timestep.

        Delay durations are fixed in milliseconds; changing ``dt_ms`` alters
        how many buffer slots are needed.

        Args:
            dt_ms: New timestep in milliseconds (must be positive).
        """
        old_dt_ms = self.dt_ms
        self.dt_ms = dt_ms
        self._delay_buffer.resize_for_new_dt(  # type: ignore[attr-defined]
            new_dt_ms=dt_ms,
            delay_ms=self.spec.delay_ms,
            old_dt_ms=old_dt_ms,
        )
