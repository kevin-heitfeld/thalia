"""Batched Short-Term Plasticity (STP) for the entire brain.

Concatenates all STP instances across all regions into contiguous global
arrays and runs one fused C++ kernel per timestep, eliminating hundreds
of individual Python STP.forward() calls.

Individual ShortTermPlasticity modules' ``u`` and ``x`` tensors become
views into the global arrays, so any code that inspects STP state
(diagnostics, logging) continues to work transparently.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from thalia.brain.synapses.stp import ShortTermPlasticity
from thalia.typing import BrainOutput, SynapseId, SynapticInput
from thalia.utils.stp_fused import is_available as _stp_cpp_available, stp_step as _stp_step_cpp

logger = logging.getLogger(__name__)


class STPBatch:
    """Batched STP state for all connections in the brain.

    Holds global contiguous arrays for ``u``, ``x``, and per-neuron decay
    parameters.  Individual :class:`ShortTermPlasticity` modules' ``u``
    and ``x`` become views into these arrays so state stays in sync.
    """

    def __init__(
        self,
        stp_entries: list[tuple[SynapseId, ShortTermPlasticity]],
        device: torch.device,
    ) -> None:
        # Sort for deterministic ordering
        entries = sorted(stp_entries, key=lambda e: str(e[0]))

        # Build registry: synapse_id → (offset, count)
        total = 0
        self.registry: dict[SynapseId, tuple[int, int]] = {}
        for synapse_id, stp in entries:
            self.registry[synapse_id] = (total, stp.n_pre)
            total += stp.n_pre

        self.total_neurons = total
        self.device = device
        self._entries = entries  # Keep reference for update_temporal_parameters

        if total == 0:
            self._empty = True
            return
        self._empty = False

        # Global state arrays
        self.u = torch.empty(total, device=device)
        self.x = torch.empty(total, device=device)

        # Per-neuron parameters (constant within each STP instance, replicated)
        self.U = torch.empty(total, device=device)
        self.decay_d = torch.empty(total, device=device)
        self.decay_f = torch.empty(total, device=device)
        self.recovery_d = torch.empty(total, device=device)
        self.recovery_f = torch.empty(total, device=device)

        # Pre-spikes buffer (filled each step)
        self.pre_spikes = torch.zeros(total, device=device)

        # Copy state and params from individual modules
        for synapse_id, stp in entries:
            offset, count = self.registry[synapse_id]
            s = slice(offset, offset + count)

            # Copy current state
            self.u[s] = stp.u
            self.x[s] = stp.x

            # Copy parameters
            self.U[s] = stp.U.item()
            self.decay_d[s] = stp.decay_d.item()
            self.decay_f[s] = stp.decay_f.item()
            self.recovery_d[s] = stp.recovery_d.item()
            self.recovery_f[s] = stp.recovery_f.item()

            # Replace individual module state with views into global arrays
            stp.u = self.u[s]
            stp.x = self.x[s]

        # Pre-build spike resolution lookup table.
        # Each entry is (offset, count, synapse_id, source_region, source_pop, is_intra).
        # Pre-classifying inter vs intra avoids per-step dict probing and
        # attribute access on SynapseId objects.
        self._spike_lookup: list[tuple[int, int, SynapseId, str, str, bool]] = []
        for synapse_id, (offset, count) in self.registry.items():
            is_intra = synapse_id.source_region == synapse_id.target_region
            self._spike_lookup.append((
                offset, count, synapse_id,
                synapse_id.source_region if is_intra else synapse_id.target_region,
                synapse_id.source_population,
                is_intra,
            ))

        # Pre-build stable efficacy result dict with views into a persistent tensor.
        # Avoids allocating a new dict + slicing every step.
        self._efficacy_buf = torch.empty(total, device=device)
        self._result_views: dict[SynapseId, torch.Tensor] = {}
        for synapse_id, (offset, count) in self.registry.items():
            self._result_views[synapse_id] = self._efficacy_buf[offset:offset + count]

        self._use_cpp = _stp_cpp_available()
        logger.info(
            "STPBatch: %d instances, %d total neurons, C++=%s",
            len(entries), total, self._use_cpp,
        )

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(
        self,
        all_region_inputs: dict[str, SynapticInput],
        last_brain_output: Optional[BrainOutput],
    ) -> dict[SynapseId, torch.Tensor]:
        """Update all STP state and return efficacy for every STP connection.

        Args:
            all_region_inputs: ``region_inputs`` dict from Brain.forward(),
                mapping ``region_name → {synapse_id → spike_tensor}``.
                Contains inter-region spikes from axonal tracts.
            last_brain_output: Previous step's BrainOutput, used to gather
                intra-region spikes (source_region == target_region).

        Returns:
            Dict mapping each STP synapse_id to its efficacy tensor ``[n_pre]``.
        """
        if self._empty:
            return {}

        # Gather pre-spikes into the global buffer using pre-built lookup table.
        # Each entry pre-classifies inter vs intra and caches lookup keys.
        pre_spikes = self.pre_spikes
        for offset, count, synapse_id, lookup_key, source_pop, is_intra in self._spike_lookup:
            spikes: Optional[torch.Tensor] = None
            if is_intra:
                # Intra-region: look up in last_brain_output[source_region][source_pop]
                if last_brain_output is not None:
                    region_out = last_brain_output.get(lookup_key)
                    if region_out is not None:
                        spikes = region_out.get(source_pop)
            else:
                # Inter-region: look up in all_region_inputs[target_region][synapse_id]
                region_dict = all_region_inputs.get(lookup_key)
                if region_dict is not None:
                    spikes = region_dict.get(synapse_id)

            if spikes is not None:
                pre_spikes[offset:offset + count] = spikes
            else:
                pre_spikes[offset:offset + count] = 0.0

        # Run kernel
        if self._use_cpp:
            efficacy = _stp_step_cpp(
                self.u, self.x, self.U,
                self.decay_d, self.decay_f,
                self.recovery_d, self.recovery_f,
                pre_spikes, self.total_neurons,
            )
        else:
            efficacy = self._step_python()

        # Copy into persistent buffer so pre-built views are valid
        self._efficacy_buf.copy_(efficacy)
        return self._result_views

    def _step_python(self) -> torch.Tensor:
        """Pure-Python fallback matching the C++ kernel exactly."""
        # Continuous decay
        self.u.mul_(self.decay_f).add_(self.recovery_f)
        self.x.mul_(self.decay_d).add_(self.recovery_d)

        # Pre-spike efficacy
        efficacy = self.u * self.x

        # Spike-triggered dynamics
        self.x.addcmul_(self.pre_spikes, efficacy, value=-1.0)
        self.u.addcmul_(self.pre_spikes, self.U - self.U * self.u)

        # Clamp
        self.u.clamp_(0.0, 1.0)
        self.x.clamp_(0.0, 1.0)

        return efficacy

    # ------------------------------------------------------------------
    # Temporal parameter updates
    # ------------------------------------------------------------------

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Recompute decay/recovery factors when dt changes."""
        if self._empty:
            return

        for synapse_id, stp in self._entries:
            offset, count = self.registry[synapse_id]
            s = slice(offset, offset + count)

            # Let the individual module recompute its scalars
            stp.update_temporal_parameters(dt_ms)

            # Copy updated scalars into the global arrays
            self.decay_d[s] = stp.decay_d.item()
            self.decay_f[s] = stp.decay_f.item()
            self.recovery_d[s] = stp.recovery_d.item()
            self.recovery_f[s] = stp.recovery_f.item()
