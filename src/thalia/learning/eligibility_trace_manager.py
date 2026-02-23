"""
Eligibility Trace Utilities - Consolidated STDP eligibility trace computation.

This module provides shared utilities for computing STDP-based eligibility traces
with soft bounds and exponential decay.

Key Concept:
============
Eligibility traces capture correlations between pre- and post-synaptic activity
over time. When a neuromodulatory signal (e.g., dopamine) arrives later, it
gates weight changes based on the accumulated eligibility.

    Eligibility ← decay(Eligibility) + STDP(pre_trace, post_trace)
    ΔWeight ← Dopamine × Eligibility

Biological Accuracy:
====================
- True exponential decay: exp(-dt/tau), not the linear approximation 1 - dt/tau.
  The linear form gives negative decay factors when dt > tau, producing oscillating
  traces with no biological basis.
- Separate tau_plus (LTP pre-trace) and tau_minus (LTD post-trace) — independently
  tunable per pathway (e.g., thalamocortical: tau_plus=20ms, tau_minus=40ms).
- Heterosynaptic plasticity (weak learning in non-active synapses).
- Local computation (no global error signals).
- All trace state registered as nn.Module buffers: correct .to(device) and state_dict.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from thalia.learning.strategies import STDPConfig


class EligibilityTraceManager(nn.Module):
    """
    Manages STDP eligibility traces with exponential decay.

    Stores pre-trace, post-trace, and eligibility accumulator as registered
    nn.Module buffers so that:
    - .to(device) correctly moves all trace state.
    - state_dict() / load_state_dict() preserves traces across checkpoints.
    - The parent STDPStrategy automatically picks up this submodule.

    Decay uses the mathematically correct exp(-dt/tau), not the linear
    approximation 1 - dt/tau which produces negative values for dt > tau.

    Separate decay constants for LTP (tau_plus, applied to pre-trace) and
    LTD (tau_minus, applied to post-trace) match biological STDP asymmetry.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        config: STDPConfig,
        device: torch.device,
    ):
        """
        Initialize eligibility trace manager.

        Args:
            n_input: Number of presynaptic neurons
            n_output: Number of postsynaptic neurons
            config: STDP configuration (from thalia.learning.strategies)
            device: Torch device
        """
        super().__init__()

        self.n_input = n_input
        self.n_output = n_output
        self.config = config

        # All trace state is registered as buffers:
        # - moves with .to(device)
        # - serialised in state_dict()
        self.register_buffer("input_trace",  torch.zeros(n_input,          device=device))
        self.register_buffer("output_trace", torch.zeros(n_output,         device=device))
        self.register_buffer("eligibility",  torch.zeros(n_output, n_input, device=device))

        # Pre-compute decay scalars (updated via update_temporal_parameters)
        self._decay_plus:  float = math.exp(-1.0 / config.tau_plus)
        self._decay_minus: float = math.exp(-1.0 / config.tau_minus)
        self._decay_elig:  float = math.exp(-1.0 / config.eligibility_tau_ms)

    # ------------------------------------------------------------------
    # Temporal parameter management
    # ------------------------------------------------------------------

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Recompute decay scalars when the simulation timestep changes."""
        self._decay_plus  = math.exp(-dt_ms / self.config.tau_plus)
        self._decay_minus = math.exp(-dt_ms / self.config.tau_minus)
        self._decay_elig  = math.exp(-dt_ms / self.config.eligibility_tau_ms)

    # ------------------------------------------------------------------
    # Trace update
    # ------------------------------------------------------------------

    def update_traces(
        self,
        input_spikes: torch.Tensor,
        output_spikes: torch.Tensor,
        dt_ms: float,
    ) -> None:
        """
        Update input/output spike traces with exponential decay.

        Trace(t) = Trace(t-1) * exp(-dt/tau) + Spikes(t)

        LTP pre-trace uses tau_plus; LTD post-trace uses tau_minus.
        Decay factors are pre-computed and updated via update_temporal_parameters().

        Args:
            input_spikes: Presynaptic spikes [n_input] (bool or float)
            output_spikes: Postsynaptic spikes [n_output] (bool or float)
            dt_ms: Timestep in milliseconds (used only to refresh cached decays if changed)
        """
        # Refresh decay if dt changed (rare but safe)
        expected_plus = math.exp(-dt_ms / self.config.tau_plus)
        if abs(self._decay_plus - expected_plus) > 1e-7:
            self.update_temporal_parameters(dt_ms)

        # Convert to float if needed
        input_float  = input_spikes.float()  if input_spikes.dtype  == torch.bool else input_spikes
        output_float = output_spikes.float() if output_spikes.dtype == torch.bool else output_spikes

        # LTP pre-trace: decays with tau_plus
        # LTD post-trace: decays with tau_minus
        self.input_trace  = self.input_trace  * self._decay_plus  + input_float
        self.output_trace = self.output_trace * self._decay_minus + output_float

    def compute_ltp_ltd_separate(
        self,
        input_spikes: torch.Tensor,
        output_spikes: torch.Tensor,
    ) -> Tuple[torch.Tensor | int, torch.Tensor | int]:
        """
        Compute separate LTP and LTD eligibility components without combining.

        This method returns raw LTP and LTD tensors WITHOUT:
        - Combining them
        - Applying soft bounds
        - Applying learning rates
        - Applying heterosynaptic scaling

        This allows the caller to apply custom modulations (dopamine, ACh, NE, etc.)
        to LTP and LTD independently before combining.

        Args:
            input_spikes: Current input spikes [n_input]
            output_spikes: Current output spikes [n_output]

        Returns:
            ltp: Long-term potentiation component [n_output, n_input], or 0 if no output spikes
            ltd: Long-term depression component [n_output, n_input], or 0 if no input spikes

        Note:
            Returns 0 (scalar) instead of zero tensor when no spikes occur,
            which broadcasts correctly in arithmetic operations.
        """
        cfg = self.config

        # OPTIMIZATION: Check for any spikes before computing outer products
        # Avoid unnecessary computation when no spikes occur
        has_output_spikes = output_spikes.any() if output_spikes.dtype == torch.bool else output_spikes.sum() > 0
        has_input_spikes = input_spikes.any() if input_spikes.dtype == torch.bool else input_spikes.sum() > 0

        # LTP: post spike when pre trace is high → post fires AFTER pre
        # If post fires now and pre_trace is high, pre fired recently → strengthen
        if has_output_spikes:
            # OPTIMIZATION: Fuse float conversion with outer product
            output_float = output_spikes.float() if output_spikes.dtype == torch.bool else output_spikes
            ltp = torch.outer(output_float, self.input_trace) * cfg.a_plus
        else:
            ltp = 0

        # LTD: pre spike when post trace is high → pre fires AFTER post
        # If pre fires now and post_trace is high, post fired recently → weaken
        if has_input_spikes:
            # OPTIMIZATION: Fuse float conversion with outer product
            input_float = input_spikes.float() if input_spikes.dtype == torch.bool else input_spikes
            ltd = torch.outer(self.output_trace, input_float) * cfg.a_minus
        else:
            ltd = 0

        return ltp, ltd

    def accumulate_eligibility(
        self,
        eligibility_update: torch.Tensor,
        dt_ms: float,
    ) -> None:
        """
        Add eligibility update to accumulated eligibility with exponential decay.

        Eligibility(t) = Eligibility(t-1) * exp(-dt/tau_elig) + Update(t)

        Uses the pre-computed _decay_elig factor (refreshed by update_temporal_parameters).

        Args:
            eligibility_update: Eligibility increment [n_output, n_input]
            dt_ms: Timestep in milliseconds (used only to refresh cached decay if changed)
        """
        # Refresh if dt changed
        expected = math.exp(-dt_ms / self.config.eligibility_tau_ms)
        if abs(self._decay_elig - expected) > 1e-7:
            self.update_temporal_parameters(dt_ms)
        self.eligibility = self.eligibility * self._decay_elig + eligibility_update

    # ------------------------------------------------------------------
    # .to(device) is handled automatically by nn.Module via register_buffer.
    # No hand-written override needed.
    # ------------------------------------------------------------------
