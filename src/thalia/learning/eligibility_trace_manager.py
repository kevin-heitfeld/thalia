"""
Eligibility Trace Utilities - Consolidated STDP eligibility trace computation.

This module provides shared utilities for computing STDP-based eligibility traces
with soft bounds and exponential decay. These patterns are used across:
- Striatum (D1/D2 pathways)
- Pathways (inter-region connections)
- Cortex, Hippocampus, Cerebellum (regional plasticity)

Key Concept:
============
Eligibility traces capture correlations between pre- and post-synaptic activity
over time. When a neuromodulatory signal (e.g., dopamine) arrives later, it
gates weight changes based on the accumulated eligibility.

    Eligibility ← decay(Eligibility) + STDP(pre_trace, post_trace)
    ΔWeight ← Dopamine × Eligibility

Biological Accuracy:
====================
- Exponential decay with biological time constants (100-1000ms)
- Soft bounds prevent saturation (LTP weakens near w_max, LTD weakens near w_min)
- Heterosynaptic plasticity (weak learning in non-active synapses)
- Local computation (no global error signals)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

if TYPE_CHECKING:
    from thalia.learning.strategies import STDPConfig


class EligibilityTraceManager:
    """
    Manages STDP eligibility traces with soft bounds and exponential decay.

    This class consolidates the repeated pattern of:
    1. Decay spike traces (pre/post)
    2. Add current spikes to traces
    3. Compute STDP (LTP/LTD via outer products)
    4. Apply soft bounds (prevent saturation)
    5. Accumulate into eligibility traces
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
        self.n_input = n_input
        self.n_output = n_output
        self.config = config
        self.device = device

        # Initialize traces
        self.input_trace = torch.zeros(n_input, device=device)
        self.output_trace = torch.zeros(n_output, device=device)
        self.eligibility = torch.zeros(n_output, n_input, device=device)

    def update_traces(
        self,
        input_spikes: torch.Tensor,
        output_spikes: torch.Tensor,
        dt_ms: float,
    ) -> None:
        """
        Update input/output spike traces with exponential decay.

        Trace(t) = Trace(t-1) * exp(-dt/tau) + Spikes(t)

        Args:
            input_spikes: Presynaptic spikes [n_input] (bool or float)
            output_spikes: Postsynaptic spikes [n_output] (bool or float)
            dt_ms: Timestep in milliseconds
        """
        # Compute decay factor
        trace_decay = 1.0 - dt_ms / self.config.tau_plus

        # Convert to float if needed
        input_float = input_spikes.float() if input_spikes.dtype == torch.bool else input_spikes
        output_float = output_spikes.float() if output_spikes.dtype == torch.bool else output_spikes

        # Decay and add current spikes
        self.input_trace = self.input_trace * trace_decay + input_float
        self.output_trace = self.output_trace * trace_decay + output_float

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
        Add eligibility update to accumulated eligibility with decay.

        Eligibility(t) = Eligibility(t-1) * exp(-dt/tau_elig) + Update(t)

        Args:
            eligibility_update: Eligibility increment [n_output, n_input]
            dt_ms: Timestep in milliseconds
        """
        eligibility_decay = 1.0 - dt_ms / self.config.eligibility_tau_ms
        self.eligibility = self.eligibility * eligibility_decay + eligibility_update

    def to(self, device: torch.device) -> EligibilityTraceManager:
        """Move all tensors to specified device."""
        self.device = device
        self.input_trace = self.input_trace.to(device)
        self.output_trace = self.output_trace.to(device)
        self.eligibility = self.eligibility.to(device)
        return self
