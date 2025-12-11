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

Author: Thalia Project
Date: December 2025
"""

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class STDPConfig:
    """Configuration for STDP eligibility computation."""
    
    # Time constants (ms)
    stdp_tau_ms: float = 20.0          # Spike trace decay time
    eligibility_tau_ms: float = 1000.0 # Eligibility decay time
    
    # Learning rates
    stdp_lr: float = 0.01              # Base STDP learning rate
    a_plus: float = 1.0                # LTP amplitude
    a_minus: float = 0.012             # LTD amplitude (default ~1.2%)
    
    # Weight bounds
    w_min: float = 0.0
    w_max: float = 1.0
    
    # Heterosynaptic plasticity
    heterosynaptic_ratio: float = 0.3  # Fraction of learning in non-active synapses


class EligibilityTraceManager:
    """
    Manages STDP eligibility traces with soft bounds and exponential decay.
    
    This class consolidates the repeated pattern of:
    1. Decay spike traces (pre/post)
    2. Add current spikes to traces
    3. Compute STDP (LTP/LTD via outer products)
    4. Apply soft bounds (prevent saturation)
    5. Accumulate into eligibility traces
    
    Usage:
    ======
    
        manager = EligibilityTraceManager(
            n_input=256,
            n_output=128,
            config=STDPConfig(),
            device=device
        )
        
        # In forward() loop:
        manager.update_traces(input_spikes, output_spikes, dt_ms=1.0)
        
        # When reward arrives:
        dw = manager.get_eligibility() * dopamine
        weights += dw
        manager.decay_eligibility(dt_ms=1.0)
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
            config: STDP configuration
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
        trace_decay = 1.0 - dt_ms / self.config.stdp_tau_ms
        
        # Convert to float if needed
        input_float = input_spikes.float() if input_spikes.dtype == torch.bool else input_spikes
        output_float = output_spikes.float() if output_spikes.dtype == torch.bool else output_spikes
        
        # Decay and add current spikes
        self.input_trace = self.input_trace * trace_decay + input_float
        self.output_trace = self.output_trace * trace_decay + output_float
    
    def compute_stdp_eligibility(
        self,
        weights: torch.Tensor,
        lr_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute STDP eligibility with soft bounds.
        
        STDP Rule:
            LTP (Long-Term Potentiation): Post fires after Pre → strengthen
            LTD (Long-Term Depression): Pre fires after Post → weaken
        
        Soft Bounds:
            - LTP weakens as weight approaches w_max
            - LTD weakens as weight approaches w_min
            - Prevents saturation and maintains dynamic range
        
        Args:
            weights: Current weight matrix [n_output, n_input]
            lr_scale: Learning rate scaling factor (e.g., d1_lr_scale, d2_lr_scale)
        
        Returns:
            eligibility_update: Eligibility increment [n_output, n_input]
        """
        cfg = self.config
        
        # STDP: outer products of traces
        # LTP: post × pre_trace (post fires when pre trace is high)
        # LTD: post_trace × pre (pre fires when post trace is high)
        ltp = torch.outer(self.output_trace, self.input_trace)
        ltd = torch.outer(self.output_trace, self.input_trace)  # Simplified: use same for both
        
        # Normalize weights to [0, 1]
        w_normalized = (weights - cfg.w_min) / (cfg.w_max - cfg.w_min)
        
        # Soft bounds: LTP weakens near max, LTD weakens near min
        ltp_factor = 1.0 - w_normalized
        ltd_factor = w_normalized
        
        # Apply soft bounds
        soft_ltp = ltp * ltp_factor * cfg.a_plus
        soft_ltd = ltd * ltd_factor * cfg.a_minus
        
        # Combine LTP and LTD
        eligibility_update = cfg.stdp_lr * lr_scale * (
            soft_ltp - cfg.heterosynaptic_ratio * soft_ltd
        )
        
        return eligibility_update
    
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
        
        Usage:
            ltp, ltd = manager.compute_ltp_ltd_separate(input_spikes, output_spikes)
            # Apply custom modulations
            ltp = ltp * dopamine_factor * ach_factor
            ltd = ltd * (1 - dopamine_factor) * phase_factor
            # Combine with learning rate
            dw = lr * (ltp - hetero_ratio * ltd)
        
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
        
        # Convert to float if needed
        input_float = input_spikes.float() if input_spikes.dtype == torch.bool else input_spikes
        output_float = output_spikes.float() if output_spikes.dtype == torch.bool else output_spikes
        
        # LTP: post spike when pre trace is high → post fires AFTER pre
        # If post fires now and pre_trace is high, pre fired recently → strengthen
        ltp = torch.outer(output_float, self.input_trace) * cfg.a_plus if output_float.sum() > 0 else 0
        
        # LTD: pre spike when post trace is high → pre fires AFTER post
        # If pre fires now and post_trace is high, post fired recently → weaken
        ltd = torch.outer(self.output_trace, input_float) * cfg.a_minus if input_float.sum() > 0 else 0
        
        return ltp, ltd
    
    def compute_stdp_eligibility_separate_ltd(
        self,
        input_spikes: torch.Tensor,
        output_spikes: torch.Tensor,
        weights: torch.Tensor,
        lr_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute STDP eligibility with separate LTP/LTD outer products.
        
        This version computes LTP and LTD separately using current spikes
        and traces, which is more biologically accurate for timing-dependent
        plasticity.
        
        Args:
            input_spikes: Current input spikes [n_input]
            output_spikes: Current output spikes [n_output]
            weights: Current weight matrix [n_output, n_input]
            lr_scale: Learning rate scaling factor
        
        Returns:
            eligibility_update: Eligibility increment [n_output, n_input]
        """
        cfg = self.config
        
        # Convert to float if needed
        input_float = input_spikes.float() if input_spikes.dtype == torch.bool else input_spikes
        output_float = output_spikes.float() if output_spikes.dtype == torch.bool else output_spikes
        
        # STDP with proper timing:
        # LTP: post fires, pre_trace is high → post fires AFTER pre
        # LTD: pre fires, post_trace is high → pre fires AFTER post
        ltp = torch.outer(output_float, self.input_trace)
        ltd = torch.outer(self.output_trace, input_float)
        
        # Normalize weights to [0, 1]
        w_normalized = (weights - cfg.w_min) / (cfg.w_max - cfg.w_min)
        
        # Soft bounds
        ltp_factor = 1.0 - w_normalized
        ltd_factor = w_normalized
        
        # Apply soft bounds
        soft_ltp = ltp * ltp_factor * cfg.a_plus
        soft_ltd = ltd * ltd_factor * cfg.a_minus
        
        # Combine
        eligibility_update = cfg.stdp_lr * lr_scale * (
            soft_ltp - cfg.heterosynaptic_ratio * soft_ltd
        )
        
        return eligibility_update
    
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
    
    def decay_eligibility(self, dt_ms: float) -> None:
        """
        Decay eligibility traces without adding new updates.
        
        Used after reward delivery or during periods without learning.
        
        Args:
            dt_ms: Timestep in milliseconds
        """
        eligibility_decay = 1.0 - dt_ms / self.config.eligibility_tau_ms
        self.eligibility = self.eligibility * eligibility_decay
    
    def get_eligibility(self) -> torch.Tensor:
        """
        Get current eligibility traces.
        
        Returns:
            eligibility: Current eligibility [n_output, n_input]
        """
        return self.eligibility
    
    def reset_traces(self) -> None:
        """Reset all traces to zero (e.g., at episode boundaries)."""
        self.input_trace.zero_()
        self.output_trace.zero_()
        self.eligibility.zero_()
    
    def reset_eligibility(self) -> None:
        """Reset eligibility only (keep spike traces)."""
        self.eligibility.zero_()
    
    def to(self, device: torch.device) -> 'EligibilityTraceManager':
        """Move all tensors to specified device."""
        self.device = device
        self.input_trace = self.input_trace.to(device)
        self.output_trace = self.output_trace.to(device)
        self.eligibility = self.eligibility.to(device)
        return self
    
    def add_neurons(self, n_new: int, dimension: str = 'output') -> 'EligibilityTraceManager':
        """
        Grow traces to accommodate new neurons.
        
        Args:
            n_new: Number of new neurons to add
            dimension: Which dimension to grow ('input' or 'output')
        
        Returns:
            New EligibilityTraceManager with expanded traces
        """
        if dimension == 'output':
            # Expand output dimension
            new_n_output = self.n_output + n_new
            new_output_trace = torch.zeros(n_new, device=self.device)
            expanded_output_trace = torch.cat([self.output_trace, new_output_trace], dim=0)
            
            # Expand eligibility [n_output, n_input]
            new_eligibility = torch.zeros(n_new, self.n_input, device=self.device)
            expanded_eligibility = torch.cat([self.eligibility, new_eligibility], dim=0)
            
            # Create new manager
            new_manager = EligibilityTraceManager(
                n_input=self.n_input,
                n_output=new_n_output,
                config=self.config,
                device=self.device,
            )
            new_manager.output_trace = expanded_output_trace
            new_manager.eligibility = expanded_eligibility
            new_manager.input_trace = self.input_trace  # Keep existing
            return new_manager
            
        elif dimension == 'input':
            # Expand input dimension
            new_n_input = self.n_input + n_new
            new_input_trace = torch.zeros(n_new, device=self.device)
            expanded_input_trace = torch.cat([self.input_trace, new_input_trace], dim=0)
            
            # Expand eligibility [n_output, n_input]
            new_eligibility = torch.zeros(self.n_output, n_new, device=self.device)
            expanded_eligibility = torch.cat([self.eligibility, new_eligibility], dim=1)
            
            # Create new manager
            new_manager = EligibilityTraceManager(
                n_input=new_n_input,
                n_output=self.n_output,
                config=self.config,
                device=self.device,
            )
            new_manager.input_trace = expanded_input_trace
            new_manager.eligibility = expanded_eligibility
            new_manager.output_trace = self.output_trace  # Keep existing
            return new_manager
        else:
            raise ValueError(f"dimension must be 'input' or 'output', got {dimension}")

