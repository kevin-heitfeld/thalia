"""
Theta oscillation dynamics for hippocampal processing.

Theta oscillations (6-10 Hz) are critical for:
1. Separating encoding vs retrieval in time (phase separation)
2. Organizing memory sequences (phase precession)
3. Coordinating hippocampal-cortical communication

This module provides:
- ThetaState: Tracks theta phase and provides encoding/retrieval strengths
- ThetaConfig: Configuration for theta oscillations
- TrialPhase: Enum for task phases (ENCODE, DELAY, RETRIEVE)
- FeedforwardInhibition: Computes transient inhibition at stimulus changes
- TemporalIntegrationLayer: Models EC layer II/III for cortex→hippocampus

NOTE: ThetaState, ThetaConfig, and TrialPhase are now re-exported from
      thalia.core.event_system for consistency. This module is maintained
      for backward compatibility and provides additional utility classes.

References:
- Hasselmo et al. (2002): Theta rhythm and encoding/retrieval
- Colgin (2013): Theta-gamma coupling in hippocampus
- Siegle & Wilson (2014): Enhancement of encoding and retrieval
"""

from typing import Optional
import math

import torch

# Re-export canonical implementations from event_system
from ..core.event_system import (
    ThetaGenerator as ThetaState,  # ThetaState is now an alias for ThetaGenerator
    ThetaConfig,
    TrialPhase,
)

# Export for backward compatibility
__all__ = [
    "ThetaState",
    "ThetaConfig", 
    "TrialPhase",
    "FeedforwardInhibition",
    "TemporalIntegrationLayer",
]


class FeedforwardInhibition:
    """
    Computes feedforward inhibition triggered by stimulus changes.
    
    In biological circuits, the arrival of a new stimulus triggers
    strong transient inhibition via fast-spiking interneurons. This:
    
    1. Clears ongoing activity (no explicit reset needed!)
    2. Sharpens temporal precision of responses
    3. Enables clean separation between stimuli
    
    The inhibition strength is proportional to how much the input changed.
    
    Example:
        ffi = FeedforwardInhibition(threshold=0.5, decay=0.9)
        
        for stimulus in stimuli:
            # Compute inhibition based on stimulus change
            inhibition = ffi.compute(stimulus)
            
            # Apply to membrane potentials
            membrane = membrane - inhibition * max_inhibition
    """
    
    def __init__(
        self,
        threshold: float = 0.3,
        max_inhibition: float = 5.0,
        decay_rate: float = 0.8,
        steepness: float = 10.0,
    ):
        """
        Initialize feedforward inhibition.
        
        Args:
            threshold: Input change threshold to trigger inhibition
            max_inhibition: Maximum inhibition strength
            decay_rate: How fast the previous input trace decays
            steepness: Steepness of sigmoid for change detection
        """
        self.threshold = threshold
        self.max_inhibition = max_inhibition
        self.decay_rate = decay_rate
        self.steepness = steepness
        
        # Track previous input for change detection
        self._prev_input: Optional[torch.Tensor] = None
        self._current_inhibition: float = 0.0
    
    def compute(
        self, 
        current_input: torch.Tensor,
        return_tensor: bool = True,
    ) -> torch.Tensor:
        """
        Compute feedforward inhibition for current input.
        
        Args:
            current_input: Current input tensor
            return_tensor: If True, return tensor matching input shape
        
        Returns:
            Inhibition strength (scalar or tensor matching input shape)
        """
        if self._prev_input is None:
            # First input - no inhibition
            self._prev_input = current_input.detach().clone()
            if return_tensor:
                return torch.zeros_like(current_input)
            return torch.tensor(0.0)
        
        # Compute input change magnitude (normalized by input size)
        input_diff = (current_input - self._prev_input).abs()
        change_magnitude = input_diff.sum() / (current_input.numel() + 1e-6)
        
        # Sigmoid activation based on change magnitude
        # High change → high inhibition
        inhibition = torch.sigmoid(
            (change_magnitude - self.threshold) * self.steepness
        ) * self.max_inhibition
        
        # Update previous input (with decay for smooth tracking)
        self._prev_input = (
            self.decay_rate * self._prev_input + 
            (1 - self.decay_rate) * current_input.detach().clone()
        )
        
        self._current_inhibition = inhibition.item()
        
        if return_tensor:
            # Return per-neuron inhibition proportional to local change
            return input_diff * inhibition / (input_diff.max() + 1e-6)
        return inhibition
    
    @property
    def current_inhibition(self) -> float:
        """Current inhibition level."""
        return self._current_inhibition


class TemporalIntegrationLayer:
    """
    Temporal integration layer between cortex and hippocampus.
    
    This models the entorhinal cortex (EC) layer II/III, which has slow
    membrane dynamics that integrate cortical input over ~100ms (roughly
    one theta cycle) before projecting to hippocampus.
    
    Problem this solves:
        Cortex L2/3 output is sparse and temporally variable (1-8 spikes
        per timestep, different neurons each time). The hippocampus NMDA
        mechanism needs consistent patterns to detect coincidence.
    
    Solution:
        1. Accumulate cortex spikes over time (leaky integration)
        2. Convert to stable firing rate representation
        3. Re-encode as consistent spike pattern
        4. Hippocampus sees the same pattern each timestep
    
    Biological basis:
        - EC layer II stellate cells have slow membrane τ (~50-100ms)
        - Grid cells and other EC neurons show persistent activity
        - EC serves as a buffer between cortex and hippocampus
    
    Example:
        integrator = TemporalIntegrationLayer(n_neurons=96, tau=50.0)
        
        for t in range(n_timesteps):
            cortex_spikes = cortex.forward(input)
            # Get stable representation for hippocampus
            stable_input = integrator.integrate(cortex_spikes)
            hippo_out = hippocampus.forward(stable_input)
    """
    
    def __init__(
        self,
        n_neurons: int,
        tau: float = 50.0,
        threshold: float = 0.5,
        gain: float = 2.0,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize temporal integration layer.
        
        Args:
            n_neurons: Number of neurons to integrate
            tau: Integration time constant in ms (higher = more smoothing)
            threshold: Threshold for spike generation from rate
            gain: Gain factor for rate-to-spike conversion
            device: Torch device
        """
        self.n_neurons = n_neurons
        self.tau = tau
        self.threshold = threshold
        self.gain = gain
        self.device = device
        
        # Leaky integration trace (accumulates spikes over time)
        self._trace: Optional[torch.Tensor] = None
        
        # Precompute decay factor
        self._decay = math.exp(-1.0 / tau)  # Decay per ms (assuming dt=1ms)
    
    def integrate(
        self,
        spikes: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        Integrate spike input and produce stable output pattern.
        
        The output is a consistent spike pattern where the SAME neurons
        fire each timestep (based on accumulated activity), rather than
        the variable pattern from raw cortex output.
        
        Args:
            spikes: Input spike pattern [batch, n_neurons]
            dt: Timestep in ms
        
        Returns:
            Integrated spike pattern [batch, n_neurons]
        """
        if spikes.dim() == 1:
            spikes = spikes.unsqueeze(0)
        
        batch_size = spikes.shape[0]
        
        # Initialize trace if needed
        if self._trace is None or self._trace.shape[0] != batch_size:
            self._trace = torch.zeros(batch_size, self.n_neurons, device=self.device)
        
        # Leaky integration: trace = trace * decay + new_spikes
        decay = math.exp(-dt / self.tau)
        self._trace = self._trace * decay + spikes.float()
        
        # Convert rate (trace) to spikes
        # Higher trace = higher probability of spiking
        # Use gain to control overall firing rate
        rate = self._trace * self.gain
        
        # Threshold to get consistent spike pattern
        # Neurons with accumulated activity above threshold spike
        output_spikes = (rate > self.threshold).float()
        
        return output_spikes
    
    def get_rate(self) -> Optional[torch.Tensor]:
        """Get current integration trace (for debugging)."""
        return self._trace.clone() if self._trace is not None else None

    def reset_state(self) -> None:
        """Reset integration state (for hard episode boundaries only)."""
        self._trace = None
