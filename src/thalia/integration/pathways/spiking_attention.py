"""
Spiking Attention Pathway - Fully spiking PFC → Cortex attention modulation.

This pathway implements top-down attention using spiking neurons with
temporal coding. The PFC attention signal modulates cortical processing
through precise spike timing.

Key features:
1. LIF neurons for realistic spike dynamics
2. Temporal coding of attention strength
3. Phase coupling to theta oscillations
4. Spike-timing-dependent gain modulation
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from ..spiking_pathway import SpikingPathway, SpikingPathwayConfig, TemporalCoding


@dataclass
class SpikingAttentionPathwayConfig(SpikingPathwayConfig):
    """Configuration for spiking attention pathway."""

    # Override defaults for attention-specific behavior
    temporal_coding: TemporalCoding = TemporalCoding.PHASE  # Phase coding for attention rhythms
    axonal_delay_ms: float = 2.0  # Faster for top-down signals

    # Attention-specific parameters
    attention_gain: float = 2.0  # How much attention amplifies responses
    attention_threshold: float = 0.3  # Minimum firing rate for attention
    beta_oscillation_freq: float = 20.0  # Beta band for attention (Hz)

    # Gain modulation parameters
    multiplicative_gain: bool = True  # Multiplicative vs additive modulation
    gain_nonlinearity: str = "sigmoid"  # sigmoid, relu, or linear

    # Dimension parameters
    input_size: int = 256  # Input dimension (before attention)
    cortex_size: int = 128  # Cortex dimension (PFC output size)


class SpikingAttentionPathway(SpikingPathway):
    """
    Spiking pathway for PFC → Cortex attention modulation.

    Uses phase-coded spikes to implement gain modulation:
    - High PFC firing → enhanced cortical responses
    - Precise spike timing encodes attention priority
    - Beta rhythm provides temporal structure

    Architecture:
      PFC activity (source_size=pfc_size)
        → SpikingPathway → target_size spikes
        → attention_encoder → attention signal
        → gain_output → gain for input modulation
    """

    def __init__(self, config: SpikingAttentionPathwayConfig):
        """Initialize spiking attention pathway."""
        # Store attention-specific config before super().__init__
        self.input_size = config.input_size
        self.attention_gain = config.attention_gain
        self.attention_threshold = config.attention_threshold
        self.beta_freq = config.beta_oscillation_freq
        self.multiplicative_gain = config.multiplicative_gain
        self.gain_nonlinearity = config.gain_nonlinearity

        # Initialize base spiking pathway
        # source_size = pfc_size, target_size = cortex_size
        super().__init__(config)

        # Input projection (from input space to pathway)
        self.input_projection = nn.Linear(config.input_size, config.source_size)
        nn.init.orthogonal_(self.input_projection.weight, gain=0.5)

        # Attention computation layer (takes spiking pathway output)
        # SpikingPathway output is target_size, so input should be target_size
        self.attention_encoder = nn.Sequential(
            nn.Linear(config.target_size, config.target_size),
            nn.LayerNorm(config.target_size),
        )

        # Gain modulation output (to match input size for modulation)
        self.gain_output = nn.Linear(config.target_size, config.input_size)
        nn.init.zeros_(self.gain_output.weight)  # Start with no modulation
        nn.init.ones_(self.gain_output.bias)  # Bias = 1 for multiplicative

        # Beta phase tracking
        self.register_buffer("beta_phase", torch.tensor(0.0))
        self.register_buffer("beta_period", torch.tensor(1.0 / config.beta_oscillation_freq))
        # Attention strength tracking
        self.current_attention: Optional[torch.Tensor] = None

    def _compute_beta_phase(self, dt: float) -> torch.Tensor:
        """Compute current beta oscillation phase."""
        self.beta_phase = (self.beta_phase + dt) % self.beta_period
        return 2 * torch.pi * self.beta_phase / self.beta_period

    def compute_attention(
        self,
        pfc_activity: torch.Tensor,
        dt: float = 1.0
    ) -> torch.Tensor:
        """
        Compute attention signal from PFC activity.

        Args:
            pfc_activity: PFC spike rates or potentials [batch, cortex_size]
            dt: Time step in ms

        Returns:
            attention: Attention weights [batch, target_size]
        """
        # =====================================================================
        # SHAPE ASSERTIONS - catch dimension mismatches early with clear messages
        # =====================================================================
        # Note: pfc_activity can be smaller than source_size, we'll pad it below
        assert pfc_activity.dim() == 2, (
            f"SpikingAttentionPathway.compute_attention: pfc_activity must be 2D [batch, size], "
            f"got shape {pfc_activity.shape}"
        )
        
        batch_size = pfc_activity.shape[0]
        
        # Update beta phase
        beta_phase = self._compute_beta_phase(dt)
        
        # Process through spiking pathway
        # First project to source size if needed
        if pfc_activity.shape[-1] != self.config.source_size:
            # PFC activity might be smaller, expand
            source_input = torch.zeros(batch_size, self.config.source_size, device=pfc_activity.device)
            source_input[:, :pfc_activity.shape[-1]] = pfc_activity
        else:
            source_input = pfc_activity

        # Get spiking dynamics - forward returns just output spikes
        output_spikes = self.forward(source_input.squeeze(), dt=dt)

        # Encode attention from output spikes
        attention_raw = self.attention_encoder(output_spikes)

        # Phase modulation - attention peaks at specific beta phase
        phase_modulation = 0.5 * (1 + torch.cos(beta_phase))
        attention = attention_raw * phase_modulation

        # Store for diagnostics
        self.current_attention = attention.detach()

        return attention

    def modulate(
        self,
        input_signal: torch.Tensor,
        pfc_activity: torch.Tensor,
        dt: float = 1.0
    ) -> torch.Tensor:
        """
        Apply attention-based gain modulation to input signal.

        Args:
            input_signal: Input to modulate [batch, input_size]
            pfc_activity: PFC activity for attention [batch, cortex_size]
            dt: Time step in ms

        Returns:
            modulated: Gain-modulated signal [batch, input_size]
        """
        # Compute attention signal
        attention = self.compute_attention(pfc_activity, dt)

        # Map attention to gain values
        gain_raw = self.gain_output(attention)

        # Apply nonlinearity
        if self.gain_nonlinearity == "sigmoid":
            gain = 1 + (self.attention_gain - 1) * torch.sigmoid(gain_raw)
        elif self.gain_nonlinearity == "relu":
            gain = 1 + torch.relu(gain_raw) * self.attention_gain
        else:  # linear
            gain = 1 + gain_raw

        # Apply modulation
        if self.multiplicative_gain:
            modulated = input_signal * gain
        else:
            modulated = input_signal + gain - 1  # Additive offset

        return modulated

    def get_diagnostics(self) -> dict:
        """Get attention-specific diagnostics."""
        diag = super().get_diagnostics()

        if self.current_attention is not None:
            diag.update({
                "attention_mean": self.current_attention.mean().item(),
                "attention_std": self.current_attention.std().item(),
                "attention_max": self.current_attention.max().item(),
                "beta_phase": (self.beta_phase / self.beta_period * 360).item(),  # degrees
            })

        return diag

    def get_state(self) -> Dict[str, Any]:
        """Get attention pathway state for checkpointing.

        Extends base SpikingPathway state with attention-specific components:
        - input_projection: Weights and biases
        - attention_encoder: Weights and biases for attention computation
        - gain_output: Weights and biases for gain modulation
        - beta_phase: Beta oscillation phase
        """
        # Get base pathway state
        state = super().get_state()

        # Add attention-specific state
        state["attention_state"] = {
            "input_projection": {
                "weight": self.input_projection.weight.data.clone(),
                "bias": self.input_projection.bias.data.clone(),
            },
            "attention_encoder": {
                "0.weight": self.attention_encoder[0].weight.data.clone(),
                "0.bias": self.attention_encoder[0].bias.data.clone(),
                "1.weight": self.attention_encoder[1].weight.data.clone(),
                "1.bias": self.attention_encoder[1].bias.data.clone(),
            },
            "gain_output": {
                "weight": self.gain_output.weight.data.clone(),
                "bias": self.gain_output.bias.data.clone(),
            },
            "beta_phase": self.beta_phase.clone(),
        }

        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load attention pathway state from checkpoint.

        Args:
            state: State dictionary from get_state()

        Note:
            Restores base pathway state plus attention-specific components.
        """
        # Load base pathway state
        super().load_state(state)

        # Load attention-specific state
        if "attention_state" in state:
            device = self.weights.device
            attention_state = state["attention_state"]

            # Restore input projection
            input_proj = attention_state["input_projection"]
            self.input_projection.weight.data.copy_(input_proj["weight"].to(device))
            self.input_projection.bias.data.copy_(input_proj["bias"].to(device))

            # Restore attention encoder
            encoder = attention_state["attention_encoder"]
            self.attention_encoder[0].weight.data.copy_(encoder["0.weight"].to(device))
            self.attention_encoder[0].bias.data.copy_(encoder["0.bias"].to(device))
            self.attention_encoder[1].weight.data.copy_(encoder["1.weight"].to(device))
            self.attention_encoder[1].bias.data.copy_(encoder["1.bias"].to(device))

            # Restore gain output
            gain = attention_state["gain_output"]
            self.gain_output.weight.data.copy_(gain["weight"].to(device))
            self.gain_output.bias.data.copy_(gain["bias"].to(device))

            # Restore beta phase
            self.beta_phase.copy_(attention_state["beta_phase"].to(device))
