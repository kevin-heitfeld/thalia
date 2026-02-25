"""Norepinephrine Neuron: Specialized Neuron Type for LC with Synchronized Bursting.

Norepinephrine neurons in the Locus Coeruleus (LC) exhibit characteristic firing
patterns that encode arousal and uncertainty through two distinct modes:

1. **Tonic Pacemaking** (1-3 Hz baseline):
   - Lower baseline than DA neurons (less intrinsic drive)
   - Represents low arousal/alert-but-relaxed state
   - Provides background norepinephrine tone

2. **Phasic Bursting**:
   - **Burst** (10-15 Hz): High uncertainty, novelty, or task difficulty
   - **Duration**: 500ms (longer than DA bursts)
   - **Synchronization**: Strong gap junction coupling → population bursts

Biophysical Mechanisms:
=======================
- **I_h (HCN channels)**: Weaker than DA neurons (lower baseline rate)
- **Gap junctions**: Electrical coupling between LC neurons (synchronized bursts)
- **SK channels**: Calcium-activated K+ channels (adaptation)
- **Small nucleus**: Only ~1,600 neurons in humans (highly synchronized)

This specialized neuron type is used exclusively by the LC region to encode
uncertainty and arousal through long, synchronized bursts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch

from thalia.typing import ConductanceTensor, GapJunctionReversal, VoltageTensor

from .conductance_lif_neuron import ConductanceLIF, ConductanceLIFConfig


@dataclass
class NorepinephrineNeuronConfig(ConductanceLIFConfig):
    """Configuration for norepinephrine neurons with pacemaking and gap junction coupling.

    Extends base LIF with parameters for:
    - Weaker I_h pacemaking (lower baseline than DA)
    - Gap junction coupling (electrical synapses)
    - SK calcium-activated K+ channels (adaptation)
    - Uncertainty-driven burst modulation
    """
    # =========================================================================
    # Membrane properties
    # =========================================================================
    tau_mem: Union[float, torch.Tensor] = 18.0  # Slightly slower than DA neurons
    v_reset: float = -0.12  # Slightly deeper hyperpolarization
    v_threshold: Union[float, torch.Tensor] = 1.0
    tau_ref: float = 2.5  # Slightly longer refractory period
    g_L: float = 0.056  # Leak conductance

    # =========================================================================
    # Noise
    # =========================================================================
    noise_std: float = 0.08

    # =========================================================================
    # I_h (HCN) pacemaker current parameters
    # =========================================================================
    # I_h pacemaking current (HCN channels) - WEAKER than DA
    # Lower conductance → lower baseline firing rate (1-3 Hz vs 4-5 Hz)
    i_h_conductance: float = 0.20  # Boosted for reliable tonic firing
    i_h_reversal: float = 0.75

    # =========================================================================
    # Gap junction parameters (for interneuron coupling)
    # =========================================================================
    # Gap junction coupling (electrical synapses)
    # LC neurons are densely coupled via gap junctions → synchronized bursts
    gap_junction_strength: float = 0.05  # Coupling strength
    gap_junction_neighbor_radius: int = 50  # Neurons within radius are coupled

    # Voltage-dependent gap junction gating
    # Only couple when neurons are depolarized (near threshold)
    # This prevents hyperpolarization spread while allowing burst synchronization
    gap_junction_v_activation: float = 0.7  # Activation threshold (70% of spike threshold)
    gap_junction_v_slope: float = 0.1  # Sigmoid slope (sharpness of gating)

    # =========================================================================
    # SK calcium-activated K+ channels (spike-frequency adaptation)
    # =========================================================================
    sk_conductance: float = 0.025  # Slightly stronger than DA (longer bursts)
    sk_reversal: float = -0.5
    ca_decay: float = 0.93  # Slower calcium decay (longer burst duration)
    ca_influx_per_spike: float = 0.18

    # Uncertainty modulation parameters
    uncertainty_to_current_gain: float = 20.0  # mV equivalent per uncertainty unit
    # High uncertainty → depolarization → burst
    # Low uncertainty → hyperpolarization → pause

    # Disable adaptation from base class (we use SK instead)
    adapt_increment: float = 0.0


class NorepinephrineNeuron(ConductanceLIF):
    """Norepinephrine neuron with gap junction coupling and long bursts.

    Key features:
    1. Autonomous firing at 1-3 Hz (weaker I_h than DA)
    2. Long synchronized bursts (10-15 Hz for 500ms) on high uncertainty
    3. Gap junction coupling → population synchronization
    4. SK adaptation allows sustained bursting
    5. Return to baseline after burst
    """

    def __init__(self, n_neurons: int, config: NorepinephrineNeuronConfig, device: str = "cpu"):
        """Initialize norepinephrine neuron population.

        Args:
            n_neurons: Number of NE neurons (~1,600 in human LC)
            config: Configuration with pacemaking and gap junction parameters
            device: PyTorch device for tensor allocation
        """
        super().__init__(n_neurons, config, device)

        # SK channel state (calcium-activated K+ for adaptation)
        self.ca_concentration = torch.zeros(n_neurons, device=device)
        self.sk_activation = torch.zeros(n_neurons, device=device)

        # Gap junction coupling matrix (electrical synapses)
        # LC neurons within proximity are electrically coupled
        self.gap_junction_matrix = self._create_gap_junction_matrix(n_neurons, config)

        # Initialize with varied phases (prevent artificial synchronization at start)
        self.v_mem = torch.rand(n_neurons, device=device) * config.v_threshold * 0.3

        # Store current uncertainty for conductance calculation
        self._current_uncertainty = 0.0

    def _create_gap_junction_matrix(self, n_neurons: int, config: NorepinephrineNeuronConfig) -> torch.Tensor:
        """Create gap junction coupling matrix.

        LC neurons are spatially organized and coupled via gap junctions.
        This creates sparse local connectivity that enables synchronized bursting.

        Args:
            n_neurons: Number of neurons
            config: Configuration with gap junction parameters

        Returns:
            Sparse coupling matrix [n_neurons, n_neurons]
        """
        device = self.V_soma.device

        # Create sparse local connectivity (neurons coupled to nearby neighbors)
        matrix = torch.zeros(n_neurons, n_neurons, device=device)

        # For each neuron, couple to neighbors within radius
        for i in range(n_neurons):
            # Compute distance to other neurons (1D topology for simplicity)
            distances = torch.abs(torch.arange(n_neurons, device=device) - i)

            # Couple to neurons within radius
            coupled = distances <= config.gap_junction_neighbor_radius
            coupled[i] = False  # Don't self-couple

            # Set coupling strength
            matrix[i, coupled] = config.gap_junction_strength

        return matrix

    @torch.no_grad()
    def forward(
        self,
        g_ampa_input: Optional[ConductanceTensor],
        g_nmda_input: Optional[ConductanceTensor],
        g_gaba_a_input: Optional[ConductanceTensor],
        g_gaba_b_input: Optional[ConductanceTensor],
        uncertainty_drive: float,
    ) -> tuple[torch.Tensor, VoltageTensor]:
        """Update norepinephrine neurons with uncertainty modulation.

        Args:
            g_ampa_input: AMPA (fast excitatory) conductance input [n_neurons]
            g_nmda_input: NMDA (slow excitatory) conductance input [n_neurons] (not used for NE neurons)
            g_gaba_a_input: GABA_A (fast inhibitory) conductance input [n_neurons]
            g_gaba_b_input: GABA_B (slow inhibitory) conductance input [n_neurons] (not used for NE neurons)
            uncertainty_drive: Uncertainty/novelty drive (normalized scalar)
                             +1.0 = high uncertainty (burst)
                             -1.0 = low uncertainty (pause)
                              0.0 = moderate uncertainty (tonic)

        Returns:
            (spikes, membrane): Spike tensor and membrane potentials
        """
        # Store uncertainty for conductance calculation
        self._current_uncertainty = uncertainty_drive

        # === Compute Gap Junction Conductances ===
        # FIX: Voltage-gated coupling prevents pacemaker quenching
        #
        # Problem: When all neurons are hyperpolarized (below threshold), gap junctions
        # pull them DOWN further, preventing I_h from driving pacemaking.
        #
        # Solution: Voltage-dependent gating - only couple when neurons are depolarized.
        # This allows:
        # - Burst synchronization (when neurons are near threshold)
        # - Independent pacemaking (when neurons are subthreshold)
        #
        # Biology: Some gap junctions show rectification (voltage-dependent conductance)
        g_gap_total: Optional[ConductanceTensor] = None
        E_gap_effective: Optional[GapJunctionReversal] = None

        # Voltage-dependent activation: sigmoid function of membrane potential
        # Activation increases as neurons approach threshold
        # - Below 0.7 × threshold: weak/no coupling (independent pacemaking)
        # - Above 0.7 × threshold: strong coupling (burst synchronization)
        v_activation = self.config.gap_junction_v_activation
        v_slope = self.config.gap_junction_v_slope
        gap_activation = torch.sigmoid((self.v_mem - v_activation) / v_slope)

        # Apply voltage-dependent gating to gap junction matrix
        # Both pre and post neurons must be depolarized for strong coupling
        # Use geometric mean of activations: sqrt(act_i × act_j)
        gap_activation_matrix = torch.sqrt(
            gap_activation.unsqueeze(1) * gap_activation.unsqueeze(0)
        )

        # Modulate gap junction strength by voltage activation
        g_gap_matrix = self.gap_junction_matrix * gap_activation_matrix

        # Total gap conductance per neuron (voltage-gated)
        g_gap_total_raw = g_gap_matrix.sum(dim=1)

        # Effective reversal = weighted average of neighbor voltages
        # Only compute if g_gap_total > 0 to avoid division by zero
        mask = g_gap_total_raw > 1e-6
        if mask.any():
            E_gap_effective_raw = torch.zeros(self.n_neurons, device=self.V_soma.device)
            neighbor_weighted_v = torch.matmul(g_gap_matrix, self.v_mem)
            E_gap_effective_raw[mask] = neighbor_weighted_v[mask] / g_gap_total_raw[mask]

            # Only pass gap junctions if non-zero
            g_gap_total = ConductanceTensor(g_gap_total_raw)
            E_gap_effective = GapJunctionReversal(E_gap_effective_raw)

        # Call parent's forward with gap junction conductances
        spikes, _ = super().forward(
            g_ampa_input=g_ampa_input,
            g_nmda_input=g_nmda_input,
            g_gaba_a_input=g_gaba_a_input,
            g_gaba_b_input=g_gaba_b_input,
            g_gap_input=g_gap_total,
            E_gap_reversal=E_gap_effective,
        )

        # === Update Calcium and SK Activation ===
        # Calcium influx on spike
        self.ca_concentration += spikes.float() * self.config.ca_influx_per_spike

        # Calcium decay
        self.ca_concentration *= self.config.ca_decay

        # SK activation (sigmoidal function of calcium)
        # High calcium → high SK → hyperpolarization → limits burst duration
        self.sk_activation = self.ca_concentration / (self.ca_concentration + 0.4)

        # Store spikes for diagnostic access
        self.spikes = spikes

        return spikes, self.V_soma

    def _get_additional_conductances(self) -> list[tuple[torch.Tensor, float]]:
        """Compute I_h and SK conductances.

        Gap junctions are now handled directly in forward() as explicit parameters.

        Returns:
            List of (conductance, reversal) tuples
        """
        # I_h pacemaker (modulated by uncertainty)
        g_ih_modulated = self.config.i_h_conductance * (1.0 + 0.4 * self._current_uncertainty)
        g_ih = torch.full((self.n_neurons,), max(0.0, g_ih_modulated), device=self.V_soma.device)

        # SK adaptation
        g_sk = self.config.sk_conductance * self.sk_activation

        # Return conductances with reversals
        return [
            (g_ih, self.config.i_h_reversal),  # I_h pacemaker
            (g_sk, self.config.sk_reversal),   # SK adaptation
        ]
