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

References:
- Aston-Jones & Cohen (2005): LC-NE system and adaptive gain theory
- Berridge & Waterhouse (2003): LC system function and dysfunction
- Sara (2009): LC function in attention and memory

Author: Thalia Project
Date: February 2026
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from thalia.components.neurons.neuron import ConductanceLIF, ConductanceLIFConfig


@dataclass
class NorepinephrineNeuronConfig(ConductanceLIFConfig):
    """Configuration for norepinephrine neurons with pacemaking and gap junction coupling.

    Extends base LIF with parameters for:
    - Weaker I_h pacemaking (lower baseline than DA)
    - Gap junction coupling (electrical synapses)
    - SK calcium-activated K+ channels (adaptation)
    - Uncertainty-driven burst modulation

    Biological parameters based on:
    - Aston-Jones & Cohen (2005): LC-NE system electrophysiology
    - Berridge & Waterhouse (2003): LC neuron properties
    - Sara & Bouret (2012): Phasic vs tonic NE modes
    """

    # Membrane properties (similar to DA neurons)
    tau_mem: float = 18.0  # Slightly slower than DA neurons
    v_rest: float = 0.0
    v_reset: float = -0.12  # Slightly deeper hyperpolarization
    v_threshold: float = 1.0
    tau_ref: float = 2.5  # Slightly longer refractory period

    # Leak conductance (tuned for lower tonic firing)
    g_L: float = 0.056  # tau_m = C_m/g_L = 18ms

    # Reversal potentials
    E_L: float = 0.0
    E_E: float = 3.0
    E_I: float = -0.5

    # Synaptic time constants
    tau_E: float = 5.0
    tau_I: float = 10.0

    # I_h pacemaking current (HCN channels) - WEAKER than DA
    # Lower conductance → lower baseline firing rate (1-3 Hz vs 4-5 Hz)
    i_h_conductance: float = 0.015  # Half of DA neurons
    i_h_reversal: float = 0.7

    # Gap junction coupling (electrical synapses)
    # LC neurons are densely coupled via gap junctions → synchronized bursts
    gap_junction_strength: float = 0.05  # Voltage coupling coefficient
    gap_junction_neighbor_radius: int = 50  # Neurons within radius are coupled

    # SK calcium-activated K+ channels (spike-frequency adaptation)
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

    # Noise for biological realism
    noise_std: float = 0.015  # Slightly higher than DA


class NorepinephrineNeuron(ConductanceLIF):
    """Norepinephrine neuron with gap junction coupling and long bursts.

    Key features:
    1. Autonomous firing at 1-3 Hz (weaker I_h than DA)
    2. Long synchronized bursts (10-15 Hz for 500ms) on high uncertainty
    3. Gap junction coupling → population synchronization
    4. SK adaptation allows sustained bursting
    5. Return to baseline after burst

    Usage:
        ```python
        ne_neurons = NorepinephrineNeuron(
            n_neurons=1600,
            config=NorepinephrineNeuronConfig(),
            device=torch.device("cpu")
        )

        # Tonic firing (low arousal)
        ne_neurons.forward(i_synaptic=0.0, uncertainty_drive=0.0)

        # Burst on high uncertainty
        ne_neurons.forward(i_synaptic=0.0, uncertainty_drive=1.0)
        ```
    """

    def __init__(
        self,
        n_neurons: int,
        config: NorepinephrineNeuronConfig,
        device: torch.device,
    ):
        """Initialize norepinephrine neuron population.

        Args:
            n_neurons: Number of NE neurons (~1,600 in human LC)
            config: Configuration with pacemaking and gap junction parameters
            device: PyTorch device for tensor allocation
        """
        super().__init__(n_neurons, config, device)

        # Store specialized config
        self.ne_config = config

        # SK channel state (calcium-activated K+ for adaptation)
        self.ca_concentration = torch.zeros(n_neurons, device=device)
        self.sk_activation = torch.zeros(n_neurons, device=device)

        # Gap junction coupling matrix (electrical synapses)
        # LC neurons within proximity are electrically coupled
        self.gap_junction_matrix = self._create_gap_junction_matrix(n_neurons, config)

        # Initialize with varied phases (prevent artificial synchronization at start)
        self.v_mem = torch.rand(n_neurons, device=device) * config.v_threshold * 0.3

    def _create_gap_junction_matrix(
        self, n_neurons: int, config: NorepinephrineNeuronConfig
    ) -> torch.Tensor:
        """Create gap junction coupling matrix.

        LC neurons are spatially organized and coupled via gap junctions.
        This creates sparse local connectivity that enables synchronized bursting.

        Args:
            n_neurons: Number of neurons
            config: Configuration with gap junction parameters

        Returns:
            Sparse coupling matrix [n_neurons, n_neurons]
        """
        # Create sparse local connectivity (neurons coupled to nearby neighbors)
        matrix = torch.zeros(n_neurons, n_neurons, device=self.device)

        # For each neuron, couple to neighbors within radius
        for i in range(n_neurons):
            # Compute distance to other neurons (1D topology for simplicity)
            distances = torch.abs(torch.arange(n_neurons, device=self.device) - i)

            # Couple to neurons within radius
            coupled = distances <= config.gap_junction_neighbor_radius
            coupled[i] = False  # Don't self-couple

            # Set coupling strength
            matrix[i, coupled] = config.gap_junction_strength

        return matrix

    def forward(
        self,
        i_synaptic: torch.Tensor | float = 0.0,
        uncertainty_drive: float = 0.0,
    ) -> torch.Tensor:
        """Update norepinephrine neurons with uncertainty modulation.

        Args:
            i_synaptic: Synaptic input current [n_neurons] or scalar
                       (LC receives input from PFC, hippocampus for uncertainty signal)
            uncertainty_drive: Uncertainty/novelty drive (normalized scalar)
                             +1.0 = high uncertainty (burst)
                             -1.0 = low uncertainty (pause)
                              0.0 = moderate uncertainty (tonic)

        Returns:
            Spike tensor [n_neurons], dtype=bool
        """
        # Convert scalar synaptic input to tensor if needed
        if isinstance(i_synaptic, (int, float)):
            i_synaptic = torch.full(
                (self.n_neurons,), float(i_synaptic), device=self.device
            )

        # === Intrinsic Currents ===

        # I_h pacemaking current (weaker than DA → lower baseline)
        i_pacemaker = self.ne_config.i_h_conductance * (
            self.ne_config.i_h_reversal - self.v_mem
        )

        # SK adaptation current (allows sustained bursting)
        i_adaptation = (
            -self.ne_config.sk_conductance
            * self.sk_activation
            * (self.v_mem - self.ne_config.sk_reversal)
        )

        # === Gap Junction Coupling ===
        # Electrical coupling to nearby neurons (synchronized bursts)
        # I_gap = g_gap * (V_neighbor - V_self)
        i_gap_junction = torch.zeros(self.n_neurons, device=self.device)
        if self.gap_junction_matrix is not None:
            # Compute voltage difference from all coupled neighbors
            # [n, n] @ [n, 1] → [n, 1]
            v_neighbors = torch.matmul(
                self.gap_junction_matrix, self.v_mem.unsqueeze(1)
            ).squeeze(1)
            i_gap_junction = v_neighbors - self.v_mem * self.gap_junction_matrix.sum(
                dim=1
            )

        # === Uncertainty Modulation ===
        # Convert uncertainty to current drive
        # High uncertainty → positive current → depolarization → burst
        # Low uncertainty → negative current → hyperpolarization → pause
        i_uncertainty = uncertainty_drive * self.ne_config.uncertainty_to_current_gain

        # === Total Current ===
        i_total = i_synaptic + i_pacemaker + i_adaptation + i_gap_junction + i_uncertainty

        # Add noise for biological realism
        if self.ne_config.noise_std > 0:
            noise = torch.randn_like(i_total) * self.ne_config.noise_std
            i_total = i_total + noise

        # Call parent's forward to update membrane potential and generate spikes
        spikes, _ = super().forward(i_total)

        # === Update Calcium and SK Activation ===
        # Calcium influx on spike
        self.ca_concentration += spikes.float() * self.ne_config.ca_influx_per_spike

        # Calcium decay
        self.ca_concentration *= self.ne_config.ca_decay

        # SK activation (sigmoidal function of calcium)
        # High calcium → high SK → hyperpolarization → limits burst duration
        self.sk_activation = self.ca_concentration / (self.ca_concentration + 0.4)

        # Store spikes for diagnostic access
        self.spikes = spikes

        return spikes

    def get_firing_rate_hz(self, window_ms: int = 100) -> float:
        """Get average firing rate over recent history.

        Args:
            window_ms: Time window for rate estimation (ms, not used in single timestep)

        Returns:
            Average firing rate in Hz
        """
        # Check if spikes have been computed
        if not hasattr(self, "spikes") or self.spikes is None:
            return 0.0

        # Single timestep rate
        spike_rate = self.spikes.float().mean().item()

        # Convert to Hz (spikes per second)
        firing_rate_hz = spike_rate * (1000.0 / self.ne_config.dt_ms)

        return firing_rate_hz

    def reset_state(self):
        """Reset neuron state to baseline."""
        super().reset_state()
        self.ca_concentration.zero_()
        self.sk_activation.zero_()
        # Re-randomize phases
        self.v_mem = (
            torch.rand(self.n_neurons, device=self.device)
            * self.ne_config.v_threshold
            * 0.3
        )
