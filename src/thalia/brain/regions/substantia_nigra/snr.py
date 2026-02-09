"""Substantia Nigra pars Reticulata (SNr) - Basal Ganglia Output Nucleus.

The SNr is one of the two primary output nuclei of the basal ganglia (along with
GPi - globus pallidus internal segment). It consists of tonically-active GABAergic
neurons that provide inhibitory output to thalamus and VTA.

Biological Background:
======================
**Anatomy:**
- Location: Midbrain, ventral to VTA
- ~10,000-15,000 GABAergic neurons in humans
- Adjacent to dopamine-producing SNc (substantia nigra pars compacta)

**Function:**
- Output gate of basal ganglia motor/cognitive loop
- Tonic inhibition of thalamus (suppresses unwanted actions)
- Tonic inhibition of VTA (regulates dopamine bursting)
- Disinhibition mechanism: Striatum D1 → SNr inhibition → thalamus release

**Firing Pattern:**
- Tonic baseline: 50-70 Hz (high spontaneous rate)
- Pauses: During action selection (D1 pathway activation)
- Bursts: During action suppression (D2/indirect pathway)

**Inputs:**
- Striatum D1 (direct pathway): Inhibitory (GABAergic)
- Striatum D2 via GPe/STN (indirect pathway): Net excitatory
- STN (subthalamic nucleus): Excitatory (glutamatergic)
- GPe (globus pallidus external): Inhibitory (GABAergic)

**Outputs:**
- Thalamus (motor VA/VL nuclei): Inhibitory (motor gating)
- Superior colliculus: Inhibitory (saccade gating)
- VTA: Inhibitory (dopamine regulation)
- Pedunculopontine nucleus: Inhibitory (locomotion)

**Computational Role in Basal Ganglia:**
===========================================
SNr firing rate encodes **negative value** (inverse of state value):
- High SNr rate → action suppression (NoGo state)
- Low SNr rate → action release (Go state)

In TD learning framework:
- SNr activity ∝ 1 - V(s)  (inverse state value)
- VTA uses SNr as V(s) estimate for RPE computation:
  RPE = r + γ·V(s') - V(s)
      ≈ r - (1 - SNr_rate)

This creates a closed-loop system:
```
VTA (dopamine) → Striatum (learning) → SNr (value) → VTA (RPE feedback)
```

**Simplified Implementation (Phase 1):**
========================================
- D1 pathway: Striatum D1 MSNs directly inhibit SNr
- D2 pathway: Striatum D2 MSNs excite SNr (via GPe/STN, simplified as direct)
- Future: Add explicit GPe, STN, GPi for full basal ganglia anatomy

Author: Thalia Project
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from thalia.brain.configs import SNrConfig
from thalia.components.neurons import ConductanceLIF, ConductanceLIFConfig
from thalia.typing import PortName, RegionLayerSizes, RegionSpikesDict

from ..neural_region import NeuralRegion
from ..region_registry import register_region


@register_region(
    "snr",
    aliases=["substantia_nigra", "snr", "substantia_nigra_reticulata"],
    description="Substantia nigra pars reticulata - basal ganglia output nucleus",
    version="1.0",
    author="Thalia Project",
    config_class=SNrConfig,
)
class SubstantiaNigra(NeuralRegion[SNrConfig]):
    """Substantia Nigra pars Reticulata - Basal Ganglia Output Nucleus.

    Tonically-active GABAergic neurons that:
    1. Gate thalamic output (motor/cognitive actions)
    2. Provide value feedback to VTA for TD learning
    3. Integrate striatal D1 (inhibitory) and D2 (excitatory) pathways

    Output Ports:
    -------------
    - "vta_feedback": Inhibitory projection to VTA (value signal)
    - "thalamus_output": Inhibitory projection to thalamus (motor gating)

    Input Ports (from connections):
    --------------------------------
    - "striatum_d1": Direct pathway (Go) - inhibits SNr
    - "striatum_d2": Indirect pathway (NoGo) - excites SNr via GPe/STN

    Computational Properties:
    -------------------------
    - High firing rate (50-70 Hz) = low state value (action suppression)
    - Low firing rate (<30 Hz) = high state value (action release)
    - Value estimate = inverse function of firing rate
    """

    OUTPUT_PORTS: Dict[PortName, str] = {
        "vta_feedback": "n_neurons",
        "thalamus_output": "n_neurons",
    }

    def __init__(self, config: SNrConfig, region_layer_sizes: RegionLayerSizes):
        super().__init__(config, region_layer_sizes)

        # GABAergic output neurons (tonically active)
        self.neurons = self._create_snr_neurons()

        # Tonic drive for baseline firing
        self.baseline_drive = torch.full(
            (config.n_neurons,), config.baseline_drive, device=self.device
        )

        # Synaptic weights will be managed by parent class's synaptic_weights dict
        # Initialized when connections are established by BrainBuilder

        # Track firing rate for value estimation
        self._firing_rate_history: list[float] = []

    def _create_snr_neurons(self) -> ConductanceLIF:
        """Create tonically-active GABAergic output neurons.

        These are specialized for high-frequency tonic firing with fast dynamics.
        """
        neuron_config = ConductanceLIFConfig(
            tau_mem=self.config.tau_mem,
            v_threshold=self.config.v_threshold,
            v_reset=0.0,
            v_rest=0.0,
            tau_ref=self.config.tau_ref,
            g_L=0.125,  # High leak for fast dynamics
            E_L=0.0,
            E_E=3.0,
            E_I=-0.5,
            tau_E=3.0,  # Fast excitatory
            tau_I=6.0,  # Fast inhibitory
            noise_std=0.02,  # Small noise for biological variability
            device=self.config.device,
        )

        return ConductanceLIF(
            n_neurons=self.config.n_neurons, config=neuron_config, device=self.device
        )

    def _forward_internal(self, inputs: RegionSpikesDict) -> None:
        """Update SNr neurons based on striatal input.

        Args:
            inputs: Dictionary of input spike tensors with keys:
                - "striatum_d1": D1 pathway spikes (inhibitory)
                - "striatum_d2": D2 pathway spikes (excitatory via GPe/STN)
        """
        # Get striatal inputs (via connections established by BrainBuilder)
        d1_spikes = inputs.get("striatum_d1")
        d2_spikes = inputs.get("striatum_d2")

        # Initialize total current with baseline tonic drive
        total_current = self.baseline_drive.clone()

        # D1 pathway (direct, Go): Inhibits SNr
        if d1_spikes is not None and "striatum_d1" in self.synaptic_weights:
            weights_d1 = self.synaptic_weights["striatum_d1"]  # [n_snr, n_d1]
            i_d1 = torch.matmul(weights_d1, d1_spikes.float())  # [n_snr]

            # Inhibitory current (reduces SNr activity)
            total_current -= i_d1 * self.config.d1_inhibition_weight

        # D2 pathway (indirect, NoGo): Excites SNr via GPe→STN→SNr
        # Simplified: D2 spikes directly increase SNr activity
        if d2_spikes is not None and "striatum_d2" in self.synaptic_weights:
            weights_d2 = self.synaptic_weights["striatum_d2"]  # [n_snr, n_d2]
            i_d2 = torch.matmul(weights_d2, d2_spikes.float())  # [n_snr]

            # Excitatory current (increases SNr activity)
            total_current += i_d2 * self.config.d2_excitation_weight

        # Update neurons
        self.neurons.forward(total_current)

        # SNr output spikes (same to both VTA and thalamus for now)
        output_spikes = self.neurons.spikes

        # Track firing rate for value estimation
        self._update_firing_rate()

        # Set output ports
        self._set_port_output("vta_feedback", output_spikes)
        self._set_port_output("thalamus_output", output_spikes)

    def get_value_estimate(self) -> float:
        """Compute state value estimate from SNr firing rate.

        Biology: SNr firing rate inversely encodes state value
        - High SNr rate (60+ Hz) = low value (action suppression)
        - Low SNr rate (<30 Hz) = high value (action release)
        - Baseline (50-60 Hz) = neutral value (~0.5)

        Returns:
            Value estimate in range [0, 1]
        """
        # Get current firing rate
        spike_rate = self.neurons.spikes.float().mean().item()

        # Convert to Hz (spikes per second)
        firing_rate_hz = spike_rate * (1000.0 / self.config.dt_ms)

        # Map firing rate to value (inverse relationship)
        # Baseline: 60 Hz → 0.5 value
        # High: 80 Hz → 0.0 value (low state value)
        # Low: 20 Hz → 1.0 value (high state value)

        baseline_hz = 60.0
        value = 1.0 - (firing_rate_hz / (2 * baseline_hz))

        # Clamp to [0, 1]
        value = max(0.0, min(1.0, value))

        return value

    def get_firing_rate_hz(self) -> float:
        """Get current population firing rate in Hz.

        Returns:
            Mean firing rate across population
        """
        spike_rate = self.neurons.spikes.float().mean().item()
        return spike_rate * (1000.0 / self.config.dt_ms)

    def _update_firing_rate(self):
        """Track firing rate history for monitoring."""
        firing_rate = self.get_firing_rate_hz()
        self._firing_rate_history.append(firing_rate)

        # Keep last 1000 timesteps only
        if len(self._firing_rate_history) > 1000:
            self._firing_rate_history.pop(0)

    def get_mean_firing_rate(self) -> float:
        """Get mean firing rate over recent history.

        Returns:
            Mean firing rate in Hz, or 0.0 if no history
        """
        if not self._firing_rate_history:
            return 0.0
        return sum(self._firing_rate_history) / len(self._firing_rate_history)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information for this region."""
        return {
            "firing_rate_hz": self.get_firing_rate_hz(),
            "value_estimate": self.get_value_estimate(),
            "mean_firing_rate_hz": self.get_mean_firing_rate(),
            "mean_membrane_potential": self.neurons.v_mem.mean().item(),
        }
