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
"""

from __future__ import annotations

from typing import Dict

import torch

from thalia.brain.configs import SNrConfig
from thalia.components.neurons import ConductanceLIF, ConductanceLIFConfig
from thalia.typing import PopulationName, PopulationSizes, RegionSpikesDict
from thalia.units import ConductanceTensor

from ..neural_region import NeuralRegion
from ..region_registry import register_region


@register_region(
    "snr",
    aliases=["substantia_nigra", "substantia_nigra_reticulata"],
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

    Input Populations:
    ------------------
    - "striatum_d1": Direct pathway (Go) - inhibits SNr
    - "striatum_d2": Indirect pathway (NoGo) - excites SNr via GPe/STN

    Output Populations:
    -------------------
    - "vta_feedback": Inhibitory projection to VTA (value signal)
    - "thalamus_output": Inhibitory projection to thalamus (motor gating)

    Computational Properties:
    -------------------------
    - High firing rate (50-70 Hz) = low state value (action suppression)
    - Low firing rate (<30 Hz) = high state value (action release)
    - Value estimate = inverse function of firing rate
    """

    OUTPUT_POPULATIONS: Dict[PopulationName, str] = {
        "vta_feedback": "n_neurons",
        "thalamus_output": "n_neurons",
    }

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, config: SNrConfig, population_sizes: PopulationSizes):
        super().__init__(config, population_sizes)

        # Store input layer sizes as attributes for connection routing
        self.d1_input_size = population_sizes.get("d1_input", 0)
        self.d2_input_size = population_sizes.get("d2_input", 0)

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

        self.__post_init__()

    def _create_snr_neurons(self) -> ConductanceLIF:
        """Create tonically-active GABAergic output neurons.

        These are specialized for high-frequency tonic firing with fast dynamics.

        Tonic firing mechanism:
        - Subthreshold baseline drive (~0.02 normalized conductance)
        - Moderate noise pushes neurons over threshold stochastically
        - Membrane tau (15ms) provides realistic integration timescale
        - Refractory period (2.0ms) limits max frequency to ~500 Hz (biological ceiling)
        - Actual firing rate: 50-70 Hz baseline (biologically realistic)

        This creates biologically realistic irregular tonic firing that can be
        modulated by striatal D1 (inhibition) and D2 (excitation) inputs.
        """
        neuron_config = ConductanceLIFConfig(
            tau_mem=self.config.tau_mem,  # 15ms - realistic SNr membrane tau
            v_threshold=self.config.v_threshold,  # 1.0 - standard threshold
            v_reset=0.0,
            v_rest=0.0,
            tau_ref=self.config.tau_ref,  # 2.0ms - biological refractory period
            g_L=0.10,  # Moderate leak (tau_m = C_m/g_L = 1.0/0.10 = 10ms effective)
            E_L=0.0,
            E_E=3.0,
            E_I=-0.5,
            tau_E=5.0,  # AMPA-like kinetics
            tau_I=10.0,  # GABA_A-like kinetics
            noise_std=0.05,  # Moderate noise for stochastic tonic firing
            device=self.config.device,
        )

        self.n_neurons = self.config.n_neurons  # Store for test access
        return ConductanceLIF(
            n_neurons=self.config.n_neurons, config=neuron_config, device=self.device
        )

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def forward(self, region_inputs: RegionSpikesDict) -> RegionSpikesDict:
        """Update SNr neurons based on striatal input.

        Args:
            region_inputs: Dictionary of input spike tensors with keys:
                - "striatum_d1": D1 pathway spikes (inhibitory)
                - "striatum_d2": D2 pathway spikes (excitatory via GPe/STN)
        """
        self._pre_forward(region_inputs)

        # Get striatal inputs (via connections established by BrainBuilder)
        d1_spikes = region_inputs.get("striatum_d1")
        d2_spikes = region_inputs.get("striatum_d2")

        # Initialize conductances
        # CRITICAL: Weights already represent per-spike conductances (normalized by g_L)
        # DO NOT apply additional gain factors - they create current-like quantities!
        g_exc = torch.zeros(self.config.n_neurons, device=self.device)
        g_inh = torch.zeros(self.config.n_neurons, device=self.device)

        # D1 pathway (direct, Go): Inhibits SNr
        if d1_spikes is not None and "striatum_d1" in self.synaptic_weights:
            weights_d1 = self.synaptic_weights["striatum_d1"]  # [n_snr, n_d1]
            # Weights represent per-spike inhibitory conductances directly
            g_inh += torch.matmul(weights_d1, d1_spikes.float())

        # D2 pathway (indirect, NoGo): Excites SNr via GPe→STN→SNr
        # Simplified: D2 spikes directly increase SNr activity
        if d2_spikes is not None and "striatum_d2" in self.synaptic_weights:
            weights_d2 = self.synaptic_weights["striatum_d2"]  # [n_snr, n_d2]
            # Weights represent per-spike excitatory conductances directly
            g_exc += torch.matmul(weights_d2, d2_spikes.float())

        # Add baseline drive as excitatory conductance to maintain tonic firing (~50-70Hz)
        g_exc_total = ConductanceTensor(g_exc + self.baseline_drive)
        g_inh_total = ConductanceTensor(g_inh)

        # Update neurons using conductance-based dynamics
        output_spikes, new_v = self.neurons.forward(g_exc_total, g_inh_total)

        # Store current output for diagnostics (as tuple)
        self._current_output = (output_spikes, new_v)

        # Track firing rate for value estimation
        self._update_firing_rate()

        region_outputs: RegionSpikesDict = {
            "vta_feedback": output_spikes,
            "thalamus_output": output_spikes,
        }

        return self._post_forward(region_outputs)

    # =========================================================================
    # TEMPORAL PARAMETER MANAGEMENT
    # =========================================================================

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters when brain timestep changes."""
        super().update_temporal_parameters(dt_ms)
        self.neurons.update_temporal_parameters(dt_ms)

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def get_value_estimate(self) -> float:
        """Compute state value estimate from SNr firing rate.

        Biology: SNr firing rate inversely encodes state value
        - High SNr rate (60+ Hz) = low value (action suppression)
        - Low SNr rate (<30 Hz) = high value (action release)
        - Baseline (50-60 Hz) = neutral value (~0.5)

        Returns:
            Value estimate in range [0, 1]
        """
        # Get current firing rate from stored output
        if hasattr(self, '_current_output'):
            spikes = self._current_output[0] if isinstance(self._current_output, tuple) else self._current_output
            spike_rate = spikes.float().mean().item()
        else:
            return 0.5  # Default neutral value

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
        if hasattr(self, '_current_output'):
            spikes = self._current_output[0] if isinstance(self._current_output, tuple) else self._current_output
            spike_rate = spikes.float().mean().item()
            return spike_rate * (1000.0 / self.config.dt_ms)
        return 60.0  # Default baseline

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
