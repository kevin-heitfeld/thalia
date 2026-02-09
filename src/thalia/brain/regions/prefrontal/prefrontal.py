"""
Prefrontal Cortex - Gated Working Memory and Executive Control.

The prefrontal cortex (PFC) specializes in cognitive control and flexible behavior:
- **Working memory maintenance**: Actively maintain information over delays
- **Rule learning**: Learn context-dependent stimulus-response mappings
- **Executive control**: Top-down attention and behavioral inhibition
- **Goal-directed behavior**: Plan and execute multi-step goal hierarchies

**Key Features**:
=================
1. **GATED WORKING MEMORY**:
   - Active maintenance against decay via recurrent excitation
   - Dopamine gates what enters/updates working memory
   - Similar to LSTM/GRU gating in deep learning, but biological
   - Persistent activity emerges from network dynamics

2. **DOPAMINE GATING MECHANISM**:
   - DA burst (>threshold) → "update gate open" → new info enters WM
   - DA baseline → "maintain" → protect current WM contents
   - DA dip → "clear" → allow WM to decay
   - Gates both learning AND maintenance

3. **CONTEXT-DEPENDENT LEARNING**:
   - Rule neurons represent abstract task rules
   - Same input → different outputs based on context/rule
   - Enables flexible behavior switching
   - Supports cognitive flexibility and set-shifting

4. **SLOW INTEGRATION**:
   - Longer time constants than sensory cortex (τ ~500ms)
   - Integrates information over longer timescales
   - Supports temporal abstraction and planning

Biological Basis:
=================
- Layer 2/3 recurrent circuits for WM maintenance
- D1/D2 receptors modulate gain and gating
- Strong connections with striatum (for action selection)
- Connections with hippocampus (for episodic retrieval)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from thalia.brain.configs import PrefrontalConfig
from thalia.components import (
    ConductanceLIF,
    ConductanceLIFConfig,
    ShortTermPlasticity,
    STPConfig,
    STPType,
    WeightInitializer,
)
from thalia.constants import DEFAULT_DT_MS
from thalia.diagnostics import compute_plasticity_metrics
from thalia.errors import ConfigurationError
from thalia.learning import (
    LearningStrategyRegistry,
    STDPConfig,
    UnifiedHomeostasis,
    UnifiedHomeostasisConfig,
)
from thalia.typing import (
    LayerName,
    RegionLayerSizes,
    RegionSpikesDict,
    SpikesSourceKey,
)
from thalia.utils import compute_ne_gain

from .goal_emergence import EmergentGoalSystem

from ..neural_region import NeuralRegion
from ..region_registry import register_region


def sample_heterogeneous_wm_neurons(
    n_neurons: int,
    stability_cv: float = 0.3,
    tau_mem_min: float = 100.0,
    tau_mem_max: float = 500.0,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Sample heterogeneous working memory neuron properties.

    Creates a distribution of neurons with varying maintenance capabilities:
    - Stable neurons: Strong recurrence, long time constants (~500ms)
    - Flexible neurons: Weak recurrence, short time constants (~100ms)

    This heterogeneity enables:
    - Stable neurons maintain context/goals over long delays
    - Flexible neurons enable rapid updating for new information
    - Mixed selectivity for distributed representations

    Biological motivation:
    - Real PFC neurons show 2-10× variability in maintenance properties
    - Heterogeneity provides robustness and mixed selectivity
    - Enables both persistent representations and flexible updating

    References:
    - Rigotti et al. (2013): Mixed selectivity in prefrontal cortex
    - Murray et al. (2017): Stable population coding for working memory
    - Wasmuht et al. (2018): Intrinsic neuronal dynamics in PFC

    Args:
        n_neurons: Number of neurons to sample
        stability_cv: Coefficient of variation for recurrent strength (default 0.3)
        tau_mem_min: Minimum membrane time constant in ms (default 100.0)
        tau_mem_max: Maximum membrane time constant in ms (default 500.0)
        device: Device for tensors ('cpu' or 'cuda')
        seed: Random seed for reproducibility (optional)

    Returns:
        Dictionary with:
        - recurrent_strength: [n_neurons] tensor of recurrent weights (0.2-1.0 range)
        - tau_mem: [n_neurons] tensor of membrane time constants (100-500ms)
        - neuron_type: [n_neurons] tensor of 0 (flexible) or 1 (stable) labels
    """
    if seed is not None:
        torch.manual_seed(seed)

    device_obj = torch.device(device)

    # Sample recurrent strength from lognormal distribution
    # Mean = 0.6 (moderate recurrence), CV = stability_cv
    # This creates a distribution with:
    # - Lower tail: Flexible neurons (weak recurrence ~0.2-0.4)
    # - Upper tail: Stable neurons (strong recurrence ~0.8-1.0)
    mean_recurrent = 0.6
    std_recurrent = mean_recurrent * stability_cv

    # Lognormal parameters: log_mean, log_std
    # For lognormal: mean = exp(μ + σ²/2), var = [exp(σ²) - 1] * exp(2μ + σ²)
    # Solve for μ, σ given desired mean and std
    log_var = torch.log(torch.tensor(1.0 + (std_recurrent / mean_recurrent) ** 2))
    log_std = torch.sqrt(log_var)
    log_mean = torch.log(torch.tensor(mean_recurrent)) - log_var / 2

    # Sample from lognormal
    recurrent_strength = torch.distributions.LogNormal(log_mean, log_std).sample((n_neurons,))

    # Clamp to reasonable range [0.2, 1.0]
    # 0.2 = minimum for any persistent activity
    # 1.0 = maximum stability (approaches attractor)
    recurrent_strength = torch.clamp(recurrent_strength, 0.2, 1.0)

    # Tau_mem scales with recurrent strength
    # Stable neurons (high recurrence) have longer time constants
    # Linear mapping: recurrent 0.2→100ms, recurrent 1.0→500ms
    tau_mem = tau_mem_min + (tau_mem_max - tau_mem_min) * (recurrent_strength - 0.2) / 0.8

    # Classify neurons as flexible (0) or stable (1)
    # Threshold at median: lower half = flexible, upper half = stable
    median_strength = torch.median(recurrent_strength)
    neuron_type = (recurrent_strength > median_strength).long()

    # Move to device
    recurrent_strength = recurrent_strength.to(device_obj)
    tau_mem = tau_mem.to(device_obj)
    neuron_type = neuron_type.to(device_obj)

    return {
        "recurrent_strength": recurrent_strength,
        "tau_mem": tau_mem,
        "neuron_type": neuron_type,
    }


class DopamineGatingSystem:
    """Dopamine-based gating for working memory updates.

    Unlike striatal dopamine (which determines LTP vs LTD direction),
    prefrontal dopamine gates what information enters working memory:
    - High DA → gate open → update WM with new input
    - Low DA → gate closed → maintain current WM
    """

    def __init__(
        self,
        n_neurons: int,
        tau_ms: float = 100.0,
        baseline: float = 0.2,
        threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.n_neurons = n_neurons
        self.tau_ms = tau_ms
        self.baseline = baseline
        self.threshold = threshold
        self.device = torch.device(device)

        self.level = baseline  # Current DA level

    def update(self, signal: float, dt_ms: float = DEFAULT_DT_MS) -> float:
        """Update dopamine level with new signal.

        Args:
            signal: External dopamine signal (-1 to 1)
            dt_ms: Timestep in ms

        Returns:
            Current dopamine level
        """
        # Decay toward baseline
        decay = torch.exp(torch.tensor(-dt_ms / self.tau_ms)).item()
        self.level = self.baseline + (self.level - self.baseline) * decay

        # Add signal
        self.level += signal

        # Clamp to valid range
        self.level = max(0.0, min(1.0, self.level))

        return self.level

    def get_gate(self) -> float:
        """Get current gating value (0-1).

        Returns smooth gate value based on dopamine level.
        """
        # Sigmoid around threshold
        gate_value: float = 1.0 / (
            1.0 + torch.exp(torch.tensor(-10 * (self.level - self.threshold))).item()
        )
        return gate_value


@register_region(
    "prefrontal",
    aliases=["pfc", "prefrontal_cortex"],
    description="Working memory and executive control with dopamine-gated updates and rule learning",
    version="1.0",
    author="Thalia Project",
    config_class=PrefrontalConfig,
)
class Prefrontal(NeuralRegion[PrefrontalConfig]):
    """Prefrontal cortex with dopamine-gated working memory.

    Implements:
    - Working memory maintenance via recurrent connections
    - Dopamine gating of updates (similar to LSTM gates)
    - Rule learning and context-dependent behavior
    - Slow integration for temporal abstraction
    """

    # Declarative output ports (auto-registered by base class)
    OUTPUT_PORTS = {
        "executive": "executive_size",
    }

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, config: PrefrontalConfig, region_layer_sizes: RegionLayerSizes):
        """Initialize prefrontal cortex."""
        super().__init__(config=config, region_layer_sizes=region_layer_sizes)

        # =====================================================================
        # EXTRACT LAYER SIZES
        # =====================================================================
        self.executive_size = self.region_layer_sizes["executive_size"]

        # =====================================================================
        # INITIALIZE STATE FIELDS
        # =====================================================================
        # PFC-specific state fields
        self.working_memory: Optional[torch.Tensor] = None
        self.update_gate: Optional[torch.Tensor] = None
        self.active_rule: Optional[torch.Tensor] = None

        # =====================================================================
        # HETEROGENEOUS WORKING MEMORY NEURONS
        # =====================================================================
        # Sample heterogeneous recurrent strengths and time constants
        # mimicking biological diversity in WM stability across neurons
        wm_properties = sample_heterogeneous_wm_neurons(
            n_neurons=self.executive_size,
            stability_cv=config.stability_cv,
            tau_mem_min=config.tau_mem_min,
            tau_mem_max=config.tau_mem_max,
            device=self.device,
            seed=None,  # Use random seed for variability
        )
        self._recurrent_strength: Optional[torch.Tensor] = wm_properties["recurrent_strength"]
        self._tau_mem_heterogeneous: Optional[torch.Tensor] = wm_properties["tau_mem"]
        self._neuron_type: Optional[torch.Tensor] = wm_properties["neuron_type"]  # 0=flexible, 1=stable

        # =====================================================================
        # D1/D2 RECEPTOR SUBTYPES
        # =====================================================================
        self._d1_neurons: Optional[torch.Tensor] = None
        self._d2_neurons: Optional[torch.Tensor] = None

        # Split neurons into D1-dominant (excitatory DA response) and
        # D2-dominant (inhibitory DA response) populations
        if config.use_d1_d2_subtypes:
            n_d1 = int(self.executive_size * config.d1_fraction)
            self._d1_neurons = torch.arange(n_d1, device=self.device)
            self._d2_neurons = torch.arange(n_d1, self.executive_size, device=self.device)

        # Override neurons to add STP (NeuralRegion creates basic neurons)
        self.neurons = self._create_neurons()

        # Recurrent weights for WM maintenance
        self.rec_weights = nn.Parameter(
            WeightInitializer.gaussian(
                n_output=self.executive_size,
                n_input=self.executive_size,
                mean=0.0,
                std=0.1,
                device=self.device,
            ),
            requires_grad=False,
        )
        # Initialize with self-excitation (heterogeneous)
        # Scale diagonal by heterogeneous recurrent strengths
        diag_strength = torch.diag(self._recurrent_strength)
        self.rec_weights.data += diag_strength

        # Lateral inhibition weights
        self.inhib_weights = nn.Parameter(
            torch.ones(self.executive_size, self.executive_size, device=self.device)
            * config.recurrent_inhibition
        )
        self.inhib_weights.data.fill_diagonal_(0.0)

        # Dopamine gating system
        self.dopamine_system = DopamineGatingSystem(
            n_neurons=self.executive_size,
            tau_ms=config.dopamine_tau_ms,
            baseline=config.dopamine_baseline,
            threshold=config.gate_threshold,
            device=config.device,
        )

        # Initialize learning strategy (STDP with dopamine gating)
        self.learning_strategy = LearningStrategyRegistry.create(
            "stdp",
            STDPConfig(
                learning_rate=config.learning_rate,
                a_plus=config.a_plus,
                a_minus=config.a_minus,
                tau_plus=config.tau_plus_ms,
                tau_minus=config.tau_minus_ms,
                w_min=config.w_min,
                w_max=config.w_max,
            ),
        )

        # Homeostasis for synaptic scaling
        # Created lazily in add_input_source when n_input is known
        self.homeostasis: Optional[UnifiedHomeostasis] = None
        self._homeostasis_config = config  # Store for lazy init

        # =====================================================================
        # HOMEOSTATIC INTRINSIC PLASTICITY
        # =====================================================================
        # Track firing rates and adaptive gains (Turrigiano 2008)
        self._target_rate = config.target_firing_rate
        self._gain_lr = config.gain_learning_rate
        self._firing_rate_alpha = config.dt_ms / config.gain_tau_ms

        # Firing rate tracker (exponential moving average)
        self.register_buffer("firing_rate", torch.zeros(self.executive_size, device=self.device))

        # Adaptive gains (learnable parameters)
        self.gain = nn.Parameter(torch.ones(self.executive_size, device=self.device, requires_grad=False))

        # Store gain bounds
        self._baseline_noise = config.baseline_noise_current

        # Adaptive threshold plasticity
        self._threshold_lr = config.threshold_learning_rate
        self._threshold_min = config.threshold_min
        self._threshold_max = config.threshold_max

        # =====================================================================
        # EMERGENT HIERARCHICAL GOALS (Biologically Plausible)
        # =====================================================================
        # Split neurons into abstract (rostral PFC) and concrete (caudal PFC)
        # This implements the biological rostral-caudal hierarchy
        n_abstract = int(self.executive_size * 0.3)  # 30% abstract (long tau, slow)
        n_concrete = self.executive_size - n_abstract  # 70% concrete (short tau, fast)

        # Goals emerge from WM patterns - no symbolic Goal objects
        self.emergent_goals = EmergentGoalSystem(
            n_wm_neurons=self.executive_size,
            n_abstract=n_abstract,
            n_concrete=n_concrete,
            device=str(self.device),
        )

        # =====================================================================
        # POST-INITIALIZATION
        # =====================================================================
        self.__post_init__()

    def _create_neurons(self) -> ConductanceLIF:
        """Create conductance-based LIF neurons with slow dynamics and SFA.

        PFC neurons have significantly different dynamics than standard pyramidal neurons:
        - Much slower leak (τ_m ≈ 50ms vs 20ms) for temporal integration
        - Slower synaptic time constants for sustained integration
        - Spike-frequency adaptation for stable working memory

        If heterogeneous WM is enabled, neurons will have varying membrane time constants
        (tau_mem) to create populations of stable vs flexible neurons.
        """
        cfg = self.config
        # Custom config for PFC-specific slow dynamics
        neuron_config = ConductanceLIFConfig(
            g_L=0.02,  # Default leak (will be overridden if heterogeneous)
            tau_E=10.0,  # Slower excitatory (for integration)
            tau_I=15.0,  # Slower inhibitory
            adapt_increment=cfg.adapt_increment,  # SFA enabled!
            tau_adapt=cfg.adapt_tau,
        )
        neurons = ConductanceLIF(self.executive_size, neuron_config, device=self.device)

        # =====================================================================
        # APPLY HETEROGENEOUS MEMBRANE TIME CONSTANTS
        # =====================================================================
        if self._tau_mem_heterogeneous is not None:
            # Convert tau_mem to g_L: tau_mem = C_m / g_L  =>  g_L = C_m / tau_mem
            # C_m = 1.0 (default), so g_L = 1 / tau_mem_ms
            C_m = neurons.C_m.item()

            # Compute per-neuron leak conductance from heterogeneous tau_mem
            # tau_mem in ms, g_L dimensionless (leak per timestep)
            g_L_heterogeneous = C_m / self._tau_mem_heterogeneous

            # Replace scalar g_L buffer with per-neuron tensor
            # Remove the old buffer and register new per-neuron buffer
            delattr(neurons, "g_L")
            neurons.register_buffer("g_L", g_L_heterogeneous.to(self.device))

        # =====================================================================
        # SHORT-TERM PLASTICITY for feedforward connections
        # =====================================================================
        # PFC feedforward connections show SHORT-TERM FACILITATION/DEPRESSION
        # for temporal filtering and gain control during encoding.
        # Created with size=0, will grow when sources are added
        self.stp_feedforward = ShortTermPlasticity(
            n_pre=0,  # Grow when sources added
            n_post=self.executive_size,
            config=STPConfig.from_type(STPType.FACILITATING),
            per_synapse=True,
        )
        self.stp_feedforward.to(self.device)

        # =====================================================================
        # SHORT-TERM PLASTICITY for recurrent connections
        # =====================================================================
        # PFC recurrent connections show SHORT-TERM DEPRESSION, preventing
        # frozen attractors. This allows working memory to be updated.
        self.stp_recurrent = ShortTermPlasticity(
            n_pre=self.executive_size,
            n_post=self.executive_size,
            config=STPConfig.from_type(STPType.DEPRESSING),
            per_synapse=True,
        )
        self.stp_recurrent.to(self.device)

        return neurons

    # =========================================================================
    # SYNAPTIC WEIGHT MANAGEMENT
    # =========================================================================

    def add_input_source(
        self,
        source_name: SpikesSourceKey,
        target_layer: LayerName,
        n_input: int,
        sparsity: float = 0.8,
        weight_scale: float = 1.0,
    ) -> None:
        """Override to create per-source feedforward weights.

        Each input source gets separate feedforward weights to PFC neurons.
        This enables proper multi-source architecture.

        Args:
            source_name: Name of input source (e.g., "cortex:l23", "hippocampus:ca1")
            n_input: Size of input from this source
            sparsity: Connection sparsity (0-1, higher = more sparse)
            weight_scale: Scaling factor for weight initialization
        """
        # Call parent to register source
        super().add_input_source(source_name, target_layer, n_input, sparsity, weight_scale)

        # Grow STP modules
        if n_input > 0:
            self.stp_feedforward.grow(n_input, target="pre")

        # Initialize homeostasis if this is first source
        self.homeostasis = UnifiedHomeostasis(UnifiedHomeostasisConfig(
            weight_budget=self._homeostasis_config.weight_budget,
            w_min=self._homeostasis_config.w_min,
            w_max=self._homeostasis_config.w_max,
            soft_normalization=self._homeostasis_config.soft_normalization,
            normalization_rate=self._homeostasis_config.normalization_rate,
            device=self.device,
        ))

        # Create per-source feedforward weights with sparse_random initialization
        # CHANGED from Xavier (2026-01-27): Xavier produced ~0.056 mean weights (100x smaller than cortex ~0.7)
        # causing PFC to be completely silent during cold-start. Now using same initialization as Cortex.
        self.synaptic_weights[source_name] = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=self.executive_size,
                n_input=n_input,
                sparsity=sparsity,
                weight_scale=weight_scale,
                device=self.device,
            )
        )

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def _forward_internal(self, inputs: RegionSpikesDict) -> None:
        """Process input through prefrontal cortex."""
        # =====================================================================
        # MULTI-SOURCE SYNAPTIC INTEGRATION
        # =====================================================================
        # Use base class helper for standardized multi-source integration
        # No STP per-source (PFC only has STP on recurrent connections)
        # No modulation callback needed (unlike cortex's alpha suppression)
        ff_integrated = self._integrate_multi_source_synaptic_inputs(
            inputs=inputs,
            n_neurons=self.executive_size,
            weight_key_suffix="",  # No suffix needed for PFC
            apply_stp=False,  # PFC doesn't use per-source STP
        )

        # Get timestep from config for temporal dynamics
        dt_ms = self.config.dt_ms

        gate = self.dopamine_system.get_gate()

        # =====================================================================
        # GAIN MODULATION
        # =====================================================================
        # Theta modulation now emerges from hippocampal-cortical interactions
        # PFC working memory gating depends on thalamic/cortical input patterns
        # that carry implicit theta structure from septal-hippocampal circuit
        ff_gain = 0.5  # Baseline
        rec_gain = 0.5  # Baseline

        # Feedforward input - modulated by encoding phase
        # Use pre-integrated multi-source input
        # NOTE: STP for multi-source is more complex - simplified here for now
        # TODO: Could apply per-source STP if needed
        ff_input = ff_integrated * ff_gain

        # =====================================================================
        # NOREPINEPHRINE GAIN MODULATION (Locus Coeruleus)
        # =====================================================================
        # High NE (arousal/uncertainty): Increase gain → more responsive WM
        # Low NE (baseline): Normal gain
        # Biological: β-adrenergic receptors modulate PFC excitability and
        # working memory flexibility (Arnsten 2009)
        ne_level = self.neuromodulator_state.norepinephrine
        # NE gain: 1.0 (baseline) to 1.5 (high arousal)
        ne_gain = compute_ne_gain(ne_level)
        ff_input = ff_input * ne_gain

        # =====================================================================
        # RECURRENT INPUT WITH STP (prevents frozen WM attractors)
        # =====================================================================
        # Without STP, the same WM pattern is reinforced forever.
        # With DEPRESSING STP, frequently-used synapses get temporarily weaker,
        # allowing WM to be updated with new information.
        if self.working_memory is not None:
            # Apply STP to recurrent connections (1D → 2D per-synapse efficacy)
            # stp_efficacy has shape [executive_size, executive_size] - per-synapse modulation
            stp_efficacy = self.stp_recurrent(self.working_memory.float())
            # Effective weights: element-wise multiply rec_weights with STP efficacy
            # rec_weights is [executive_size, executive_size], stp_efficacy is [executive_size, executive_size]
            effective_rec_weights = self.rec_weights * stp_efficacy.t()
            # Recurrent: weights[executive_size, executive_size] @ wm[executive_size] → [executive_size]
            rec_input = (effective_rec_weights @ self.working_memory.float()) * rec_gain
        else:
            # Recurrent input from working memory - modulated by retrieval phase
            # rec_weights[executive_size, executive_size] @ wm[executive_size] → [executive_size]
            wm = (
                self.working_memory.float()
                if self.working_memory is not None
                else torch.zeros(self.executive_size, device=self.device)
            )
            rec_input = (self.rec_weights @ wm) * rec_gain

        # Lateral inhibition: inhib_weights[executive_size, executive_size] @ wm[executive_size] → [executive_size]
        wm = (
            self.working_memory.float()
            if self.working_memory is not None
            else torch.zeros(self.executive_size, device=self.device)
        )
        inhib = self.inhib_weights @ wm

        # Total excitation and inhibition
        g_exc = (ff_input + rec_input).clamp(min=0)
        g_inh = inhib.clamp(min=0)

        # Apply homeostatic gain and baseline noise (Turrigiano 2008)
        # Add baseline noise (spontaneous miniature EPSPs)
        if self._baseline_noise > 0:
            g_exc = g_exc + self._baseline_noise
        # Apply adaptive gain
        g_exc = g_exc * self.gain

        # Run through neurons (returns 1D bool spikes)
        output_spikes, _ = self.neurons(g_exc, g_inh)

        # Update homeostatic gain for PFC
        # Update firing rate (exponential moving average)
        self.firing_rate = (
            1 - self._firing_rate_alpha
        ) * self.firing_rate + self._firing_rate_alpha * output_spikes.float()
        # Compute rate error
        rate_error = self._target_rate - self.firing_rate
        # Update gain (increase gain if firing too low, decrease if too high)
        # Clamp to prevent negative gains
        self.gain.data = (self.gain + self._gain_lr * rate_error).clamp(min=0.001)

        # Adaptive threshold: lower threshold when underactive
        self.neurons.v_threshold.data = torch.clamp(
            self.neurons.v_threshold - self._threshold_lr * rate_error,
            min=self._threshold_min,
            max=self._threshold_max,
        )

        # =====================================================================
        # D1/D2 RECEPTOR SUBTYPES - Differential Dopamine Modulation
        # =====================================================================
        # D1-dominant neurons: DA increases excitability (excitatory response)
        # D2-dominant neurons: DA decreases excitability (inhibitory response)
        # Biological: D1 receptors increase cAMP → enhanced firing
        #            D2 receptors decrease cAMP → reduced firing
        if self.config.use_d1_d2_subtypes and da_level != 0.0:
            # Create output buffer for modulated activity
            modulated_output = output_spikes.float().clone()

            # D1 neurons: Excitatory DA response (gain boost)
            d1_gain = 1.0 + self.config.d1_da_gain * da_level
            modulated_output[self._d1_neurons] *= d1_gain

            # D2 neurons: Inhibitory DA response (gain reduction)
            d2_gain = 1.0 - self.config.d2_da_gain * da_level
            modulated_output[self._d2_neurons] *= d2_gain

            # Convert back to spikes (probabilistic based on modulated activity)
            # High activity → more likely to spike
            spike_probs = modulated_output.clamp(0, 1)
            output_spikes = (torch.rand_like(spike_probs) < spike_probs).bool()

        # Update working memory with gating
        # High gate (high DA) → update with new activity
        # Low gate (low DA) → maintain current WM

        # Initialize working memory if needed (first forward pass)
        if self.working_memory is None:
            self.working_memory = torch.zeros(self.executive_size, device=self.device, dtype=torch.float32)

        gate_tensor = torch.full_like(self.working_memory, gate)
        self.update_gate = gate_tensor

        # WM decay
        decay = torch.exp(torch.tensor(-dt_ms / self.config.wm_decay_tau_ms))

        # Gated update: WM = gate * new_input + (1-gate) * decayed_old
        new_wm = (
            gate_tensor * output_spikes.float()
            + (1 - gate_tensor) * self.working_memory * decay
        )

        # Add noise for stochasticity
        noise = torch.randn_like(new_wm) * self.config.wm_noise_std
        self.working_memory = (new_wm + noise).clamp(min=0, max=1)

        # Output shape check
        assert output_spikes.shape == (self.executive_size,), (
            f"PrefrontalCortex.forward: output_spikes has shape {output_spikes.shape} "
            f"but expected ({self.executive_size},). "
            f"Check PFC neuron or weight configuration."
        )
        assert output_spikes.dtype == torch.bool, (
            f"PrefrontalCortex.forward: output_spikes must be bool (ADR-004), "
            f"got {output_spikes.dtype}"
        )

        # Apply continuous plasticity (learning happens as part of forward dynamics)
        # NOTE: Multi-source plasticity needs per-source learning
        # For now, skip plasticity until per-source learning strategies are implemented
        # TODO: Implement per-source STDP learning like striatum

        # =====================================================================
        # EMERGENT GOAL SYSTEM: Tag active WM patterns as goals
        # =====================================================================
        if self.working_memory is not None:
            # Tag currently active goal patterns (similar to hippocampal synaptic tagging)
            self.emergent_goals.update_goal_tags(self.working_memory)

        # Store output (NeuralRegion pattern)
        self.output_spikes = output_spikes

        # =====================================================================
        # SET PORT OUTPUTS
        # =====================================================================
        self.set_port_output("executive", output_spikes)

    def _apply_plasticity(
        self,
        input_spikes: torch.Tensor,
        output_spikes: torch.Tensor,
    ) -> None:
        """Apply dopamine-gated STDP learning using strategy pattern.

        Uses the learning strategy system for consistent plasticity application.
        """

        # IMPORTANT TODO: This method never gets called currently!
        # Need to integrate with multi-source learning in forward().
        # For now, plasticity is skipped in PFC until multi-source STDP is implemented.

        cfg = self.config
        # Input/output are already 1D bool tensors (ADR-005)

        # Apply STDP learning via strategy
        # Dopamine modulation is handled automatically by _apply_strategy_learning
        metrics = self._apply_strategy_learning(
            pre_activity=input_spikes,
            post_activity=output_spikes,
            weights=self.synaptic_weights["default"],
        )

        # Optional: Apply synaptic scaling for homeostasis
        if metrics:
            self.synaptic_weights["default"].data = self.homeostasis.normalize_weights(
                self.synaptic_weights["default"].data, dim=1
            )

        # ======================================================================
        # Update recurrent weights to strengthen WM patterns
        # ======================================================================
        # This simple Hebbian update for recurrent connections maintains WM patterns
        if self.working_memory is not None:
            wm = self.working_memory  # [executive_size]
            dW_rec = cfg.rule_lr * torch.outer(wm, wm)  # [executive_size, executive_size]
            self.rec_weights.data += dW_rec
            self.rec_weights.data.fill_diagonal_(cfg.recurrent_strength)  # Maintain self-excitation
            self.rec_weights.data.clamp_(0.0, 1.0)

        # ======================================================================
        # EMERGENT GOAL LEARNING: Learn transitions and consolidate values
        # ======================================================================
        if self.working_memory is not None:
            # Extract abstract and concrete patterns from WM
            abstract_pattern = self.working_memory[self.emergent_goals.abstract_neurons]
            concrete_pattern = self.working_memory[self.emergent_goals.concrete_neurons]

            # Learn goal transitions via Hebbian association
            # When abstract pattern A is active and concrete pattern B follows,
            # strengthen the transition A→B (emergent goal decomposition)
            if abstract_pattern.sum() > 0.1 and concrete_pattern.sum() > 0.1:
                self.emergent_goals.learn_transition(
                    abstract_pattern,
                    concrete_pattern,
                    learning_rate=cfg.rule_lr,
                )

            # Consolidate valuable goal patterns with dopamine
            # High dopamine → strengthen value associations for tagged goals
            self.emergent_goals.consolidate_valuable_goals(
                dopamine=self.neuromodulator_state.dopamine,
                learning_rate=cfg.rule_lr,
            )

    # =========================================================================
    # GOAL SYSTEM (EMERGENT HIERARCHICAL GOALS)
    # =========================================================================

    def get_current_goal_pattern(self) -> torch.Tensor:
        """Return current working memory pattern (= current goal).

        Goals ARE working memory patterns - no symbolic representation needed.

        Returns:
            Current WM pattern [executive_size] representing active goal

        Example:
            goal_pattern = pfc.get_current_goal_pattern()
            # This IS the goal - a sustained activity pattern
        """
        if self.working_memory is None:
            return torch.zeros(self.executive_size, device=self.device)
        return self.working_memory

    def predict_next_subgoal(self) -> torch.Tensor:
        """Predict next subgoal from current abstract WM pattern.

        Uses learned transitions (emergent goal decomposition), not explicit
        subgoal lists.

        Returns:
            Predicted concrete subgoal pattern [n_concrete]
        """
        assert self.emergent_goals is not None

        if self.working_memory is None:
            return torch.zeros(self.emergent_goals.n_concrete, device=self.device)

        # Extract abstract pattern from WM
        abstract_pattern = self.working_memory[self.emergent_goals.abstract_neurons]

        # Predict subgoal via learned transitions
        return self.emergent_goals.predict_subgoal(abstract_pattern)

    def get_goal_value(self, goal_pattern: torch.Tensor) -> float:
        """Get learned value for a goal pattern.

        Value is learned from experience with dopamine, not computed explicitly.

        Args:
            goal_pattern: Goal pattern to evaluate [executive_size]

        Returns:
            Estimated value of goal pattern

        Raises:
            ConfigurationError: If emergent goals not enabled
        """
        if self.emergent_goals is None:
            raise ConfigurationError("Emergent goals not enabled.")

        # Compute value as dot product with learned value weights
        return float(torch.sum(goal_pattern * self.emergent_goals.value_weights).item())

    # =========================================================================
    # TEMPORAL PARAMETER MANAGEMENT
    # =========================================================================

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters when brain timestep changes.

        Propagates dt update to neurons, STP components, and learning strategies.

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        super().update_temporal_parameters(dt_ms)

        # Update neurons
        if hasattr(self, "neurons") and self.neurons is not None:
            self.neurons.update_temporal_parameters(dt_ms)

        # Update STP components
        if hasattr(self, "stp_feedforward") and self.stp_feedforward is not None:
            self.stp_feedforward.update_temporal_parameters(dt_ms)
        if hasattr(self, "stp_recurrent") and self.stp_recurrent is not None:
            self.stp_recurrent.update_temporal_parameters(dt_ms)

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information for this region."""
        cfg = self.config

        # Plasticity metrics (if learning enabled)
        plasticity = None
        if self.config.learning_rate > 0:
            # Average across all weight matrices
            all_weights = torch.cat(
                [
                    self.synaptic_weights["default"].flatten(),
                    self.rec_weights.flatten(),
                    self.inhib_weights.flatten(),
                ]
            )
            plasticity = compute_plasticity_metrics(
                all_weights, learning_rate=self.config.learning_rate
            )

        return {
            "plasticity": plasticity,
            "gate_mean": (
                self.update_gate.mean().item() if self.update_gate is not None else 0.0
            ),
            "gate_std": (
                self.update_gate.std().item() if self.update_gate is not None else 0.0
            ),
            "wm_mean": (
                self.working_memory.mean().item()
                if self.working_memory is not None
                else 0.0
            ),
            "wm_std": (
                self.working_memory.std().item()
                if self.working_memory is not None
                else 0.0
            ),
            "wm_active": (
                (self.working_memory > 0.1).sum().item()
                if self.working_memory is not None
                else 0
            ),
            "config_w_min": cfg.w_min,
            "config_w_max": cfg.w_max,
        }
