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

from typing import Dict

import torch

from thalia.brain.configs import PrefrontalConfig
from thalia.brain.regions.population_names import PrefrontalPopulation
from thalia.components import (
    ConductanceLIF,
    ConductanceLIFConfig,
    NeuromodulatorReceptor,
    STPConfig,
    WeightInitializer,
)
from thalia.learning import STDPConfig, STDPStrategy
from thalia.typing import (
    NeuromodulatorInput,
    PopulationSizes,
    RegionName,
    RegionOutput,
    SynapseId,
    SynapticInput,
)
from thalia.units import ConductanceTensor
from thalia.utils import (
    CircularDelayBuffer,
    clamp_weights,
    compute_ach_recurrent_suppression,
    compute_ne_gain,
)

from .goal_emergence import EmergentGoalSystem

from ..neural_region import NeuralRegion
from ..region_registry import register_region


def sample_heterogeneous_wm_neurons(
    n_neurons: int,
    stability_cv: float,
    tau_mem_min: float,
    tau_mem_max: float,
    device: torch.device,
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
        device: Device for output tensors

    Returns:
        Dictionary with:
        - recurrent_strength: [n_neurons] tensor of recurrent weights (0.2-1.0 range)
        - tau_mem: [n_neurons] tensor of membrane time constants (100-500ms)
    """
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

    # Move to device
    recurrent_strength = recurrent_strength.to(device)
    tau_mem = tau_mem.to(device)

    return {
        "recurrent_strength": recurrent_strength,
        "tau_mem": tau_mem,
    }


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

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, config: PrefrontalConfig, population_sizes: PopulationSizes, region_name: RegionName):
        """Initialize prefrontal cortex."""
        super().__init__(config, population_sizes, region_name)

        # =====================================================================
        # EXTRACT LAYER SIZES
        # =====================================================================
        self.executive_size = population_sizes[PrefrontalPopulation.EXECUTIVE.value]

        # =====================================================================
        # INITIALIZE STATE FIELDS
        # =====================================================================
        self.working_memory: torch.Tensor = torch.zeros(self.executive_size, device=self.device, dtype=torch.float32)

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
        )

        # =====================================================================
        # D1/D2 RECEPTOR SUBTYPES
        # =====================================================================
        # Split neurons into D1-dominant (excitatory DA response) and
        # D2-dominant (inhibitory DA response) populations
        n_d1 = int(self.executive_size * config.d1_fraction)
        self._d1_neurons: torch.Tensor = torch.arange(n_d1, device=self.device)
        self._d2_neurons: torch.Tensor = torch.arange(n_d1, self.executive_size, device=self.device)

        self.neurons = ConductanceLIF(
            n_neurons=self.executive_size,
            config=ConductanceLIFConfig(
                region_name=self.region_name,
                population_name=PrefrontalPopulation.EXECUTIVE.value,
                device=self.device,
                tau_mem=wm_properties["tau_mem"],  # Per-neuron tensor (100-500ms)!
                g_L=0.02,  # Leak conductance (will work with per-neuron tau_mem in dynamics)
                tau_E=10.0,  # Slower excitatory (for integration)
                tau_I=15.0,  # Slower inhibitory
                adapt_increment=self.config.adapt_increment,  # SFA enabled!
                tau_adapt=self.config.adapt_tau,
            ),
        )

        # =====================================================================
        # SHORT-TERM PLASTICITY for recurrent connections
        # =====================================================================
        # PFC recurrent connections show BALANCED depression/facilitation.
        # CRITICAL: Working memory requires FAST recovery (tau_d=200ms)
        # Standard PSEUDOLINEAR (tau_d=400ms) caused 99.7% depletion in 100ms!
        # Custom config:
        # - Light depression (U=0.2) minimizes depletion
        # - Fast recovery (tau_d=200ms) enables sustained activity
        # - Moderate facilitation (tau_f=150ms) supports working memory
        # Biology: Layer 2/3 pyramidal show fast recovery for persistent activity

        # Recurrent weights for WM maintenance
        # CONDUCTANCE-BASED: Weights must be non-negative (represent excitatory conductances)
        rec_weights = WeightInitializer.sparse_gaussian(
            n_input=self.executive_size,
            n_output=self.executive_size,
            connectivity=1.0,
            mean=0.0,
            std=0.0005,
            device=self.device,
        )
        # Initialize with self-excitation (heterogeneous)
        # Scale diagonal by heterogeneous recurrent strengths
        # CRITICAL: Also multiply by GLOBAL_WEIGHT_SCALE to respect global disable
        rec_weights = rec_weights + torch.diag(wm_properties["recurrent_strength"]) * WeightInitializer.GLOBAL_WEIGHT_SCALE
        self._add_internal_connection(
            source_population=PrefrontalPopulation.EXECUTIVE.value,
            target_population=PrefrontalPopulation.EXECUTIVE.value,
            weights=rec_weights,
            stp_config=STPConfig(U=0.2, tau_d=200.0, tau_f=150.0),
            is_inhibitory=False,
        )

        # PFC recurrent delay buffer (prevents instant feedback oscillations)
        recurrent_delay_steps = int(config.recurrent_delay_ms / config.dt_ms)
        self._recurrent_buffer = CircularDelayBuffer(
            max_delay=recurrent_delay_steps,
            size=self.executive_size,
            device=str(self.device),
            dtype=torch.float32,  # Working memory values (continuous)
        )

        # Lateral inhibition weights
        inhib_weights = WeightInitializer.sparse_random(
            n_input=self.executive_size,
            n_output=self.executive_size,
            connectivity=1.0,  # Fully connected lateral inhibition
            weight_scale=0.08,
            device=self.device,
        )
        inhib_weights.fill_diagonal_(0.0)
        self._add_internal_connection(
            source_population=PrefrontalPopulation.EXECUTIVE.value,
            target_population=PrefrontalPopulation.EXECUTIVE.value,
            weights=inhib_weights,
            stp_config=None,
            is_inhibitory=True,
        )

        # =====================================================================
        # DOPAMINE RECEPTORS (Spiking DA from VTA)
        # =====================================================================
        # Convert VTA dopamine neuron spikes to synaptic concentration.
        # Biology: PFC receives strong mesocortical DA projection from VTA
        # - D1 receptors: Modulate WM maintenance and gating
        # - DA rise: ~10-20 ms (fast release)
        # - DA decay: ~200 ms (slow DAT reuptake)
        # Primarily D1-type receptors in PFC (unlike striatum's D1/D2 balance)
        self.da_receptor = NeuromodulatorReceptor(
            n_receptors=self.executive_size,
            tau_rise_ms=15.0,  # Moderate release
            tau_decay_ms=200.0,  # Slow reuptake via DAT
            spike_amplitude=0.15,  # Moderate amplitude
            device=self.device,
        )
        self._da_concentration = torch.zeros(self.executive_size, device=self.device)

        # =====================================================================
        # NOREPINEPHRINE RECEPTORS (Spiking NE from LC)
        # =====================================================================
        # Convert LC norepinephrine spikes to synaptic concentration.
        # Biology: PFC receives dense NE innervation from LC
        # - NE modulates attention, arousal, and task engagement
        # - High NE → increased gain, enhanced signal detection
        # - NE rise: ~8 ms, decay: ~150 ms (NET reuptake)
        self.ne_receptor = NeuromodulatorReceptor(
            n_receptors=self.executive_size,
            tau_rise_ms=8.0,
            tau_decay_ms=150.0,
            spike_amplitude=0.12,
            device=self.device,
        )
        self._ne_concentration = torch.zeros(self.executive_size, device=self.device)

        # =====================================================================
        # ACETYLCHOLINE RECEPTORS (Spiking ACh from NB)
        # =====================================================================
        # Convert NB acetylcholine spikes to synaptic concentration.
        # Biology: PFC receives ACh from nucleus basalis
        # - ACh modulates attention and working memory encoding
        # - High ACh → encoding mode, enhance sensory processing
        # - ACh rise: ~5 ms, decay: ~50 ms (AChE fast degradation)
        self.ach_receptor = NeuromodulatorReceptor(
            n_receptors=self.executive_size,
            tau_rise_ms=5.0,
            tau_decay_ms=50.0,
            spike_amplitude=0.2,
            device=self.device,
        )
        self._ach_concentration = torch.zeros(self.executive_size, device=self.device)

        # =====================================================================
        # LEARNING STRATEGY: STDP with dopamine gating
        # =====================================================================
        self.learning_strategy = STDPStrategy(STDPConfig(
            learning_rate=config.learning_rate,
            a_plus=config.a_plus,
            a_minus=config.a_minus,
            tau_plus=config.tau_plus_ms,
            tau_minus=config.tau_minus_ms,
            w_min=config.w_min,
            w_max=config.w_max,
        ))

        # =====================================================================
        # HOMEOSTATIC INTRINSIC PLASTICITY
        # =====================================================================
        # Firing rate tracker (exponential moving average)
        self.register_buffer("firing_rate", torch.zeros(self.executive_size, device=self.device))

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
        # REGISTER NEURON POPULATIONS
        # =====================================================================
        self._register_neuron_population(PrefrontalPopulation.EXECUTIVE.value, self.neurons)

        # =====================================================================
        # POST-INITIALIZATION
        # =====================================================================
        self.__post_init__()

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    @torch.no_grad()
    def forward(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Process input through prefrontal cortex.

        Args:
            synaptic_inputs: Point-to-point synaptic connections from cortex, hippocampus, striatum
            neuromodulator_inputs: Broadcast neuromodulatory signals (DA, NE, ACh)
        """
        self._pre_forward(synaptic_inputs, neuromodulator_inputs)

        cfg = self.config
        dt_ms = cfg.dt_ms

        if cfg.enable_neuromodulation:
            # =====================================================================
            # DOPAMINE RECEPTOR PROCESSING (from VTA)
            # =====================================================================
            # Process VTA dopamine spikes → concentration dynamics
            # PFC receives widespread DA innervation for working memory gating
            vta_da_spikes = neuromodulator_inputs.get('da', None)
            self._da_concentration = self.da_receptor.update(vta_da_spikes)

            # =====================================================================
            # NOREPINEPHRINE RECEPTOR PROCESSING (from LC)
            # =====================================================================
            # Process LC norepinephrine spikes → gain modulation
            # NE increases arousal and task engagement in PFC
            lc_ne_spikes = neuromodulator_inputs.get('ne', None)
            self._ne_concentration = self.ne_receptor.update(lc_ne_spikes)

            # =====================================================================
            # ACETYLCHOLINE RECEPTOR PROCESSING (from NB)
            # =====================================================================
            # Process NB acetylcholine spikes → encoding/attention modulation
            # ACh switches PFC between encoding and retrieval modes
            nb_ach_spikes = neuromodulator_inputs.get('ach', None)
            self._ach_concentration = self.ach_receptor.update(nb_ach_spikes)
        else:
            # Neuromodulation disabled: keep baseline concentrations
            pass

        # =====================================================================
        # MULTI-SOURCE SYNAPTIC INTEGRATION
        # =====================================================================
        # Use base class helper for standardized multi-source integration
        # Apply per-source STP for temporal filtering and gain control
        # No modulation callback needed (unlike cortex's alpha suppression)
        ff_input = self._integrate_synaptic_inputs_at_dendrites(synaptic_inputs, n_neurons=self.executive_size)

        # =====================================================================
        # NOREPINEPHRINE GAIN MODULATION (Locus Coeruleus)
        # =====================================================================
        # High NE (arousal/uncertainty): Increase gain → more responsive WM
        # Low NE (baseline): Normal gain
        # Biological: β-adrenergic receptors modulate PFC excitability and
        # working memory flexibility (Arnsten 2009)
        ne_level = self._ne_concentration.mean().item()  # Average across neurons
        # NE gain: 1.0 (baseline) to 1.5 (high arousal)
        ne_gain = compute_ne_gain(ne_level)
        ff_input = ff_input * ne_gain

        # =====================================================================
        # RECURRENT INPUT WITH STP (prevents frozen WM attractors)
        # =====================================================================
        # Without STP, the same WM pattern is reinforced forever.
        # With DEPRESSING STP, frequently-used synapses get temporarily weaker,
        # allowing WM to be updated with new information.

        executive_rec_exc_synapse = SynapseId(
            source_region=self.region_name,
            source_population=PrefrontalPopulation.EXECUTIVE.value,
            target_region=self.region_name,
            target_population=PrefrontalPopulation.EXECUTIVE.value,
            is_inhibitory=False,
        )
        executive_rec_inhib_synapse = SynapseId(
            source_region=self.region_name,
            source_population=PrefrontalPopulation.EXECUTIVE.value,
            target_region=self.region_name,
            target_population=PrefrontalPopulation.EXECUTIVE.value,
            is_inhibitory=True,
        )

        # Read delayed working memory from buffer (returns zeros if buffer not initialized)
        wm_delayed = self._recurrent_buffer.read(self._recurrent_buffer.max_delay)

        stp_efficacy = self.stp_modules[executive_rec_exc_synapse].forward(wm_delayed)
        effective_rec_weights = self.get_synaptic_weights(executive_rec_exc_synapse) * stp_efficacy.t()

        # ACh modulation applied here (region-level neuromodulation):
        # High ACh (attention/encoding): Suppress recurrence to prioritize sensory input
        # Low ACh (maintenance): Enable recurrence for stable working memory attractors
        ach_recurrent_modulation = compute_ach_recurrent_suppression(self._ach_concentration.mean().item())

        rec_input = (effective_rec_weights @ wm_delayed).clamp(min=0) * ach_recurrent_modulation

        # Total excitation
        g_exc = ff_input + rec_input
        if cfg.baseline_noise_conductance_enabled:
            noise = torch.randn_like(g_exc) * 0.007
            g_exc = (g_exc + noise).clamp(min=0)

        g_ampa, g_nmda = self._split_excitatory_conductance(g_exc)

        # Lateral inhibition: use delayed working memory for causality
        # inhib_weights[executive_size, executive_size] @ wm[executive_size] → [executive_size]
        g_inh = self.get_synaptic_weights(executive_rec_inhib_synapse) @ wm_delayed
        g_inh = g_inh.clamp(min=0)

        output_spikes, _ = self.neurons.forward(
            g_ampa_input=ConductanceTensor(g_ampa),
            g_gaba_a_input=ConductanceTensor(g_inh),
            g_nmda_input=ConductanceTensor(g_nmda),
        )

        self._update_homeostasis(spikes=output_spikes, firing_rate=self.firing_rate, neurons=self.neurons)

        # =====================================================================
        # D1/D2 RECEPTOR SUBTYPES - Differential Dopamine Modulation
        # =====================================================================
        # D1-dominant neurons: DA increases excitability (excitatory response)
        # D2-dominant neurons: DA decreases excitability (inhibitory response)
        # Biological: D1 receptors increase cAMP → enhanced firing
        #            D2 receptors decrease cAMP → reduced firing
        # Use per-neuron DA concentration from VTA spikes
        da_levels = self._da_concentration  # [executive_size]

        # Create output buffer for modulated activity
        modulated_output = output_spikes.float().clone()

        # D1 neurons: Excitatory DA response (gain boost)
        d1_gain = 1.0 + cfg.d1_da_gain * da_levels[self._d1_neurons]
        modulated_output[self._d1_neurons] *= d1_gain

        # D2 neurons: Inhibitory DA response (gain reduction)
        d2_gain = 1.0 - cfg.d2_da_gain * da_levels[self._d2_neurons]
        modulated_output[self._d2_neurons] *= d2_gain

        # Convert back to spikes (probabilistic based on modulated activity)
        # High activity → more likely to spike
        spike_probs = modulated_output.clamp(0, 1)
        output_spikes = (torch.rand_like(spike_probs) < spike_probs).bool()

        # Update working memory with gating
        # High gate (high DA) → update with new activity
        # Low gate (low DA) → maintain current WM

        # Compute DA gate from receptor-based concentration (per-neuron)
        # Sigmoid gating: gate = sigmoid(10 * (da_level - threshold))
        # This creates smooth transition around threshold:
        # - da_level < threshold → gate ≈ 0 (maintain)
        # - da_level > threshold → gate ≈ 1 (update)
        gate_tensor = torch.sigmoid(10 * (self._da_concentration - cfg.gate_threshold))
        wm_decay = torch.exp(torch.tensor(-dt_ms / cfg.wm_decay_tau_ms))

        # Gated update: WM = gate * new_input + (1-gate) * decayed_old
        new_wm = gate_tensor * output_spikes.float() + (1 - gate_tensor) * self.working_memory * wm_decay
        # Add noise for stochasticity
        wm_noise = torch.randn_like(new_wm) * cfg.wm_noise_std

        self.working_memory = (new_wm + wm_noise).clamp(min=0, max=1)

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

        # =====================================================================
        # CONTINUOUS PLASTICITY (Per-Source Learning)
        # =====================================================================
        # Apply STDP learning per-source with dopamine modulation
        # Biology: Plasticity happens continuously based on spike timing,
        # modulated by dopamine concentration from VTA
        if WeightInitializer.GLOBAL_LEARNING_ENABLED:
            if cfg.learning_rate > 0:
                # Get dopamine modulation (average across neurons for learning rate scaling)
                da_modulation = self._da_concentration.mean().item()
                effective_lr = cfg.learning_rate * (1.0 + da_modulation)

                # Apply per-source STDP learning to feedforward connections
                for synapse_id, source_spikes in synaptic_inputs.items():
                    # Skip if this source doesn't have weights
                    if not self.has_synaptic_weights(synapse_id):
                        continue

                    # Apply STDP learning via strategy
                    weights = self.get_synaptic_weights(synapse_id)
                    weights.data = self.learning_strategy.compute_update(
                        weights=weights.data,
                        pre_spikes=source_spikes,
                        post_spikes=output_spikes,
                        learning_rate=effective_lr,
                        neuromodulator=da_modulation,
                    )
                    clamp_weights(weights.data, cfg.w_min, cfg.w_max)

                # ======================================================================
                # RECURRENT WEIGHT LEARNING: Strengthen active WM patterns
                # ======================================================================
                # Simple Hebbian update for recurrent connections maintains WM patterns
                # Biology: Persistent activity is stabilized by recurrent excitation
                dW_rec = effective_lr * torch.outer(self.working_memory, self.working_memory)  # [executive_size, executive_size]
                executive_rec_exc_synapse = SynapseId(
                    source_region=self.region_name,
                    source_population=PrefrontalPopulation.EXECUTIVE.value,
                    target_region=self.region_name,
                    target_population=PrefrontalPopulation.EXECUTIVE.value,
                    is_inhibitory=False,
                )
                executive_rec_exc_weights = self.get_synaptic_weights(executive_rec_exc_synapse)
                executive_rec_exc_weights.data += dW_rec
                executive_rec_exc_weights.data.fill_diagonal_(0.15)  # Maintain self-excitation
                executive_rec_exc_weights.data.clamp_(0.0, 1.0)

                # TODO: Add learning to inhibitory weights for balance (homeostatic inhibitory plasticity)

            # =====================================================================
            # EMERGENT GOAL LEARNING
            # =====================================================================
            # Tag currently active goal patterns (similar to hippocampal synaptic tagging)
            self.emergent_goals.update_goal_tags(self.working_memory)

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
                    learning_rate=cfg.learning_rate,
                )

            # Consolidate valuable goal patterns with dopamine
            # High dopamine → strengthen value associations for tagged goals
            self.emergent_goals.consolidate_valuable_goals(
                dopamine=self._da_concentration.mean().item(),  # Average across neurons
                learning_rate=cfg.learning_rate,
            )

        region_outputs: RegionOutput = {
            PrefrontalPopulation.EXECUTIVE.value: output_spikes,
        }

        # Write current working memory to buffer for next timestep (causality)
        self._recurrent_buffer.write_and_advance(self.working_memory)

        return self._post_forward(region_outputs)

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
        self.neurons.update_temporal_parameters(dt_ms)
