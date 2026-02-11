"""
Striatum - Reinforcement Learning with Three-Factor Rule

The striatum (part of basal ganglia) learns through dopamine-modulated
plasticity, implementing the classic three-factor learning rule for
reinforcement learning.

Key Features:
=============
1. THREE-FACTOR LEARNING: Δw = eligibility × dopamine
   - Pre-post activity creates eligibility traces
   - Eligibility alone does NOT cause plasticity
   - Dopamine arriving later converts eligibility to weight change
   - DA burst → LTP, DA dip → LTD, No DA → no learning

2. DOPAMINE as REWARD PREDICTION ERROR:
   - Burst: "Better than expected" → reinforce recent actions
   - Dip: "Worse than expected" → weaken recent actions
   - Baseline: "As expected" → maintain current policy

3. LONG ELIGIBILITY TRACES:
   - Biological tau: 500-2000ms (Yagishita et al., 2014)
   - Allows credit assignment for delayed rewards
   - Synaptic tag persists until dopamine arrives

4. **ACTION SELECTION** (Winner-Take-All):
   - Lateral inhibition creates competition between action neurons
   - Winning action's synapses become eligible for learning
   - Dopamine retroactively credits (burst) or blames (dip) the winner
   - Losers' eligibility decays without reinforcement

**Biological Basis**:
====================
- **Medium Spiny Neurons (MSNs)**: 95% of striatal neurons
- **D1-MSNs (direct pathway)**: Express D1 receptors, DA → LTP → "Go" signal
- **D2-MSNs (indirect pathway)**: Express D2 receptors, DA → LTD → "No-Go" signal
- **Opponent Processing**: D1 promotes, D2 suppresses actions
"""

from __future__ import annotations

import weakref
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from thalia.brain.configs import StriatumConfig
from thalia.components import (
    ConductanceLIF,
    GapJunctionConfig,
    GapJunctionCoupling,
    NeuronFactory,
    ShortTermPlasticity,
    WeightInitializer,
    get_stp_config,
    create_heterogeneous_stp_configs,
)
from thalia.components.synapses.neuromodulator_receptor import NeuromodulatorReceptor
from thalia.typing import (
    PopulationName,
    PopulationSizes,
    RegionSpikesDict,
    SpikesSourceKey,
)
from thalia.units import ConductanceTensor
from thalia.utils import CircularDelayBuffer, compute_ne_gain

from .pathway import StriatumPathway, StriatumPathwayConfig
from .state_tracker import StriatumStateTracker

from ..neural_region import NeuralRegion
from ..region_registry import register_region


GAIN_MINIMUM_STRIATUM = 0.1  # Increased from 0.01 to allow recovery
GAIN_MAXIMUM_STRIATUM = 5.0  # Added upper bound to prevent explosion


@register_region(
    "striatum",
    aliases=["basal_ganglia"],
    description="Reinforcement learning via dopamine-modulated three-factor rule with D1/D2 opponent pathways",
    version="1.0",
    author="Thalia Project",
    config_class=StriatumConfig,
)
class Striatum(NeuralRegion[StriatumConfig]):
    """Striatal region with three-factor reinforcement learning.

    Inherits from NeuralRegion with biologically-accurate synaptic
    weight placement at target dendrites (not in axonal pathways).

    Implements dopamine-modulated learning:
    - Eligibility traces tag recently active synapses
    - Dopamine signal converts eligibility to plasticity
    - No learning without dopamine (unlike Hebbian)
    - Synaptic weights stored per-source in synaptic_weights dict
    """

    OUTPUT_POPULATIONS: Dict[PopulationName, str] = {
        "d1": "d1_size",
        "d2": "d2_size",
    }

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def d1_neurons(self) -> ConductanceLIF:
        """D1 neuron population (delegates to d1_pathway)."""
        return self.d1_pathway.neurons

    @property
    def d2_neurons(self) -> ConductanceLIF:
        """D2 neuron population (delegates to d2_pathway)."""
        return self.d2_pathway.neurons

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, config: StriatumConfig, population_sizes: PopulationSizes):
        """Initialize Striatum with D1/D2 opponent pathways."""
        super().__init__(config=config, population_sizes=population_sizes)

        # =====================================================================
        # EXTRACT LAYER SIZES
        # =====================================================================
        self.d1_size = population_sizes["d1_size"]
        self.d2_size = population_sizes["d2_size"]
        self.n_actions = population_sizes["n_actions"]
        self.neurons_per_action = population_sizes["neurons_per_action"]

        total_msn_neurons = self.d1_size + self.d2_size

        # =====================================================================
        # MULTI-SOURCE ELIGIBILITY TRACES
        # =====================================================================
        # Per-source-pathway eligibility traces for multi-source learning
        # Structure: {"source_d1": tensor, "source_d2": tensor, ...}
        #
        # Multi-timescale eligibility traces
        # Biology: Synaptic tags (eligibility) exist at multiple timescales:
        # - Fast traces (~500ms): Immediate coincidence detection (STDP-like)
        # - Slow traces (~60s): Consolidated long-term tags for delayed reward
        # Combined eligibility = fast + α*slow enables both rapid and multi-second
        # credit assignment. (Yagishita et al. 2014, Shindou et al. 2019)
        self._eligibility_d1: Dict[str, torch.Tensor] = {}
        self._eligibility_d2: Dict[str, torch.Tensor] = {}

        # Multi-timescale eligibility (optional, enabled via config)
        if config.use_multiscale_eligibility:
            self._eligibility_d1_fast: Dict[str, torch.Tensor] = {}
            self._eligibility_d2_fast: Dict[str, torch.Tensor] = {}
            self._eligibility_d1_slow: Dict[str, torch.Tensor] = {}
            self._eligibility_d2_slow: Dict[str, torch.Tensor] = {}
        else:
            # Single-timescale mode: fast traces are the regular eligibility
            # (no separate fast/slow dicts)
            pass

        # Source-specific eligibility tau configuration (optional overrides)
        # If not set, uses biological defaults in _get_source_eligibility_tau()
        self._source_eligibility_tau: Dict[str, float] = {}

        # =====================================================================
        # STATE TRACKER - Temporal State Management
        # =====================================================================
        self.state_tracker = StriatumStateTracker(
            n_actions=self.n_actions,
            n_output=total_msn_neurons,
            device=self.device,
        )

        # =====================================================================
        # EXPLORATION (UCB + Adaptive Exploration)
        # =====================================================================
        # UCB tracking
        self._action_counts = torch.zeros(self.n_actions, device=self.device)
        self._total_trials = 0

        # Adaptive exploration tracking
        self._recent_rewards: List[float] = []
        self._recent_accuracy = 0.0
        self.tonic_dopamine = self.config.tonic_dopamine

        # =====================================================================
        # D1/D2 PATHWAYS - Separate MSN Populations
        # =====================================================================
        # Create pathway-specific configuration with size=0
        # Will grow when sources are added
        # Pathways operate at MSN level (d1_size/d2_size), not action level
        d1_pathway_config = StriatumPathwayConfig(
            n_input=0,  # Grow when sources added
            n_output=self.d1_size,
            w_min=config.w_min,
            w_max=config.w_max,
            eligibility_tau_ms=config.eligibility_tau_ms,
            learning_rate=config.learning_rate,
            device=str(self.device),
        )
        d2_pathway_config = StriatumPathwayConfig(
            n_input=0,  # Grow when sources added
            n_output=self.d2_size,
            w_min=config.w_min,
            w_max=config.w_max,
            eligibility_tau_ms=config.eligibility_tau_ms,
            learning_rate=config.learning_rate,
            device=str(self.device),
        )

        # Create D1 and D2 pathways (neurons only, weights stored in parent)
        self.d1_pathway = StriatumPathway.create_d1(d1_pathway_config)
        self.d2_pathway = StriatumPathway.create_d2(d2_pathway_config)

        # =====================================================================
        # FSI (FAST-SPIKING INTERNEURONS) - Parvalbumin+ Interneurons
        # =====================================================================
        # FSI are ~2% of striatal neurons, provide feedforward inhibition
        # Critical for action selection timing (Koós & Tepper 1999)
        # Gap junction networks enable ultra-fast synchronization (<0.1ms)

        self.fsi_size = int(total_msn_neurons * self.config.fsi_ratio)
        self.fsi_neurons: ConductanceLIF = NeuronFactory.create_fast_spiking_neurons(
            n_neurons=self.fsi_size,
            device=self.device,
        )
        self.gap_junctions_fsi: Optional[GapJunctionCoupling] = None  # Will be initialized after weights

        # =====================================================================
        # MSN→FSI CONNECTIONS (Excitatory Feedback, Koos & Tepper 1999)
        # =====================================================================
        # Biology: MSNs excite FSI via glutamatergic collaterals (~30% connectivity)
        # This creates the feedback loop needed for winner-take-all dynamics:
        # - Winning action's MSNs fire more → excite FSI more
        # - FSI depolarizes → voltage-dependent Ca²⁺ channels open
        # - More GABA release → stronger inhibition of losing actions
        #
        # Implementation: Sparse connectivity from both D1 and D2 MSNs to FSI
        # Shape: [fsi_size, d1_size+d2_size] - FSI receive from all MSNs

        # D1 MSNs → FSI (excitatory, ~30% connectivity)
        self.d1_to_fsi_weights = WeightInitializer.sparse_random(
            n_output=self.fsi_size,
            n_input=self.d1_size,
            sparsity=0.3,  # ~30% MSN-FSI connectivity
            weight_scale=0.2,  # Moderate excitation
            device=self.device,
        )
        # Positive for excitation
        self.d1_to_fsi_weights = torch.abs(self.d1_to_fsi_weights)

        # D2 MSNs → FSI (excitatory, ~30% connectivity)
        self.d2_to_fsi_weights = WeightInitializer.sparse_random(
            n_output=self.fsi_size,
            n_input=self.d2_size,
            sparsity=0.3,
            weight_scale=0.2,  # Matches D1
            device=self.device,
        )
        self.d2_to_fsi_weights = torch.abs(self.d2_to_fsi_weights)

        # =====================================================================
        # FSI→MSN CONNECTIONS (Per-Neuron, Moyer 2014)
        # =====================================================================
        # Biology: Each MSN receives ~116 feedforward connections from ~18 FSIs
        # FSI inputs are 4-10× STRONGER than MSN lateral inputs
        #
        # Implementation: Sparse connectivity matrix from FSI → MSNs
        # Shape: [msn_size, fsi_size] - which FSI neurons connect to which MSNs
        # NOT a global broadcast - each MSN gets different FSI subset

        # FSI → D1 connections
        # ~15% connectivity to match ~18 FSI per MSN (18/120 ≈ 0.15)
        self.fsi_to_d1_weights = WeightInitializer.sparse_random(
            n_output=self.d1_size,
            n_input=self.fsi_size,
            sparsity=0.15,  # Sparse FSI connections
            weight_scale=0.8,  # Moderate strength (reduced from 2.5)
            device=self.device,
        )
        # Negative for inhibition
        self.fsi_to_d1_weights = -torch.abs(self.fsi_to_d1_weights)

        # FSI → D2 connections (same structure)
        self.fsi_to_d2_weights = WeightInitializer.sparse_random(
            n_output=self.d2_size,
            n_input=self.fsi_size,
            sparsity=0.15,
            weight_scale=0.8,  # Moderate strength (matches D1)
            device=self.device,
        )
        self.fsi_to_d2_weights = -torch.abs(self.fsi_to_d2_weights)

        # =====================================================================
        # HOMEOSTATIC INTRINSIC PLASTICITY (Adaptive Gain)
        # =====================================================================
        # Per-pathway adaptive gains to overcome bootstrap problem
        # EMA tracking of firing rates (per pathway)
        self.register_buffer("d1_firing_rate", torch.zeros(self.d1_size, device=self.device))
        self.register_buffer("d2_firing_rate", torch.zeros(self.d2_size, device=self.device))

        # Adaptive gains (per pathway)
        self.d1_gain = nn.Parameter(torch.ones(self.d1_size, device=self.device), requires_grad=False)
        self.d2_gain = nn.Parameter(torch.ones(self.d2_size, device=self.device), requires_grad=False)

        # Configuration
        self._target_rate = config.target_firing_rate
        self._gain_lr = config.gain_learning_rate
        self._baseline_noise = config.baseline_noise_current

        # EMA alpha for firing rate tracking
        self._firing_rate_alpha = config.dt_ms / config.gain_tau_ms

        # Adaptive threshold plasticity (complementary to gain adaptation)
        self._threshold_lr = config.threshold_learning_rate
        self._threshold_min = config.threshold_min
        self._threshold_max = config.threshold_max

        # =====================================================================
        # SHORT-TERM PLASTICITY (Per-Source)
        # =====================================================================
        # Multi-source architecture: Each source-pathway has its own STP module.
        # Different sources have different dynamics:
        # - Cortical inputs: DEPRESSING (U=0.4) - context filtering
        # - Thalamic inputs: FACILITATING (U=0.25) - phasic amplification
        # - Hippocampal inputs: DEPRESSING (U=0.35) - episodic filtering

        # NOTE: STP modules will be added per-source in add_input_source()
        # Each source-pathway (e.g., "cortex:l5_d1", "hippocampus_d2") gets its own STP
        self.stp_modules: Dict[str, ShortTermPlasticity] = {}

        # =====================================================================
        # GOAL-CONDITIONED VALUES
        # =====================================================================
        # Enable PFC goal context to modulate striatal action values via gating.
        # Biology: PFC working memory → Striatum modulation (Miller & Cohen 2001)
        # Learning: Three-factor rule extended with goal context:
        #   Δw = eligibility × dopamine × goal_context

        # NOTE: These are initialized lazily when add_input_source()
        # is called, enabling automatic size inference from actual PFC connection.
        self.pfc_modulation_d1: Optional[nn.Parameter] = None
        self.pfc_modulation_d2: Optional[nn.Parameter] = None

        # Initialize recent_spikes tensor for trial activity tracking
        self.recent_spikes = torch.zeros(total_msn_neurons, device=self.device)

        # =====================================================================
        # D1/D2 PATHWAY DELAY BUFFERS (Temporal Competition)
        # =====================================================================
        # Implement biologically-accurate transmission delays for opponent pathways:
        # - D1 direct pathway: ~15ms (Striatum → GPi/SNr → Thalamus)
        # - D2 indirect pathway: ~25ms (Striatum → GPe → STN → GPi/SNr → Thalamus)
        # D1 arrives ~10ms before D2, creating temporal competition window.

        # Calculate delay steps from millisecond delays
        self._d1_delay_steps = int(self.config.d1_to_output_delay_ms / self.config.dt_ms)
        self._d2_delay_steps = int(self.config.d2_to_output_delay_ms / self.config.dt_ms)

        # Initialize CircularDelayBuffer for D1 and D2 pathways
        # These buffer the action votes (n_actions) not individual neuron spikes
        self._d1_delay_buffer = CircularDelayBuffer(
            max_delay=self._d1_delay_steps,
            size=self.n_actions,
            device=str(self.device),
            dtype=torch.float32,  # Votes are float (spike counts per action)
        )
        self._d2_delay_buffer = CircularDelayBuffer(
            max_delay=self._d2_delay_steps,
            size=self.n_actions,
            device=str(self.device),
            dtype=torch.float32,
        )

        # Initialize current spikes for delay buffer input (will be updated each forward pass)
        self.d1_spikes = torch.zeros(self.d1_size, dtype=torch.bool, device=self.device)
        self.d2_spikes = torch.zeros(self.d2_size, dtype=torch.bool, device=self.device)

        # =====================================================================
        # DOPAMINE RECEPTORS (Spiking DA from VTA)
        # =====================================================================
        # Convert VTA dopamine neuron spikes to synaptic concentration.
        # Biology:
        # - D1 receptors: Gs-coupled → increase cAMP → facilitate LTP
        # - D2 receptors: Gi-coupled → decrease cAMP → facilitate LTD
        # - DA rise time: ~10-20 ms (fast release)
        # - DA decay time: ~200 ms (slow DAT reuptake)
        # Both pathways receive same DA spikes, but receptors have opposite effects.

        # Create D1 and D2 DA receptors (both receive same VTA spikes)
        self.da_receptor_d1 = NeuromodulatorReceptor(
            n_receptors=self.d1_size,
            tau_rise_ms=10.0,  # Fast release (ms)
            tau_decay_ms=200.0,  # Slow reuptake via DAT (ms)
            spike_amplitude=0.15,  # Moderate amplitude for summation
            device=self.device,
        )
        self.da_receptor_d2 = NeuromodulatorReceptor(
            n_receptors=self.d2_size,
            tau_rise_ms=10.0,
            tau_decay_ms=200.0,
            spike_amplitude=0.15,
            device=self.device,
        )

        # DA concentration state (updated each timestep from VTA spikes)
        self._da_concentration_d1 = torch.zeros(self.d1_size, device=self.device)
        self._da_concentration_d2 = torch.zeros(self.d2_size, device=self.device)

        # =====================================================================
        # NOREPINEPHRINE RECEPTORS (Spiking NE from LC)
        # =====================================================================
        # Convert LC norepinephrine neuron spikes to synaptic concentration.
        # Biology:
        # - NE modulates gain: high NE → increased excitability
        # - NE promotes exploration: high NE → more random action selection
        # - NE rise time: ~8 ms (fast release)
        # - NE decay time: ~150 ms (NET reuptake)
        # Both D1 and D2 MSNs receive NE modulation.

        self.ne_receptor_d1 = NeuromodulatorReceptor(
            n_receptors=self.d1_size,
            tau_rise_ms=8.0,  # Fast release (ms)
            tau_decay_ms=150.0,  # NET reuptake (ms)
            spike_amplitude=0.12,  # Moderate amplitude
            device=self.device,
        )
        self.ne_receptor_d2 = NeuromodulatorReceptor(
            n_receptors=self.d2_size,
            tau_rise_ms=8.0,
            tau_decay_ms=150.0,
            spike_amplitude=0.12,
            device=self.device,
        )

        # NE concentration state (updated from LC spikes)
        self._ne_concentration_d1 = torch.zeros(self.d1_size, device=self.device)
        self._ne_concentration_d2 = torch.zeros(self.d2_size, device=self.device)

        # =====================================================================
        # POST-INITIALIZATION
        # =====================================================================
        self.__post_init__()

    # =========================================================================
    # SYNAPTIC WEIGHT MANAGEMENT
    # =========================================================================

    def add_input_source(
        self,
        source_name: SpikesSourceKey,
        target_population: PopulationName,
        n_input: int,
        sparsity: float = 0.0,
        weight_scale: float = 1.0,
    ) -> None:
        """Add input source with automatic D1/D2 pathway weight creation.

        This is the primary method for connecting input sources to the striatum.
        It creates BOTH D1 and D2 pathway weights for the given source.

        Args:
            source_name: Source name (e.g., "cortex:l5", "hippocampus", "thalamus")
            n_input: Input size from this source
            sparsity: Connection sparsity (0.0 = fully connected)
            weight_scale: Initial weight scale multiplier
        """
        # Neuromodulator inputs (vta_da, lc_ne) don't create synaptic weights
        # They're processed directly by receptors in forward()
        if target_population in ("vta_da", "lc_ne"):
            # Just register the source for validation, but skip weight creation
            self.input_sources[source_name] = n_input
            return

        # REGISTER BASE SOURCE NAME in input_sources for validation
        # The forward() method receives base names (e.g., "cortex:l5")
        # and internally maps them to pathway-specific keys (e.g., "cortex:l5_d1")
        if source_name not in self.input_sources:
            self.input_sources[source_name] = n_input

        # Initialize D1 weights for this source
        # Use sparse random initialization with positive weights for reliable excitation
        # Biology: Most synapses are excitatory (glutamatergic) in corticostriatal pathways
        # With sparsity=0.8, weight_scale, and w_max: mean ≈ 0.8 * (weight_scale/2) * w_max
        # Multi-source scaling: striatum receives 4 sources, so scale down by 1/4 to prevent over-drive
        # Target cumulative mean: 4 sources × 0.0625 = 0.25 (same as single-source regions)
        # D1/D2 asymmetry: D1 starts stronger (3×) for exploration before D2 learns refinement
        d1_weights = (
            WeightInitializer.sparse_random(
                n_output=self.d1_size,
                n_input=n_input,
                sparsity=0.8,  # 80% connectivity - biologically realistic
                weight_scale=0.468 * weight_scale,  # 3× stronger: Mean ≈ 0.1875 * w_max (for exploration)
                device=self.device,
            )
            * self.config.w_max
        )

        # Initialize D2 weights for this source (weaker for later refinement)
        d2_weights = (
            WeightInitializer.sparse_random(
                n_output=self.d2_size,
                n_input=n_input,
                sparsity=0.8,  # 80% connectivity - biologically realistic
                weight_scale=0.156 * weight_scale,  # Standard: Mean ≈ 0.0625 * w_max (for refinement)
                device=self.device,
            )
            * self.config.w_max
        )

        # Register D1 pathway weights (internal key with suffix)
        d1_key = f"{source_name}_d1"
        # NOTE: We DON'T call super().add_input_source() for pathway-specific keys
        # because we already registered the base source name above
        self.synaptic_weights[d1_key] = nn.Parameter(d1_weights, requires_grad=False)

        # Register D2 pathway weights (internal key with suffix)
        d2_key = f"{source_name}_d2"
        self.synaptic_weights[d2_key] = nn.Parameter(d2_weights, requires_grad=False)

        # Link pathways to parent on first source (for checkpoint compatibility)
        # Pathways need _parent_striatum_ref and _weight_source to access weights
        if self.d1_pathway._parent_striatum_ref is None:
            self.d1_pathway._parent_striatum_ref = weakref.ref(self)
            self.d1_pathway._weight_source = d1_key
        if self.d2_pathway._parent_striatum_ref is None:
            self.d2_pathway._parent_striatum_ref = weakref.ref(self)
            self.d2_pathway._weight_source = d2_key

        # =====================================================================
        # LAZY PFC MODULATION INITIALIZATION
        # =====================================================================
        # If this is the PFC source and goal conditioning is enabled,
        # initialize PFC modulation weights with dynamically inferred size.
        if source_name.lower() == "pfc" and self.config.use_goal_conditioning:
            # Initialize PFC → D1 modulation weights [d1_size, n_input]
            self.pfc_modulation_d1 = nn.Parameter(
                WeightInitializer.sparse_random(
                    n_output=self.d1_size,
                    n_input=n_input,  # Inferred from actual PFC connection
                    sparsity=0.3,
                    device=self.device,
                ),
                requires_grad=False,
            )
            # Initialize PFC → D2 modulation weights [d2_size, n_input]
            self.pfc_modulation_d2 = nn.Parameter(
                WeightInitializer.sparse_random(
                    n_output=self.d2_size,
                    n_input=n_input,  # Inferred from actual PFC connection
                    sparsity=0.3,
                    device=self.device,
                ),
                requires_grad=False,
            )

        # =====================================================================
        # CREATE STP MODULES FOR SOURCE-PATHWAY
        # =====================================================================
        # Each source-pathway gets its own STP module with source-specific config.
        # Biology: Different input pathways have different short-term dynamics.
        # Heterogeneous STP enables per-synapse parameter variability.
        # Determine STP type based on source name
        if "cortex" in source_name or "cortical" in source_name:
            stp_type = "corticostriatal"  # Depressing (U=0.4)
        elif "thalamus" in source_name or "thalamic" in source_name:
            stp_type = "thalamostriatal"  # Facilitating (U=0.25)
        elif "hippocampus" in source_name or "hippoc" in source_name:
            stp_type = "schaffer_collateral"  # Depressing (U=0.46) - hippocampal preset
        else:
            stp_type = "corticostriatal"  # Default to cortical

        # Create STP configs (heterogeneous if enabled)
        if self.config.heterogeneous_stp:
            # Sample per-synapse STP parameters from distributions
            # Biology: 10-fold variability in U within same pathway
            # D1 pathway: Create list of per-synapse STP configs
            d1_configs = create_heterogeneous_stp_configs(
                base_preset=stp_type,
                n_synapses=n_input * self.d1_size,  # Total synapses
                variability=self.config.stp_variability,
                seed=self.config.stp_seed,
            )

            # D2 pathway: Create list of per-synapse STP configs
            d2_configs = create_heterogeneous_stp_configs(
                base_preset=stp_type,
                n_synapses=n_input * self.d2_size,  # Total synapses
                variability=self.config.stp_variability,
                seed=self.config.stp_seed,
            )

            # Create STP modules with heterogeneous configs
            d1_stp = ShortTermPlasticity(
                n_pre=n_input,
                n_post=self.d1_size,
                config=d1_configs,  # List of configs for heterogeneous STP
                per_synapse=True,
            )

            d2_stp = ShortTermPlasticity(
                n_pre=n_input,
                n_post=self.d2_size,
                config=d2_configs,  # List of configs for heterogeneous STP
                per_synapse=True,
            )
        else:
            # Standard uniform STP parameters
            d1_stp = ShortTermPlasticity(
                n_pre=n_input,
                n_post=self.d1_size,
                config=get_stp_config(stp_type),
                per_synapse=True,
            )

            d2_stp = ShortTermPlasticity(
                n_pre=n_input,
                n_post=self.d2_size,
                config=get_stp_config(stp_type),
                per_synapse=True,
            )

        # Register STP modules
        d1_stp.to(self.device)
        self.stp_modules[d1_key] = d1_stp

        d2_stp.to(self.device)
        self.stp_modules[d2_key] = d2_stp

        # =====================================================================
        # ADD FSI SOURCE (Feedforward Inhibition)
        # =====================================================================
        # FSI neurons need to receive input from ALL sources to provide
        # feedforward inhibition. If FSI are enabled, automatically create
        # FSI input weights for this source.
        if self.fsi_size > 0:
            self.add_fsi_source(source_name, n_input, weight_scale=weight_scale)

    def add_fsi_source(
        self,
        source_name: SpikesSourceKey,
        n_input: int,
        weight_scale: float = 1.0,
    ) -> None:
        """Add FSI input source (no D1/D2 separation for interneurons).

        FSI (fast-spiking interneurons) are parvalbumin+ interneurons that provide
        feedforward inhibition. Unlike MSNs, FSI don't have D1/D2 separation.

        Args:
            source_name: Source name (e.g., "cortex", "thalamus")
            n_input: Input size from this source
            weight_scale: Initial weight scale multiplier
        """
        if self.fsi_size == 0:
            return  # FSI disabled

        # Initialize FSI weights (input → FSI)
        # Use sparse_random for positive excitatory weights (FSI receive glutamatergic input)
        # FSI need strong weights because they're highly excitable and fire from weak inputs
        fsi_weights = (
            WeightInitializer.sparse_random(
                n_output=self.fsi_size,
                n_input=n_input,
                sparsity=0.5,  # 50% connectivity (FSI are highly connected)
                weight_scale=1.0 * weight_scale,  # Strong weights for FSI (increased from 0.4)
                device=self.device,
            )
            * self.config.w_max
        )

        # Register FSI source (internal key with prefix)
        fsi_key = f"fsi_{source_name}"
        # NOTE: We DON'T call super().add_input_source() for FSI keys
        # because the base source name is already registered in add_input_source()
        self.synaptic_weights[fsi_key] = nn.Parameter(fsi_weights, requires_grad=False)

        # Create gap junction coupling (if enabled and this is first FSI source)
        if self.gap_junctions_fsi is None:
            gap_config_fsi = GapJunctionConfig(
                coupling_strength=self.config.gap_junction_strength,
                connectivity_threshold=self.config.gap_junction_threshold,
                max_neighbors=self.config.gap_junction_max_neighbors,
            )
            # Use first FSI source weights for gap junction neighborhood computation
            self.gap_junctions_fsi = GapJunctionCoupling(
                n_neurons=self.fsi_size,
                afferent_weights=fsi_weights,  # Use current source weights
                config=gap_config_fsi,
                device=self.device,
            )

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def forward(self, region_inputs: RegionSpikesDict) -> RegionSpikesDict:
        """Process input and select action using separate D1/D2 populations."""
        self._pre_forward(region_inputs)

        # =====================================================================
        # DOPAMINE RECEPTOR UPDATE (VTA Spikes → Concentration)
        # =====================================================================
        # Convert spiking DA from VTA to synaptic concentration for learning.
        vta_da_spikes = region_inputs.get("vta:da_output", None)

        if vta_da_spikes is not None:
            # Update D1 and D2 receptors (both receive same VTA spikes)
            self._da_concentration_d1 = self.da_receptor_d1.update(vta_da_spikes)
            self._da_concentration_d2 = self.da_receptor_d2.update(vta_da_spikes)
        else:
            # No VTA connection: decay toward tonic baseline
            self._da_concentration_d1 = self.da_receptor_d1.update(None)
            self._da_concentration_d2 = self.da_receptor_d2.update(None)

        # =====================================================================
        # NOREPINEPHRINE RECEPTOR UPDATE (LC Spikes → Concentration)
        # =====================================================================
        # Convert spiking NE from LC to synaptic concentration for gain modulation.
        # NE increases excitability and promotes exploration.
        lc_ne_spikes = region_inputs.get("lc:ne_output", None)

        if lc_ne_spikes is not None:
            # Update D1 and D2 receptors (both receive same LC spikes)
            self._ne_concentration_d1 = self.ne_receptor_d1.update(lc_ne_spikes)
            self._ne_concentration_d2 = self.ne_receptor_d2.update(lc_ne_spikes)
        else:
            # No LC connection: decay toward baseline
            self._ne_concentration_d1 = self.ne_receptor_d1.update(None)
            self._ne_concentration_d2 = self.ne_receptor_d2.update(None)

        # =====================================================================
        # MULTI-SOURCE SYNAPTIC INTEGRATION
        # =====================================================================
        # Each source (cortex:l5, hippocampus, thalamus) has separate weights
        # for D1 and D2 pathways. Accumulate synaptic currents separately.
        # Biology: D1 and D2 MSNs are distinct neurons with independent synaptic
        # weights, so integration must remain separate per pathway.
        d1_current = self._integrate_multi_source_inputs(region_inputs, "_d1", self.d1_size)
        d2_current = self._integrate_multi_source_inputs(region_inputs, "_d2", self.d2_size)

        # =====================================================================
        # FSI (FAST-SPIKING INTERNEURONS) - Feedforward Inhibition
        # =====================================================================
        # FSI process inputs in parallel with MSNs but with:
        # 1. Gap junction coupling for synchronization (<0.1ms)
        # 2. Feedforward inhibition to MSNs (sharpens action timing)
        # Biology: FSI are parvalbumin+ interneurons (~2% of striatum)
        if self.fsi_size > 0:
            # Accumulate FSI currents from all sources
            fsi_current = torch.zeros(self.fsi_size, device=self.device)
            for source_name, source_spikes in region_inputs.items():
                fsi_key = f"fsi_{source_name}"
                if fsi_key in self.synaptic_weights:
                    source_spikes_float = (
                        source_spikes.float()
                        if source_spikes.dtype == torch.bool
                        else source_spikes
                    )
                    fsi_current += self.synaptic_weights[fsi_key] @ source_spikes_float

            # Apply gap junction coupling
            if (self.gap_junctions_fsi is not None
                and self.fsi_neurons.membrane is not None
            ):
                gap_current = self.gap_junctions_fsi(self.fsi_neurons.membrane)
                fsi_current = fsi_current + gap_current

            # =====================================================================
            # MSN→FSI EXCITATION (Winner-Take-All Feedback)
            # =====================================================================
            # Biology: MSN collaterals excite FSI via glutamatergic synapses
            # This creates positive feedback for winner-take-all:
            # - Winning action's MSNs fire more → excite FSI more
            # - FSI depolarizes → voltage-dependent GABA release increases
            # - Increased GABA → suppresses losing action more
            # - Gap widens → runaway to winner-take-all state

            # Compute excitation from D1 and D2 MSNs to FSI
            # Get spikes from current or previous timestep
            d1_to_fsi_input = self.d1_to_fsi_weights @ self.d1_spikes.float()
            d2_to_fsi_input = self.d2_to_fsi_weights @ self.d2_spikes.float()

            # Add MSN excitation to FSI input
            fsi_current = fsi_current + d1_to_fsi_input + d2_to_fsi_input

            # Update FSI neurons (fast kinetics, tau_mem ~5ms)
            fsi_spikes, fsi_membrane = self.fsi_neurons(
                g_exc_input=ConductanceTensor(fsi_current),
                g_inh_input=ConductanceTensor(torch.zeros_like(fsi_current)),  # FSI receive minimal inhibition
            )

            # =====================================================================
            # PER-NEURON FSI→MSN INHIBITION WITH VOLTAGE-DEPENDENT GABA RELEASE
            # =====================================================================
            # Biology: Each MSN receives ~116 feedforward connections from ~18 FSIs
            # FSI inputs are 4-10× STRONGER than MSN lateral inputs
            # CRITICAL: GABA release is voltage-dependent (Ca²⁺ channel dynamics)!
            #
            # Mechanism:
            # 1. FSI spikes create baseline inhibition via fsi_to_msn_weights
            # 2. FSI membrane voltage modulates release strength (Ca²⁺-dependent)
            # 3. Depolarized FSI (recent high activity) → more GABA release
            # 4. Hyperpolarized FSI (recent low activity) → less GABA release
            #
            # This creates BISTABILITY:
            # - Competitive state (~-60 mV): weak GABA, balanced competition
            # - Winner-take-all state (~-50 mV): strong GABA, losers suppressed
            #
            # NO EXPLICIT DETECTION - system naturally settles into attractor states!

            # Compute voltage-dependent inhibition scaling factor
            # fsi_membrane is [fsi_size] tensor of membrane voltages
            # Returns [fsi_size] tensor of scaling factors (0.1 to 0.8)
            inhibition_scale = self._fsi_membrane_to_inhibition_strength(fsi_membrane)

            # Average scaling across FSI population (they're synchronized via gap junctions)
            # This gives single scaling factor for whole network
            avg_inhibition_scale = inhibition_scale.mean()

            # FSI → D1 inhibition (per-neuron, voltage-dependent)
            # Shape: fsi_to_d1_weights [d1_size, fsi_size] @ fsi_spikes [fsi_size] = [d1_size]
            # Then scale by voltage-dependent factor
            fsi_inhibition_d1 = (self.fsi_to_d1_weights @ fsi_spikes.float()) * avg_inhibition_scale

            # FSI → D2 inhibition (per-neuron, voltage-dependent)
            fsi_inhibition_d2 = (self.fsi_to_d2_weights @ fsi_spikes.float()) * avg_inhibition_scale
        else:
            # FSI disabled → no FSI inhibition
            fsi_inhibition_d1 = torch.zeros(self.d1_size, device=self.device)
            fsi_inhibition_d2 = torch.zeros(self.d2_size, device=self.device)

        # =====================================================================
        # MSN→MSN LATERAL INHIBITION (Moyer 2014)
        # =====================================================================
        # Biology: Each MSN receives ~636 lateral inhibitory connections from ~430 other MSNs
        # Creates action competition: neurons of one action inhibit neurons of other actions
        # Mechanism: GABAergic collaterals with action-specific spatial organization

        # Get previous MSN spikes for lateral inhibition
        # Use previous timestep to avoid instantaneous feedback loops
        if self.d1_spikes is not None and self.d2_spikes is not None:
            prev_d1_spikes = self.d1_spikes.float()
            prev_d2_spikes = self.d2_spikes.float()
        else:
            # First timestep - no previous spikes
            prev_d1_spikes = torch.zeros(self.d1_size, device=self.device)
            prev_d2_spikes = torch.zeros(self.d2_size, device=self.device)

        # =====================================================================
        # MSN→MSN LATERAL INHIBITION (now EXTERNAL via population routing)
        # =====================================================================
        # D1/D2 lateral inhibition now arrives via external AxonalProjection
        # with proper 1.5ms axonal delays and sparse connectivity
        # Input via "d1_lateral" and "d2_lateral" populations

        # D1 lateral inhibition: Receive from "d1_lateral" population
        d1_lateral_input = region_inputs.get("d1_lateral", None)
        if d1_lateral_input is not None:
            d1_lateral_inhibition = d1_lateral_input.float()  # Already weighted and delayed
        else:
            d1_lateral_inhibition = torch.zeros_like(prev_d1_spikes, dtype=torch.float32)

        # D2 lateral inhibition: Receive from "d2_lateral" population
        d2_lateral_input = region_inputs.get("d2_lateral", None)
        if d2_lateral_input is not None:
            d2_lateral_inhibition = d2_lateral_input.float()  # Already weighted and delayed
        else:
            d2_lateral_inhibition = torch.zeros_like(prev_d2_spikes, dtype=torch.float32)

        # =====================================================================
        # D1/D2 NEURON ACTIVATION with Modulation
        # =====================================================================
        # Apply all modulation (theta, dopamine, NE, PFC, homeostasis) to currents
        # before neuron execution

        # Theta modulation emerges from hippocampal-cortical projections to striatum
        # D1/D2 balance determined by dopamine, inputs, and circuit dynamics
        # (no explicit encoding/retrieval phase modulation)

        # Dopamine gain modulation (per-neuron from receptors)
        # D1: DA increases excitability (Gs-coupled)
        # D2: DA decreases excitability (Gi-coupled) - inverted gain
        d1_da_gain = 1.0 + 0.3 * (self._da_concentration_d1 - 0.5)  # Centered at 0.5 baseline
        d2_da_gain = 1.0 - 0.2 * (self._da_concentration_d2 - 0.5)  # Inverted for Gi-coupling

        # NE gain modulation (average across neurons)
        ne_gain = compute_ne_gain(self._ne_concentration_d1.mean().item())

        d1_current = d1_current * d1_da_gain * ne_gain
        d2_current = d2_current * d2_da_gain * ne_gain

        # Apply ALL inhibition sources (FSI + MSN lateral)
        # Both are negative values, so addition makes currents more negative (stronger inhibition)
        d1_current = d1_current + fsi_inhibition_d1 + d1_lateral_inhibition
        d2_current = d2_current + fsi_inhibition_d2 + d2_lateral_inhibition

        # =====================================================================
        # HOMEOSTATIC INTRINSIC PLASTICITY (Bootstrap Solution)
        # =====================================================================
        # Add baseline noise + apply adaptive gains to overcome silent network problem
        # Add baseline noise (spontaneous miniature EPSPs)
        d1_noise = torch.randn_like(d1_current) * self._baseline_noise
        d2_noise = torch.randn_like(d2_current) * self._baseline_noise
        d1_current = d1_current + d1_noise
        d2_current = d2_current + d2_noise

        # Apply per-neuron adaptive gains
        d1_current = d1_current * self.d1_gain
        d2_current = d2_current * self.d2_gain

        # Execute D1 and D2 MSN populations
        d1_spikes, _ = self.d1_pathway.neurons(
            g_exc_input=ConductanceTensor(d1_current.clamp(min=0)),
            g_inh_input=ConductanceTensor(torch.zeros_like(d1_current)),
        )
        d2_spikes, _ = self.d2_pathway.neurons(
            g_exc_input=ConductanceTensor(d2_current.clamp(min=0)),
            g_inh_input=ConductanceTensor(torch.zeros_like(d2_current)),
        )

        # =====================================================================
        # HOMEOSTATIC GAIN UPDATE (After Spiking)
        # =====================================================================
        # Update firing rate EMA and adapt gains to maintain target rates
        # BIOLOGICAL STRATEGY: Regulate COMBINED D1+D2 rate, not independently
        # - Allows natural D1/D2 balance to emerge from competition
        # - Prevents asymmetric gain drift that causes weight divergence
        # - If total rate too high: reduce both gains proportionally
        # - If total rate too low: increase both gains proportionally
        # Update D1 firing rate (EMA)
        self.d1_firing_rate.data = (
            (1 - self._firing_rate_alpha) * self.d1_firing_rate
            + self._firing_rate_alpha * d1_spikes.float()
        )
        # Update D2 firing rate (EMA)
        self.d2_firing_rate.data = (
            (1 - self._firing_rate_alpha) * self.d2_firing_rate
            + self._firing_rate_alpha * d2_spikes.float()
        )

        # Compute COMBINED firing rate (D1 + D2 together)
        # Biology: Striatum as a whole should maintain sparse coding
        combined_rate = (self.d1_firing_rate.mean() + self.d2_firing_rate.mean()) / 2.0
        target_combined = self._target_rate  # 0.08 average across both pathways

        # Rate error for combined population (negative = firing too low)
        combined_rate_error = combined_rate - target_combined

        # Update BOTH D1 and D2 gains proportionally based on combined error
        # This maintains D1/D2 balance while regulating total activity
        # If total too high: reduce both gains
        # If total too low: increase both gains
        self.d1_gain.data = (self.d1_gain - self._gain_lr * combined_rate_error).clamp(min=GAIN_MINIMUM_STRIATUM, max=GAIN_MAXIMUM_STRIATUM)
        self.d2_gain.data = (self.d2_gain - self._gain_lr * combined_rate_error).clamp(min=GAIN_MINIMUM_STRIATUM, max=GAIN_MAXIMUM_STRIATUM)

        # Adaptive threshold update (complementary to gain adaptation)
        # Also use combined error to maintain balance
        # Adjust thresholds based on combined activity, not independently
        # Lower threshold when underactive, raise when overactive
        threshold_update = self._threshold_lr * combined_rate_error

        self.d1_neurons.v_threshold.data.add_(threshold_update).clamp_(
            min=self._threshold_min, max=self._threshold_max
        )
        self.d2_neurons.v_threshold.data.add_(threshold_update).clamp_(
            min=self._threshold_min, max=self._threshold_max
        )

        # =====================================================================
        # ACTION SELECTION: D1 - D2 (GO - NOGO) WITH TEMPORAL COMPETITION
        # =====================================================================
        # For each action, compute NET = D1_activity - D2_activity
        # Select action with highest NET value (or sample from softmax)
        # This is the key biological insight: D1 and D2 populations COMPETE

        # Count votes from current timestep spikes
        d1_votes_current = self._count_population_votes(d1_spikes)
        d2_votes_current = self._count_population_votes(d2_spikes)

        # Store current timestep votes BEFORE delays (for per-timestep action selection)
        # This allows both accumulated (for trial-level) and current (for timestep-level) decisions
        self.last_d1_votes = d1_votes_current.clone()
        self.last_d2_votes = d2_votes_current.clone()

        # =====================================================================
        # APPLY D1/D2 PATHWAY DELAYS (Biological Realism)
        # =====================================================================
        # D1 direct pathway: ~15ms (arrives first!)
        # D2 indirect pathway: ~25ms (arrives ~10ms later)
        # This creates temporal competition where D1 "Go" signal arrives before
        # D2 "No-Go" signal, explaining impulsivity and action selection timing.

        # Apply D1 delay (if configured)
        if self._d1_delay_steps > 0:
            # Write current votes to buffer
            self._d1_delay_buffer.write(d1_votes_current)
            # Read delayed votes
            d1_votes = self._d1_delay_buffer.read(self._d1_delay_steps)
            # Advance buffer to next timestep
            self._d1_delay_buffer.advance()
        else:
            d1_votes = d1_votes_current

        # Apply D2 delay (if configured, typically LONGER than D1)
        if self._d2_delay_steps > 0:
            # Write current votes to buffer
            self._d2_delay_buffer.write(d2_votes_current)
            # Read delayed votes (arrives LATER than D1!)
            d2_votes = self._d2_delay_buffer.read(self._d2_delay_steps)
            # Advance buffer to next timestep
            self._d2_delay_buffer.advance()
        else:
            d2_votes = d2_votes_current

        # ACCUMULATE delayed D1/D2 votes across timesteps for trial-level decision
        # This integrates sparse spiking evidence over time WITH proper temporal dynamics
        self.state_tracker.accumulate_votes(d1_votes, d2_votes)

        # =====================================================================
        # OUTPUT SPIKES: Return BOTH D1 and D2 (concatenated)
        # =====================================================================
        # Biologically: Both D1-MSNs and D2-MSNs are PROJECTION neurons that send
        # axons out of the striatum to different targets:
        # - D1 neurons → GPi/SNr (direct pathway, facilitates movement)
        # - D2 neurons → GPe (indirect pathway, suppresses movement)
        #
        # Output format: [D1_neuron_0, ..., D1_neuron_N, D2_neuron_0, ..., D2_neuron_M]
        # Total size: d1_size + d2_size
        #
        # Action selection (which action to take) happens in finalize_action() at
        # trial end, but both pathways' activity is visible to downstream regions.
        output_spikes = torch.cat([d1_spikes, d2_spikes], dim=0)

        # =====================================================================
        # UPDATE ELIGIBILITY TRACES (for all active neurons)
        # =====================================================================
        # Update D1/D2 STDP-style eligibility (always enabled)
        # Eligibility accumulates for ALL neurons that fire during the trial.
        # Learning is applied continuously in _apply_dopamine_modulated_learning()
        # using these eligibility traces and the current dopamine level.
        # Pass inputs dict to eligibility update (multi-source aware)
        self._update_pathway_eligibility(region_inputs, d1_spikes, "_d1", "_eligibility_d1")
        self._update_pathway_eligibility(region_inputs, d2_spikes, "_d2", "_eligibility_d2")

        # Update recent spikes and trial activity via state_tracker
        self.state_tracker.update_recent_spikes(d1_spikes, d2_spikes, decay=0.9)

        # =====================================================================
        # CONTINUOUS LEARNING (Biologically Accurate)
        # =====================================================================
        # Apply three-factor learning EVERY timestep using current dopamine level
        # Biology: Plasticity is continuous, not discrete. The three-factor rule
        # (Δw = eligibility × dopamine × lr) runs constantly in real synapses.
        # No separate "learning trigger" exists - dopamine levels fluctuate
        # continuously and weight changes happen continuously.s
        self._apply_dopamine_modulated_learning()

        # Store output spikes
        self.output_spikes = output_spikes

        region_outputs: RegionSpikesDict = {
            "d1": d1_spikes,
            "d2": d2_spikes,
        }

        return self._post_forward(region_outputs)

    def _apply_dopamine_modulated_learning(self) -> Dict[str, Any]:
        """Apply three-factor learning rule using current dopamine level.

        **Continuous Biological Plasticity**: This method implements the core
        three-factor learning rule (Δw = eligibility × dopamine × lr) that runs
        continuously in real neurons.

        **Multi-Source Learning**: Applies dopamine-modulated plasticity to each
        source-pathway combination separately using their respective eligibility traces.

        **Multi-Step Credit Assignment**: Credit naturally distributes to all recently
        active synapses via eligibility trace decay. Earlier actions receive weaker
        credit due to exponential trace decay (tau ~1000ms).

        **Spiking DA Receptors**: Uses per-neuron DA concentration from VTA spikes
        instead of scalar broadcast, enabling spatially-heterogeneous learning.

        Returns:
            Metrics dict with dopamine level and weight changes per source.
        """
        # Use per-neuron DA concentration from receptors (not scalar)
        # Each MSN neuron has its own DA receptor with local concentration
        d1_da_level = self._da_concentration_d1  # [d1_size]
        d2_da_level = self._da_concentration_d2  # [d2_size]

        # Apply learning per source-pathway using eligibility traces
        # Three-factor rule: Δw = eligibility × dopamine × learning_rate
        # Biology: Dopamine broadcasts globally to ALL synapses with eligibility traces.
        # No action-specific masking - this matches biological dopamine modulation.
        # Action differentiation emerges from differences in eligibility (which neurons spiked most)
        d1_total_ltp = 0.0
        d1_total_ltd = 0.0
        d2_total_ltp = 0.0
        d2_total_ltd = 0.0

        # D1 pathway learning (DA+ → LTP, DA- → LTD)
        for source_key, eligibility in self._eligibility_d1.items():
            if source_key in self.synaptic_weights:
                # Use combined eligibility if multi-timescale enabled
                if self.config.use_multiscale_eligibility:
                    # Combined eligibility = fast + α*slow
                    # Biology: Fast traces for immediate learning, slow traces for delayed reward
                    fast_trace = self._eligibility_d1_fast.get(source_key, eligibility)
                    slow_trace = self._eligibility_d1_slow.get(
                        source_key, torch.zeros_like(eligibility)
                    )
                    combined_eligibility = (
                        fast_trace + self.config.slow_trace_weight * slow_trace
                    )
                else:
                    # Single-timescale mode: use standard eligibility
                    combined_eligibility = eligibility

                # Compute weight update: Δw = eligibility × dopamine × lr
                # Biology: Per-neuron DA concentration modulates plasticity
                # Shape: [d1_size, n_input] × [d1_size, 1] = [d1_size, n_input]
                weight_update = combined_eligibility * d1_da_level.unsqueeze(1) * self.config.learning_rate

                # Apply update with weight bounds
                new_weights = torch.clamp(
                    self.synaptic_weights[source_key] + weight_update,
                    min=self.config.w_min,
                    max=self.config.w_max,
                )
                self.synaptic_weights[source_key].data = new_weights

                # Track LTP/LTD for diagnostics (per-neuron DA, use mean for reporting)
                mean_da = d1_da_level.mean().item()
                if mean_da > 0:
                    d1_total_ltp += weight_update.sum().item()
                else:
                    d1_total_ltd += weight_update.sum().item()

        # D2 pathway learning (DA+ → LTD, DA- → LTP - INVERTED!)
        for source_key, eligibility in self._eligibility_d2.items():
            if source_key in self.synaptic_weights:
                # Use combined eligibility if multi-timescale enabled
                if self.config.use_multiscale_eligibility:
                    # Combined eligibility = fast + α*slow
                    fast_trace = self._eligibility_d2_fast.get(source_key, eligibility)
                    slow_trace = self._eligibility_d2_slow.get(
                        source_key, torch.zeros_like(eligibility)
                    )
                    combined_eligibility = (
                        fast_trace + self.config.slow_trace_weight * slow_trace
                    )
                else:
                    # Single-timescale mode: use standard eligibility
                    combined_eligibility = eligibility

                # Compute weight update with INVERTED dopamine (D2 = Gi-coupled)
                # Biology: Per-neuron DA concentration modulates plasticity inversely
                # Shape: [d2_size, n_input] × [d2_size, 1] = [d2_size, n_input]
                weight_update = combined_eligibility * (-d2_da_level.unsqueeze(1)) * self.config.learning_rate

                # Apply update with weight bounds
                new_weights = torch.clamp(
                    self.synaptic_weights[source_key] + weight_update,
                    min=self.config.w_min,
                    max=self.config.w_max,
                )
                self.synaptic_weights[source_key].data = new_weights

                # Track LTP/LTD for diagnostics (note inverted dopamine, use mean for reporting)
                mean_da = d2_da_level.mean().item()
                if mean_da > 0:
                    d2_total_ltd += weight_update.sum().item()
                else:
                    d2_total_ltp += weight_update.sum().item()

        # Return metrics with mean DA levels for diagnostics
        return {
            "d1_ltp": d1_total_ltp,
            "d1_ltd": d1_total_ltd,
            "d2_ltp": d2_total_ltp,
            "d2_ltd": d2_total_ltd,
            "net_change": d1_total_ltp + d1_total_ltd + d2_total_ltp + d2_total_ltd,
            "dopamine": d1_da_level.mean().item(),  # Mean DA concentration for reporting
        }

    def _get_source_eligibility_tau(self, source_name: str) -> float:
        """Get source-specific eligibility trace tau.

        Different input sources have different temporal dynamics:
        - Cortical inputs: Long traces (1000ms) for sustained context
        - Hippocampal inputs: Fast traces (300ms) for episodic snapshots
        - Thalamic inputs: Intermediate (500ms) for phasic signals

        Args:
            source_name: Source identifier (e.g., "cortex:l5", "hippocampus")

        Returns:
            Eligibility tau in milliseconds
        """
        # Check if custom tau is configured
        if hasattr(self, "_source_eligibility_tau") and source_name in self._source_eligibility_tau:
            tau_value = self._source_eligibility_tau[source_name]
            # Ensure it's a float (handle tensor or other numeric types)
            if isinstance(tau_value, torch.Tensor):
                return float(tau_value.item())
            return float(tau_value)

        # Apply biological defaults based on source type
        if "cortex" in source_name:
            return 1000.0  # Standard corticostriatal (long traces)
        elif "hippocampus" in source_name or "hippoc" in source_name:
            return 300.0  # Fast episodic context
        elif "thalamus" in source_name or "thal" in source_name:
            return 500.0  # Intermediate phasic signals
        else:
            # Default to config value
            return (
                self.config.eligibility_tau_ms
                if self.config.eligibility_tau_ms is not None
                else 1000.0
            )

    def _update_pathway_eligibility(
        self,
        inputs: RegionSpikesDict,
        post_spikes: torch.Tensor,
        pathway_suffix: str,
        eligibility_attr: str,
    ) -> None:
        """Update eligibility traces for one pathway (D1 or D2).

        This helper consolidates eligibility update logic that is identical between
        D1 and D2 pathways. Biological separation is maintained:
        - D1 and D2 have separate eligibility trace dictionaries
        - D1 and D2 have different post-synaptic spike patterns
        - Dopamine modulation (applied later) differs between pathways

        Multi-timescale eligibility (optional)
        When use_multiscale_eligibility is enabled:
        - Fast traces (~500ms): Immediate coincidence detection
        - Slow traces (~60s): Consolidated long-term tags
        - Consolidation: slow ← slow*decay + fast*consolidation_rate
        - Combined eligibility = fast + α*slow (used in continuous learning)

        Args:
            inputs: Dict of source_name -> spike_tensor
            post_spikes: Post-synaptic spikes for this pathway [n_neurons]
            pathway_suffix: "_d1" or "_d2" to select pathway weights
            eligibility_attr: "_eligibility_d1" or "_eligibility_d2"

        Biological note:
            Eligibility traces are the "synaptic tags" that mark recently active
            synapses. When dopamine arrives (seconds later), only tagged synapses
            undergo plasticity. This implements the three-factor learning rule.
        """
        post_spikes = post_spikes.float()  # Ensure float for outer product

        # Multi-timescale mode
        if self.config.use_multiscale_eligibility:
            # Get or initialize eligibility dicts for fast and slow traces
            fast_attr = f"{eligibility_attr}_fast"
            slow_attr = f"{eligibility_attr}_slow"

            if not hasattr(self, fast_attr):
                setattr(self, fast_attr, {})
            if not hasattr(self, slow_attr):
                setattr(self, slow_attr, {})

            fast_dict = getattr(self, fast_attr)
            slow_dict = getattr(self, slow_attr)

            # Decay constants
            fast_decay = torch.exp(
                torch.tensor(-self.config.dt_ms / self.config.fast_eligibility_tau_ms)
            )
            slow_decay = torch.exp(
                torch.tensor(-self.config.dt_ms / self.config.slow_eligibility_tau_ms)
            )
            consolidation_rate = self.config.eligibility_consolidation_rate

            for source_name, source_spikes in inputs.items():
                # Ensure 1D
                if source_spikes.dim() != 1:
                    source_spikes = source_spikes.squeeze()
                source_spikes_float = source_spikes.float()

                # Get pathway-specific key
                key = f"{source_name}{pathway_suffix}"
                if key not in self.synaptic_weights:
                    continue

                # Initialize traces if needed
                weight_shape = self.synaptic_weights[key].shape
                if key not in fast_dict:
                    fast_dict[key] = torch.zeros(weight_shape, device=self.device)
                if key not in slow_dict:
                    slow_dict[key] = torch.zeros(weight_shape, device=self.device)

                # Compute STDP-style eligibility update
                eligibility_update = torch.outer(post_spikes, source_spikes_float)

                # Update fast trace: decay + immediate tagging
                # NOTE: Do NOT apply learning rate here - it's applied during weight update
                fast_dict[key] = fast_dict[key] * fast_decay + eligibility_update

                # Update slow trace: decay + consolidation from fast trace
                # Biology: Fast tags consolidate into persistent slow tags
                slow_dict[key] = slow_dict[key] * slow_decay + fast_dict[key] * consolidation_rate

            # For backward compatibility: set regular eligibility to fast traces
            # (continuous learning uses combined eligibility when multi-timescale enabled)
            if not hasattr(self, eligibility_attr):
                setattr(self, eligibility_attr, {})
            eligibility_dict = getattr(self, eligibility_attr)
            for key in fast_dict:
                eligibility_dict[key] = fast_dict[key].clone()

        else:
            # Single-timescale mode (original implementation)
            # Get or initialize eligibility dict for this pathway
            if not hasattr(self, eligibility_attr):
                setattr(self, eligibility_attr, {})
            eligibility_dict = getattr(self, eligibility_attr)

            for source_name, source_spikes in inputs.items():
                # Ensure 1D
                if source_spikes.dim() != 1:
                    source_spikes = source_spikes.squeeze()

                # Get float version
                source_spikes_float = source_spikes.float()

                # Get pathway-specific key (e.g., "cortex:l5_d1", "hippocampus_d2")
                key = f"{source_name}{pathway_suffix}"
                if key not in self.synaptic_weights:
                    continue

                # Initialize eligibility trace if needed
                if key not in eligibility_dict:
                    weight_shape = self.synaptic_weights[key].shape
                    eligibility_dict[key] = torch.zeros(weight_shape, device=self.device)

                # Compute STDP-style eligibility: outer product of post and pre
                # This marks synapses where pre-spike and post-spike co-occurred
                eligibility_update = torch.outer(post_spikes, source_spikes_float)

                # Get source-specific decay tau (cortex=1000ms, hippoc=300ms, etc.)
                tau_ms = self._get_source_eligibility_tau(source_name)
                decay = torch.exp(torch.tensor(-self.config.dt_ms / tau_ms))

                # Decay old eligibility and add new trace
                # NOTE: Do NOT apply learning rate here - it's applied during weight update
                # Three-factor rule: Δw = eligibility × dopamine × lr (lr applied in _apply_dopamine_modulated_learning)
                eligibility_dict[key] = eligibility_dict[key] * decay + eligibility_update

    def set_source_eligibility_tau(self, source_name: str, tau_ms: float) -> None:
        """Configure custom eligibility tau for a specific source.

        Override the biological default with a custom value for specific sources.
        Useful for fine-tuning learning dynamics per input pathway.

        Args:
            source_name: Source identifier (e.g., "cortex:l5", "hippocampus")
            tau_ms: Eligibility trace decay time constant in milliseconds
        """
        self._source_eligibility_tau[source_name] = tau_ms

    # =========================================================================
    # MULTI-SOURCE SYNAPTIC INTEGRATION
    # =========================================================================

    def _fsi_membrane_to_inhibition_strength(self, fsi_membrane_v: torch.Tensor) -> torch.Tensor:
        """Convert FSI membrane potential to inhibition scaling factor.

        Biology: GABA release is voltage-dependent!
        - Calcium influx increases with depolarization
        - More Ca²⁺ → more vesicle fusion → more GABA release
        - This creates nonlinear relationship between membrane voltage and inhibition

        NO RATE COMPUTATION - the membrane potential itself carries temporal history:
        - Membrane time constant (tau_mem ~5ms) naturally integrates recent spikes
        - Depolarized membrane = recent high activity
        - Hyperpolarized membrane = recent low activity

        This voltage-dependent release creates BISTABILITY:
        - Below ~-58 mV: weak GABA release, competitive dynamics
        - Above ~-52 mV: strong GABA release, winner-take-all dynamics

        Implementation:
        - Baseline (0.1): Minimum inhibition at rest (~-65 mV)
        - Maximum (0.8): Maximum inhibition near threshold (~-45 mV)
        - Inflection (-55 mV): Where transition happens (FSI saturation)
        - Steepness (3.0 mV): How sharp the transition is

        Args:
            fsi_membrane_v: FSI membrane potentials [n_fsi] in mV

        Returns:
            Per-FSI inhibition scaling factors [n_fsi] (0.1 to 0.8)
        """
        baseline = 0.1  # Minimum inhibition (at rest, ~-65 mV)
        maximum = 0.8  # Maximum inhibition (near threshold, ~-45 mV)
        inflection = -55.0  # Voltage where transition happens (mV)
        steepness = 3.0  # How sharp the transition is (mV)

        # Sigmoid based on membrane voltage (voltage-dependent GABA release)
        # σ(v) = baseline + (maximum - baseline) / (1 + exp(-(v - inflection) / steepness))
        sigmoid = torch.sigmoid((fsi_membrane_v - inflection) / steepness)
        return baseline + (maximum - baseline) * sigmoid

    def _integrate_multi_source_inputs(
        self,
        inputs: RegionSpikesDict,
        pathway_suffix: str,
        n_neurons: int,
    ) -> torch.Tensor:
        """Integrate synaptic currents from multiple sources for D1 or D2 pathway.

        This helper consolidates the synaptic integration logic that is identical
        between D1 and D2 pathways, while maintaining biological separation:
        - D1 and D2 have separate synaptic weights (independent plasticity)
        - D1 and D2 have separate STP modules (pathway-specific dynamics)
        - Same input creates different currents due to different weights

        Args:
            inputs: Dict mapping source names to spike tensors
            pathway_suffix: "_d1" or "_d2" to select pathway weights
            n_neurons: Number of neurons in this pathway (d1_size or d2_size)

        Returns:
            Total synaptic current for this pathway [n_neurons]

        Biological note:
            This consolidation is for code quality only. Biologically, D1 and D2
            MSNs are distinct neurons with independent synaptic weights, so the
            computation MUST remain separate per pathway.
        """
        current = torch.zeros(n_neurons, device=self.device)

        for source_name, source_spikes in inputs.items():
            # Ensure 1D (ADR-005)
            if source_spikes.dim() != 1:
                source_spikes = source_spikes.squeeze()

            # Convert to float for matrix multiplication
            source_spikes_float = (
                source_spikes.float() if source_spikes.dtype == torch.bool else source_spikes
            )

            # Get pathway-specific weights (e.g., "cortex:l5_d1" or "hippocampus_d2")
            key = f"{source_name}{pathway_suffix}"
            if key not in self.synaptic_weights:
                continue

            weights = self.synaptic_weights[key]

            # Apply source-specific STP if enabled
            if key in self.stp_modules:
                # STP returns [n_input, n_neurons] efficacy matrix
                efficacy = self.stp_modules[key](source_spikes_float)
                # Modulate weights: effective_w = w * efficacy.T
                # weights: [n_neurons, n_input], efficacy.T: [n_neurons, n_input]
                effective_weights = weights * efficacy.T
                current += effective_weights @ source_spikes_float
            else:
                # No STP: direct weight multiplication
                current += weights @ source_spikes_float

        return current

    # =========================================================================
    # ACTION SELECTION HELPERS
    # =========================================================================

    def _get_action_population_indices(self, action: int) -> slice:
        """Get the slice of neuron indices for a given action.

        With population coding enabled:
        - Action 0 → neurons 0 to neurons_per_action-1
        - Action 1 → neurons neurons_per_action to 2*neurons_per_action-1
        - etc.
        """
        start = action * self.neurons_per_action
        end = start + self.neurons_per_action
        return slice(start, end)

    def _count_population_votes(self, spikes: torch.Tensor) -> torch.Tensor:
        """Count spike votes for each action population.

        Used for D1-D2 subtraction to compute NET signal per action.

        Args:
            spikes: Spike tensor [n_output] (1D)

        Returns:
            Tensor of shape [n_actions] with vote counts per action.
        """
        # Ensure 1D
        if spikes.dim() != 1:
            spikes = spikes.squeeze()

        votes = torch.zeros(self.n_actions, device=self.device)

        for action in range(self.n_actions):
            pop_slice = self._get_action_population_indices(action)
            votes[action] = spikes[pop_slice].sum()

        return votes

    def update_action_counts(self, action: int) -> None:
        """Update UCB action counts after a trial completes.

        This should be called ONCE per trial by the brain_system after
        action selection is finalized. Not called inside forward() because
        forward() runs multiple times per timestep.

        Args:
            action: The action that was selected (0 to n_actions-1)
        """
        self._action_counts[action] += 1
        self._total_trials += 1

    def update_performance(self, reward: float) -> None:
        """Update performance history for adaptive exploration.

        This should be called ONCE per trial by the brain_system after
        reward is received. Not called inside forward() because forward()
        runs multiple times per timestep.

        Args:
            reward: Reward received for the selected action
        """
        self._recent_rewards.append(reward)
        if len(self._recent_rewards) > self.config.performance_window:
            self._recent_rewards.pop(0)

        # Update running accuracy
        window = min(len(self._recent_rewards), self.config.performance_window)
        if window > 0:
            correct_count = sum(1 for r in self._recent_rewards[-window:] if r > 0)
            self._recent_accuracy = correct_count / window

        # Adjust tonic dopamine based on performance
        if self.config.adaptive_exploration:
            # Poor performance → higher tonic DA → more exploration
            # Good performance → lower tonic DA → more exploitation
            if self._recent_accuracy < 0.5:
                self.tonic_dopamine = self.config.max_tonic_dopamine
            elif self._recent_accuracy > 0.8:
                self.tonic_dopamine = self.config.min_tonic_dopamine
            else:
                # Linear interpolation
                self.tonic_dopamine = (
                    self.config.min_tonic_dopamine
                    + (self.config.max_tonic_dopamine - self.config.min_tonic_dopamine)
                    * (0.8 - self._recent_accuracy)
                    / 0.3
                )

    def _compute_ucb_bonus(self) -> torch.Tensor:
        """Compute UCB (Upper Confidence Bound) exploration bonus.

        UCB formula: c * sqrt(ln(total_trials) / action_count)

        Returns:
            UCB bonus per action [n_actions]
        """
        if not self.config.ucb_exploration or self._total_trials == 0:
            return torch.zeros(self.n_actions, device=self.device)

        log_t = torch.log(
            torch.tensor(self._total_trials + 1, dtype=torch.float32, device=self.device)
        )
        ucb_bonus = self.config.ucb_coefficient * torch.sqrt(log_t / (self._action_counts + 1.0))
        return ucb_bonus

    def finalize_action(self, explore: bool = True) -> Dict[str, Any]:
        """Finalize action selection at the end of a trial.

        This consolidates the accumulated NET votes, applies UCB bonus,
        performs softmax (or argmax) selection, updates action counts ONCE,
        and returns diagnostics.

        Args:
            explore: Whether to allow exploration (bias-correcting + tonic DA)

        Returns:
            Dict with keys: selected_action, probs (if softmax), ucb_bonus,
            net_votes, exploring (bool), exploration_prob
        """
        net_votes = self.state_tracker.get_net_votes()
        ucb_bonus = self._compute_ucb_bonus()

        selection_values = net_votes + ucb_bonus

        # Compute bias-correcting exploration probability (same as forward)
        exploration_prob = 0.0
        if explore:
            action_nets = selection_values.tolist()
            if len(action_nets) > 1:
                net_range = max(action_nets) - min(action_nets)
                temperature = self.config.uncertainty_temperature
                bias_factor = (
                    net_range / (temperature + net_range) if (temperature + net_range) > 0 else 0.0
                )
                min_boost = self.config.min_exploration_boost
                max_boost = 0.5
                exploration_prob = min_boost + bias_factor * (max_boost - min_boost)

            # tonic modulation
            if self.config.tonic_modulates_exploration:
                tonic_boost = self.tonic_dopamine * self.config.tonic_exploration_scale
                exploration_prob = min(0.6, exploration_prob + tonic_boost)

        self.state_tracker.exploring = False
        probs = None
        selected_action = 0

        if explore and torch.rand(1).item() < exploration_prob:
            # Random exploration
            self.state_tracker.exploring = True
            selected_action = int(torch.randint(0, self.n_actions, (1,)).item())
        else:
            if self.config.softmax_action_selection:
                temperature = self.config.softmax_temperature
                selection_values_norm = selection_values - selection_values.max()
                probs = torch.softmax(selection_values_norm / temperature, dim=0)
                selected_action = int(torch.multinomial(probs, 1).item())
            else:
                max_val = selection_values.max().item()
                max_indices = (selection_values == max_val).nonzero(as_tuple=True)[0]
                if len(max_indices) > 1:
                    idx = int(torch.randint(len(max_indices), (1,)).item())
                    selected_action = int(max_indices[idx].item())
                else:
                    selected_action = int(max_indices[0].item())

        # Update bookkeeping ONCE per trial
        self.state_tracker.set_last_action(selected_action, self.state_tracker.exploring)
        self.state_tracker.update_exploration_stats(exploration_prob=exploration_prob)
        self.update_action_counts(selected_action)

        return {
            "selected_action": selected_action,
            "probs": probs,
            "ucb_bonus": ucb_bonus,
            "net_votes": net_votes,
            "exploring": self.state_tracker.exploring,
            "exploration_prob": exploration_prob,
        }

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
        self.d1_neurons.update_temporal_parameters(dt_ms)
        self.d2_neurons.update_temporal_parameters(dt_ms)
        self.fsi_neurons.update_temporal_parameters(dt_ms)
