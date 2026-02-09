"""
Cortex - Multi-layer cortical microcircuit.

This implements a biologically realistic cortical column with distinct layers:

Architecture (based on canonical cortical microcircuit):
=========================================================

         Feedback from higher areas
                    │
                    ▼
    ┌───────────────────────────────────┐
    │          LAYER 2/3                │ ← Superficial pyramidal cells
    │   (Cortico-cortical output)       │ → To other cortical areas
    │   - Receives from L4              │ → Gamma attention gating
    │   - Lateral recurrent connections │ → Gap junctions for sync
    │   - Top-down feedback target      │
    └───────────────┬───────────────────┘
                    │
    ┌───────────────┴───────────────────┐
    │          LAYER 4                  │ ← Spiny stellate cells
    │   (Feedforward input layer)       │ ← From thalamus/lower areas
    │   - Main sensory input recipient  │
    │   - No recurrent connections      │
    │   - Fast, feedforward processing  │
    └───────────────┬───────────────────┘
                    │
    ┌───────────────┴───────────────────┐
    │          LAYER 5                  │ ← Deep pyramidal cells
    │   (Subcortical output layer)      │ → To striatum, brainstem, etc.
    │   - Receives from L2/3            │ → Motor/action-related output
    │   - Different output pathway      │
    │   - Burst-capable neurons         │
    └───────────────┬───────────────────┘
                    │
    ┌───────────────┴───────────────────┐
    │          LAYER 6a                 │ ← CT Type I (low gamma)
    │   (Corticothalamic → TRN)         │ → Spatial attention via TRN
    │   - Receives from L2/3            │ → Inhibitory modulation
    │   - Projects to thalamic TRN      │
    │   - Sparse, low-gamma firing      │
    └───────────────────────────────────┘
                    │
    ┌───────────────┴───────────────────┐
    │          LAYER 6b                 │ ← CT Type II (high gamma)
    │   (Corticothalamic → Relay)       │ → Fast gain modulation
    │   - Receives from L2/3            │ → Direct relay excitation
    │   - Projects to thalamic relay    │
    │   - Dense, high-gamma firing      │
    └───────────────────────────────────┘
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from thalia.brain.configs import CortexConfig, CortexLayer
from thalia.components import (
    GapJunctionCoupling,
    WeightInitializer,
    NeuronFactory,
)
from thalia.diagnostics import compute_plasticity_metrics
from thalia.learning import (
    LearningStrategy,
    UnifiedHomeostasis,
    UnifiedHomeostasisConfig,
    BCMConfig,
    BCMStrategy,
    STDPConfig,
    STDPStrategy,
    CompositeStrategy,
)
from thalia.typing import (
    LayerName,
    RegionLayerSizes,
    RegionSpikesDict,
    SpikesSourceKey,
)
from thalia.utils import (
    CircularDelayBuffer,
    clamp_weights,
    compute_ach_recurrent_suppression,
    compute_ne_gain,
    validate_spike_tensor,
)

from .inhibitory_network import InhibitoryNetwork

from ..neural_region import NeuralRegion
from ..region_registry import register_region
from ..stimulus_gating import StimulusGating


@register_region(
    "cortex",
    aliases=["layered_cortex"],
    description="Multi-layer cortical microcircuit with L2/3/L4/L5/L6a/L6b structure",
    version="1.0",
    author="Thalia Project",
    config_class=CortexConfig,
)
class Cortex(NeuralRegion[CortexConfig]):
    """Multi-layer cortical microcircuit with proper layer separation and routing.

    Implements a canonical cortical column with distinct computational layers:

    **Layer Architecture** (based on mammalian cortex):
    - **L2/3**: Processing layer - recurrent computation, outputs to other cortex
    - **L4**: Input layer - receives thalamic/sensory input, feedforward processing
    - **L5**: Output layer - projects to subcortical structures (striatum, etc.)
    - **L6a**: CT Type I - projects to thalamic TRN (spatial attention, low gamma)
    - **L6b**: CT Type II - projects to thalamic relay (gain modulation, high gamma)

    **Information Flow**:
    1. External input → L4 (feedforward processing)
    2. L4 → L2/3 (local integration with axonal delays)
    3. L2/3 → L2/3 (recurrent processing with STP and gap junctions)
    4. L2/3 → L5 (deep projection with axonal delays)
    5. L2/3 → L6a (corticothalamic type I with axonal delays)
    6. L2/3 → L6b (corticothalamic type II with axonal delays)
    7. L2/3 → Other cortex (cortico-cortical output)
    8. L5 → Subcortical (striatum, thalamus, brainstem)
    9. L6a → Thalamus TRN (spatial attention via inhibitory modulation)
    10. L6b → Thalamus Relay (fast gain modulation via excitatory drive)
    """

    # Declarative output ports (auto-registered by base class)
    OUTPUT_PORTS = {
        "l23": "l23_size",
        "l4": "l4_size",
        "l5": "l5_size",
        "l6a": "l6a_size",
        "l6b": "l6b_size",
    }

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, config: CortexConfig, region_layer_sizes: RegionLayerSizes):
        """Initialize cortex."""
        super().__init__(config=config, region_layer_sizes=region_layer_sizes)

        # =====================================================================
        # EXTRACT LAYER SIZES AND CALCULATE FSI POPULATIONS
        # =====================================================================
        # Layer sizes represent PYRAMIDAL neurons (excitatory, what projects out)
        # FSI (fast-spiking interneurons) are additive internal circuitry:
        #   - FSI = 25% of pyramidal count = 20% of total (80/20 split)
        #   - Example: 400 pyr + 100 FSI = 500 total (80% pyr, 20% FSI)

        # Pyramidal sizes
        self.l23_size: int = region_layer_sizes["l23_size"]
        self.l4_size: int = region_layer_sizes["l4_size"]
        self.l5_size: int = region_layer_sizes["l5_size"]
        self.l6a_size: int = region_layer_sizes["l6a_size"]
        self.l6b_size: int = region_layer_sizes["l6b_size"]

        # Pyramidal neuron counts (same as layer sizes for clarity)
        self.l4_pyr_size = self.l4_size
        self.l23_pyr_size = self.l23_size
        self.l5_pyr_size = self.l5_size
        self.l6a_pyr_size = self.l6a_size
        self.l6b_pyr_size = self.l6b_size

        # FSI counts (25% of pyramidal = 20% of total)
        # FSIs generate gamma through fast rhythmic inhibition
        self.l4_fsi_size = max(int(self.l4_size * 0.25), 10)  # Minimum 10 FSI
        self.l23_fsi_size = max(int(self.l23_size * 0.25), 10)
        self.l5_fsi_size = max(int(self.l5_size * 0.25), 10)
        self.l6a_fsi_size = max(int(self.l6a_size * 0.25), 10)
        self.l6b_fsi_size = max(int(self.l6b_size * 0.25), 10)

        # =====================================================================
        # INITIALIZE STATE FIELDS
        # =====================================================================
        # Spikes per layer (initialized in _init_layers)
        self.l23_spikes: Optional[torch.Tensor] = None  # L2/3 processing layer
        self.l4_spikes: Optional[torch.Tensor] = None  # L4 input layer
        self.l5_spikes: Optional[torch.Tensor] = None  # L5 output layer
        self.l6a_spikes: Optional[torch.Tensor] = None  # L6a → TRN pathway
        self.l6b_spikes: Optional[torch.Tensor] = None  # L6b → relay pathway

        # L2/3 recurrent activity (accumulated over time)
        self.l23_recurrent_activity: Optional[torch.Tensor] = None

        # =====================================================================
        # INITIALIZE SUBCOMPONENTS
        # =====================================================================
        # Initialize layers
        self._init_layers()

        # Initialize inter-layer weights
        self._init_weights()

        # Initialize stimulus gating (transient inhibition) - always enabled
        self.stimulus_gating = StimulusGating(
            threshold=config.ffi_threshold,
            max_inhibition=config.ffi_strength * 10.0,
            decay_rate=1.0 - (1.0 / config.ffi_tau),
        )

        # STDP+BCM learning strategies for each layer
        # Custom STDP config using biologically appropriate amplitudes from config
        stdp_cfg = STDPConfig(
            learning_rate=config.learning_rate,
            a_plus=config.a_plus,
            a_minus=config.a_minus,
            tau_plus=config.tau_plus_ms,
            tau_minus=config.tau_minus_ms,
            w_min=config.w_min,
            w_max=config.w_max,
        )

        # BCM config for homeostatic modulation
        # CRITICAL: BCM LR must be much slower than STDP to prevent runaway potentiation
        # Use 2% of STDP learning rate (50:1 ratio) to reduce BCM influence
        # Previous 10:1 ratio caused runaway depression overwhelming STDP
        bcm_cfg = BCMConfig(
            learning_rate=config.learning_rate * 0.02,  # 50x slower than STDP
            w_min=config.w_min,
            w_max=config.w_max,
            tau_theta=100000.0,  # Very slow threshold adaptation (100s)
            theta_init=0.01,  # Higher initial threshold
            theta_max=0.5,  # Prevent saturation
            weight_decay=0.0002,  # L2 toward zero (proper weight decay)
            activity_threshold=0.001,  # 0.1% - allows LTD in sparse networks
        )

        # Create BCM+STDP strategies for each layer
        self.bcm_l23: LearningStrategy = CompositeStrategy([STDPStrategy(stdp_cfg), BCMStrategy(bcm_cfg)])
        self.bcm_l4: LearningStrategy = CompositeStrategy([STDPStrategy(stdp_cfg), BCMStrategy(bcm_cfg)])
        self.bcm_l5: LearningStrategy = CompositeStrategy([STDPStrategy(stdp_cfg), BCMStrategy(bcm_cfg)])
        self.bcm_l6a: LearningStrategy = CompositeStrategy([STDPStrategy(stdp_cfg), BCMStrategy(bcm_cfg)])
        self.bcm_l6b: LearningStrategy = CompositeStrategy([STDPStrategy(stdp_cfg), BCMStrategy(bcm_cfg)])

        # Intrinsic plasticity tracking (initialized in _init_layers)
        self._l23_threshold_offset: Optional[torch.Tensor] = None
        self._l23_activity_history: Optional[torch.Tensor] = None

        # Homeostasis for synaptic scaling and intrinsic plasticity
        # Weight budget computed from total L4 synaptic connections (across all sources)
        self.homeostasis = UnifiedHomeostasis(UnifiedHomeostasisConfig(
            weight_budget=config.weight_budget * self.l4_size,  # Based on L4 capacity
            w_min=config.w_min,
            w_max=config.w_max,
            soft_normalization=config.soft_normalization,
            normalization_rate=config.normalization_rate,
            activity_target=config.activity_target,
            device=str(self.device),
        ))

        # =====================================================================
        # POST-INITIALIZATION
        # =====================================================================
        self.__post_init__()

    def _init_layers(self) -> None:
        """Initialize conductance-based LIF neurons for each layer.

        Uses ConductanceLIF for biologically realistic gain control:
        - Separate excitatory and inhibitory conductances
        - Shunting inhibition (divisive effect)
        - Natural saturation at reversal potentials
        - No need for artificial divisive normalization

        **Layer-Specific Heterogeneity**:
        Applies distinct electrophysiological properties per layer
        (tau_mem, v_threshold, adaptation) reflecting biological
        diversity of cortical cell types.
        """
        cfg = self.config

        # =====================================================================
        # LAYER-SPECIFIC HETEROGENEITY
        # =====================================================================
        # Prepare layer-specific overrides if heterogeneity enabled
        l23_overrides = {}
        l4_overrides = {}
        l5_overrides = {}
        l6a_overrides = {}
        l6b_overrides = {}

        # L2/3: Integration and association
        l23_overrides = {
            "tau_mem": cfg.layer_tau_mem[CortexLayer.L23],
            "v_threshold": cfg.layer_v_threshold[CortexLayer.L23],
            "adapt_increment": cfg.layer_adaptation[CortexLayer.L23],
        }

        # L4: Fast sensory processing
        l4_overrides = {
            "tau_mem": cfg.layer_tau_mem[CortexLayer.L4],
            "v_threshold": cfg.layer_v_threshold[CortexLayer.L4],
            "adapt_increment": cfg.layer_adaptation[CortexLayer.L4],
        }

        # L5: Output generation
        l5_overrides = {
            "tau_mem": cfg.layer_tau_mem[CortexLayer.L5],
            "v_threshold": cfg.layer_v_threshold[CortexLayer.L5],
            "adapt_increment": cfg.layer_adaptation[CortexLayer.L5],
        }

        # L6a: TRN feedback (low gamma)
        l6a_overrides = {
            "tau_mem": cfg.layer_tau_mem[CortexLayer.L6A],
            "v_threshold": cfg.layer_v_threshold[CortexLayer.L6A],
            "adapt_increment": cfg.layer_adaptation[CortexLayer.L6A],
        }

        # L6b: Relay feedback (high gamma)
        l6b_overrides = {
            "tau_mem": cfg.layer_tau_mem[CortexLayer.L6B],
            "v_threshold": cfg.layer_v_threshold[CortexLayer.L6B],
            "adapt_increment": cfg.layer_adaptation[CortexLayer.L6B],
        }

        # Create pyramidal neurons using factory functions with heterogeneous properties
        self.l4_neurons = NeuronFactory.create_cortical_layer_neurons(
            self.l4_pyr_size, "L4", self.device, **l4_overrides
        )
        self.l23_neurons = NeuronFactory.create_cortical_layer_neurons(
            self.l23_pyr_size,
            "L2/3",
            self.device,
            adapt_increment=l23_overrides["adapt_increment"],
            tau_adapt=cfg.adapt_tau,
            **({k: v for k, v in l23_overrides.items() if k != "adapt_increment"}),
        )
        self.l5_neurons = NeuronFactory.create_cortical_layer_neurons(
            self.l5_pyr_size, "L5", self.device, **l5_overrides
        )

        # L6 split into two subtypes:
        # - L6a (corticothalamic type I): Projects to TRN (inhibitory modulation)
        # - L6b (corticothalamic type II): Projects to relay (excitatory modulation)
        self.l6a_neurons = NeuronFactory.create_cortical_layer_neurons(
            self.l6a_pyr_size, "L6a", self.device, **l6a_overrides
        )
        self.l6b_neurons = NeuronFactory.create_cortical_layer_neurons(
            self.l6b_pyr_size, "L6b", self.device, **l6b_overrides
        )

        # =====================================================================
        # EXPLICIT INHIBITORY NETWORKS (PV, SST, VIP INTERNEURONS)
        # =====================================================================
        # Replace old FSI-only approach with explicit inhibitory networks containing
        # multiple cell types: PV (basket), SST (Martinotti), and VIP (disinhibitory)
        # Each layer gets its own inhibitory network with E→I, I→E, and I→I connectivity

        # L4 inhibitory network
        self.l4_inhibitory = InhibitoryNetwork(
            layer_name="L4",
            pyr_size=self.l4_pyr_size,
            total_inhib_fraction=0.25,  # 25% of pyr = 20% of total
            device=str(self.device),
            dt_ms=cfg.dt_ms,
        )
        self.l4_fsi_size = self.l4_inhibitory.get_total_size()  # For backward compat

        # L2/3 inhibitory network
        self.l23_inhibitory = InhibitoryNetwork(
            layer_name="L2/3",
            pyr_size=self.l23_pyr_size,
            total_inhib_fraction=0.25,
            device=str(self.device),
            dt_ms=cfg.dt_ms,
        )
        self.l23_fsi_size = self.l23_inhibitory.get_total_size()

        # L5 inhibitory network
        self.l5_inhibitory = InhibitoryNetwork(
            layer_name="L5",
            pyr_size=self.l5_pyr_size,
            total_inhib_fraction=0.25,
            device=str(self.device),
            dt_ms=cfg.dt_ms,
        )
        self.l5_fsi_size = self.l5_inhibitory.get_total_size()

        # L6a inhibitory network
        self.l6a_inhibitory = InhibitoryNetwork(
            layer_name="L6a",
            pyr_size=self.l6a_pyr_size,
            total_inhib_fraction=0.25,
            device=str(self.device),
            dt_ms=cfg.dt_ms,
        )
        self.l6a_fsi_size = self.l6a_inhibitory.get_total_size()

        # L6b inhibitory network
        self.l6b_inhibitory = InhibitoryNetwork(
            layer_name="L6b",
            pyr_size=self.l6b_pyr_size,
            total_inhib_fraction=0.25,
            device=str(self.device),
            dt_ms=cfg.dt_ms,
        )
        self.l6b_fsi_size = self.l6b_inhibitory.get_total_size()

        # =====================================================================
        # GAP JUNCTIONS for L2/3 interneuron synchronization
        # =====================================================================
        # Basket cells and chandelier cells have dense gap junction networks
        # Critical for cortical gamma oscillations (30-80 Hz) and precise timing
        # ~70-80% of cortical gap junctions are interneuron-interneuron
        # TODO: This is never instantiated. Need to create after inhibitory networks are initialized to determine connectivity.
        self.gap_junctions_l23: Optional[GapJunctionCoupling] = None

        # =====================================================================
        # HOMEOSTATIC INTRINSIC PLASTICITY (Adaptive Gain Control)
        # =====================================================================
        # Like thalamus, cortex adapts neuron gains to maintain target firing rate.
        # Prevents complete silencing during early learning.
        self._target_rate = cfg.target_firing_rate
        self._gain_lr = cfg.gain_learning_rate

        # Compute decay factor for firing rate averaging
        self._firing_rate_alpha = cfg.dt_ms / cfg.gain_tau_ms

        # Adaptive threshold plasticity (complementary to gain adaptation)
        self._threshold_lr = cfg.threshold_learning_rate
        self._threshold_min = cfg.threshold_min
        self._threshold_max = cfg.threshold_max

        # Per-layer firing rate trackers (EMA) - pyramidal neurons only
        self.register_buffer("l4_firing_rate", torch.zeros(self.l4_size, device=self.device))
        self.register_buffer("l23_firing_rate", torch.zeros(self.l23_size, device=self.device))
        self.register_buffer("l5_firing_rate", torch.zeros(self.l5_size, device=self.device))

        # Per-layer adaptive gains (start at 1.0, will adapt based on activity)
        # Pyramidal neurons only (FSIs have fixed fast dynamics)
        self.l4_gain = nn.Parameter(torch.ones(self.l4_size, device=self.device, requires_grad=False))
        self.l23_gain = nn.Parameter(torch.ones(self.l23_size, device=self.device, requires_grad=False))
        self.l5_gain = nn.Parameter(torch.ones(self.l5_size, device=self.device, requires_grad=False))

        # =====================================================================
        # SYNAPTIC SCALING (Global Multiplicative Scaling)
        # =====================================================================
        # Biology: Chronically underactive neurons scale up ALL input synapses
        # This is distinct from Hebbian learning (input-specific) and works with
        # gain adaptation. Turrigiano & Nelson 2004. Critical for bootstrap.
        self._synaptic_scaling_enabled = cfg.synaptic_scaling_enabled
        self._synaptic_scaling_lr = cfg.synaptic_scaling_lr
        self._synaptic_scaling_min_activity = cfg.synaptic_scaling_min_activity
        self._synaptic_scaling_max_factor = cfg.synaptic_scaling_max_factor

        # Per-layer synaptic scaling factors (multiplicative, start at 1.0)
        self.l4_weight_scale = nn.Parameter(torch.ones(1, device=self.device, requires_grad=False))
        self.l23_weight_scale = nn.Parameter(torch.ones(1, device=self.device, requires_grad=False))
        self.l5_weight_scale = nn.Parameter(torch.ones(1, device=self.device, requires_grad=False))

        # =====================================================================
        # INTER-LAYER AXONAL DELAYS (using CircularDelayBuffer utility)
        # =====================================================================
        # Create delay buffers for biological signal propagation within layers
        # L4→L2/3 delay: Short vertical projection (~2ms biologically)
        # L2/3→L5: Longer vertical projection (~2ms biologically)
        # L2/3→L6a/L6b: Within column (~2ms biologically)

        # Calculate delay steps
        self._l4_l23_delay_steps: int = int(cfg.l4_to_l23_delay_ms / cfg.dt_ms)
        self._l23_l5_delay_steps: int = int(cfg.l23_to_l5_delay_ms / cfg.dt_ms)
        self._l23_l6a_delay_steps: int = int(cfg.l23_to_l6a_delay_ms / cfg.dt_ms)
        self._l23_l6b_delay_steps: int = int(cfg.l23_to_l6b_delay_ms / cfg.dt_ms)
        self._l5_l4_delay_steps: int = int(cfg.l5_to_l4_delay_ms / cfg.dt_ms)
        self._l6_l4_delay_steps: int = int(cfg.l6_to_l4_delay_ms / cfg.dt_ms)

        # Initialize CircularDelayBuffer for each pathway
        self._l4_l23_buffer = CircularDelayBuffer(
            max_delay=self._l4_l23_delay_steps,
            size=self.l4_size,
            device=str(self.device),
            dtype=torch.bool,
        )
        self._l23_l5_buffer = CircularDelayBuffer(
            max_delay=self._l23_l5_delay_steps,
            size=self.l23_size,
            device=str(self.device),
            dtype=torch.bool,
        )
        self._l23_l6a_buffer = CircularDelayBuffer(
            max_delay=self._l23_l6a_delay_steps,
            size=self.l23_size,
            device=str(self.device),
            dtype=torch.bool,
        )
        self._l23_l6b_buffer = CircularDelayBuffer(
            max_delay=self._l23_l6b_delay_steps,
            size=self.l23_size,
            device=str(self.device),
            dtype=torch.bool,
        )

        # Predictive coding feedback delays (L5/L6 → L4)
        self._l5_l4_buffer = CircularDelayBuffer(
            max_delay=self._l5_l4_delay_steps,
            size=self.l5_size,
            device=str(self.device),
            dtype=torch.bool,
        )
        self._l6_l4_buffer = CircularDelayBuffer(
            max_delay=self._l6_l4_delay_steps,
            size=self.l6a_size + self.l6b_size,  # Combined L6
            device=str(self.device),
            dtype=torch.bool,
        )

    def _init_weights(self) -> None:
        """Initialize inter-layer weight matrices.

        Feedforward weights need to be strong enough to propagate sparse activity.
        With low initial activity, we use generous weight initialization to ensure
        signal propagation through the cortical layers.
        """
        # Expected number of active inputs given sparsity
        expected_active_l4 = max(1, int(self.l4_size * self.config.l4_sparsity))
        expected_active_l23 = max(1, int(self.l23_size * self.config.l23_sparsity))

        # Feedforward weights: Strong initialization for reliable propagation
        # Scale factor multiplier to ensure signal gets through even with sparse activity
        boost_factor = 10.0  # Multiply standard scaling by 10x for initial reliability

        # L4 → L2/3 needs extra boost to overcome dead layer problem
        # Diagnostic showed L2/3 firing 0% with default boost_factor
        # Increasing to 50x enables reliable L2/3 activation and BCM learning
        l4_l23_boost = 50.0  # 5x stronger than other inter-layer connections

        # L4 → L2/3: positive excitatory weights (AT L2/3 DENDRITES)
        w_scale_l4_l23 = l4_l23_boost / expected_active_l4
        l4_l23_weights = torch.abs(
            WeightInitializer.gaussian(
                n_output=self.l23_size,
                n_input=self.l4_size,
                mean=0.0,
                std=w_scale_l4_l23,
                device=self.device,
            )
        )
        self.synaptic_weights["l4_l23"] = nn.Parameter(l4_l23_weights)

        # L2/3 → L5: positive excitatory weights (AT L5 DENDRITES)
        w_scale_l23_l5 = boost_factor / expected_active_l23
        l23_l5_weights = torch.abs(
            WeightInitializer.gaussian(
                n_output=self.l5_size,
                n_input=self.l23_size,
                mean=0.0,
                std=w_scale_l23_l5,
                device=self.device,
            )
        )
        self.synaptic_weights["l23_l5"] = nn.Parameter(l23_l5_weights)

        # L2/3 → L6a: positive excitatory weights (corticothalamic type I → TRN) (AT L6a DENDRITES)
        w_scale_l23_l6a = boost_factor / expected_active_l23
        l23_l6a_weights = torch.abs(
            WeightInitializer.gaussian(
                n_output=self.l6a_size,
                n_input=self.l23_size,
                mean=0.0,
                std=w_scale_l23_l6a,
                device=self.device,
            )
        )
        self.synaptic_weights["l23_l6a"] = nn.Parameter(l23_l6a_weights)

        # L2/3 → L6b: positive excitatory weights (corticothalamic type II → relay) (AT L6b DENDRITES)
        w_scale_l23_l6b = boost_factor / expected_active_l23
        l23_l6b_weights = torch.abs(
            WeightInitializer.gaussian(
                n_output=self.l6b_size,
                n_input=self.l23_size,
                mean=0.0,
                std=w_scale_l23_l6b,
                device=self.device,
            )
        )
        self.synaptic_weights["l23_l6b"] = nn.Parameter(l23_l6b_weights)

        # =====================================================================
        # PREDICTIVE CODING: L5/L6 → L4 FEEDBACK
        # =====================================================================
        # Deep layers generate predictions that inhibit L4 when correct
        # This makes L4 naturally compute prediction errors (input - prediction)
        # L2/3 then propagates these errors up the hierarchy

        # L5 → L4 inhibitory prediction (AT L4 DENDRITES)
        # Anti-Hebbian: When both fire, prediction was wrong, strengthen inhibition
        # Start with weak inhibition to allow initial activity to develop
        l5_l4_weights = torch.abs(
            WeightInitializer.gaussian(
                n_output=self.l4_size,
                n_input=self.l5_size,
                mean=0.0,
                std=0.01,  # Very weak initially to avoid suppressing L4
                device=self.device,
            )
        )
        self.synaptic_weights["l5_l4_pred"] = nn.Parameter(l5_l4_weights)

        # L6 (combined) → L4 inhibitory prediction (AT L4 DENDRITES)
        l6_combined_size = self.l6a_size + self.l6b_size
        l6_l4_weights = torch.abs(
            WeightInitializer.gaussian(
                n_output=self.l4_size,
                n_input=l6_combined_size,
                mean=0.0,
                std=0.01,  # Very weak initially to avoid suppressing L4
                device=self.device,
            )
        )
        self.synaptic_weights["l6_l4_pred"] = nn.Parameter(l6_l4_weights)

    # =========================================================================
    # SYNAPTIC WEIGHT MANAGEMENT
    # =========================================================================

    def add_input_source(
        self,
        source_name: SpikesSourceKey,
        target_layer: LayerName,
        n_input: int,
        sparsity: float = 0.5,  # Biological sparsity - 50% connectivity
        weight_scale: float = 3.0,  # Strong weights - ensures reliable L4 activation
    ) -> None:
        """Add synaptic weights for a new input source."""
        super().add_input_source(
            source_name=source_name,
            target_layer=target_layer,
            n_input=n_input,
            sparsity=sparsity,
            weight_scale=weight_scale,
        )

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def _forward_internal(self, inputs: RegionSpikesDict) -> None:
        """Process input through layered cortical circuit."""
        cfg = self.config

        # =====================================================================
        # SEPARATE EXTERNAL vs CORTICAL FEEDBACK INPUTS
        # =====================================================================
        # External inputs have source names ("thalamus:relay", "hippocampus:ca1")
        # Cortical recurrent uses this region's name ("cortex:l23", "cortex:l5", etc.)
        # External inputs have learnable synaptic weights that need plasticity updates
        # Cortical recurrent connections are handled separately (not in external plasticity)
        external_inputs = {
            name: spikes
            for name, spikes in inputs.items()
            if name in self.synaptic_weights and not name.startswith("cortex:")
        }
        cortical_feedback = {
            name: spikes
            for name, spikes in inputs.items()
            if name.startswith("cortex:") or name not in self.synaptic_weights
        }

        # =====================================================================
        # ALPHA-BASED ATTENTION GATING
        # =====================================================================
        # Alpha oscillations (8-13 Hz) suppress processing in cortical areas.
        # High alpha = attention directed elsewhere (suppress this region)
        # Low alpha = attention focused here (normal processing)
        #
        # Biological basis: Alpha power is inversely related to cortical
        # excitability. Regions with high alpha are "idling" or suppressed
        # to prevent interference with attended regions.

        # Alpha signal ranges [-1, 1], convert to suppression [0, 0.5]
        # High positive alpha (near 1.0) → max suppression (50%)
        # Low/negative alpha → minimal suppression
        alpha_magnitude = max(0.0, self._alpha_signal)  # Only positive values
        alpha_suppression = 1.0 - (alpha_magnitude * 0.5)  # Scale to 50-100%

        # =====================================================================
        # MULTI-SOURCE ROUTING: L4 (bottom-up) vs L2/3 (top-down)
        # =====================================================================
        # Biology: Different cortical layers receive inputs from different sources
        # - L4 receives thalamic/sensory inputs (bottom-up)
        # - L2/3 receives feedback from higher cortical areas (top-down)

        # Filter inputs to exclude L2/3 direct inputs (those will be processed later)
        # Use target_layer metadata (biological: thalamus→L4, PFC→L2/3)
        l4_inputs = {
            name: spikes
            for name, spikes in external_inputs.items()
            if self._source_target_layers.get(name) == "l4"
        }

        # Use base class helper with alpha suppression modulation
        def alpha_gate(spikes: torch.Tensor, source_name: str) -> torch.Tensor:
            """Apply alpha attention gating to each source."""
            return spikes * alpha_suppression

        l4_g_exc_full = self._integrate_multi_source_synaptic_inputs(
            inputs=l4_inputs,
            n_neurons=self.l4_size,  # Full layer size (pyramidal + FSI)
            weight_key_suffix="",
            apply_stp=False,  # Cortex doesn't use per-source STP (only recurrent)
            modulation_fn=alpha_gate,  # Apply alpha suppression per source
        )

        # Split input between pyramidal and FSI neurons
        l4_g_exc = l4_g_exc_full[:self.l4_pyr_size]  # Pyramidal neurons
        l4_fsi_g_exc_input = l4_g_exc_full[:self.l4_fsi_size]  # FSI neurons (broadcast from first part)

        # Apply global scaling to pyramidal (theta emerges from circuit dynamics)
        l4_g_exc = l4_g_exc * cfg.input_to_l4_strength

        # Add baseline noise (spontaneous miniature EPSPs)
        if cfg.baseline_noise_current > 0:
            l4_g_exc = l4_g_exc + cfg.baseline_noise_current

        # Apply homeostatic gain adaptation (like thalamus)
        l4_g_exc = l4_g_exc * self.l4_gain

        # Inhibitory conductance: ~25% of excitatory (4:1 E/I ratio)
        # This provides feedback inhibition for gain control
        l4_g_inh = l4_g_exc * 0.25

        # =====================================================================
        # PREDICTIVE CODING: L5/L6 → L4 INHIBITORY PREDICTIONS
        # =====================================================================
        # Deep layers predict what L4 should receive (from previous timestep)
        # Good prediction → strong inhibition → L4 silent (no error)
        # Bad prediction → weak inhibition → L4 fires (error signal)
        # This makes L4 naturally compute: error = input - prediction
        #
        # PRECISION WEIGHTING:
        # Predictions are weighted by confidence (population activity level):
        # - High activity in deep layers → strong, confident prediction
        # - Low activity in deep layers → weak, uncertain prediction
        # This allows attention-like modulation of prediction strength.

        # L5 → L4 prediction (with delay)
        if self.l5_spikes is not None:
            if self._l5_l4_delay_steps > 0:
                self._l5_l4_buffer.write(self.l5_spikes)
                l5_delayed = self._l5_l4_buffer.read(self._l5_l4_delay_steps)
                self._l5_l4_buffer.advance()
            else:
                l5_delayed = self.l5_spikes

            # Apply L5 prediction as inhibition
            pred_from_l5 = torch.matmul(
                self.synaptic_weights["l5_l4_pred"],
                l5_delayed.float()
            )

            # Precision weighting: Scale by L5 activity (confidence)
            l5_activity = torch.mean(l5_delayed.float()).item()  # Population activity [0, 1]
            # Map activity to precision weight [precision_min, precision_max]
            l5_precision = cfg.precision_min + (cfg.precision_max - cfg.precision_min) * l5_activity
            pred_from_l5 = pred_from_l5 * l5_precision

            l4_g_inh = l4_g_inh + pred_from_l5 * cfg.l5_to_l4_pred_strength

        # L6 (combined) → L4 prediction (with delay)
        if self.l6a_spikes is not None and self.l6b_spikes is not None:
            l6_combined = torch.cat([self.l6a_spikes, self.l6b_spikes], dim=-1)

            if self._l6_l4_delay_steps > 0:
                self._l6_l4_buffer.write(l6_combined)
                l6_delayed = self._l6_l4_buffer.read(self._l6_l4_delay_steps)
                self._l6_l4_buffer.advance()
            else:
                l6_delayed = l6_combined

            # Apply L6 prediction as inhibition
            pred_from_l6 = torch.matmul(
                self.synaptic_weights["l6_l4_pred"],
                l6_delayed.float()
            )

            # Precision weighting: Scale by L6 activity (confidence)
            l6_activity = torch.mean(l6_delayed.float()).item()  # Population activity [0, 1]
            # Map activity to precision weight [precision_min, precision_max]
            l6_precision = cfg.precision_min + (cfg.precision_max - cfg.precision_min) * l6_activity
            pred_from_l6 = pred_from_l6 * l6_precision

            l4_g_inh = l4_g_inh + pred_from_l6 * cfg.l6_to_l4_pred_strength

        # =====================================================================
        # NOREPINEPHRINE GAIN MODULATION (Locus Coeruleus)
        # =====================================================================
        # High NE (arousal/uncertainty): Increase gain → more responsive
        # Low NE (baseline): Normal gain
        # Biological: β-adrenergic receptors increase neuronal excitability
        ne_level = self.neuromodulator_state.norepinephrine
        # NE gain: 1.0 (baseline) to 1.5 (high arousal)
        ne_gain = compute_ne_gain(ne_level)
        l4_g_exc = l4_g_exc * ne_gain

        # ConductanceLIF automatically handles shunting inhibition
        l4_spikes, l4_membrane = self.l4_neurons(l4_g_exc, l4_g_inh)
        l4_spikes = self._apply_sparsity(l4_spikes, cfg.l4_sparsity)

        # =====================================================================
        # L4 INHIBITORY NETWORK (PV, SST, VIP INTERNEURONS)
        # =====================================================================
        # Explicit inhibitory network with multiple cell types:
        # - PV (basket cells): Fast perisomatic inhibition → gamma oscillations
        # - SST (Martinotti): Slower dendritic inhibition → feedback modulation
        # - VIP: Disinhibitory neurons → attention/ACh-gated disinhibition

        # Run inhibitory network (E→I, I→E, I→I connectivity)
        l4_inhib_output = self.l4_inhibitory(
            pyr_spikes=l4_spikes,
            pyr_membrane=l4_membrane,
            external_excitation=l4_g_exc,  # Pass pyramidal excitation, not FSI slice
            acetylcholine=self.neuromodulator_state.acetylcholine,
        )

        # Apply inhibition to pyramidal spikes
        # PV cells create gamma modulation through fast rhythmic inhibition
        perisomatic_inh = l4_inhib_output["perisomatic_inhibition"]  # From PV cells
        _dendritic_inh = l4_inhib_output["dendritic_inhibition"]  # From SST cells

        # Gamma gating: PV inhibition creates rhythmic suppression
        l4_gamma_modulation = 1.0 / (1.0 + perisomatic_inh * 0.5)
        l4_spikes = (l4_spikes.float() * l4_gamma_modulation > 0.5).bool()

        # Homeostatic gain adaptation: Update L4 firing rate and adjust gains
        # Update exponential moving average of firing rate
        current_rate = l4_spikes.float()  # [l4_size], instantaneous rate (0 or 1)
        self.l4_firing_rate.mul_(1.0 - self._firing_rate_alpha).add_(
            current_rate * self._firing_rate_alpha
        )

        # Compute gain update: increase gain when underactive, decrease when overactive
        rate_error = self._target_rate - self.l4_firing_rate  # [l4_size]
        gain_update = self._gain_lr * rate_error  # [l4_size]

        # Apply update with minimum floor to prevent negative gains
        self.l4_gain.data.add_(gain_update).clamp_(min=0.001)

        # Adaptive threshold update (using same rate_error)
        # Lower threshold when underactive, raise when overactive
        threshold_update = -self._threshold_lr * rate_error  # [l4_size]
        self.l4_neurons.v_threshold.data.add_(threshold_update).clamp_(
            min=self._threshold_min, max=self._threshold_max
        )

        # =====================================================================
        # SYNAPTIC SCALING (Multiplicative Homeostasis)
        # =====================================================================
        # Biology: Chronically underactive layers scale UP all input weights
        # This complements gain adaptation and helps bootstrap from silence
        # Turrigiano & Nelson 2004: "Homeostatic Plasticity in the Developing NS"
        if self._synaptic_scaling_enabled:
            # Compute layer-wide average activity (not per-neuron)
            layer_avg_rate = self.l4_firing_rate.mean()

            # Scale up weights when chronically below threshold
            if layer_avg_rate < self._synaptic_scaling_min_activity:
                # Compute scaling update (slow, multiplicative)
                rate_deficit = self._synaptic_scaling_min_activity - layer_avg_rate
                scale_update = self._synaptic_scaling_lr * rate_deficit

                # Apply multiplicative scaling (1.0 -> 1.001 -> 1.002, etc.)
                self.l4_weight_scale.data.mul_(1.0 + scale_update).clamp_(
                    min=1.0, max=self._synaptic_scaling_max_factor
                )

                # Scale ALL input weights to L4 (global, non-specific)
                for source_name in inputs.keys():
                    if source_name in self.synaptic_weights:
                        self.synaptic_weights[source_name].data.mul_(1.0 + scale_update)

        # Inter-layer shape check: L4 output (pyramidal only)
        assert l4_spikes.shape == (self.l4_pyr_size,), (
            f"Cortex: L4 spikes have shape {l4_spikes.shape} "
            f"but expected ({self.l4_pyr_size},). "
            f"Check L4 sparsity or input→L4 weights shape."
        )

        # =====================================================================
        # APPLY L4→L2/3 AXONAL DELAY
        # =====================================================================
        # Apply biological transmission delay for L4→L2/3 vertical projection
        # If delay is 0, l4_spikes_delayed = l4_spikes
        if self._l4_l23_delay_steps > 0:
            self._l4_l23_buffer.write(l4_spikes)
            l4_spikes_delayed = self._l4_l23_buffer.read(self._l4_l23_delay_steps)
            self._l4_l23_buffer.advance()
        else:
            l4_spikes_delayed = l4_spikes

        # L2/3: Processing with recurrence
        # NOTE: Use delayed L4 spikes for biological accuracy
        l23_ff = (
            torch.matmul(l4_spikes_delayed.float(), self.synaptic_weights["l4_l23"].t())
            * cfg.l4_to_l23_strength
        )

        # Stimulus gating (transient inhibition - always enabled)
        # Compute total input activity from all sources
        # Use l4_size as consistent representation for gating (biological: gating happens at L4)
        if inputs:
            # Sum activity across all sources and distribute over l4 neurons
            total_activity = sum(s.float().sum() for s in inputs.values())
            # Normalize by l4_size to get average activity per neuron
            gating_input = torch.full((self.l4_size,), total_activity / self.l4_size, device=self.device)
        else:
            gating_input = torch.zeros(self.l4_size, device=self.device)

        ffi = self.stimulus_gating.compute(gating_input, return_tensor=False)
        raw_ffi = ffi.item() if hasattr(ffi, "item") else float(ffi)
        effective_ffi_strength = min(1.0, raw_ffi / self.stimulus_gating.max_inhibition) * cfg.ffi_strength
        ffi_suppression = 1.0 - effective_ffi_strength

        # =====================================================================
        # RECURRENT PROCESSING IN L2/3 (Local Horizontal Connections)
        # =====================================================================
        # ACh modulation applied here (region-level neuromodulation):
        # High ACh (encoding): Suppress recurrence to prevent interference
        # Low ACh (retrieval): Enable recurrence for pattern completion
        ach_level = self.neuromodulator_state.acetylcholine
        ach_recurrent_modulation = compute_ach_recurrent_suppression(ach_level)

        # Get recurrent input from external connection if present
        if "l23_recurrent" in inputs:
            # Already integrated by parent's multi-source integration
            # Extract just this input for proper routing
            l23_rec_raw = self._integrate_multi_source_synaptic_inputs(
                inputs={"l23_recurrent": inputs["l23_recurrent"]},
                n_neurons=self.l23_size,
                weight_key_suffix="_l23",
                apply_stp=False,  # STP already applied at AxonalProjection
            )
            l23_rec = (
                l23_rec_raw
                * cfg.l23_recurrent_strength
                * 0.5  # recurrent_scale
                * ffi_suppression
                * ach_recurrent_modulation
            )
        else:
            l23_rec = torch.zeros_like(l23_ff)

        # =====================================================================
        # TOP-DOWN MODULATION TO L2/3 (Multi-Source)
        # =====================================================================
        # Biology: L2/3 receives direct feedback from higher cortical areas
        # (e.g., prefrontal, parietal) that modulates processing without going
        # through L4. This implements predictive coding and attentional modulation.

        # Filter inputs to only those targeting L2/3 (top-down modulation)
        # Use target_layer metadata (biological: PFC→L2/3 apical dendrites)
        l23_td_inputs = {
            name: spikes
            for name, spikes in external_inputs.items()
            if self._source_target_layers.get(name) == "l23"
        }

        # Use base class helper for top-down integration
        # Note: No alpha suppression on top-down (already processed by PFC)
        l23_td = self._integrate_multi_source_synaptic_inputs(
            inputs=l23_td_inputs,
            n_neurons=self.l23_size,
            weight_key_suffix="",  # No suffix - weights keyed by source_name directly
            apply_stp=False,
        )

        # Apply global scaling
        l23_td = l23_td * cfg.l23_top_down_strength

        # Integrate all L2/3 inputs
        l23_input = l23_ff + l23_rec + l23_td

        # GAP JUNCTION COUPLING (L2/3 interneuron synchronization)
        # Ultra-fast electrical coupling for gamma synchronization
        # Apply gap junction current based on previous timestep's membrane potentials
        if self.gap_junctions_l23 is not None and self.l23_neurons.membrane is not None:
            # Get coupling current from neighboring interneurons
            gap_current = self.gap_junctions_l23(self.l23_neurons.membrane)
            # Add gap junction depolarization to input
            l23_input = l23_input + gap_current

        # Add baseline noise (spontaneous miniature EPSPs)
        if cfg.baseline_noise_current > 0:
            l23_input = l23_input + cfg.baseline_noise_current

        # Apply homeostatic gain adaptation
        l23_input = l23_input * self.l23_gain

        # INTRINSIC PLASTICITY: Apply per-neuron threshold adjustment (UnifiedHomeostasis)
        # Under-firing neurons get lower thresholds (more excitable)
        # Over-firing neurons get higher thresholds (less excitable)
        # This is applied temporarily for this timestep only - doesn't modify stored thresholds
        cfg = self.config
        if self._l23_threshold_offset is not None:
            # Temporarily adjust thresholds for this forward pass
            self.l23_neurons.adjust_thresholds(
                self._l23_threshold_offset,
                min_threshold=self._threshold_min,
                max_threshold=self._threshold_max,
            )

        l23_spikes, l23_membrane = self.l23_neurons(F.relu(l23_input), F.relu(l23_input) * 0.25)
        l23_spikes = self._apply_sparsity(l23_spikes, cfg.l23_sparsity)

        # =====================================================================
        # L2/3 INHIBITORY NETWORK (PV, SST, VIP INTERNEURONS)
        # =====================================================================
        # Primary site of gamma oscillations and attentional modulation
        # VIP cells provide ACh-gated disinhibition for selective attention

        # Run inhibitory network
        l23_inhib_output = self.l23_inhibitory(
            pyr_spikes=l23_spikes,
            pyr_membrane=l23_membrane,
            external_excitation=l23_input,
            acetylcholine=self.neuromodulator_state.acetylcholine,
        )

        # Apply inhibition (PV for gamma, SST for feedback modulation)
        perisomatic_inh = l23_inhib_output["perisomatic_inhibition"]

        # Gamma gating from PV cells
        l23_gamma_modulation = 1.0 / (1.0 + perisomatic_inh * 0.5)
        l23_spikes = (l23_spikes.float() * l23_gamma_modulation) > 0.5

        # Homeostatic gain adaptation: Update L23 firing rate and adjust gains
        # Update exponential moving average of firing rate
        current_rate = l23_spikes.float()  # [l23_size], instantaneous rate (0 or 1)
        self.l23_firing_rate.mul_(1.0 - self._firing_rate_alpha).add_(
            current_rate * self._firing_rate_alpha
        )

        # Compute gain update: increase gain when underactive, decrease when overactive
        rate_error = self._target_rate - self.l23_firing_rate  # [l23_size]
        gain_update = self._gain_lr * rate_error  # [l23_size]

        # Apply update (no artificial bounds)
        self.l23_gain.data.add_(gain_update)

        # Adaptive threshold update (using same rate_error)
        # Lower threshold when underactive, raise when overactive
        threshold_update = -self._threshold_lr * rate_error  # [l23_size]
        self.l23_neurons.v_threshold.data.add_(threshold_update).clamp_(
            min=self._threshold_min, max=self._threshold_max
        )

        # Synaptic scaling for L2/3 (scale L4→L2/3 weights only)
        if self._synaptic_scaling_enabled:
            layer_avg_rate = self.l23_firing_rate.mean()
            if layer_avg_rate < self._synaptic_scaling_min_activity:
                rate_deficit = self._synaptic_scaling_min_activity - layer_avg_rate
                scale_update = self._synaptic_scaling_lr * rate_deficit
                self.l23_weight_scale.data.mul_(1.0 + scale_update).clamp_(
                    min=1.0, max=self._synaptic_scaling_max_factor
                )
                # Scale L4→L2/3 weights (recurrent weights are external now)
                self.synaptic_weights["l4_l23"].data.mul_(1.0 + scale_update)

        # Inter-layer shape check: L2/3 → L5
        assert l23_spikes.shape == (self.l23_size,), (
            f"Cortex: L2/3 spikes have shape {l23_spikes.shape} "
            f"but expected ({self.l23_size},). "
            f"Check L2/3 sparsity or L4→L2/3 weights shape."
        )

        # Update recurrent activity trace
        if self.l23_recurrent_activity is not None:
            self.l23_recurrent_activity = (
                self.l23_recurrent_activity * cfg.l23_recurrent_decay + l23_spikes.float()
            )
        else:
            self.l23_recurrent_activity = l23_spikes.float()

        # =====================================================================
        # APPLY L2/3→L5 AXONAL DELAY
        # =====================================================================
        # Apply biological transmission delay for L2/3→L5 vertical projection
        # If delay is 0, l23_spikes_delayed = l23_spikes (instant, backward compatible)
        if self._l23_l5_delay_steps > 0:
            self._l23_l5_buffer.write(l23_spikes)
            l23_spikes_delayed = self._l23_l5_buffer.read(self._l23_l5_delay_steps)
            self._l23_l5_buffer.advance()
        else:
            l23_spikes_delayed = l23_spikes

        # L5: Subcortical output (conductance-based)
        # NOTE: Use delayed L2/3 spikes for biological accuracy
        l5_g_exc = (
            torch.matmul(self.synaptic_weights["l23_l5"], l23_spikes_delayed.float())
            * cfg.l23_to_l5_strength
        )

        # Add baseline noise (spontaneous miniature EPSPs)
        if cfg.baseline_noise_current > 0:
            l5_g_exc = l5_g_exc + cfg.baseline_noise_current

        # Apply homeostatic gain adaptation
        l5_g_exc = l5_g_exc * self.l5_gain

        # L5 inhibition: ~25% of excitation (4:1 E/I ratio)
        l5_g_inh = l5_g_exc * 0.25

        l5_spikes, l5_membrane = self.l5_neurons(l5_g_exc, l5_g_inh)
        l5_spikes = self._apply_sparsity(l5_spikes, cfg.l5_sparsity)

        # =====================================================================
        # L5 INHIBITORY NETWORK (PV, SST, VIP INTERNEURONS)
        # =====================================================================
        # L5 inhibitory network for subcortical output pathway

        # Run inhibitory network
        l5_inhib_output = self.l5_inhibitory(
            pyr_spikes=l5_spikes,
            pyr_membrane=l5_membrane,
            external_excitation=l5_g_exc,
            acetylcholine=self.neuromodulator_state.acetylcholine,
        )

        # Apply gamma gating from PV cells
        perisomatic_inh = l5_inhib_output["perisomatic_inhibition"]
        l5_gamma_modulation = 1.0 / (1.0 + perisomatic_inh * 0.5)
        l5_spikes = (l5_spikes.float() * l5_gamma_modulation > 0.5).bool()

        # Homeostatic gain adaptation: Update L5 firing rate and adjust gains
        # Update exponential moving average of firing rate
        current_rate = l5_spikes.float()  # [l5_size], instantaneous rate (0 or 1)
        self.l5_firing_rate.mul_(1.0 - self._firing_rate_alpha).add_(
            current_rate * self._firing_rate_alpha
        )

        # Compute gain update: increase gain when underactive, decrease when overactive
        rate_error = self._target_rate - self.l5_firing_rate  # [l5_size]
        gain_update = self._gain_lr * rate_error  # [l5_size]

        # Apply update (no artificial bounds)
        self.l5_gain.data.add_(gain_update)

        # Adaptive threshold update (using same rate_error)
        # Lower threshold when underactive, raise when overactive
        threshold_update = -self._threshold_lr * rate_error  # [l5_size]
        self.l5_neurons.v_threshold.data.add_(threshold_update).clamp_(
            min=self._threshold_min, max=self._threshold_max
        )

        # Synaptic scaling for L5 (scale L2/3→L5 weights)
        if self._synaptic_scaling_enabled:
            layer_avg_rate = self.l5_firing_rate.mean()
            if layer_avg_rate < self._synaptic_scaling_min_activity:
                rate_deficit = self._synaptic_scaling_min_activity - layer_avg_rate
                scale_update = self._synaptic_scaling_lr * rate_deficit
                self.l5_weight_scale.data.mul_(1.0 + scale_update).clamp_(
                    min=1.0, max=self._synaptic_scaling_max_factor
                )
                # Scale L2/3→L5 weights
                self.synaptic_weights["l23_l5"].data.mul_(1.0 + scale_update)

        # Inter-layer shape check: L5 output
        assert l5_spikes.shape == (self.l5_size,), (
            f"Cortex: L5 spikes have shape {l5_spikes.shape} "
            f"but expected ({self.l5_size},). "
            f"Check L5 sparsity or L2/3→L5 weights shape."
        )

        # =====================================================================
        # LAYER 6: CORTICOTHALAMIC FEEDBACK (SPLIT INTO L6a AND L6b)
        # =====================================================================
        # L6a (corticothalamic type I): Projects to TRN (inhibitory modulation, low gamma)
        # L6b (corticothalamic type II): Projects to relay (excitatory modulation, high gamma)
        # This dual feedback implements both spatial attention (via TRN) and
        # fast gain modulation (direct relay excitation).

        # =====================================================================
        # L6a: Apply L2/3→L6a axonal delay
        # =====================================================================
        if self._l23_l6a_delay_steps > 0:
            self._l23_l6a_buffer.write(l23_spikes)
            l23_spikes_for_l6a = self._l23_l6a_buffer.read(self._l23_l6a_delay_steps)
            self._l23_l6a_buffer.advance()
        else:
            l23_spikes_for_l6a = l23_spikes

        # L6a forward pass (corticothalamic type I → TRN)
        l6a_g_exc = (
            torch.matmul(self.synaptic_weights["l23_l6a"], l23_spikes_for_l6a.float())
            * cfg.l23_to_l6a_strength
        )
        l6a_g_inh = l6a_g_exc * 0.8  # Strong local inhibition for sparse low-gamma firing

        l6a_spikes, l6a_membrane = self.l6a_neurons(l6a_g_exc, l6a_g_inh)
        l6a_spikes = self._apply_sparsity(l6a_spikes, cfg.l6a_sparsity)

        # =====================================================================
        # L6a INHIBITORY NETWORK (PV, SST, VIP INTERNEURONS)
        # =====================================================================
        # L6a inhibitory network for TRN pathway (low gamma)

        # Run inhibitory network
        l6a_inhib_output = self.l6a_inhibitory(
            pyr_spikes=l6a_spikes,
            pyr_membrane=l6a_membrane,
            external_excitation=l6a_g_exc,
            acetylcholine=self.neuromodulator_state.acetylcholine,
        )

        # Apply gamma gating from PV cells
        perisomatic_inh = l6a_inhib_output["perisomatic_inhibition"]
        l6a_gamma_modulation = 1.0 / (1.0 + perisomatic_inh * 0.5)
        l6a_spikes = (l6a_spikes.float() * l6a_gamma_modulation > 0.5).bool()

        # =====================================================================
        # L6b: Apply L2/3→L6b axonal delay
        # =====================================================================
        if self._l23_l6b_delay_steps > 0:
            self._l23_l6b_buffer.write(l23_spikes)
            l23_spikes_for_l6b = self._l23_l6b_buffer.read(self._l23_l6b_delay_steps)
            self._l23_l6b_buffer.advance()
        else:
            l23_spikes_for_l6b = l23_spikes

        # L6b forward pass (corticothalamic type II → relay)
        l6b_g_exc = (
            torch.matmul(self.synaptic_weights["l23_l6b"], l23_spikes_for_l6b.float())
            * cfg.l23_to_l6b_strength
        )
        l6b_g_inh = l6b_g_exc * 0.15  # Minimal local inhibition

        l6b_spikes, l6b_membrane = self.l6b_neurons(l6b_g_exc, l6b_g_inh)
        l6b_spikes = self._apply_sparsity(l6b_spikes, cfg.l6b_sparsity)

        # =====================================================================
        # L6b INHIBITORY NETWORK (PV, SST, VIP INTERNEURONS)
        # =====================================================================
        # L6b inhibitory network for relay pathway (high gamma)

        # Run inhibitory network
        l6b_inhib_output = self.l6b_inhibitory(
            pyr_spikes=l6b_spikes,
            pyr_membrane=l6b_membrane,
            external_excitation=l6b_g_exc,
            acetylcholine=self.neuromodulator_state.acetylcholine,
        )

        # Apply gamma gating from PV cells
        perisomatic_inh = l6b_inhib_output["perisomatic_inhibition"]
        l6b_gamma_modulation = 1.0 / (1.0 + perisomatic_inh * 0.5)
        l6b_spikes = (l6b_spikes.float() * l6b_gamma_modulation) > 0.5

        # Inter-layer shape checks
        assert l6a_spikes.shape == (self.l6a_size,), (
            f"Cortex: L6a spikes have shape {l6a_spikes.shape} "
            f"but expected ({self.l6a_size},)."
        )
        assert l6b_spikes.shape == (self.l6b_size,), (
            f"Cortex: L6b spikes have shape {l6b_spikes.shape} "
            f"but expected ({self.l6b_size},)."
        )

        # =====================================================================
        # PREDICTIVE CODING: ANTI-HEBBIAN LEARNING
        # =====================================================================
        # L5/L6 → L4 weights learn to predict and suppress L4 activity
        # When prediction correct: L4 silent → no weight change (good)
        # When prediction wrong: L4 fires → strengthen inhibition (learn)
        # This is anti-Hebbian: co-activation → increase inhibition

        pred_lr = self.get_effective_learning_rate(cfg.prediction_learning_rate)

        if pred_lr > 1e-8:  # Only learn if not suppressed
            # L5 → L4 anti-Hebbian learning
            if (l5_spikes is not None and
                l4_spikes is not None and
                "l5_l4_pred" in self.synaptic_weights):
                # Positive correlation → strengthen inhibitory weight
                # This makes L5 better at predicting and suppressing L4
                dW_l5 = torch.outer(l4_spikes.float(), l5_spikes.float())
                self.synaptic_weights["l5_l4_pred"].data.add_(pred_lr * dW_l5)

                # Keep weights positive (inhibitory strength)
                self.synaptic_weights["l5_l4_pred"].data.clamp_(min=0.0, max=2.0)

            # L6 → L4 anti-Hebbian learning
            if (l6a_spikes is not None and
                l6b_spikes is not None and
                l4_spikes is not None and
                "l6_l4_pred" in self.synaptic_weights
            ):
                l6_combined = torch.cat([l6a_spikes, l6b_spikes], dim=-1)
                dW_l6 = torch.outer(l4_spikes.float(), l6_combined.float())
                self.synaptic_weights["l6_l4_pred"].data.add_(pred_lr * dW_l6)

                # Keep weights positive (inhibitory strength)
                self.synaptic_weights["l6_l4_pred"].data.clamp_(min=0.0, max=2.0)

        # Apply continuous plasticity (learning happens as part of forward dynamics)
        # Pass external inputs (thalamus, hippocampus, etc.) for BCM/STDP learning
        # Cortical feedback (l23, l4, etc.) doesn't have learnable weights
        self._apply_plasticity(
            inputs=external_inputs,
            l23_spikes=l23_spikes,
            l4_spikes=l4_spikes,
            l5_spikes=l5_spikes,
            l6a_spikes=l6a_spikes,
            l6b_spikes=l6b_spikes,
        )

        # Store spikes for output ports and potential use in next timestep's processing
        self.l23_spikes = l23_spikes
        self.l4_spikes = l4_spikes
        self.l5_spikes = l5_spikes
        self.l6a_spikes = l6a_spikes
        self.l6b_spikes = l6b_spikes

        # =====================================================================
        # SET PORT OUTPUTS
        # =====================================================================
        self.set_port_output("l23", l23_spikes)
        self.set_port_output("l4", l4_spikes)
        self.set_port_output("l5", l5_spikes)
        self.set_port_output("l6a", l6a_spikes)
        self.set_port_output("l6b", l6b_spikes)

    def _apply_plasticity(
        self,
        inputs: RegionSpikesDict,
        l23_spikes: Optional[torch.Tensor] = None,
        l4_spikes: Optional[torch.Tensor] = None,
        l5_spikes: Optional[torch.Tensor] = None,
        l6a_spikes: Optional[torch.Tensor] = None,
        l6b_spikes: Optional[torch.Tensor] = None,
    ) -> None:
        """Apply continuous STDP learning with BCM modulation.

        This is called automatically at each forward() timestep.
        Learning rate is modulated by dopamine (via get_effective_learning_rate).

        In biological cortex, synaptic plasticity happens continuously based on
        pre/post spike timing. Dopamine doesn't trigger learning - it modulates
        how much weight change occurs from the spike-timing-based plasticity.
        """
        # Early exit if no activity yet
        if l4_spikes is None:
            return

        cfg = self.config

        # =====================================================================
        # LAYER-SPECIFIC DOPAMINE MODULATION
        # =====================================================================
        # Apply layer-specific dopamine scaling to learning rates.
        # Different layers have different DA receptor densities (relative sensitivity).
        base_dopamine = self.neuromodulator_state.dopamine
        l23_dopamine = base_dopamine * 0.3
        l4_dopamine = base_dopamine * 0.2
        l5_dopamine = base_dopamine * 0.4
        l6_dopamine = base_dopamine * 0.1

        # Get base learning rate
        base_lr = cfg.learning_rate

        # Use STDP+BCM composite strategies for proper spike-timing-dependent learning
        # Per-source learning: Each input source (thalamus, hippocampus, etc.) learns independently
        # Route to appropriate layer based on target_layer metadata (biological: synapses define pathway)

        # Route external inputs to appropriate layers for plasticity
        for source_name, source_spikes in inputs.items():
            # Skip if this source doesn't have weights
            if source_name not in self.synaptic_weights:
                continue

            # Determine target layer from stored metadata
            target_layer = self._source_target_layers.get(source_name, None)

            # Route to appropriate learning strategy and postsynaptic spikes
            if target_layer == "l23":
                # Top-down inputs → L2/3 plasticity
                if l23_spikes is not None:
                    l23_lr = base_lr * (1.0 + l23_dopamine)
                    updated_weights, _ = self.bcm_l23.compute_update(
                        weights=self.synaptic_weights[source_name].data,
                        pre_spikes=source_spikes,
                        post_spikes=l23_spikes,
                        learning_rate=l23_lr,
                    )
                    self.synaptic_weights[source_name].data.copy_(updated_weights)
                    clamp_weights(self.synaptic_weights[source_name].data, cfg.w_min, cfg.w_max)

            elif target_layer == "l4":
                # Feedforward inputs → L4 plasticity
                if l4_spikes is not None:
                    l4_lr = base_lr * (1.0 + l4_dopamine)
                    updated_weights, _ = self.bcm_l4.compute_update(
                        weights=self.synaptic_weights[source_name].data,
                        pre_spikes=source_spikes,
                        post_spikes=l4_spikes,
                        learning_rate=l4_lr,
                    )
                    self.synaptic_weights[source_name].data.copy_(updated_weights)
                    clamp_weights(self.synaptic_weights[source_name].data, cfg.w_min, cfg.w_max)

            elif target_layer == "l5":
                # Hippocampal inputs → L5 plasticity
                if l5_spikes is not None:
                    l5_lr = base_lr * (1.0 + l5_dopamine)
                    updated_weights, _ = self.bcm_l23.compute_update(  # Use L2/3 BCM for L5 (same supragranular strategy)
                        weights=self.synaptic_weights[source_name].data,
                        pre_spikes=source_spikes,
                        post_spikes=l5_spikes,
                        learning_rate=l5_lr,
                    )
                    self.synaptic_weights[source_name].data.copy_(updated_weights)
                    clamp_weights(self.synaptic_weights[source_name].data, cfg.w_min, cfg.w_max)

            else:
                raise ValueError(f"Unknown target layer '{target_layer}' for source '{source_name}' in plasticity routing.")

        # L4 → L2/3 - L2/3-specific dopamine
        if l23_spikes is not None:
            # L2/3 learning rate with layer-specific dopamine
            l23_lr = base_lr * (1.0 + l23_dopamine)
            updated_weights, _ = self.bcm_l23.compute_update(
                weights=self.synaptic_weights["l4_l23"].data,
                pre_spikes=l4_spikes,
                post_spikes=l23_spikes,
                learning_rate=l23_lr,
            )
            self.synaptic_weights["l4_l23"].data.copy_(updated_weights)
            clamp_weights(self.synaptic_weights["l4_l23"].data, cfg.w_min, cfg.w_max)

            # =================================================================
            # INTRINSIC PLASTICITY: Update per-neuron threshold offsets
            # =================================================================
            # Initialize if needed
            if self._l23_activity_history is None:
                self._l23_activity_history = torch.zeros(self.l23_size, device=self.device)
            if self._l23_threshold_offset is None:
                self._l23_threshold_offset = torch.zeros(self.l23_size, device=self.device)

            # Update activity history (exponential moving average)
            self._l23_activity_history = 0.99 * self._l23_activity_history + (1.0 - 0.99) * l23_spikes.float()

            # Compute threshold modulation using UnifiedHomeostasis
            threshold_mod = self.homeostasis.compute_excitability_modulation(
                activity_history=self._l23_activity_history,
                tau=100.0
            )
            # Convert gain modulation to threshold offset:
            # Under-firing → mod > 1.0 → negative offset → lower threshold → easier to spike
            # Over-firing → mod < 1.0 → positive offset → higher threshold → harder to spike
            # Formula: offset = (1.0 - modulation) inverts the sign correctly
            self._l23_threshold_offset = (1.0 - threshold_mod).clamp(-0.5, 0.5)

            # L2/3 → L5 - L5-specific dopamine (highest modulation)
            if l5_spikes is not None:
                # L5 learning rate with layer-specific dopamine (100% modulation)
                l5_lr = base_lr * (1.0 + l5_dopamine)
                updated_weights, _ = self.bcm_l5.compute_update(
                    weights=self.synaptic_weights["l23_l5"].data,
                    pre_spikes=l23_spikes,
                    post_spikes=l5_spikes,
                    learning_rate=l5_lr,
                )
                self.synaptic_weights["l23_l5"].data.copy_(updated_weights)
                clamp_weights(self.synaptic_weights["l23_l5"].data, cfg.w_min, cfg.w_max)

            # L2/3 → L6a (corticothalamic type I → TRN) - L6-specific dopamine
            if l6a_spikes is not None:
                # L6 learning rate with layer-specific dopamine (20% modulation - stable feedback)
                l6_lr = base_lr * (1.0 + l6_dopamine)
                updated_weights, _ = self.bcm_l6a.compute_update(
                    weights=self.synaptic_weights["l23_l6a"].data,
                    pre_spikes=l23_spikes,
                    post_spikes=l6a_spikes,
                    learning_rate=l6_lr,
                )
                self.synaptic_weights["l23_l6a"].data.copy_(updated_weights)
                clamp_weights(self.synaptic_weights["l23_l6a"].data, cfg.w_min, cfg.w_max)

            # L2/3 → L6b (corticothalamic type II → relay) - L6-specific dopamine
            if l6b_spikes is not None:
                # L6 learning rate with layer-specific dopamine (20% modulation - stable feedback)
                l6_lr = base_lr * (1.0 + l6_dopamine)
                updated_weights, _ = self.bcm_l6b.compute_update(
                    weights=self.synaptic_weights["l23_l6b"].data,
                    pre_spikes=l23_spikes,
                    post_spikes=l6b_spikes,
                    learning_rate=l6_lr,
                )
                self.synaptic_weights["l23_l6b"].data.copy_(updated_weights)
                clamp_weights(self.synaptic_weights["l23_l6b"].data, cfg.w_min, cfg.w_max)

    def _apply_sparsity(self, spikes: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """Apply winner-take-all sparsity to spike tensor."""
        validate_spike_tensor(spikes)

        n_neurons = spikes.shape[0]
        k = max(1, int(n_neurons * target_sparsity))

        sparse_spikes = torch.zeros_like(spikes)
        active = spikes.nonzero(as_tuple=True)[0]

        if len(active) > k:
            keep_indices = active[torch.randperm(len(active))[:k]]
            sparse_spikes[keep_indices] = spikes[keep_indices]
        else:
            sparse_spikes = spikes

        return sparse_spikes

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

        # Update neurons in all layers
        if hasattr(self, "l23_neurons") and self.l23_neurons is not None:
            self.l23_neurons.update_temporal_parameters(dt_ms)
        if hasattr(self, "l4_neurons") and self.l4_neurons is not None:
            self.l4_neurons.update_temporal_parameters(dt_ms)
        if hasattr(self, "l5_neurons") and self.l5_neurons is not None:
            self.l5_neurons.update_temporal_parameters(dt_ms)
        if hasattr(self, "l6a_neurons") and self.l6a_neurons is not None:
            self.l6a_neurons.update_temporal_parameters(dt_ms)
        if hasattr(self, "l6b_neurons") and self.l6b_neurons is not None:
            self.l6b_neurons.update_temporal_parameters(dt_ms)

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information for this region."""
        cfg = self.config

        # Compute plasticity metrics from L4→L2/3 (most dynamic feedforward pathway)
        # Note: L2/3 recurrent weights are now external (in AxonalProjection)
        plasticity = compute_plasticity_metrics(
            weights=self.synaptic_weights["l4_l23"].data,
            learning_rate=cfg.learning_rate,
        )
        # Add per-source input weight statistics (multi-source architecture)
        for source_name in self.input_sources:
            if source_name in self.synaptic_weights:
                key = f"{source_name}_l4_mean"
                plasticity[key] = float(self.synaptic_weights[source_name].data.mean().item())

        # Add inter-layer pathway statistics
        plasticity["l4_l23_mean"] = float(self.synaptic_weights["l4_l23"].data.mean().item())
        plasticity["l23_l5_mean"] = float(self.synaptic_weights["l23_l5"].data.mean().item())
        plasticity["l23_l6a_mean"] = float(self.synaptic_weights["l23_l6a"].data.mean().item())
        plasticity["l23_l6b_mean"] = float(self.synaptic_weights["l23_l6b"].data.mean().item())

        # Recurrent activity
        recurrent_dynamics = {}
        if self.l23_recurrent_activity is not None:
            recurrent_dynamics["l23_recurrent_mean"] = float(
                self.l23_recurrent_activity.mean().item()
            )

        return {
            "plasticity": plasticity,
            "architecture": {
                "l4_size": self.l4_size,
                "l23_size": self.l23_size,
                "l5_size": self.l5_size,
                "l6a_size": self.l6a_size,
                "l6b_size": self.l6b_size,
            },
            "recurrent_dynamics": recurrent_dynamics,
            "homeostasis": {
                "l4_gain": float(self.l4_gain.mean().item()),
                "l23_gain": float(self.l23_gain.mean().item()),
                "l5_gain": float(self.l5_gain.mean().item()),
                "target_rate": self._target_rate,
            },
            "synaptic_scaling": {
                "l4_weight_scale": float(self.l4_weight_scale.item()),
                "l23_weight_scale": float(self.l23_weight_scale.item()),
                "l5_weight_scale": float(self.l5_weight_scale.item()),
            },
        }
