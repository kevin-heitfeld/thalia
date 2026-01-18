"""
Cerebellum - Supervised Error-Corrective Learning for Precise Motor Control.

The cerebellum learns through supervised error signals from climbing fibers,
enabling fast, precise learning of input-output mappings without trial-and-error.

**Key Features**:
=================
1. **ERROR-CORRECTIVE LEARNING** (Delta Rule):
   - Δw ∝ pre × (target - actual)
   - Direct teaching signal (not reward/punishment like RL)
   - Can learn arbitrary mappings in 1-10 trials (vs hundreds for RL)

2. **CLIMBING FIBER ERROR SIGNAL**:
   - Inferior olive computes mismatch between intended and actual movement
   - Climbing fiber activates Purkinje cell → LTD on active parallel fibers
   - Absence of climbing fiber → LTP (strengthen correct associations)
   - Binary teacher: "correct" or "incorrect"

3. **PRECISE TIMING AND COORDINATION**:
   - Cerebellum is master of temporal precision
   - Can learn sub-millisecond timing patterns
   - Critical for smooth, coordinated movements
   - Predictive timing (anticipates sensory consequences)

4. **FAST SUPERVISED LEARNING**:
   - Unlike RL (needs many trials with delayed rewards)
   - Cerebellum learns in 1-10 trials with immediate feedback
   - Supervised signal provides direct gradient information
   - No exploration needed - direct instruction

FILE ORGANIZATION (759 lines)
==============================
Lines 1-80:    Module docstring, imports
Lines 81-170:  CerebellumConfig dataclass
Lines 171-280: Cerebellum class __init__, weight initialization
Lines 281-370: Forward pass (parallel fibers → Purkinje cells)
Lines 371-500: Error learning (climbing fiber supervision, delta rule)
Lines 501-650: Growth and neurogenesis (grow_output)
Lines 651-730: Diagnostics and health monitoring
Lines 731-759: Utility methods (reset_state, get_full_state)

NAVIGATION TIP: Use VSCode's "Go to Symbol" (Ctrl+Shift+O) to jump between methods.

Biological Basis:
=================
- Marr (1969) and Albus (1971): Cerebellar learning theory
- Parallel fibers (inputs) → Purkinje cells (outputs)
- Climbing fibers carry error/teaching signals
- LTD at parallel fiber-Purkinje synapses when climbing fiber active

When to Use:
============
- You have explicit target outputs (labels)
- You want to learn arbitrary input→output mappings
- You need fast learning (few trials)
- Precise timing is important
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from thalia.components.neurons import (
    ConductanceLIF,
    ConductanceLIFConfig,
)
from thalia.components.synapses import ShortTermPlasticity, STPConfig, STPType, WeightInitializer
from thalia.config.learning_config import ErrorCorrectiveLearningConfig
from thalia.constants.neuromodulation import (
    ACH_BASELINE,
    DA_BASELINE_STANDARD,
    NE_BASELINE,
    compute_ne_gain,
)
from thalia.constants.neuron import (
    E_EXCITATORY,
    E_INHIBITORY,
    E_LEAK,
    V_RESET_STANDARD,
    V_THRESHOLD_STANDARD,
)
from thalia.core.base.component_config import NeuralComponentConfig
from thalia.core.component_state import NeuralComponentState
from thalia.core.neural_region import NeuralRegion
from thalia.core.region_state import BaseRegionState
from thalia.learning import EligibilitySTDPConfig as STDPConfig
from thalia.learning import EligibilityTraceManager
from thalia.learning.homeostasis.synaptic_homeostasis import (
    UnifiedHomeostasis,
    UnifiedHomeostasisConfig,
)
from thalia.managers.component_registry import register_region
from thalia.utils.core_utils import clamp_weights
from thalia.utils.input_routing import InputRouter
from thalia.utils.oscillator_utils import compute_theta_encoding_retrieval

from .deep_nuclei import DeepCerebellarNuclei
from .granule_layer import GranuleCellLayer
from .purkinje_cell import EnhancedPurkinjeCell


@dataclass
class CerebellumConfig(NeuralComponentConfig, ErrorCorrectiveLearningConfig):
    """Configuration specific to cerebellar regions.

    The cerebellum implements ERROR-CORRECTIVE learning through:
    1. Parallel fiber → Purkinje cell connections (learned)
    2. Climbing fiber error signals from inferior olive
    3. LTD when climbing fiber active with parallel fiber

    Key biological features:
    - Error signal triggers immediate learning (not delayed like RL)
    - Can learn arbitrary input-output mappings quickly
    - Uses eligibility traces for temporal credit assignment

    Inherits from NeuralComponentConfig (structural) then ErrorCorrectiveLearningConfig (behavioral):
    - learning_rate_ltp: LTP rate (default 0.01)
    - learning_rate_ltd: LTD rate (default 0.01)
    - error_threshold: Minimum error (default 0.01)
    - use_eligibility_traces: Enable traces (default True)
    - eligibility_tau_ms: Trace decay (default 20.0)

    **Size Specification** (Semantic-First):
    - Sizes passed via sizes dict: input_size, granule_size, purkinje_size
    - Computed at instantiation: output_size (= purkinje_size), total_neurons (granule + purkinje or just purkinje)
    - Config contains only behavioral parameters (learning rates, circuit flags, etc.)
    """

    # Temporal processing
    temporal_window_ms: float = 10.0  # Window for coincidence detection

    # Cerebellum uses weaker heterosynaptic competition for faster convergence:
    heterosynaptic_ratio: float = 0.2  # Override base (0.3) - weaker competition

    # Input trace parameters
    input_trace_tau_ms: float = 20.0  # Input trace decay

    # =========================================================================
    # ENHANCED MICROCIRCUIT (optional, for increased biological detail)
    # =========================================================================
    # When enabled, uses granule→Purkinje→DCN circuit instead of direct
    # parallel fiber→Purkinje mapping. Provides:
    # - 4× sparse expansion in granule layer (pattern separation)
    # - Dendritic computation in Purkinje cells (complex/simple spikes)
    # - DCN integration (Purkinje sculpts tonic output)
    use_enhanced_microcircuit: bool = True

    granule_sparsity: float = 0.03  # Fraction of granule cells active (3%)
    purkinje_n_dendrites: int = 100  # Simplified dendritic compartments

    # =========================================================================
    # SHORT-TERM PLASTICITY (STP) - CRITICAL FOR CEREBELLAR TIMING
    # =========================================================================
    # Biologically, cerebellar synapses show distinct STP properties that are
    # CRITICAL for temporal processing and motor timing:
    #
    # 1. PARALLEL FIBERS→PURKINJE: DEPRESSING (U=0.5-0.7)
    #    - Implements temporal high-pass filter
    #    - Fresh inputs signal new patterns
    #    - Sustained inputs fade → cerebellum detects CHANGES, not steady-state
    #    - Enables sub-millisecond timing discrimination
    #    - WITHOUT THIS: Cerebellar timing precision COLLAPSES
    #
    # 2. MOSSY FIBERS→GRANULE CELLS: FACILITATING (U=0.15-0.25)
    #    - Burst detection for sparse coding
    #    - Amplifies repeated mossy fiber activity
    #    - Enhances pattern separation in granule layer
    #
    # 3. CLIMBING FIBERS→PURKINJE: RELIABLE (U≈0.9, minimal STP)
    #    - Error signal must be unambiguous
    #    - No adaptation - every climbing fiber spike matters
    #
    # References:
    # - Dittman et al. (2000): Nature 403:530-534 - Classic PF→Purkinje STP paper
    # - Atluri & Regehr (1996): Delayed release at granule cell synapses
    # - Isope & Barbour (2002): Facilitation at mossy fiber synapses
    #
    # BIOLOGICAL IMPORTANCE: This is perhaps the MOST important STP in the brain
    # for motor learning and timing. The cerebellar cortex is the brain's master
    # clock, and STP is essential for its temporal precision.
    stp_enabled: bool = True
    stp_pf_purkinje_type: STPType = STPType.DEPRESSING  # Parallel fiber depression
    stp_mf_granule_type: STPType = STPType.FACILITATING  # Mossy fiber facilitation

    # =========================================================================
    # GAP JUNCTIONS (Inferior Olive Synchronization)
    # =========================================================================
    # Inferior olive (IO) neurons are electrically coupled via gap junctions,
    # creating synchronized complex spikes across multiple Purkinje cells.
    # This coordination is critical for motor learning and timing precision.
    #
    # Biology:
    # - IO neurons form one of the densest gap junction networks in the brain
    # - Synchronization time: <1ms (ultra-fast electrical coupling)
    # - Functional role: Coordinates learning across multiple cerebellar modules
    # - Complex spikes arrive synchronously at related Purkinje cells
    #
    # References:
    # - Llinás & Yarom (1981): Electrophysiology of IO gap junctions
    # - De Zeeuw et al. (1998): Gap junctions in IO create synchronous climbing fiber activity
    # - Leznik & Llinás (2005): Role of gap junctions in IO oscillations
    # - Schweighofer et al. (1999): Computational role of IO synchronization
    gap_junctions_enabled: bool = True
    """Enable gap junction coupling in inferior olive neurons."""

    gap_junction_strength: float = 0.18
    """Gap junction conductance for IO neurons (biological: 0.1-0.3, IO has stronger coupling)."""

    gap_junction_threshold: float = 0.20
    """Connectivity threshold for gap junction coupling (shared error patterns)."""

    gap_junction_max_neighbors: int = 12
    """Maximum gap junction neighbors per IO neuron (biological: 6-15, IO is densely coupled)."""

    # =========================================================================
    # COMPLEX SPIKE DYNAMICS (Phase 2B Enhancement)
    # =========================================================================
    # Climbing fibers trigger complex spikes in Purkinje cells - bursts of 2-7
    # spikes with very short inter-spike intervals (1-2ms). These bursts encode
    # ERROR MAGNITUDE: larger errors → longer bursts → more calcium influx.
    #
    # This provides GRADED ERROR SIGNALING instead of binary (error/no-error),
    # enabling more nuanced learning: small errors trigger small corrections,
    # large errors trigger large corrections.
    #
    # Biological Evidence:
    # - Mathy et al. (2009): Complex spikes have 2-7 spikelets per burst
    # - Davie et al. (2008): Number of spikelets correlates with error magnitude
    # - Najafi & Medina (2013): Graded complex spikes enable graded learning
    # - Yang & Lisberger (2014): Complex spike amplitude predicts learning rate
    #
    # Mechanism:
    # 1. Climbing fiber error → complex spike burst (not single spike)
    # 2. Each spikelet triggers dendritic calcium influx (~0.2 units)
    # 3. Total calcium = n_spikes × ca2_per_spike
    # 4. LTD magnitude ∝ total calcium (graded learning)
    #
    # Example:
    # - Small error (0.2): 3 spikes → Ca²⁺ = 0.6 → small LTD
    # - Large error (0.8): 6 spikes → Ca²⁺ = 1.2 → large LTD
    #
    # References:
    # - Mathy et al. (2009): Nature Neuroscience 12:1378-1387
    # - Najafi & Medina (2013): J. Neuroscience 33:15825-15840
    # - Yang & Lisberger (2014): Neuron 82:1389-1401
    # - Davie et al. (2008): Nature Neuroscience 11:838-848
    use_complex_spike_bursts: bool = False
    """Enable complex spike burst dynamics (graded error signaling)."""

    min_complex_spike_count: int = 2
    """Minimum spikelets per complex spike burst (biological: 2-3)."""

    max_complex_spike_count: int = 7
    """Maximum spikelets per complex spike burst (biological: 5-8)."""

    complex_spike_isi_ms: float = 1.5
    """Inter-spike interval within burst (biological: 1-2ms, very fast)."""

    ca2_per_spikelet: float = 0.2
    """Calcium influx per spikelet (arbitrary units, scaled for learning)."""


@dataclass
class CerebellumState(BaseRegionState):
    """Complete state for Cerebellum region.

    Stores all cerebellar state including:
    - Eligibility traces (input, output, STDP)
    - Climbing fiber error signal
    - Neuron state (classic mode) OR enhanced microcircuit state
    - Short-term plasticity state

    Note: Neuromodulators (dopamine, acetylcholine, norepinephrine) are
    inherited from BaseRegionState.

    Classic Mode Fields (use_enhanced_microcircuit=False):
    - v_mem, g_exc, g_inh: Direct Purkinje cell neuron state
    - stp_pf_purkinje_state: STP for parallel fiber→Purkinje synapses

    Enhanced Mode Fields (use_enhanced_microcircuit=True):
    - granule_layer_state: Granule cell layer state
    - purkinje_cells_state: List of enhanced Purkinje cell states
    - deep_nuclei_state: Deep cerebellar nuclei state
    - stp_mf_granule_state: STP for mossy fiber→granule synapses
    """

    # ========================================================================
    # TRACE MANAGER STATE (both modes)
    # ========================================================================
    input_trace: Optional[torch.Tensor] = None  # [n_input] or [n_granule] if enhanced
    output_trace: Optional[torch.Tensor] = None  # [n_output]
    stdp_eligibility: Optional[torch.Tensor] = None  # [n_output, n_input/n_granule]

    # ========================================================================
    # CLIMBING FIBER ERROR (both modes)
    # ========================================================================
    climbing_fiber_error: Optional[torch.Tensor] = None  # [n_output] - Error signal from IO
    io_membrane: Optional[torch.Tensor] = None  # [n_output] - IO membrane for gap junctions

    # ========================================================================
    # CLASSIC MODE NEURON STATE (use_enhanced=False)
    # ========================================================================
    v_mem: Optional[torch.Tensor] = None  # [n_output] - Membrane voltage
    g_exc: Optional[torch.Tensor] = None  # [n_output] - Excitatory conductance
    g_inh: Optional[torch.Tensor] = None  # [n_output] - Inhibitory conductance

    # ========================================================================
    # SHORT-TERM PLASTICITY STATE
    # ========================================================================
    stp_pf_purkinje_state: Optional[Dict[str, torch.Tensor]] = None  # Classic mode STP
    stp_mf_granule_state: Optional[Dict[str, torch.Tensor]] = None  # Enhanced mode STP

    # ========================================================================
    # ENHANCED MICROCIRCUIT STATE (use_enhanced=True)
    # ========================================================================
    granule_layer_state: Optional[Dict[str, Any]] = None  # GranuleCellLayer state
    purkinje_cells_state: Optional[list] = None  # List of EnhancedPurkinjeCell states
    deep_nuclei_state: Optional[Dict[str, Any]] = None  # DeepCerebellarNuclei state

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary.

        Returns:
            Dictionary containing all state fields.
        """
        return {
            # Traces
            "input_trace": self.input_trace,
            "output_trace": self.output_trace,
            "stdp_eligibility": self.stdp_eligibility,
            # Error signal
            "climbing_fiber_error": self.climbing_fiber_error,
            "io_membrane": self.io_membrane,
            # Classic neuron state
            "v_mem": self.v_mem,
            "g_exc": self.g_exc,
            "g_inh": self.g_inh,
            # STP state
            "stp_pf_purkinje_state": self.stp_pf_purkinje_state,
            "stp_mf_granule_state": self.stp_mf_granule_state,
            # Neuromodulators
            "dopamine": self.dopamine,
            "acetylcholine": self.acetylcholine,
            "norepinephrine": self.norepinephrine,
            # Enhanced microcircuit
            "granule_layer_state": self.granule_layer_state,
            "purkinje_cells_state": self.purkinje_cells_state,
            "deep_nuclei_state": self.deep_nuclei_state,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: str = "cpu") -> CerebellumState:
        """Deserialize state from dictionary.

        Args:
            data: Dictionary from to_dict()
            device: Target device for tensors

        Returns:
            CerebellumState instance with tensors on specified device.
        """
        dev = torch.device(device)

        # Helper to transfer tensors to device
        def to_device(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return tensor.to(dev) if tensor is not None else None

        # Helper for STP state dicts
        def stp_to_device(
            stp_state: Optional[Dict[str, torch.Tensor]],
        ) -> Optional[Dict[str, torch.Tensor]]:
            if stp_state is None:
                return None
            return {k: v.to(dev) if v is not None else None for k, v in stp_state.items()}

        return cls(
            # Traces (may be None if state is uninitialized)
            input_trace=to_device(data.get("input_trace")),
            output_trace=to_device(data.get("output_trace")),
            stdp_eligibility=to_device(data.get("stdp_eligibility")),
            # Error signal (may be None if state is uninitialized)
            climbing_fiber_error=to_device(data.get("climbing_fiber_error")),
            # Classic neuron state (optional)
            v_mem=to_device(data.get("v_mem")),
            g_exc=to_device(data.get("g_exc")),
            g_inh=to_device(data.get("g_inh")),
            # STP state (optional)
            stp_pf_purkinje_state=stp_to_device(data.get("stp_pf_purkinje_state")),
            stp_mf_granule_state=stp_to_device(data.get("stp_mf_granule_state")),
            # Neuromodulators (base tonic baseline)
            dopamine=data.get("dopamine", DA_BASELINE_STANDARD),
            acetylcholine=data.get("acetylcholine", ACH_BASELINE),
            norepinephrine=data.get("norepinephrine", NE_BASELINE),
            # Enhanced microcircuit (optional)
            granule_layer_state=data.get("granule_layer_state"),
            purkinje_cells_state=data.get("purkinje_cells_state"),
            deep_nuclei_state=data.get("deep_nuclei_state"),
        )

    def reset(self) -> None:
        """Reset state in-place (clears traces, resets neurons)."""
        # Reset base state (spikes, membrane, neuromodulators)
        super().reset()

        # Clear traces (if initialized)
        if self.input_trace is not None:
            self.input_trace.zero_()
        if self.output_trace is not None:
            self.output_trace.zero_()
        if self.stdp_eligibility is not None:
            self.stdp_eligibility.zero_()

        # Clear error (if initialized)
        if self.climbing_fiber_error is not None:
            self.climbing_fiber_error.zero_()

        # Reset classic neuron state
        if self.v_mem is not None:
            self.v_mem.fill_(-70.0)  # Resting potential
        if self.g_exc is not None:
            self.g_exc.zero_()
        if self.g_inh is not None:
            self.g_inh.zero_()

        # Reset STP state
        if self.stp_pf_purkinje_state is not None:
            if "u" in self.stp_pf_purkinje_state:
                self.stp_pf_purkinje_state["u"].zero_()
            if "x" in self.stp_pf_purkinje_state:
                self.stp_pf_purkinje_state["x"].fill_(1.0)  # Resources fully available

        if self.stp_mf_granule_state is not None:
            if "u" in self.stp_mf_granule_state:
                self.stp_mf_granule_state["u"].zero_()
            if "x" in self.stp_mf_granule_state:
                self.stp_mf_granule_state["x"].fill_(1.0)

        # Note: Enhanced microcircuit states are complex nested structures
        # They should be reset through their respective subsystems


class ClimbingFiberSystem:
    """Climbing fiber error signaling system.

    Climbing fiber activity means: "You got it WRONG"
    Absence means: "You got it RIGHT (or no feedback)"

    The error signal: target - actual
    - Positive: Should have fired but didn't → strengthen inputs
    - Negative: Fired but shouldn't have → weaken inputs
    """

    def __init__(self, purkinje_size: int, device: str = "cpu"):
        self.purkinje_size = purkinje_size
        self.device = torch.device(device)
        self.error = torch.zeros(purkinje_size, device=self.device)

    def compute_error(
        self,
        actual: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute error signal (climbing fiber activity).

        Args:
            actual: Actual output [purkinje_size] (1D)
            target: Target output [purkinje_size] (1D)

        Returns:
            Error signal [purkinje_size] (1D)
        """
        # Ensure 1D
        if actual.dim() != 1:
            actual = actual.squeeze()
        if target.dim() != 1:
            target = target.squeeze()

        self.error = target - actual
        return self.error

    def reset_state(self) -> None:
        self.error = torch.zeros(self.purkinje_size, device=self.device)

    def get_state(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        return {
            "error": self.error.clone(),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.error = state["error"].to(self.device)


@register_region(
    "cerebellum",
    description="Supervised error-corrective learning via climbing fiber error signals",
    version="2.0",
    author="Thalia Project",
    config_class=CerebellumConfig,
)
class Cerebellum(NeuralRegion):
    """Cerebellar region with supervised error-corrective learning.

    Implements the cerebellar learning rule:
    - Error signal = target - actual (from climbing fibers)
    - Weight change = learning_rate × input × error
    - This is essentially the delta rule / perceptron learning

    Mixins Provide:
    ---------------
    From DiagnosticsMixin:
        - check_health() → HealthMetrics
        - get_firing_rate(spikes) → float
        - check_weight_health(weights, name) → WeightHealth
        - detect_runaway_excitation(spikes) → bool
        - detect_silence(spikes) → bool

    From LearnableComponent (abstract base):
        - forward(input, **kwargs) → Tensor [must implement]
        - reset_state() → None
        - get_diagnostics() → Dict
        - deliver_error(target) → Dict [cerebellum-specific]

    See Also:
        docs/patterns/mixins.md for detailed mixin patterns
    """

    def __init__(self, config: CerebellumConfig, sizes: Dict[str, int], device: str):
        """Initialize cerebellum.

        Args:
            config: Cerebellum configuration (behavioral parameters only)
            sizes: Size specification {'input_size': int, 'granule_size': int, 'purkinje_size': int}
            device: Device ('cpu' or 'cuda')
        """
        self.config: CerebellumConfig = config
        self.device = torch.device(device)

        # Extract sizes
        input_size = sizes["input_size"]
        self.granule_size = sizes["granule_size"]
        self.purkinje_size = sizes["purkinje_size"]
        self.n_output = self.purkinje_size  # Purkinje cells are output
        self.total_neurons = (
            (self.granule_size + self.purkinje_size)
            if config.use_enhanced_microcircuit
            else self.purkinje_size
        )

        # Initialize NeuralRegion (Phase 2 pattern)
        super().__init__(
            n_neurons=self.purkinje_size,
            device=device,
        )

        # Initialize default input source (multi-source architecture)
        # Cerebellum uses single "default" source that accepts concatenated inputs
        self.input_sources["default"] = input_size
        self._total_input_size = input_size  # Track for concatenation validation

        # Initialize state for NeuromodulatorMixin
        self.state = NeuralComponentState()

        self.climbing_fiber = ClimbingFiberSystem(
            purkinje_size=self.purkinje_size,
            device=device,
        )

        # =====================================================================
        # ENHANCED MICROCIRCUIT (optional)
        # =====================================================================
        self.use_enhanced = self.config.use_enhanced_microcircuit

        if self.use_enhanced:
            # Granule cell layer (sparse expansion)
            self.granule_layer = GranuleCellLayer(
                n_mossy_fibers=self._total_input_size,
                expansion_factor=self.granule_size / self._total_input_size,
                sparsity=self.config.granule_sparsity,
                device=device,
            )

            # Enhanced Purkinje cells (one per output neuron)
            # Pass granule layer size as n_parallel_fibers for eager initialization
            self.purkinje_cells = torch.nn.ModuleList(
                [
                    EnhancedPurkinjeCell(
                        n_parallel_fibers=self.granule_layer.n_granule,
                        n_dendrites=self.config.purkinje_n_dendrites,
                        device=device,
                    )
                    for _ in range(self.purkinje_size)
                ]
            )

            # Deep cerebellar nuclei (final output)
            self.deep_nuclei = DeepCerebellarNuclei(
                n_output=self.purkinje_size,
                n_purkinje=self.purkinje_size,
                n_mossy=self._total_input_size,
                device=device,
            )

            # Update weight dimensions for granule layer expansion
            expanded_input = self.granule_layer.n_granule
        else:
            self.granule_layer = None  # type: ignore[assignment]
            self.purkinje_cells = None  # type: ignore[assignment]
            self.deep_nuclei = None  # type: ignore[assignment]
            expanded_input = self._total_input_size

        # =====================================================================
        # ELIGIBILITY TRACE MANAGER for STDP
        # =====================================================================
        stdp_config = STDPConfig(
            stdp_tau_ms=self.config.tau_plus_ms,  # Use tau_plus_ms as STDP tau
            eligibility_tau_ms=self.config.eligibility_tau_ms,
            stdp_lr=self.config.stdp_lr,
            a_plus=1.0,
            a_minus=self.config.heterosynaptic_ratio,
            w_min=config.w_min,
            w_max=config.w_max,
            heterosynaptic_ratio=self.config.heterosynaptic_ratio,
        )
        self._trace_manager = EligibilityTraceManager(
            n_input=expanded_input,  # Use expanded size if granule layer enabled
            n_output=self.purkinje_size,
            config=stdp_config,
            device=self.device,
        )

        # Beta oscillator phase tracking for motor timing
        self._beta_phase: float = 0.0
        self._gamma_phase: float = 0.0

        # Timestep cache for trace manager calls
        self._dt_ms: float = 0.0  # Will be set by brain via update_temporal_parameters()

        # IO membrane potential for gap junction coupling
        # Only used if gap_junctions_enabled; stored at cerebellum level like climbing_fiber.error
        self._io_membrane: Optional[torch.Tensor] = None

        # Homeostasis for synaptic scaling
        homeostasis_config = UnifiedHomeostasisConfig(
            weight_budget=config.weight_budget * self._total_input_size,  # Total budget per neuron
            w_min=config.w_min,
            w_max=config.w_max,
            soft_normalization=config.soft_normalization,
            normalization_rate=config.normalization_rate,
            device=device,
        )
        self.homeostasis = UnifiedHomeostasis(homeostasis_config)

        # =====================================================================
        # INITIALIZE SYNAPTIC WEIGHTS (Phase 2 pattern)
        # =====================================================================
        # Parallel fiber (mossy fiber→Purkinje) weights stored in synaptic_weights
        # Use expanded_input if granule layer enabled, otherwise input_size
        self.synaptic_weights["default"] = self._initialize_weights_tensor(
            n_output=self.purkinje_size,
            n_input=expanded_input,
        )

        # =====================================================================
        # SHORT-TERM PLASTICITY (STP)
        # =====================================================================
        # Initialize STP modules for cerebellar pathways if enabled

        if self.config.stp_enabled:
            # Parallel Fibers→Purkinje: DEPRESSING (CRITICAL for timing)
            # This implements the temporal high-pass filter that makes the
            # cerebellum respond to CHANGES rather than sustained input.
            # Without this, cerebellar timing precision is severely impaired.
            self.stp_pf_purkinje = ShortTermPlasticity(
                n_pre=expanded_input,
                n_post=self.purkinje_size,
                config=STPConfig.from_type(self.config.stp_pf_purkinje_type),
                per_synapse=True,  # Per-synapse dynamics for maximum precision
            )
            self.stp_pf_purkinje.to(self.device)

            # Mossy Fibers→Granule Cells: FACILITATING (if enhanced circuit enabled)
            # Amplifies repeated mossy fiber activity for sparse coding
            if self.use_enhanced:
                self.stp_mf_granule = ShortTermPlasticity(
                    n_pre=self.input_size,
                    n_post=self.granule_layer.n_granule,
                    config=STPConfig.from_type(self.config.stp_mf_granule_type),
                    per_synapse=True,
                )
                self.stp_mf_granule.to(self.device)
            else:
                self.stp_mf_granule = None  # type: ignore[assignment]

        # Climbing Fibers→Purkinje: NO STP (reliable error signal)
        # Every climbing fiber spike is critical - no adaptation
        else:
            self.stp_pf_purkinje = None  # type: ignore[assignment]
            self.stp_mf_granule = None  # type: ignore[assignment]

        # Initialize for forward() storage (fixes attribute outside __init__)
        self.last_effective_input: Optional[torch.Tensor] = None

        # =====================================================================
        # GAP JUNCTIONS (Inferior Olive Synchronization)
        # =====================================================================
        # IO neurons are densely coupled via gap junctions, creating synchronized
        # complex spikes across multiple Purkinje cells. This is critical for
        # coordinated motor learning across cerebellar modules.
        #
        # We use the parallel fiber weights (or granule→Purkinje weights if enhanced)
        # to infer functional neighborhoods: Purkinje cells receiving similar inputs
        # likely receive error signals from neighboring IO neurons.
        #
        # GapJunctionCoupling expects afferent weights [n_neurons, n_input] and
        # internally computes functional connectivity neighborhoods.
        self.gap_junctions_io: Optional[GapJunctionCoupling] = None
        if self.config.gap_junctions_enabled:
            from thalia.components.gap_junctions import GapJunctionConfig, GapJunctionCoupling

            gap_config = GapJunctionConfig(
                enabled=True,
                coupling_strength=self.config.gap_junction_strength,
                connectivity_threshold=self.config.gap_junction_threshold,
                max_neighbors=self.config.gap_junction_max_neighbors,
                interneuron_only=False,  # IO neurons are not interneurons (glutamatergic)
            )

            # Pass actual parallel fiber weights to infer IO neuron neighborhoods
            # IO neurons projecting to Purkinje cells with similar input patterns
            # are anatomically close and thus coupled via gap junctions
            # weights shape: [purkinje_size, n_input] → each row is a Purkinje cell's input pattern
            self.gap_junctions_io = GapJunctionCoupling(
                n_neurons=self.purkinje_size,
                afferent_weights=self.synaptic_weights[
                    "default"
                ].detach(),  # Use actual weights, not similarity
                config=gap_config,
                interneuron_mask=None,  # All IO neurons participate
                device=self.device,
            )

        # Move all components to target device
        self.to(self.device)

    def _initialize_weights_tensor(self, n_output: int, n_input: int) -> torch.nn.Parameter:
        """Initialize weights tensor (no longer part of LearnableComponent pattern)."""
        weights = WeightInitializer.gaussian(
            n_output=n_output,
            n_input=n_input,
            mean=self.config.w_max * 0.1,
            std=0.02,
            device=self.device,
        )
        return torch.nn.Parameter(
            clamp_weights(weights, self.config.w_min, self.config.w_max, inplace=False)
        )

    @property
    def input_size(self) -> int:
        """Total input size across all sources (backward compatibility)."""
        return self._total_input_size

    @property
    def input_trace(self) -> torch.Tensor:
        """Input trace (delegated to trace manager)."""
        return self._trace_manager.input_trace

    @property
    def output_trace(self) -> torch.Tensor:
        """Output trace (delegated to trace manager)."""
        return self._trace_manager.output_trace

    @property
    def stdp_eligibility(self) -> torch.Tensor:
        """STDP eligibility (delegated to trace manager)."""
        return self._trace_manager.eligibility

    def set_oscillator_phases(
        self,
        phases: Dict[str, float],
        signals: Optional[Dict[str, float]] = None,
        theta_slot: int = 0,
        coupled_amplitudes: Optional[Dict[str, float]] = None,
    ) -> None:
        """Receive oscillator information from brain broadcast.

        Beta oscillations are critical for motor control and learning:
        - Pre-movement: beta increases during planning phase
        - Movement initiation: beta desynchronizes (ERD)
        - During movement: low beta allows rapid adjustments
        - Post-movement: beta rebounds (ERS)

        The cerebellum uses beta phase to gate climbing fiber learning:
        - Beta trough (phase = π): Peak learning window (movement initiation)
        - Beta peak (phase = 0/2π): Minimal learning (action maintenance)

        Args:
            phases: Oscillator phases in radians {'theta': ..., 'beta': ..., 'gamma': ...}
            signals: Oscillator signal values (sin/cos waveforms)
            theta_slot: Current theta slot [0, n_slots-1] for working memory
            coupled_amplitudes: Effective amplitudes per oscillator (pre-computed)

        Note:
            Called automatically by Brain before each forward() call.
            Do not call this manually.
        """
        # Use base mixin implementation to store all oscillator data
        # This populates self._theta_phase, self._beta_phase, self._gamma_phase,
        # self._beta_amplitude_effective, self._gamma_amplitude_effective, etc.
        super().set_oscillator_phases(phases, signals, theta_slot, coupled_amplitudes)

    def _compute_beta_gate(self) -> float:
        """Compute beta-gated learning modulation.

        Biology:
        - Climbing fibers from inferior olive fire most effectively during beta trough
        - Beta desynchronization (ERD) signals movement initiation window
        - This is when cerebellum should update motor predictions
        - Automatic multiplicative coupling: beta modulated by all slower oscillators

        Returns:
            Gate value [0, 1] - peak at beta trough (phase = π), modulated by coupling
        """
        # Peak learning at β = π (beta trough = movement initiation)
        phase_diff = abs(self._beta_phase - math.pi)
        width = math.pi / 4  # ±45° learning window
        gate = math.exp(-(phase_diff**2) / (2 * width**2))

        # Modulate by ALL coupling effects (theta, beta modulation)
        # This gives emergent theta-beta-gamma triple coupling automatically
        gate = gate * self._gamma_amplitude_effective

        return gate

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters when brain timestep changes.

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        self._dt_ms = dt_ms
        # Note: Trace manager doesn't cache dt_ms, it's passed per-call

    def _create_neurons(self) -> ConductanceLIF:
        """Create Purkinje-like neurons with constants from neuron_constants.py."""
        neuron_config = ConductanceLIFConfig(
            v_threshold=V_THRESHOLD_STANDARD,
            v_reset=V_RESET_STANDARD,
            E_L=E_LEAK,
            E_E=E_EXCITATORY,
            E_I=E_INHIBITORY,
            tau_E=3.0,  # Faster excitatory for precise timing
            tau_I=8.0,  # Faster inhibitory for precise timing
        )
        neurons = ConductanceLIF(
            n_neurons=self.purkinje_size,
            config=neuron_config,
            device=self.device,
        )
        return neurons

    def grow_output(
        self,
        n_new: int,
        initialization: str = "sparse_random",
        sparsity: float = 0.1,
    ) -> None:
        """Grow output dimension by adding Purkinje cells.

        Expands motor learning capacity by adding neurons.

        Args:
            n_new: Number of neurons to add
            initialization: Weight initialization strategy
            sparsity: Sparsity for new connections
        """
        old_n_output = self.purkinje_size
        new_n_output = old_n_output + n_new

        # =====================================================================
        # 1. EXPAND WEIGHTS using base helper (classic pathway only)
        # =====================================================================
        if not self.use_enhanced:
            self.synaptic_weights["default"].data = self._expand_weights(
                current_weights=self.synaptic_weights["default"].data,  # type: ignore[arg-type]
                n_new=n_new,
                initialization=initialization,
                sparsity=sparsity,
                scale=1.0,  # Default scale for cerebellum
            )

        # =====================================================================
        # 2. UPDATE CONFIG AND INSTANCE VARIABLES
        # =====================================================================
        # Update instance variables
        self.purkinje_size = new_n_output
        self.n_output = new_n_output
        self.n_neurons = new_n_output  # Always track Purkinje count (output neurons)
        if not self.use_enhanced:
            self.total_neurons = new_n_output
        # Note: granule_size doesn't change when adding Purkinje cells

        # =====================================================================
        # 3. EXPAND NEURON POPULATION (classic pathway only)
        # =====================================================================
        if not self.use_enhanced:
            # Use efficient in-place neuron growth (ConductanceLIF)
            self.neurons.grow_neurons(n_new)
        else:
            # Enhanced pathway: add new Purkinje cells with current granule layer size
            for _ in range(n_new):
                self.purkinje_cells.append(
                    EnhancedPurkinjeCell(
                        n_parallel_fibers=self.granule_layer.n_granule,
                        n_dendrites=self.config.purkinje_n_dendrites,
                        device=self.device,  # type: ignore[arg-type]
                        dt_ms=self.config.dt_ms,
                    )
                )

            # Grow DCN to accept more Purkinje inputs and produce more outputs
            self.deep_nuclei.grow_input(n_new, source="purkinje")
            self.deep_nuclei.grow_output(n_new)

        # =====================================================================
        # 4. GROW TRACE MANAGER
        # =====================================================================
        self._trace_manager = self._trace_manager.grow_dimension(n_new, dimension="output")

        # =====================================================================
        # 5. GROW STP MODULES (manual growth due to complex routing)
        # =====================================================================
        # stp_pf_purkinje tracks parallel fiber→Purkinje, needs post growth
        if self.stp_pf_purkinje is not None:
            self.stp_pf_purkinje.grow(n_new, target="post")
        # stp_mf_granule tracks mossy→granule, does NOT grow with output
        # (granule size is determined by input * expansion, not output)

        # =====================================================================
        # 6. VALIDATE GROWTH
        # =====================================================================
        # Skip neuron check since enhanced mode uses purkinje_cells list
        # instead of self.neurons, and classic mode already grew neurons
        self._validate_output_growth(old_n_output, n_new, check_neurons=not self.use_enhanced)

    def grow_source(
        self,
        source_name: str,
        new_size: int,
        initialization: str = "xavier",
        sparsity: float = 0.2,
        weight_scale: float = 0.3,
    ) -> None:
        """Grow specific input source (override for cerebellar microcircuit).

        Cerebellum has internal transformation mossy→granule→purkinje, so growing
        the input requires:
        1. Update input_sources tracking (mossy fiber count)
        2. If using enhanced microcircuit, grow granule layer
        3. Grow synaptic weights (parallel fiber→purkinje connections)

        Args:
            source_name: Name of source to grow (must be "default")
            new_size: New total size for mossy fiber input
            initialization: Weight initialization strategy
            sparsity: Connection sparsity
            weight_scale: Weight magnitude scale
        """
        if source_name != "default":
            raise ValueError(
                f"Cerebellum only supports 'default' source, got '{source_name}'. "
                "Cerebellum uses concatenated multi-source input on single 'default' port."
            )

        old_mossy_size = self._total_input_size
        n_new_mossy = new_size - old_mossy_size

        if n_new_mossy <= 0:
            return  # No growth needed

        # Update input tracking
        self.input_sources["default"] = new_size
        self._total_input_size = new_size

        if self.use_enhanced:
            # Grow granule layer to accept more mossy fibers
            old_granule_size = self.granule_layer.n_granule
            expansion_factor = self.granule_size / old_mossy_size
            new_granule_size = int(new_size * expansion_factor)
            n_new_granule = new_granule_size - old_granule_size

            # Grow granule layer
            self.granule_layer.grow_output(n_new_granule)

            # Grow parallel fiber→purkinje weights
            for pc in self.purkinje_cells:
                pc.grow_input(n_new_granule)

            # Grow deep nuclei mossy fiber input
            self.deep_nuclei.grow_input(n_new_mossy, source="mossy")
        else:
            # Classic pathway: grow parallel fiber→purkinje weights directly
            old_weights = self.synaptic_weights["default"]
            n_new = n_new_mossy  # Direct mapping in classic mode

            # Initialize new input columns
            if initialization == "xavier":
                new_cols = WeightInitializer.xavier(
                    n_output=self.purkinje_size,
                    n_input=n_new,
                    gain=weight_scale,
                    device=self.device,
                )
            else:  # sparse_random (default)
                new_cols = WeightInitializer.sparse_random(
                    n_output=self.purkinje_size,
                    n_input=n_new,
                    sparsity=sparsity,
                    scale=weight_scale,
                    device=self.device,
                )

            # Concatenate new columns to existing weights
            expanded_weights = torch.cat([old_weights, new_cols], dim=1)
            self.synaptic_weights["default"] = nn.Parameter(expanded_weights)

            # Grow STP module if enabled
            if self.stp_pf_purkinje is not None:
                self.stp_pf_purkinje.grow(n_new, target="pre")

            # Grow trace manager
            self._trace_manager = self._trace_manager.grow_dimension(n_new, dimension="input")

        # Update base class n_input for compatibility
        self.n_input = self._total_input_size

    def grow_layer(
        self,
        layer_name: str,
        n_new: int,
        initialization: str = "sparse_random",
        sparsity: float = 0.1,
    ) -> None:
        """Grow specific cerebellar layer (SEMANTIC API).

        Args:
            layer_name: Layer to grow ('purkinje', 'granule')
            n_new: Number of neurons to add
            initialization: Weight init strategy
            sparsity: Connection sparsity

        Note:
            Currently only Purkinje cell growth supported (output layer).
            Granule layer growth would require mossy fiber expansion.
        """
        if layer_name.lower() in ["purkinje", "output"]:
            self.grow_output(n_new, initialization, sparsity)
        else:
            raise NotImplementedError(
                f"Direct growth of {layer_name} not yet supported. "
                f"Use grow_layer('purkinje', n) to grow Purkinje output."
            )

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Process input through cerebellar circuit.

        Args:
            inputs: Dict mapping source names to spike tensors
                   e.g., {"cortex": [n_cortex], "hippocampus": [n_hippo]}
            **kwargs: Additional arguments (unused)

        Returns:
            Output spikes [n_output] (1D bool tensor, ADR-004/005)

        Note:
            Theta modulation and timestep (dt_ms) computed internally from config
        """
        # Concatenate all input sources
        input_spikes = InputRouter.concatenate_sources(
            inputs,
            component_name="Cerebellum",
            n_input=self._total_input_size,
            device=self.device,
        )

        # Assert 1D input
        assert input_spikes.dim() == 1, (
            f"Cerebellum.forward: input_spikes must be 1D [n_input], "
            f"got shape {input_spikes.shape}. See ADR-005: No Batch Dimension."
        )
        assert input_spikes.shape[0] == self._total_input_size, (
            f"Cerebellum.forward: input_spikes has {input_spikes.shape[0]} neurons "
            f"but expected {self._total_input_size}."
        )

        if self.neurons.membrane is None:
            self.neurons.reset_state()

        cfg = self.config

        # ======================================================================
        # COMPUTE THETA MODULATION (from oscillator phase set by Brain)
        # ======================================================================
        encoding_mod, _retrieval_mod = compute_theta_encoding_retrieval(self._theta_phase)

        # Encoding phase: stronger input drive for error detection
        # Retrieval phase: stronger output for motor correction
        input_gain = 0.7 + 0.3 * encoding_mod  # 0.7-1.0
        # Note: retrieval_mod could be used for output gain if needed for motor commands

        # =====================================================================
        # ENHANCED MICROCIRCUIT PATHWAY (if enabled)
        # =====================================================================
        if self.use_enhanced:
            # 1. Granule layer: sparse expansion (4× expansion, 3% active)
            # Apply STP to mossy fiber→granule synapses if enabled
            if self.stp_mf_granule is not None:
                # Mossy fiber facilitation amplifies repeated activity
                mf_efficacy = self.stp_mf_granule(input_spikes.float())
                # Apply efficacy to granule layer input
                granule_spikes = self.granule_layer(input_spikes, mf_efficacy=mf_efficacy)
            else:
                granule_spikes = self.granule_layer(input_spikes)  # [n_granule]

            # 2. Purkinje cells: dendritic computation
            # Each Purkinje cell receives sparse parallel fibers
            purkinje_spikes = []
            for purkinje in self.purkinje_cells:
                # Get climbing fiber error if available (passed through DCN)
                climbing_fiber = torch.tensor(0.0, device=self.device)

                # Process parallel fibers + climbing fiber
                # EnhancedPurkinjeCell returns (simple_spikes, complex_spike_occurred)
                simple_spikes, _complex_spike = purkinje(granule_spikes, climbing_fiber)
                purkinje_spikes.append(simple_spikes)

            purkinje_output = torch.stack(purkinje_spikes)  # [n_output]

            # 3. Deep cerebellar nuclei: final integration
            # DCN receives Purkinje inhibition + mossy collaterals
            output_spikes = self.deep_nuclei(
                purkinje_spikes=purkinje_output,
                mossy_spikes=input_spikes,  # Direct mossy input to DCN
            )

            # For learning: use granule spikes as effective input
            effective_input = granule_spikes
        else:
            # CLASSIC PATHWAY: parallel fiber → Purkinje directly
            # Compute synaptic input - modulated by encoding phase

            # Apply STP to parallel fiber→Purkinje if enabled
            # This is CRITICAL for cerebellar timing precision
            if self.stp_pf_purkinje is not None:
                # Parallel fiber DEPRESSION: Fresh inputs strong, sustained fade
                # This implements temporal high-pass filter for change detection
                pf_efficacy = self.stp_pf_purkinje(input_spikes.float())  # [n_input, n_output]
                # Transpose to match weight shape [n_output, n_input]
                # Modulate synaptic weights by STP efficacy
                effective_weights = self.synaptic_weights["default"] * pf_efficacy.T
            else:
                effective_weights = self.synaptic_weights["default"]

            # 1D matmul: weights[n_output, n_input] @ input[n_input] → [n_output]
            g_exc = (effective_weights @ input_spikes.float()) * input_gain

            # =====================================================================
            # NOREPINEPHRINE GAIN MODULATION (Locus Coeruleus)
            # =====================================================================
            # High NE (arousal/stress): Increase motor gain → faster reactions
            # Low NE (baseline): Normal gain
            # Biological: NE modulates cerebellar Purkinje cell excitability
            ne_level = self.state.norepinephrine
            # NE gain: 1.0 (baseline) to 1.5 (high arousal)
            ne_gain = compute_ne_gain(ne_level)
            g_exc = g_exc * ne_gain

            # Forward through neurons (returns 1D bool spikes)
            output_spikes, _ = self.neurons(g_exc, None)

            # For learning: use original input
            effective_input = input_spikes

        # NOTE: All neuromodulators (DA, ACh, NE) are now managed centrally by Brain.
        # VTA updates dopamine, LC updates NE, NB updates ACh.
        # Brain broadcasts to all regions every timestep via _update_neuromodulators().
        # No local decay needed.

        # ======================================================================
        # Update STDP eligibility using trace manager
        # ======================================================================
        # Use trace manager for consolidated STDP computation
        self._trace_manager.update_traces(
            input_spikes=effective_input,  # Use granule spikes if enhanced
            output_spikes=output_spikes,
            dt_ms=self._dt_ms,
        )

        # Compute STDP weight change direction (raw LTP/LTD without combining)
        ltp, ltd = self._trace_manager.compute_ltp_ltd_separate(
            input_spikes=effective_input,
            output_spikes=output_spikes,
        )

        # Combine LTP and LTD with learning rate and heterosynaptic ratio
        stdp_dw = cfg.stdp_lr * (ltp - cfg.heterosynaptic_ratio * ltd)

        # Accumulate into eligibility trace (with decay)
        if isinstance(stdp_dw, torch.Tensor):
            self._trace_manager.accumulate_eligibility(stdp_dw, dt_ms=self._dt_ms)

        # Store output (NeuralRegion pattern - no state.t tracking)
        self.output_spikes = output_spikes
        # Store effective input for learning (granule spikes in enhanced mode)
        self.last_effective_input = effective_input

        # Axonal delays are handled by AxonalProjection pathways, not within regions
        return output_spikes  # type: ignore[no-any-return]

    def _generate_complex_spike_burst(
        self,
        error_magnitude: torch.Tensor,  # [n_output] - Absolute error per Purkinje cell
    ) -> torch.Tensor:
        """Generate complex spike burst with graded calcium influx.

        Converts error magnitude to complex spike bursts (2-7 spikelets per burst).
        Larger errors trigger longer bursts, which produce more calcium influx,
        leading to stronger LTD. This provides GRADED ERROR SIGNALING.

        Biological mechanism:
        - Small error (0.1): 2-3 spikelets → Ca²⁺ = 0.4-0.6 → small LTD
        - Medium error (0.5): 4-5 spikelets → Ca²⁺ = 0.8-1.0 → medium LTD
        - Large error (0.9): 6-7 spikelets → Ca²⁺ = 1.2-1.4 → large LTD

        Args:
            error_magnitude: Absolute error per Purkinje cell [n_output]

        Returns:
            total_calcium: Total Ca²⁺ influx per Purkinje cell [n_output]
                          (proportional to burst length)

        References:
        - Mathy et al. (2009): Complex spike burst length encodes error
        - Najafi & Medina (2013): Graded complex spikes → graded learning
        - Yang & Lisberger (2014): Burst amplitude predicts learning rate
        """
        cfg = self.config

        # Error magnitude → burst length (linear scaling)
        # error 0.0 → min_count spikes
        # error 1.0 → max_count spikes
        spike_range = cfg.max_complex_spike_count - cfg.min_complex_spike_count
        n_spikes = cfg.min_complex_spike_count + spike_range * torch.clamp(
            error_magnitude, 0.0, 1.0
        )

        # Round to integer spike counts (stochastic rounding for biological variability)
        n_spikes_floor = n_spikes.floor()
        n_spikes_frac = n_spikes - n_spikes_floor
        stochastic_round = (torch.rand_like(n_spikes_frac) < n_spikes_frac).float()
        n_spikes_int = n_spikes_floor + stochastic_round

        # Total calcium influx = n_spikes × ca2_per_spikelet
        total_calcium = n_spikes_int * cfg.ca2_per_spikelet

        return total_calcium

    def _apply_error_learning(
        self,
        output_spikes: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, Any]:
        """Apply error-corrective learning using accumulated eligibility.

        This is the core plasticity mechanism, called by deliver_error().
        Uses STDP eligibility traces modulated by error signal:
        - Eligibility accumulates spike-timing correlations during forward()
        - Error signal (climbing fiber) gates weight changes
        - Positive error → apply eligibility (LTP for correct timing)
        - Negative error → apply anti-eligibility (LTD for incorrect timing)

        Dopamine modulates the overall learning rate (arousal/attention effect).

        Args:
            output_spikes: Output spike pattern [n_output] (1D bool, ADR-005)
            target: Target output [n_output] (1D, ADR-005)

        Returns:
            Dictionary with learning metrics
        """
        # Assert 1D inputs
        assert output_spikes.dim() == 1, (
            f"Cerebellum._apply_error_learning: output_spikes must be 1D, "
            f"got shape {output_spikes.shape}"
        )
        assert target.dim() == 1, (
            f"Cerebellum._apply_error_learning: target must be 1D, " f"got shape {target.shape}"
        )

        cfg = self.config

        # ======================================================================
        # Compute error via climbing fiber system
        # ======================================================================
        error = self.climbing_fiber.compute_error(output_spikes.float(), target.float())
        # error is already 1D from compute_error()

        # ======================================================================
        # COMPLEX SPIKE BURST DYNAMICS (Phase 2B Enhancement)
        # ======================================================================
        # Convert error magnitude to complex spike bursts (2-7 spikelets).
        # Larger errors trigger longer bursts, producing more calcium influx,
        # which leads to stronger LTD. This provides GRADED ERROR SIGNALING.
        #
        # Without this: Error is binary (present/absent) or linear magnitude
        # With this: Error is converted to calcium dynamics (biophysical)
        #
        # This modulates the effective error magnitude used for learning,
        # making it proportional to the total calcium influx from the burst.
        if cfg.use_complex_spike_bursts:
            # Generate complex spike burst: error magnitude → calcium influx
            error_magnitude = error.abs()  # [n_output]
            calcium_influx = self._generate_complex_spike_burst(error_magnitude)  # [n_output]

            # Use calcium influx as the effective error magnitude
            # The sign of the error is preserved (LTP vs LTD direction)
            error_sign = torch.sign(error)  # [n_output]
            error = error_sign * calcium_influx  # [n_output]

        # ======================================================================
        # GAP JUNCTION SYNCHRONIZATION (Inferior Olive)
        # ======================================================================
        # IO neurons synchronize error signals via gap junctions (<1ms coupling).
        # This creates synchronized complex spikes across related Purkinje cells,
        # coordinating learning in multiple cerebellar modules.
        #
        # Biological rationale:
        # - IO neurons with similar error patterns are anatomically close
        # - Gap junctions synchronize their firing to create simultaneous
        #   complex spikes in multiple Purkinje cells
        # - This coordinates motor learning across cerebellar regions
        #
        # We model IO membrane potential as proportional to error magnitude
        # (larger errors → higher membrane potential → stronger complex spikes)
        if self.gap_junctions_io is not None:
            # Use error directly as proxy for IO membrane potential
            # Gap junctions compute: I_gap = Σ g[i,j] * (V[j] - V[i])
            # This adds a coupling current that pulls voltages toward neighbors
            io_membrane = error  # [n_output] (signed error)

            # Apply gap junction coupling to synchronize neighboring IO neurons
            # This adds the coupling current to the membrane potential
            # Note: Gap junctions can reduce signal if you're higher than neighbors
            coupled_membrane = io_membrane + self.gap_junctions_io(io_membrane)  # [n_output]

            # Use the synchronized membrane potential as the error signal
            # This makes neighboring Purkinje cells receive correlated error signals
            error = coupled_membrane

            # Store absolute value for state tracking
            self._io_membrane = coupled_membrane.abs()

        if error.abs().max() < cfg.error_threshold:
            return {"error": 0.0, "ltp": 0.0, "ltd": 0.0}

        # ======================================================================
        # BETA GATING - Motor timing modulation
        # ======================================================================
        # Climbing fiber learning is most effective during beta trough (movement initiation)
        # This implements the biological observation that error signals are processed
        # during movement execution, not during movement maintenance
        beta_gate = self._compute_beta_gate()

        # ======================================================================
        # Error-modulated STDP learning
        # ======================================================================
        # The key insight: use error to select WHICH neurons update
        # - Neurons with positive error (should have fired more) → LTP
        # - Neurons with negative error (fired too much) → LTD

        # Scale eligibility by per-neuron error direction
        # Positive error → apply eligibility (strengthen timing correlations)
        # Negative error → apply anti-eligibility (weaken timing correlations)
        # error is already 1D [n_output]
        error_sign = torch.sign(error).unsqueeze(1)  # [n_output, 1]

        # Modulate eligibility by error magnitude and sign
        # Dopamine provides arousal/attention modulation (from VTA via Brain)
        # Beta gate provides motor timing modulation (movement initiation window)
        effective_lr = self.get_effective_learning_rate() * beta_gate
        dw = self.stdp_eligibility * error_sign * error.abs().unsqueeze(1) * effective_lr

        # Update weights with hard bounds
        # For enhanced cerebellum, skip weight updates (learning happens in Purkinje cells)
        # For classic cerebellum, update global weights
        if not self.use_enhanced:
            # Classic pathway: update global weights
            old_weights = self.synaptic_weights["default"].clone()
            self.synaptic_weights["default"].data = clamp_weights(
                self.synaptic_weights["default"] + dw,
                self.config.w_min,
                self.config.w_max,
                inplace=False,
            )

            # Synaptic scaling for homeostasis using UnifiedHomeostasis
            if cfg.homeostasis_enabled:
                self.synaptic_weights["default"].data = self.homeostasis.normalize_weights(
                    self.synaptic_weights["default"], dim=1
                )

            actual_dw = self.synaptic_weights["default"] - old_weights
        else:
            # Enhanced: Per-Purkinje cell dendritic learning (biologically accurate)
            # Each Purkinje cell learns its own parallel fiber→dendrite synaptic weights
            # This implements the classical LTD mechanism (Ito, 2001)
            actual_dw = torch.zeros_like(self.synaptic_weights["default"])

            # Get granule spikes (stored during forward pass)
            if not hasattr(self, "last_effective_input"):
                # No forward pass yet, skip learning
                return {"error": 0.0, "ltp": 0.0, "ltd": 0.0}

            granule_spikes = self.last_effective_input  # [n_granule]

            # Update each Purkinje cell's dendritic weights
            for i, purkinje in enumerate(self.purkinje_cells):
                if purkinje.pf_synaptic_weights is None:
                    # Weights not initialized yet (no forward pass)
                    continue

                old_pf_weights = purkinje.pf_synaptic_weights.clone()  # type: ignore[operator]

                # Compute per-cell error
                cell_error = error[i]  # Scalar error for this Purkinje cell

                # Delta rule: Δw = lr × error × pre_activity
                # Shape: [n_granule] (parallel fiber weights for this Purkinje cell)
                cell_dw = cfg.learning_rate * cell_error * granule_spikes.float()  # type: ignore[union-attr]

                # Update this Purkinje cell's dendritic weights
                new_weights = purkinje.pf_synaptic_weights.squeeze(0) + cell_dw  # type: ignore[operator]
                purkinje.pf_synaptic_weights.data = clamp_weights(
                    new_weights.unsqueeze(0), cfg.w_min, cfg.w_max, inplace=False
                )

                # Apply synaptic homeostasis per Purkinje cell
                if cfg.homeostasis_enabled:
                    purkinje.pf_synaptic_weights.data = self.homeostasis.normalize_weights(
                        purkinje.pf_synaptic_weights.data, dim=1
                    )

                # Track actual weight change for this cell (for metrics)
                # Note: actual_dw has shape [n_output, n_input] but Purkinje weights are [1, n_granule]
                # We'll just record the sum of changes per cell
                cell_weight_change = (  # type: ignore[operator]
                    (purkinje.pf_synaptic_weights.squeeze(0) - old_pf_weights.squeeze(0))
                    .sum()
                    .item()
                )
                if i < actual_dw.shape[0]:
                    # Store change magnitude in first column for metrics
                    actual_dw[i, 0] = cell_weight_change

        ltp = actual_dw[actual_dw > 0].sum().item() if (actual_dw > 0).any() else 0.0
        ltd = actual_dw[actual_dw < 0].sum().item() if (actual_dw < 0).any() else 0.0

        return {
            "error": error.abs().mean().item(),
            "ltp": ltp,
            "ltd": ltd,
            "net_change": ltp + ltd,
            "eligibility_max": self.stdp_eligibility.abs().max().item(),
            "beta_gate": beta_gate,  # Motor timing window
        }

    def deliver_error(
        self,
        target: torch.Tensor,
        output_spikes: Optional[torch.Tensor] = None,
        target_neuron: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Deliver error signal (climbing fiber) and apply learning.

        This is the main learning API for cerebellum - analogous to
        deliver_reward() for striatum. The pattern is:
        1. Run forward() to build eligibility traces (spike-timing correlations)
        2. Call deliver_error() when target/error is known

        The climbing fiber carries the error signal from inferior olive:
        - Positive error (target - actual > 0): should have fired more → LTP
        - Negative error (target - actual < 0): fired too much → LTD

        Dopamine (set via set_dopamine()) modulates learning rate,
        providing arousal/attention effects on motor learning.

        Args:
            target: Target output pattern [n_output] or [batch, n_output]
            output_spikes: Current output (if None, uses last state from forward())
            target_neuron: Single target neuron index (alternative to target tensor)

        Returns:
            Learning metrics dict with error, ltp, ltd, net_change
        """
        # Handle target_neuron convenience parameter
        if target_neuron is not None:
            target = torch.zeros(1, self.purkinje_size, device=self.device)
            target[0, target_neuron] = 1.0

        if output_spikes is None:
            output_spikes = self.output_spikes

        if output_spikes is None:
            return {"error": 0.0, "ltp": 0.0, "ltd": 0.0}

        # Apply error-corrective learning using accumulated eligibility
        return self._apply_error_learning(output_spikes, target)

    def reset_state(self) -> None:
        super().reset_state()

        # Reset subsystems (trace manager handles its own traces)
        self._reset_subsystems(
            "_trace_manager", "climbing_fiber", "stp_pf_purkinje", "stp_mf_granule"
        )

        # Initialize IO membrane for gap junctions
        if self.gap_junctions_io is not None:
            self._io_membrane = torch.zeros(self.purkinje_size, device=self.device)

    def set_training_step(self, step: int) -> None:
        """Update the current training step for neurogenesis tracking.

        This should be called by the training loop to keep track of when neurons
        are created during growth events.

        Args:
            step: Current global training step
        """
        # Cerebellum doesn't currently support neurogenesis, but we implement
        # this method for API consistency with other regions
        return

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics in standardized DiagnosticsDict format.

        Returns consolidated diagnostic information about:
        - Activity: Output spike statistics (Purkinje cells or DCN)
        - Plasticity: Weight statistics for parallel fiber connections
        - Health: Trace magnitudes, weight bounds, error signals
        - Neuromodulators: Not applicable (cerebellum uses error signals)
        - Region-specific: Granule/Purkinje/climbing fiber details, oscillations

        This is the primary diagnostic interface for the Cerebellum.
        """
        from thalia.core.diagnostics_schema import (
            compute_activity_metrics,
            compute_health_metrics,
            compute_plasticity_metrics,
        )

        cfg = self.config

        # Compute activity metrics from output spikes
        activity = compute_activity_metrics(
            output_spikes=(
                self.output_spikes
                if self.output_spikes is not None
                else torch.zeros(self.purkinje_size, device=self.device)
            ),
            total_neurons=self.purkinje_size,
        )

        # Compute plasticity metrics from parallel fiber weights
        plasticity = None
        if cfg.learning_enabled:
            plasticity = compute_plasticity_metrics(
                weights=self.synaptic_weights["default"].data,
                learning_rate=cfg.learning_rate_ltp,  # Use LTP rate (primary)
            )
            # Add LTD rate as well
            plasticity["learning_rate_ltd"] = cfg.learning_rate_ltd  # type: ignore[typeddict-item]

        # Compute health metrics
        health_tensors = {
            "weights": self.synaptic_weights["default"].data,
        }
        if self.input_trace is not None:
            health_tensors["input_trace"] = self.input_trace
        if self.output_trace is not None:
            health_tensors["output_trace"] = self.output_trace
        if self.stdp_eligibility is not None:
            health_tensors["eligibility"] = self.stdp_eligibility

        health = compute_health_metrics(
            state_tensors=health_tensors,
            firing_rate=activity.get("firing_rate", 0.0),
        )

        # Cerebellum doesn't use neuromodulators (uses error signals)
        neuromodulators: dict[str, float] = {}

        # Region-specific custom metrics
        region_specific = {
            "architecture": {
                "n_input": self._total_input_size,
                "n_output": self.n_output,
                "use_enhanced_microcircuit": cfg.use_enhanced_microcircuit,
            },
            "error_signaling": {
                "error_mean": self.climbing_fiber.get_state().get("last_error_mean", 0.0),
                "error_threshold": cfg.error_threshold,
            },
            "traces": {
                "input_mean": (
                    self.input_trace.mean().item() if self.input_trace is not None else 0.0
                ),
                "output_mean": (
                    self.output_trace.mean().item() if self.output_trace is not None else 0.0
                ),
                "eligibility_mean": (
                    self.stdp_eligibility.mean().item()
                    if self.stdp_eligibility is not None
                    else 0.0
                ),
            },
            "oscillations": {
                "beta_phase": self._beta_phase,
                "gamma_phase": self._gamma_phase,
                "theta_phase": self._theta_phase,
                "beta_amplitude": self._beta_amplitude_effective,
                "gamma_amplitude": self._gamma_amplitude_effective,
            },
        }

        # Return in standardized format
        return {
            "activity": activity,
            "plasticity": plasticity,
            "health": health,
            "neuromodulators": neuromodulators,
            "region_specific": region_specific,
        }

    def get_full_state(self) -> Dict[str, Any]:
        """Get complete state for checkpointing.

        Returns state dictionary with keys:
        - weights: Parallel fiber to Purkinje cell weights (classic) or granule/DCN (enhanced)
        - region_state: Neuron state, traces
        - learning_state: Eligibility traces, climbing fiber state
        - enhanced_state: Granule layer, Purkinje cells, DCN (if use_enhanced)
        - config: Configuration for validation
        """
        state = self.get_state()
        state_dict = state.to_dict()
        # Add config to dict for checkpoint validation
        state_dict["config"] = asdict(self.config)
        return state_dict

    def load_full_state(self, state: Dict[str, Any]) -> None:
        """Load complete state from checkpoint.

        Args:
            state: State dictionary from get_full_state()
        """
        state_obj = CerebellumState.from_dict(state, device=str(self.device))
        self.load_state(state_obj)

    # ========================================================================
    # REGIONSTATE PROTOCOL IMPLEMENTATION (Phase 3.1)
    # ========================================================================

    def get_state(self) -> CerebellumState:
        """Get current state as CerebellumState dataclass.

        Returns unified state for checkpointing following RegionState protocol.
        This replaces the dict-based get_full_state() for standardized serialization.

        Returns:
            CerebellumState with all region state.
        """
        # Get STP state for parallel fiber→Purkinje (classic mode)
        stp_pf_state = None
        if self.stp_pf_purkinje is not None:
            stp_pf_state = self.stp_pf_purkinje.get_state()

        # Get STP state for mossy fiber→granule (enhanced mode)
        stp_mf_state = None
        if self.stp_mf_granule is not None:
            stp_mf_state = self.stp_mf_granule.get_state()

        # Classic mode neuron state
        v_mem = None
        g_exc = None
        g_inh = None
        if not self.use_enhanced and self.neurons is not None:
            neuron_state = self.neurons.get_state()
            v_mem = neuron_state["membrane"]  # ConductanceLIF uses "membrane" key
            g_exc = neuron_state["g_E"]  # ConductanceLIF uses "g_E" key
            g_inh = neuron_state["g_I"]  # ConductanceLIF uses "g_I" key

        # Enhanced microcircuit state
        granule_state = None
        purkinje_state = None
        dcn_state = None
        if self.use_enhanced:
            granule_state = self.granule_layer.get_full_state()  # type: ignore[misc]
            purkinje_state = [pc.get_state() for pc in self.purkinje_cells]  # type: ignore[operator]
            dcn_state = self.deep_nuclei.get_full_state()  # type: ignore[operator]

        # Get climbing fiber error
        cf_state = self.climbing_fiber.get_state()
        climbing_error = cf_state.get("error", torch.zeros(self.purkinje_size, device=self.device))

        # Get IO membrane (gap junction coupling state)
        # If gap junctions are disabled, io_membrane will be None
        io_mem = self._io_membrane

        return CerebellumState(
            # Traces
            input_trace=self.input_trace.clone(),
            output_trace=self.output_trace.clone(),
            stdp_eligibility=self.stdp_eligibility.clone(),
            # Error signal
            climbing_fiber_error=climbing_error.clone(),
            io_membrane=io_mem.clone() if io_mem is not None else None,
            # Classic neuron state
            v_mem=v_mem.clone() if v_mem is not None else None,
            g_exc=g_exc.clone() if g_exc is not None else None,
            g_inh=g_inh.clone() if g_inh is not None else None,
            # STP state
            stp_pf_purkinje_state=stp_pf_state,  # type: ignore[arg-type]
            stp_mf_granule_state=stp_mf_state,  # type: ignore[arg-type]
            # Neuromodulators (from NeuromodulatorMixin state)
            dopamine=self.state.dopamine,
            acetylcholine=self.state.acetylcholine,
            norepinephrine=self.state.norepinephrine,
            # Enhanced microcircuit
            granule_layer_state=granule_state,  # type: ignore[arg-type]
            purkinje_cells_state=purkinje_state,
            deep_nuclei_state=dcn_state,
        )

    def load_state(self, state: CerebellumState) -> None:
        """Load state from CerebellumState dataclass.

        Restores all region state from unified dataclass following RegionState protocol.
        This replaces dict-based load_full_state() for standardized deserialization.

        Args:
            state: CerebellumState to restore from.
        """
        # Use mixin helpers for common restoration
        super().load_state(state)  # Restores: membrane, conductances, traces, neuromodulators

        # Cerebellum-specific state restoration
        self._load_custom_state(state)

    def _load_custom_state(self, state: CerebellumState) -> None:
        """Restore cerebellum-specific state components.

        Args:
            state: CerebellumState to restore from.
        """
        # Restore traces
        self._trace_manager.input_trace.copy_(state.input_trace.to(self.device))  # type: ignore[union-attr]
        self._trace_manager.output_trace.copy_(state.output_trace.to(self.device))  # type: ignore[union-attr]
        self._trace_manager.eligibility.copy_(state.stdp_eligibility.to(self.device))  # type: ignore[union-attr]

        # Restore climbing fiber error
        self.climbing_fiber.error.copy_(state.climbing_fiber_error.to(self.device))  # type: ignore[union-attr]

        # Restore IO membrane (gap junction state)
        if state.io_membrane is not None:
            self._io_membrane = state.io_membrane.to(self.device)

        # Restore classic neuron state (if not using enhanced microcircuit)
        if not self.use_enhanced and self.neurons is not None:
            if state.v_mem is not None and state.g_exc is not None and state.g_inh is not None:
                neuron_state = {
                    "membrane": state.v_mem.to(self.device),  # Map to ConductanceLIF "membrane" key
                    "g_E": state.g_exc.to(self.device),  # Map to ConductanceLIF "g_E" key
                    "g_I": state.g_inh.to(self.device),  # Map to ConductanceLIF "g_I" key
                    "g_adapt": None,  # Not tracked in CerebellumState
                    "refractory": None,  # Not tracked in CerebellumState
                }
                self.neurons.load_state(neuron_state)

        # Restore enhanced microcircuit state
        if self.use_enhanced:
            if state.granule_layer_state is not None:
                self.granule_layer.load_full_state(state.granule_layer_state)  # type: ignore[arg-type]
            if state.purkinje_cells_state is not None:
                for pc, pc_state in zip(self.purkinje_cells, state.purkinje_cells_state):
                    pc.load_state(pc_state)  # type: ignore[operator]
            if state.deep_nuclei_state is not None:
                self.deep_nuclei.load_full_state(state.deep_nuclei_state)

        # Restore STP state
        if self.stp_pf_purkinje is not None and state.stp_pf_purkinje_state is not None:
            self.stp_pf_purkinje.to(self.device)  # Move module to target device first
            # Reset state to ensure u/x are on correct device (they're not nn.Parameters)
            self.stp_pf_purkinje.reset_state()
            # Then load the actual state values
            self.stp_pf_purkinje.load_state(state.stp_pf_purkinje_state)  # type: ignore[arg-type]

        if self.stp_mf_granule is not None and state.stp_mf_granule_state is not None:
            self.stp_mf_granule.to(self.device)  # Move module to target device first
            # Reset state to ensure u/x are on correct device (they're not nn.Parameters)
            self.stp_mf_granule.reset_state()
            # Then load the actual state values
            self.stp_mf_granule.load_state(state.stp_mf_granule_state)  # type: ignore[arg-type]

        # Neuromodulators already restored by super().load_state() via _restore_neuromodulators()
