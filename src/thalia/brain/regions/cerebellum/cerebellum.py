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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.nn as nn

from thalia.brain.configs import CerebellumConfig
from thalia.components.synapses import ShortTermPlasticity, STPConfig
from thalia.diagnostics import compute_plasticity_metrics
from thalia.learning import (
    EligibilitySTDPConfig,
    EligibilityTraceManager,
)
from thalia.learning import UnifiedHomeostasis, UnifiedHomeostasisConfig
from thalia.typing import RegionLayerSizes, RegionSpikesDict

from .deep_nuclei import DeepCerebellarNuclei
from .granule_layer import GranuleCellLayer
from .purkinje_cell import EnhancedPurkinjeCell

from ..neural_region import NeuralRegion
from ..region_registry import register_region

if TYPE_CHECKING:
    from thalia.components.gap_junctions import GapJunctionCoupling


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


@register_region(
    "cerebellum",
    description="Supervised error-corrective learning via climbing fiber error signals",
    version="1.0",
    author="Thalia Project",
    config_class=CerebellumConfig,
)
class Cerebellum(NeuralRegion[CerebellumConfig]):
    """Cerebellar region with supervised error-corrective learning."""

    # Declarative output ports (auto-registered by base class)
    OUTPUT_PORTS = {
        "prediction": "purkinje_size",
    }

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, config: CerebellumConfig, region_layer_sizes: RegionLayerSizes):
        """Initialize cerebellum."""
        super().__init__(config=config, region_layer_sizes=region_layer_sizes)

        # =====================================================================
        # EXTRACT LAYER SIZES
        # =====================================================================
        self.granule_size = region_layer_sizes["granule_size"]
        self.purkinje_size = region_layer_sizes["purkinje_size"]

        # =====================================================================
        # MOSSY FIBER LAYER (Pontine Nuclei equivalent)
        # =====================================================================
        # Biology: Cortex projects to pontine nuclei → mossy fibers → cerebellum
        # We model this as an intermediate representation layer
        # Size: Typically ~10% of granule cells (biological ratio)
        self.n_mossy = max(int(self.granule_size * 0.1), 50)  # At least 50 mossy fibers

        # Mossy fiber neurons (simple pass-through for now, could add dynamics later)
        # Store as buffer for potential future mossy fiber dynamics
        self.register_buffer(
            "_mossy_fiber_state",
            torch.zeros(self.n_mossy, dtype=torch.bool, device=self.device)
        )

        # =====================================================================
        # CLIMBING FIBER SYSTEM
        # =====================================================================
        self.climbing_fiber = ClimbingFiberSystem(
            purkinje_size=self.purkinje_size,
            device=self.device,
        )

        # =====================================================================
        # ENHANCED MICROCIRCUIT (granule-Purkinje-DCN)
        # =====================================================================
        # Granule cell layer (sparse expansion from mossy fibers)
        self.granule_layer = GranuleCellLayer(
            n_mossy_fibers=self.n_mossy,
            expansion_factor=self.granule_size / self.n_mossy,
            sparsity=self.config.granule_sparsity,
            device=self.device,
        )

        # Purkinje cells (one per output neuron)
        # Pass granule layer size as n_parallel_fibers for eager initialization
        self.purkinje_cells = torch.nn.ModuleList(
            [
                EnhancedPurkinjeCell(
                    n_parallel_fibers=self.granule_layer.n_granule,
                    n_dendrites=self.config.purkinje_n_dendrites,
                    device=self.device,
                )
                for _ in range(self.purkinje_size)
            ]
        )

        # Deep cerebellar nuclei (final output)
        # Receives both Purkinje inhibition and mossy fiber collaterals
        self.deep_nuclei = DeepCerebellarNuclei(
            n_output=self.purkinje_size,
            n_purkinje=self.purkinje_size,
            n_mossy=self.n_mossy,  # Mossy fiber collaterals
            device=self.device,
        )

        # =====================================================================
        # ELIGIBILITY TRACE MANAGER for STDP
        # =====================================================================
        stdp_config = EligibilitySTDPConfig(
            stdp_tau_ms=self.config.tau_plus_ms,  # Use tau_plus_ms as STDP tau
            eligibility_tau_ms=self.config.eligibility_tau_ms,
            stdp_lr=self.config.learning_rate,
            a_plus=1.0,
            a_minus=self.config.heterosynaptic_ratio,
            w_min=config.w_min,
            w_max=config.w_max,
            heterosynaptic_ratio=self.config.heterosynaptic_ratio,
        )
        self._trace_manager = EligibilityTraceManager(
            n_input=self.granule_layer.n_granule,
            n_output=self.purkinje_size,
            config=stdp_config,
            device=self.device,
        )

        # IO membrane potential for gap junction coupling
        self._io_membrane: Optional[torch.Tensor] = None

        # Homeostasis for synaptic scaling
        self.homeostasis = UnifiedHomeostasis(UnifiedHomeostasisConfig(
            weight_budget=config.weight_budget,  # Total budget per neuron
            w_min=config.w_min,
            w_max=config.w_max,
            soft_normalization=config.soft_normalization,
            normalization_rate=config.normalization_rate,
            device=self.device,
        ))

        # =====================================================================
        # HOMEOSTATIC INTRINSIC PLASTICITY (Adaptive Gain)
        # =====================================================================
        # Purkinje cells have high spontaneous firing rates (~40-50 Hz in biology)
        # Bootstrap solution for silent network problem
        # EMA tracking of firing rates
        self.register_buffer("firing_rate", torch.zeros(self.purkinje_size, device=self.device))

        # Adaptive gains (per neuron)
        self.gain = nn.Parameter(torch.ones(self.purkinje_size, device=self.device), requires_grad=False)

        # Configuration
        self._target_rate = config.target_firing_rate
        self._gain_lr = config.gain_learning_rate
        self._baseline_noise = config.baseline_noise_current

        # EMA alpha for firing rate tracking
        self._firing_rate_alpha = self.dt_ms / config.gain_tau_ms

        # Adaptive threshold plasticity
        self._threshold_lr = config.threshold_learning_rate
        self._threshold_min = config.threshold_min
        self._threshold_max = config.threshold_max

        # =====================================================================
        # SHORT-TERM PLASTICITY (STP)
        # =====================================================================
        # Per-source STP modules for mossy fiber→granule synapses
        # Created dynamically as sources are added (see _create_source_stp)
        # Biology: Different mossy fiber types have different facilitation/depression
        self.stp_modules: Dict[str, ShortTermPlasticity] = {}

        # Parallel Fibers→Purkinje: DEPRESSING (CRITICAL for timing)
        # This implements the temporal high-pass filter that makes the
        # cerebellum respond to CHANGES rather than sustained input.
        # Without this, cerebellar timing precision is severely impaired.
        self.stp_pf_purkinje = ShortTermPlasticity(
            n_pre=self.granule_layer.n_granule,
            n_post=self.purkinje_size,
            config=STPConfig.from_type(self.config.stp_pf_purkinje_type),
            per_synapse=True,  # Per-synapse dynamics for maximum precision
        )
        self.stp_pf_purkinje.to(self.device)

        # Initialize for forward() storage (fixes attribute outside __init__)
        self.last_effective_input: Optional[torch.Tensor] = None

        # =====================================================================
        # GAP JUNCTIONS (Inferior Olive Synchronization)
        # =====================================================================
        # IO neurons are densely coupled via gap junctions, creating synchronized
        # complex spikes across multiple Purkinje cells. This is critical for
        # coordinated motor learning across cerebellar modules.
        #
        # Gap junctions will be initialized after first source is added,
        # using the granule→Purkinje connectivity pattern to infer IO neighborhoods.
        # Purkinje cells with similar parallel fiber inputs receive error signals
        # from neighboring (gap junction-coupled) IO neurons.
        self.gap_junctions_io: Optional[GapJunctionCoupling] = None

        # =====================================================================
        # POST-INITIALIZATION
        # =====================================================================
        self.__post_init__()

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def _forward_internal(self, inputs: RegionSpikesDict) -> None:
        """Process input through cerebellar circuit."""
        # =====================================================================
        # MULTI-SOURCE SYNAPTIC INTEGRATION
        # =====================================================================
        # Biology: Different sources (cortex, spinal, brainstem) project to pontine
        # nuclei, which give rise to mossy fibers. We model this as a integration stage.
        mossy_fiber_currents = self._integrate_multi_source_synaptic_inputs(
            inputs=inputs,
            n_neurons=self.n_mossy,
            weight_key_suffix="_mossy",  # E.g., "cortex:l5_mossy"
            apply_stp=True,  # Per-source facilitation/depression
        )

        # Convert currents to spikes (mossy fibers are spiking)
        # Simple threshold: current > 0 produces spike
        # TODO: Could add mossy fiber neuron dynamics here if needed
        mossy_fiber_spikes = (mossy_fiber_currents > 0.1).bool()  # [n_mossy]
        self._mossy_fiber_state = mossy_fiber_spikes  # Cache for state inspection

        # =====================================================================
        # STAGE 2: MOSSY FIBERS → GRANULE CELLS (Sparse Expansion)
        # =====================================================================
        # Biology: Mossy fibers project to granule cells (4-5x expansion)
        # Granule layer has its own weights: [n_granule, n_mossy]
        granule_spikes = self.granule_layer(mossy_fiber_spikes)  # [n_granule]

        # =====================================================================
        # STAGE 3: GRANULE CELLS → PURKINJE CELLS (Parallel Fibers)
        # =====================================================================
        # Each Purkinje cell receives sparse parallel fibers
        purkinje_spikes = []
        for purkinje in self.purkinje_cells:
            # Get climbing fiber error if available (passed through DCN)
            climbing_fiber = torch.tensor(0.0, device=self.device)

            # Process parallel fibers + climbing fiber
            # EnhancedPurkinjeCell returns (simple_spikes, complex_spike_occurred)
            simple_spikes, _complex_spike = purkinje(granule_spikes, climbing_fiber)
            purkinje_spikes.append(simple_spikes)

        purkinje_output = torch.stack(purkinje_spikes)  # [purkinje_size]

        # =====================================================================
        # STAGE 4: PURKINJE + MOSSY COLLATERALS → DCN (Final Output)
        # =====================================================================
        # Biology: DCN receives both Purkinje inhibition and mossy fiber collaterals
        # The mossy collaterals provide excitatory drive that Purkinje inhibits
        output_spikes = self.deep_nuclei(
            purkinje_spikes=purkinje_output,
            mossy_spikes=mossy_fiber_spikes,  # Proper collaterals!
        )

        # For learning: use granule spikes as effective input
        effective_input = granule_spikes

        # ======================================================================
        # Update STDP eligibility using trace manager
        # ======================================================================
        # Use trace manager for consolidated STDP computation
        self._trace_manager.update_traces(
            input_spikes=effective_input,  # Use granule spikes if enhanced
            output_spikes=output_spikes,
            dt_ms=self.dt_ms,
        )

        # Compute STDP weight change direction (raw LTP/LTD without combining)
        ltp, ltd = self._trace_manager.compute_ltp_ltd_separate(
            input_spikes=effective_input,
            output_spikes=output_spikes,
        )

        # Combine LTP and LTD with learning rate and heterosynaptic ratio
        stdp_dw = self.config.learning_rate * (ltp - self.config.heterosynaptic_ratio * ltd)

        # Accumulate into eligibility trace (with decay)
        if isinstance(stdp_dw, torch.Tensor):
            self._trace_manager.accumulate_eligibility(stdp_dw, dt_ms=self.dt_ms)

        # Store output spikes
        self.output_spikes = output_spikes
        # Store effective input for learning (granule spikes in enhanced mode)
        self.last_effective_input = effective_input

        # =====================================================================
        # SET PORT OUTPUTS
        # =====================================================================
        self.set_port_output("prediction", output_spikes)

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
        if hasattr(self, "purkinje_neurons") and self.purkinje_neurons is not None:
            self.purkinje_neurons.update_temporal_parameters(dt_ms)
        if hasattr(self, "granule_neurons") and self.granule_neurons is not None:
            self.granule_neurons.update_temporal_parameters(dt_ms)

        # Update STP components
        if hasattr(self, "stp_pf_purkinje") and self.stp_pf_purkinje is not None:
            self.stp_pf_purkinje.update_temporal_parameters(dt_ms)
        if hasattr(self, "stp_mf_granule") and self.stp_mf_granule is not None:
            self.stp_mf_granule.update_temporal_parameters(dt_ms)

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information for this region."""
        # Compute plasticity metrics from parallel fiber weights
        plasticity = compute_plasticity_metrics(
            weights=self.synaptic_weights["default"].data,
            learning_rate=self.config.learning_rate_ltp,  # Use LTP rate (primary)
        )

        return {
            "plasticity": plasticity,
            "learning_rate_ltd": self.config.learning_rate_ltd,  # Add LTD rate as well
        }
