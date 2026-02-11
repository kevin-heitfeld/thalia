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

from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.nn as nn

from thalia.brain.configs import CerebellumConfig
from thalia.components.synapses import ShortTermPlasticity, STPConfig
from thalia.learning import (
    EligibilitySTDPConfig,
    EligibilityTraceManager,
)
from thalia.learning import UnifiedHomeostasis, UnifiedHomeostasisConfig
from thalia.typing import PopulationName, PopulationSizes, RegionSpikesDict

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

    OUTPUT_POPULATIONS: Dict[PopulationName, str] = {
        "prediction": "purkinje_size",
    }

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, config: CerebellumConfig, population_sizes: PopulationSizes):
        """Initialize cerebellum."""
        super().__init__(config=config, population_sizes=population_sizes)

        # =====================================================================
        # EXTRACT LAYER SIZES
        # =====================================================================
        self.granule_size = population_sizes["granule_size"]
        self.purkinje_size = population_sizes["purkinje_size"]

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

    def forward(self, region_inputs: RegionSpikesDict) -> RegionSpikesDict:
        """Process input through cerebellar circuit."""
        self._pre_forward(region_inputs)

        # =====================================================================
        # MULTI-SOURCE SYNAPTIC INTEGRATION
        # =====================================================================
        # Biology: Different sources (cortex, spinal, brainstem) project to pontine
        # nuclei, which give rise to mossy fibers. We model this as a integration stage.
        # Note: Weights project directly to granule layer (granule_size), not to mossy
        # fiber intermediates. The granule layer handles internal expansion/sparsification.
        mossy_fiber_currents = self._integrate_multi_source_synaptic_inputs(
            inputs=region_inputs,
            n_neurons=self.granule_size,  # Weights target granule cells directly
            weight_key_suffix="",  # Fixed: weights are "cortex:l5" not "cortex:l5_mossy"
            apply_stp=True,  # Per-source facilitation/depression
        )

        # Convert currents to spikes at granule cell level
        # Since weights project directly to granule layer, skip mossy fiber intermediate
        # Add baseline noise for spontaneous activity (biology: granule cells have ~5Hz baseline)
        if self._baseline_noise > 0:
            noise = torch.randn(self.granule_size, device=self.device) * self._baseline_noise
            mossy_fiber_currents = mossy_fiber_currents + noise

        # Simple threshold: current > 0 produces spike (lowered from 0.1 to allow weaker inputs)
        granule_spikes = (mossy_fiber_currents > 0.0).bool()  # [granule_size]

        # For compatibility with state inspection, create mossy fiber approximation
        # by downsampling granule spikes to mossy fiber dimensionality
        # This is just for monitoring - not used in forward computation
        step = self.granule_size // self.n_mossy
        self._mossy_fiber_state = granule_spikes[::step][:self.n_mossy]  # [n_mossy]

        # =====================================================================
        # STAGE 2: GRANULE CELLS → PURKINJE CELLS (Parallel Fibers)
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
        # STAGE 3: PURKINJE + MOSSY COLLATERALS → DCN (Final Output)
        # =====================================================================
        # Biology: DCN receives both Purkinje inhibition and mossy fiber collaterals
        # The mossy collaterals provide excitatory drive that Purkinje inhibits
        # Use mossy fiber state (downsampled from granule spikes) for DCN input
        output_spikes = self.deep_nuclei(
            purkinje_spikes=purkinje_output,
            mossy_spikes=self._mossy_fiber_state,  # Use cached mossy fiber approximation
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

        # ======================================================================
        # APPLY CEREBELLAR LEARNING (Parallel Fiber → Purkinje)
        # ======================================================================
        # Biology: Climbing fiber error signals gate the application of parallel
        # fiber eligibility traces. When climbing fiber fires (error detected),
        # active parallel fibers undergo LTD. When no climbing fiber (correct),
        # active parallel fibers undergo LTP.
        #
        # Implementation: We use the accumulated eligibility traces and apply them
        # to each Purkinje cell's dendritic weights. In the full implementation,
        # climbing fiber error would gate these updates (error × eligibility).
        # For now, we apply the eligibility-based STDP learning.

        # Get eligibility from trace manager
        eligibility = self._trace_manager.eligibility  # [n_granule, n_purkinje]

        # Apply weight updates to each Purkinje cell's dendritic weights
        for purkinje_idx, purkinje_cell in enumerate(self.purkinje_cells):
            # Get this Purkinje cell's eligibility trace
            # eligibility shape: [n_granule, n_purkinje]
            # We need [n_granule] for this Purkinje cell
            # Index the second dimension (purkinje) to get all granule inputs for this cell
            cell_eligibility = eligibility[purkinje_idx, :]  # [n_granule]

            # Reshape to match dendritic_weights: [1, n_parallel_fibers]
            weight_update = cell_eligibility.unsqueeze(0)  # [1, n_granule]

            # Apply update with weight bounds
            # Biology: Parallel fiber synapses have limited dynamic range
            new_weights = torch.clamp(
                purkinje_cell.dendritic_weights.data + weight_update,
                min=self.config.w_min,
                max=self.config.w_max,
            )
            purkinje_cell.dendritic_weights.data = new_weights

        # ======================================================================
        # APPLY HOMEOSTATIC SYNAPTIC SCALING (Per-Source Mossy Fiber Inputs)
        # ======================================================================
        # Apply homeostatic scaling to mossy fiber input weights to prevent
        # runaway excitation or silencing. Each source (cortex, spinal, etc.)
        # has independent weights that need homeostatic regulation.
        for source_name in list(self.input_sources.keys()):
            weight_key = f"{source_name}_mossy"
            if weight_key in self.synaptic_weights:
                self.synaptic_weights[weight_key].data = self.homeostasis.normalize_weights(
                    self.synaptic_weights[weight_key].data, dim=1
                )

        # Store output spikes
        self.output_spikes = output_spikes
        # Store effective input for learning (granule spikes in enhanced mode)
        self.last_effective_input = effective_input

        region_outputs: RegionSpikesDict = {
            "prediction": output_spikes,
        }

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

        # Update neurons
        self.deep_nuclei.dcn_neurons.update_temporal_parameters(dt_ms)
        self.granule_layer.granule_neurons.update_temporal_parameters(dt_ms)
        for purkinje_cell in self.purkinje_cells:
            purkinje_cell.soma_neurons.update_temporal_parameters(dt_ms)

        # Update STP components
        self.stp_pf_purkinje.update_temporal_parameters(dt_ms)
