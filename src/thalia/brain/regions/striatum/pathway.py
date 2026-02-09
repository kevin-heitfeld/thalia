"""
Base class for striatal pathways (D1/D2).

Provides common interface for pathway-specific learning and dynamics.
D1 and D2 pathways differ in dopamine polarity and functional role.

**Architecture Note**:
Internal pathways (D1/D2) are different from external pathways (AxonalProjection):
- External pathways connect BETWEEN regions (pure axonal transmission with delays)
- Internal pathways are MSN subpopulations WITHIN striatum
- Different biological roles: external = long-range projections,
  internal = local cell type differentiation (D1 vs D2 receptors)

This class uses mixins for shared utilities while
remaining independent from the external pathway hierarchy.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from thalia.brain.configs import NeuralRegionConfig
from thalia.components.neurons import NeuronFactory, NeuronType
from thalia.learning import ThreeFactorStrategy, ThreeFactorConfig


@dataclass
class StriatumPathwayConfig(NeuralRegionConfig):
    """Configuration for a striatal pathway (D1 or D2).

    **Pathways ARE simple feedforward connectors** - they don't need semantic dimensions.
    They just route spikes from source (n_input) to target (n_output) regions.

    Unlike regions (which specify semantic dimensions like n_actions), pathways
    use physical dimensions directly:
    - n_input: Source region size
    - n_output: Target region size
    - n_neurons: Intermediate population (usually = n_output)

    Inherits common parameters from NeuralRegionConfig:
    - dt_ms, device, dtype, seed
    - w_min, w_max: Weight bounds
    - learning_rate: Learning parameters

    Each pathway has its own:
    - Weight matrix [n_output, n_input]
    - Eligibility traces
    - Neuron population
    - Learning dynamics
    """

    # =========================================================================
    # PATHWAY DIMENSIONS (simple feedforward sizing)
    # =========================================================================
    n_input: int = 128
    """Input dimension (source region size)."""

    n_output: int = 64
    """Output dimension (target region size)."""

    n_neurons: int = 0
    """Number of intermediate neurons (computed = n_output if not specified)."""

    # =========================================================================
    # PATHWAY-SPECIFIC OVERRIDES
    # =========================================================================
    axonal_delay_ms: float = 5.0
    """Axonal conduction delay in milliseconds (inter-region default).

    Biological ranges for inter-region pathways:
    - Cortico-cortical: 5-10ms
    - Thalamo-cortical: 8-15ms
    - Striato-cortical: 10-20ms
    - Hippocampo-cortical: 10-15ms

    Default of 5.0ms is appropriate for typical cortico-cortical projections.
    Specific pathways can override for longer delays (e.g., thalamus: 10ms).
    """

    # =========================================================================
    # CONNECTIVITY
    # =========================================================================
    sparsity: float = 0.1
    """Target sparsity for pathway connections (fraction of non-zero weights)."""

    # =========================================================================
    # NEURON MODEL PARAMETERS
    # =========================================================================
    tau_mem_ms: float = 20.0
    """Membrane time constant in milliseconds."""

    tau_syn_ms: float = 5.0
    """Synaptic time constant in milliseconds."""

    v_thresh: float = -50.0
    """Spike threshold voltage in mV."""

    v_reset: float = -65.0
    """Reset voltage after spike in mV."""

    v_rest: float = -70.0
    """Resting membrane potential in mV."""

    refractory_ms: float = 2.0
    """Refractory period in milliseconds."""

    # =========================================================================
    # LEARNING PARAMETERS
    # =========================================================================

    # Three-factor learning parameters
    eligibility_tau_ms: float = 1000.0  # Eligibility trace decay time

    # Neuron parameters
    threshold: float = 1.0
    reset_potential: float = 0.0
    e_leak: float = 0.0
    e_excitatory: float = 3.0
    e_inhibitory: float = -0.5

    # =========================================================================
    # POST-INIT SYNCHRONIZATION
    # =========================================================================
    def __post_init__(self):
        """Synchronize n_neurons with n_output for pathway consistency."""
        # For pathways, n_neurons should match n_output (target size)
        if self.n_neurons == 0:
            object.__setattr__(self, "n_neurons", self.n_output)


@dataclass
class StriatumPathwayState:
    """State for a striatal pathway (D1 or D2).

    Attributes:
        weights: Synaptic weight matrix [n_output, n_input]
        eligibility: Eligibility traces for three-factor learning (can be None)
        neuron_membrane: Membrane potentials [n_output] (can be None)
        neuron_g_E: Excitatory conductance [n_output] (can be None)
        neuron_g_I: Inhibitory conductance [n_output] (can be None)
        neuron_refractory: Refractory counters [n_output] (can be None)
    """

    STATE_VERSION: int = 1
    """State format version."""

    weights: torch.Tensor = torch.Tensor()
    eligibility: Optional[torch.Tensor] = None
    neuron_membrane: Optional[torch.Tensor] = None
    neuron_g_E: Optional[torch.Tensor] = None
    neuron_g_I: Optional[torch.Tensor] = None
    neuron_refractory: Optional[torch.Tensor] = None


class StriatumPathway(nn.Module):
    """
    Striatal pathway implementation for dopamine receptor subtypes.

    Creates MSN subpopulations via factory methods (create_d1, create_d2).
    Each pathway is a separate population of Medium Spiny Neurons (MSNs)
    with its own weights, eligibility traces, and learning dynamics.

    Key responsibilities:
    - Weight matrix management
    - Eligibility trace computation
    - Neuron population simulation
    - Dopamine-modulated learning
    - State management (checkpointing)
    """

    # =========================================================================
    # FACTORY METHODS
    # =========================================================================

    @classmethod
    def create_d1(cls, config: StriatumPathwayConfig) -> "StriatumPathway":
        """Factory method to create D1 pathway (direct/Go pathway).

        D1 pathways use DIRECT dopamine modulation:
        - Dopamine burst (DA+): Strengthen eligible synapses (LTP)
        - Dopamine dip (DA-): Weaken eligible synapses (LTD)

        This implements the "GO" signal: rewarded actions become stronger.

        Args:
            config: Pathway configuration with n_input, n_output, learning rates

        Returns:
            StriatumPathway configured as D1 pathway with D1-MSN neurons
        """
        pathway = cls(config, neuron_type=NeuronType.MSN_D1)
        pathway.pathway_name = "D1"
        pathway.dopamine_polarity = 1.0  # Direct modulation
        return pathway

    @classmethod
    def create_d2(cls, config: StriatumPathwayConfig) -> "StriatumPathway":
        """Factory method to create D2 pathway (indirect/NoGo pathway).

        D2 pathways use INVERTED dopamine modulation:
        - Dopamine burst (DA+): WEAKEN eligible synapses (LTD)
        - Dopamine dip (DA-): STRENGTHEN eligible synapses (LTP)

        This implements the "NOGO" signal: punished actions become more inhibited.

        Args:
            config: Pathway configuration with n_input, n_output, learning rates

        Returns:
            StriatumPathway configured as D2 pathway with D2-MSN neurons
        """
        pathway = cls(config, neuron_type=NeuronType.MSN_D2)
        pathway.pathway_name = "D2"
        pathway.dopamine_polarity = -1.0  # Inverted modulation
        return pathway

    # =========================================================================
    # PROPERTIES FOR WEIGHT ACCESS
    # =========================================================================

    @property
    def device(self) -> torch.device:
        """Device where tensors are located."""
        return torch.device(self.config.device)

    @property
    def weights(self) -> torch.Tensor:
        """Access pathway weights from parent's synaptic_weights dict.

        Pathways no longer own weights - they're stored in parent's synaptic_weights
        dict. This implements Option B (biologically accurate architecture).

        Returns:
            Weight matrix [n_output, n_input]
        """
        if self._parent_striatum_ref is None or self._weight_source is None:
            raise RuntimeError(
                f"{self.__class__.__name__} not linked to parent. "
                "Call striatum._link_pathway_weights_to_parent() after construction."
            )
        parent = self._parent_striatum_ref()
        if parent is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: Parent striatum has been garbage collected"
            )
        return parent.synaptic_weights[self._weight_source]

    @weights.setter
    def weights(self, value: torch.Tensor) -> None:
        """Update pathway weights in parent's synaptic_weights dict.

        Args:
            value: New weight matrix [n_output, n_input] (Tensor or nn.Parameter)
        """
        if self._parent_striatum_ref is None or self._weight_source is None:
            raise RuntimeError(
                f"{self.__class__.__name__} not linked to parent. "
                "Call striatum._link_pathway_weights_to_parent() after construction."
            )
        parent = self._parent_striatum_ref()
        if parent is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: Parent striatum has been garbage collected"
            )
        # Extract tensor data from nn.Parameter if needed
        if isinstance(value, nn.Parameter):
            parent.synaptic_weights[self._weight_source].data = value.data
        else:
            parent.synaptic_weights[self._weight_source].data = value

    @property
    def eligibility(self) -> Optional[torch.Tensor]:
        """Eligibility traces [n_output, n_input] (from parent striatum).

        Returns eligibility trace from parent's dict (multi-source architecture).
        """
        # Get eligibility from parent striatum's dict
        if hasattr(self, "_parent_striatum_ref") and self._parent_striatum_ref is not None:
            parent = self._parent_striatum_ref()
            if parent is not None:
                # Get pathway-specific eligibility dict (_eligibility_d1 or _eligibility_d2)
                pathway_type = type(self).__name__
                if "D1" in pathway_type:
                    elig_dict = getattr(parent, "_eligibility_d1", {})
                else:
                    elig_dict = getattr(parent, "_eligibility_d2", {})

                # Return first available eligibility trace (usually only one source in tests)
                if elig_dict:
                    return next(iter(elig_dict.values()))

        return None

    @eligibility.setter
    def eligibility(self, value: torch.Tensor) -> None:
        """Set eligibility traces (for checkpoint loading)."""
        if hasattr(self.learning_strategy, "eligibility"):
            self.learning_strategy.eligibility = value

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, config: StriatumPathwayConfig, neuron_type: NeuronType = NeuronType.MSN_D1):
        super().__init__()
        self.config = config

        self.state = StriatumPathwayState(
            weights=torch.zeros(config.n_output, config.n_input, device=self.device),
        )

        # Pathway identity (set by subclass or factory method)
        self.pathway_name: str = ""  # "D1" or "D2"
        self.dopamine_polarity: float = 1.0  # +1.0 for D1, -1.0 for D2

        # Parent reference (set by Striatum after construction)
        # Pathways access weights via parent's synaptic_weights dict
        # Use weakref to avoid circular reference during .to(device)
        self._parent_striatum_ref: Optional[weakref.ref] = None  # WeakRef to Striatum
        self._weight_source: Optional[str] = None  # e.g., "default_d1" or "default_d2"

        # Three-factor learning strategy (eligibility × dopamine)
        # Use factory function for consistent strategy creation
        self.learning_strategy = ThreeFactorStrategy(ThreeFactorConfig(
            learning_rate=config.learning_rate,
            w_min=config.w_min,
            w_max=config.w_max,
            eligibility_tau=config.eligibility_tau_ms,
        ))

        # Neuron population - use NeuronFactory for standardized MSN creation
        # Factory handles all MSN parameters (adaptation, membrane dynamics, etc.)
        self.neurons = NeuronFactory.create(
            neuron_type=neuron_type,
            n_neurons=self.config.n_output,
            device=self.device,
        )

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def forward(
        self,
        input_spikes: torch.Tensor,
        gain: float = 1.0,
        baseline_exc: float = 1.2,
        theta_contrast_mod: float = 1.0,
        lateral_inhibition: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process input spikes through pathway neurons.

        Args:
            input_spikes: Input spikes [n_input] (bool or float)
            gain: Multiplicative gain factor (from tonic DA, NE, etc.)
            baseline_exc: Baseline excitatory drive
            theta_contrast_mod: Theta phase modulation
            lateral_inhibition: Optional lateral inhibition signal [n_output]

        Returns:
            spikes: Output spikes [n_output] (bool)
            activation: Pre-neuron activation [n_output] (float)
        """
        # Convert bool to float for weight multiplication
        input_float = input_spikes.float() if input_spikes.dtype == torch.bool else input_spikes

        # Compute weighted activation [n_output]
        activation = torch.matmul(self.weights, input_float)

        # Apply modulation
        g_exc = (activation * theta_contrast_mod * gain + baseline_exc).clamp(min=0)

        # Lateral inhibition (if provided)
        g_inh = torch.zeros_like(g_exc)
        if lateral_inhibition is not None:
            g_inh = g_inh + lateral_inhibition

        # Generate spikes through neurons
        spikes, _ = self.neurons(g_exc, g_inh)

        return spikes, activation

    def update_eligibility(
        self,
        input_spikes: torch.Tensor,
        output_spikes: torch.Tensor,
    ) -> None:
        """
        Update eligibility traces using Hebbian correlation.

        Eligibility = accumulated correlation between pre and post activity.
        Actual learning happens when dopamine arrives later.

        Args:
            input_spikes: Input spikes [n_input]
            output_spikes: Output spikes [n_output]
        """
        self.learning_strategy.update_eligibility(input_spikes, output_spikes)
