"""
Base class for striatal pathways (D1/D2).

Provides common interface for pathway-specific learning and dynamics.
D1 and D2 pathways differ in dopamine polarity and functional role.

**Architecture Note**:
Internal pathways (D1/D2) are different from external pathways (SpikingPathway):
- External pathways connect BETWEEN regions (managed by PathwayManager)
- Internal pathways are MSN subpopulations WITHIN striatum
- Different biological roles: external = long-range projections with delays,
  internal = local cell type differentiation (D1 vs D2 receptors)

This class uses mixins for shared utilities (weight init, growth) while
remaining independent from the external pathway hierarchy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from thalia.core.base.component_config import PathwayConfig
from thalia.components.neurons.neuron import ConductanceLIF, ConductanceLIFConfig
from thalia.components.synapses.weight_init import WeightInitializer
from thalia.learning import ThreeFactorStrategy, ThreeFactorConfig
from thalia.components.neurons.neuron_constants import (
    V_THRESHOLD_STANDARD,
    V_RESET_STANDARD,
    E_LEAK,
    E_EXCITATORY,
    E_INHIBITORY,
)
from thalia.mixins import GrowthMixin, ResettableMixin


@dataclass
class StriatumPathwayConfig(PathwayConfig):
    """Configuration for a striatal pathway (D1 or D2).

    Each pathway has its own:
    - Weight matrix [n_output, n_input]
    - Eligibility traces
    - Neuron population
    - Learning dynamics
    """

    # Three-factor learning parameters
    eligibility_tau_ms: float = 1000.0  # Eligibility trace decay time

    # Neuron parameters
    tau_mem_ms: float = 20.0
    threshold: float = V_THRESHOLD_STANDARD
    reset_potential: float = V_RESET_STANDARD
    e_leak: float = E_LEAK
    e_excitatory: float = E_EXCITATORY
    e_inhibitory: float = E_INHIBITORY


class StriatumPathway(nn.Module, GrowthMixin, ResettableMixin, ABC):
    """
    Base class for D1 and D2 striatal pathways.

    Each pathway is a separate population of Medium Spiny Neurons (MSNs)
    with its own weights, eligibility traces, and learning dynamics.

    **Mixins Used**:
    - GrowthMixin: Provides _expand_weights() helper for grow() and grow_input()
    - ResettableMixin: Enforces reset_state() interface

    Key responsibilities:
    - Weight matrix management
    - Eligibility trace computation
    - Neuron population simulation
    - Dopamine-modulated learning
    - Growth (adding neurons or inputs)
    - State management (checkpointing)
    """

    def __init__(self, config: StriatumPathwayConfig):
        super().__init__()
        self.config = config
        self.device = config.device

        # Parent reference (set by Striatum after construction)
        # Pathways access weights via parent's synaptic_weights dict
        # Use weakref to avoid circular reference during .to(device)
        import weakref
        self._parent_striatum_ref: Optional[weakref.ref] = None  # WeakRef to Striatum
        self._weight_source: Optional[str] = None  # e.g., "default_d1" or "default_d2"

        # Three-factor learning strategy (eligibility × dopamine)
        three_factor_config = ThreeFactorConfig(
            learning_rate=config.stdp_lr,
            eligibility_tau=config.eligibility_tau_ms,
            w_min=config.w_min,
            w_max=config.w_max,
            dt=1.0,
            device=config.device,
        )
        self.learning_strategy = ThreeFactorStrategy(three_factor_config)

        # Neuron population
        self.neurons = self._create_neurons()

    # =========================================================================
    # WEIGHT ACCESS (Phase 2 - Option B)
    # =========================================================================

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
            raise RuntimeError(f"{self.__class__.__name__}: Parent striatum has been garbage collected")
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
            raise RuntimeError(f"{self.__class__.__name__}: Parent striatum has been garbage collected")
        # Extract tensor data from nn.Parameter if needed
        if isinstance(value, nn.Parameter):
            parent.synaptic_weights[self._weight_source].data = value.data
        else:
            parent.synaptic_weights[self._weight_source].data = value

    # =========================================================================
    # PROPERTIES FOR DIAGNOSTICS
    # =========================================================================

    @property
    def eligibility(self) -> Optional[torch.Tensor]:
        """Eligibility traces [n_output, n_input] (from learning strategy)."""
        return self.learning_strategy.eligibility

    @eligibility.setter
    def eligibility(self, value: torch.Tensor) -> None:
        """Set eligibility traces (for checkpoint loading)."""
        self.learning_strategy.eligibility = value

    def _create_neurons(self) -> ConductanceLIF:
        """Create neuron population for this pathway.

        Returns:
            ConductanceLIF neuron population
        """
        neuron_config = ConductanceLIFConfig(
            v_threshold=self.config.threshold,
            v_reset=self.config.reset_potential,
            E_L=self.config.e_leak,
            E_E=self.config.e_excitatory,
            E_I=self.config.e_inhibitory,
            tau_E=5.0,  # AMPA/NMDA time constant (ms)
            tau_I=5.0,  # GABA time constant (ms)
            dt_ms=1.0,
            tau_ref=2.0,  # Refractory period (ms)
        )
        neurons = ConductanceLIF(n_neurons=self.config.n_output, config=neuron_config)
        neurons.to(self.device)
        return neurons

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
        # Reset neuron state if needed
        if self.neurons.membrane is None:
            self.neurons.reset_state()

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
        # Strategy handles eligibility trace updates internally
        self.learning_strategy.update_eligibility(input_spikes, output_spikes)

    @abstractmethod
    def apply_dopamine_modulation(
        self,
        dopamine: float,
        heterosynaptic_ratio: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Apply dopamine-modulated plasticity to weights.

        D1 and D2 pathways respond OPPOSITELY to dopamine:
        - D1: DA+ → LTP (strengthen), DA- → LTD (weaken)
        - D2: DA+ → LTD (weaken), DA- → LTP (strengthen)

        Args:
            dopamine: Dopamine signal (RPE, typically -1 to +1)
            heterosynaptic_ratio: Fraction of learning applied to non-eligible synapses

        Returns:
            Metrics dict with numeric values and string metadata (pathway type, dopamine sign)
        """

    def grow(self, n_new_neurons: int, initialization: str = 'xavier') -> None:
        """
        Add new neurons to pathway (for adding new actions).

        Uses GrowthMixin._expand_weights() for standardized weight expansion.

        Expands:
        - Weight matrix (rows)
        - Eligibility traces
        - Neuron population

        Args:
            n_new_neurons: Number of neurons to add
            initialization: Weight initialization strategy

        Example:
            >>> # In Striatum.grow_output()
            >>> self.d1_pathway.grow(n_new=10)
            >>> self.d2_pathway.grow(n_new=10)
        """
        old_n_output = self.config.n_output
        new_n_output = old_n_output + n_new_neurons

        # 1. Expand weights using GrowthMixin helper
        self.weights = self._expand_weights(
            current_weights=self.weights,
            n_new=n_new_neurons,
            initialization=initialization,
            scale=self.config.w_max * 0.2,
        )

        # 2. Reset eligibility traces (strategy will reinitialize on next update)
        self.learning_strategy.reset_state()

        # 3. Update config and grow neurons using efficient in-place growth (ConductanceLIF)
        self.config.n_output = new_n_output
        self.neurons.grow_neurons(n_new_neurons)

    def grow_input(self, n_new_inputs: int, initialization: str = 'xavier') -> None:
        """
        Expand input dimension when upstream regions grow.

        NOTE: Does NOT update self.config.n_input - that's handled by the parent
        region (Striatum). Internal pathways just expand their weight matrices.

        Args:
            n_new_inputs: Number of input neurons to add
            initialization: Weight initialization strategy

        Example:
            >>> # When cortex grows from 128→148 neurons:
            >>> cortex.grow_output(20)  # cortex adds 20 neurons
            >>> cortex_to_striatum.grow_source('cortex', 148)  # pathway resizes
            >>> striatum.grow_input(20)  # Calls d1/d2.grow_input(20)
        """
        # Strategy: Create new columns by initializing new [n_output, n_new_inputs] block
        if initialization == 'xavier':
            new_cols = WeightInitializer.xavier(
                n_output=self.config.n_output,
                n_input=n_new_inputs,
                gain=0.2,
                device=self.device,
            ) * self.config.w_max
        elif initialization == 'sparse_random':
            new_cols = WeightInitializer.sparse_random(
                n_output=self.config.n_output,
                n_input=n_new_inputs,
                sparsity=0.1,
                scale=self.config.w_max * 0.2,
                device=self.device,
            )
        else:  # uniform
            new_cols = WeightInitializer.uniform(
                n_output=self.config.n_output,
                n_input=n_new_inputs,
                low=0.0,
                high=self.config.w_max * 0.2,
                device=self.device,
            )

        # Concatenate along input dimension (columns)
        expanded = torch.cat([self.weights.data, new_cols], dim=1)
        self.weights = nn.Parameter(expanded)

        # Reset eligibility traces (new dimensions)
        self.learning_strategy.reset_state()

        # NOTE: Config is NOT updated here - parent region (Striatum) manages config

    def get_state(self) -> Dict[str, Any]:
        """Get pathway state for checkpointing.

        Returns:
            State dict with weights, eligibility, neuron state
        """
        return {
            'weights': self.weights.detach().clone(),
            'eligibility': self.eligibility.clone() if self.eligibility is not None else None,
            'neuron_membrane': self.neurons.membrane.clone() if self.neurons.membrane is not None else None,
            'neuron_g_E': self.neurons.g_E.clone() if self.neurons.g_E is not None else None,
            'neuron_g_I': self.neurons.g_I.clone() if self.neurons.g_I is not None else None,
            'neuron_refractory': self.neurons.refractory.clone() if self.neurons.refractory is not None else None,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load pathway state from checkpoint.

        Args:
            state: State dict from get_state()
        """
        # Use property setter to update parent's synaptic_weights dict
        self.weights = state['weights']
        if state.get('eligibility') is not None:
            self.learning_strategy.eligibility = state['eligibility']

        if state.get('neuron_membrane') is not None:
            self.neurons.membrane = state['neuron_membrane']
        if state.get('neuron_g_E') is not None:
            self.neurons.g_E = state['neuron_g_E']
        if state.get('neuron_g_I') is not None:
            self.neurons.g_I = state['neuron_g_I']
        if state.get('neuron_refractory') is not None:
            self.neurons.refractory = state['neuron_refractory']

    def reset_state(self) -> None:
        """Reset pathway state (eligibility, neurons)."""
        self.learning_strategy.reset_state()
        self.neurons.reset_state()
