"""
Base class for striatal pathways (D1/D2).

Provides common interface for pathway-specific learning and dynamics.
D1 and D2 pathways differ in dopamine polarity and functional role.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from thalia.core.component_config import PathwayConfig
from thalia.components.neurons.neuron import ConductanceLIF, ConductanceLIFConfig
from thalia.core.weight_init import WeightInitializer
from thalia.core.eligibility_utils import EligibilityTraceManager, STDPConfig
from thalia.components.neurons.neuron_constants import (
    V_THRESHOLD_STANDARD,
    V_RESET_STANDARD,
    E_LEAK,
    E_EXCITATORY,
    E_INHIBITORY,
)


@dataclass
class StriatumPathwayConfig(PathwayConfig):
    """Configuration for a striatal pathway (D1 or D2).

    Each pathway has its own:
    - Weight matrix [n_output, n_input]
    - Eligibility traces
    - Neuron population
    - Learning dynamics
    """

    # STDP trace parameters
    stdp_tau_ms: float = 20.0  # Spike trace decay time (for STDP)

    # Eligibility trace parameters
    eligibility_tau_ms: float = 1000.0

    # Neuron parameters
    tau_mem_ms: float = 20.0
    threshold: float = V_THRESHOLD_STANDARD
    reset_potential: float = V_RESET_STANDARD
    e_leak: float = E_LEAK
    e_excitatory: float = E_EXCITATORY
    e_inhibitory: float = E_INHIBITORY


class StriatumPathway(nn.Module, ABC):
    """
    Base class for D1 and D2 striatal pathways.

    Each pathway is a separate population of Medium Spiny Neurons (MSNs)
    with its own weights, eligibility traces, and learning dynamics.

    Key responsibilities:
    - Weight matrix management
    - Eligibility trace computation
    - Neuron population simulation
    - Dopamine-modulated learning
    - Growth (adding new actions)
    - State management (checkpointing)
    """

    def __init__(self, config: StriatumPathwayConfig):
        super().__init__()
        self.config = config
        self.device = config.device

        # Initialize weights [n_output, n_input]
        self.weights = self._initialize_weights()

        # Eligibility trace manager (consolidated STDP + eligibility traces)
        stdp_config = STDPConfig(
            stdp_tau_ms=config.stdp_tau_ms,
            eligibility_tau_ms=config.eligibility_tau_ms,
            stdp_lr=config.stdp_lr,
            w_min=config.w_min,
            w_max=config.w_max,
        )
        self._trace_manager = EligibilityTraceManager(
            n_input=config.n_input,
            n_output=config.n_output,
            config=stdp_config,
            device=self.device,
        )

        # Neuron population
        self.neurons = self._create_neurons()

    # =========================================================================
    # BACKWARD COMPATIBILITY PROPERTIES
    # =========================================================================
    # External code may access traces directly. Delegate to trace_manager.

    @property
    def eligibility(self) -> torch.Tensor:
        """Eligibility traces [n_output, n_input] (delegates to trace_manager)."""
        return self._trace_manager.eligibility

    @eligibility.setter
    def eligibility(self, value: torch.Tensor) -> None:
        """Set eligibility traces directly (for checkpoint loading)."""
        self._trace_manager.eligibility = value

    @property
    def input_trace(self) -> torch.Tensor:
        """Input STDP trace [n_input] (delegates to trace_manager)."""
        return self._trace_manager.input_trace

    @input_trace.setter
    def input_trace(self, value: torch.Tensor) -> None:
        """Set input trace directly (for checkpoint loading)."""
        self._trace_manager.input_trace = value

    @property
    def output_trace(self) -> torch.Tensor:
        """Output STDP trace [n_output] (delegates to trace_manager)."""
        return self._trace_manager.output_trace

    @output_trace.setter
    def output_trace(self, value: torch.Tensor) -> None:
        """Set output trace directly (for checkpoint loading)."""
        self._trace_manager.output_trace = value

    def _initialize_weights(self) -> nn.Parameter:
        """Initialize pathway weights using Xavier initialization.

        Returns:
            Weight matrix [n_output, n_input] as nn.Parameter
        """
        weights = WeightInitializer.xavier(
            n_output=self.config.n_output,
            n_input=self.config.n_input,
            gain=0.2,  # Conservative initialization
            device=self.device,
        ) * self.config.w_max

        return nn.Parameter(weights)

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
        dt_ms: float,
    ) -> None:
        """
        Update eligibility traces using STDP-based plasticity.

        Eligibility = correlation between pre and post activity.
        Actual learning happens when dopamine arrives later.

        Args:
            input_spikes: Input spikes [n_input]
            output_spikes: Output spikes [n_output]
            dt_ms: Timestep in milliseconds
        """
        # Update traces using consolidated manager
        self._trace_manager.update_traces(input_spikes, output_spikes, dt_ms)

        # Compute STDP eligibility with soft bounds
        # Note: lr_scale=1.0 here; pathway-specific scaling (d1_lr_scale, d2_lr_scale)
        # is applied in apply_dopamine_modulation()
        eligibility_update = self._trace_manager.compute_stdp_eligibility(
            weights=self.weights,
            lr_scale=1.0,
        )

        # Accumulate into eligibility traces with decay
        self._trace_manager.accumulate_eligibility(eligibility_update, dt_ms)

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

        Expands:
        - Weight matrix
        - Eligibility traces
        - STDP traces
        - Neuron population

        Args:
            n_new_neurons: Number of neurons to add
            initialization: Weight initialization strategy
        """
        old_n_output = self.config.n_output
        new_n_output = old_n_output + n_new_neurons

        # 1. Expand weights
        if initialization == 'xavier':
            new_weights = WeightInitializer.xavier(
                n_output=n_new_neurons,
                n_input=self.config.n_input,
                gain=0.2,
                device=self.device,
            ) * self.config.w_max
        else:  # uniform
            new_weights = WeightInitializer.uniform(
                n_output=n_new_neurons,
                n_input=self.config.n_input,
                low=0.0,
                high=self.config.w_max * 0.2,
                device=self.device,
            )

        self.weights = nn.Parameter(
            torch.cat([self.weights.data, new_weights], dim=0)
        )

        # 2. Expand eligibility traces
        new_elig = torch.zeros(n_new_neurons, self.config.n_input, device=self.device)
        self.eligibility = torch.cat([self.eligibility, new_elig], dim=0)

        # 3. Expand STDP output trace
        new_trace = torch.zeros(n_new_neurons, device=self.device)
        self.output_trace = torch.cat([self.output_trace, new_trace], dim=0)

        # 4. Recreate neurons with new size (preserving old state)
        old_membrane = self.neurons.membrane.clone() if self.neurons.membrane is not None else None
        old_g_E = self.neurons.g_E.clone() if self.neurons.g_E is not None else None
        old_g_I = self.neurons.g_I.clone() if self.neurons.g_I is not None else None
        old_refractory = self.neurons.refractory.clone() if self.neurons.refractory is not None else None

        # Update config
        self.config.n_output = new_n_output

        # Create new neurons
        self.neurons = self._create_neurons()
        self.neurons.reset_state()

        # Restore old state for existing neurons
        if old_membrane is not None:
            self.neurons.membrane[:old_n_output] = old_membrane
        if old_g_E is not None:
            self.neurons.g_E[:old_n_output] = old_g_E
        if old_g_I is not None:
            self.neurons.g_I[:old_n_output] = old_g_I
        if old_refractory is not None:
            self.neurons.refractory[:old_n_output] = old_refractory

    def get_state(self) -> Dict[str, Any]:
        """Get pathway state for checkpointing.

        Returns:
            State dict with weights, eligibility, traces, neuron state
        """
        return {
            'weights': self.weights.detach().clone(),
            'eligibility': self.eligibility.clone(),
            'input_trace': self.input_trace.clone(),
            'output_trace': self.output_trace.clone(),
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
        self.weights = nn.Parameter(state['weights'])
        self.eligibility = state['eligibility']
        self.input_trace = state['input_trace']
        self.output_trace = state['output_trace']

        if state['neuron_membrane'] is not None:
            self.neurons.membrane = state['neuron_membrane']
        if state['neuron_g_E'] is not None:
            self.neurons.g_E = state['neuron_g_E']
        if state['neuron_g_I'] is not None:
            self.neurons.g_I = state['neuron_g_I']
        if state['neuron_refractory'] is not None:
            self.neurons.refractory = state['neuron_refractory']

    def reset_state(self) -> None:
        """Reset pathway state (eligibility, traces, neurons)."""
        self.eligibility.zero_()
        self.input_trace.zero_()
        self.output_trace.zero_()
        self.neurons.reset_state()
