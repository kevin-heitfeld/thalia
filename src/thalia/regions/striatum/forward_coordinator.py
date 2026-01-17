"""
Striatum Forward Pass Coordinator - D1/D2 Pathway Orchestration

This component manages the complex forward pass logic for the striatum,
extracted from the main Striatum class to isolate the ~300 lines of
forward pass coordination into a focused, maintainable module.

**Responsibilities:**
- Compute D1/D2 pathway activations from cortical/thalamic inputs
- Apply theta/beta oscillator modulation (action selection timing)
- Apply neuromodulator gain modulation (tonic DA, norepinephrine)
- Apply goal-conditioned modulation (PFC → Striatum gating)
- Apply homeostatic excitability modulation (prevent silencing/runaway)
- Run D1/D2 neurons to generate spikes
- Update recent spike history for lateral inhibition
- Coordinate between D1 pathway (Go) and D2 pathway (NoGo)

**Used By:**
- `Striatum.forward()` (main forward pass entry point)

**Coordinates With:**
- `D1Pathway`: Computes D1 activations, manages D1 weights/eligibility
- `D2Pathway`: Computes D2 activations, manages D2 weights/eligibility
- `ConductanceLIF`: D1/D2 neuron populations (membrane dynamics)
- `StriatumHomeostasisComponent`: Provides excitability scaling factors
- `StriatumStateTracker`: Updates recent_spikes and last spikes
- PFC modulation weights: Goal-conditioned gating of striatal activity

**Why Extracted:**
- Complexity: Forward pass was ~300 lines in main class
- Orthogonal concern: Activation computation separate from learning/decisions
- Testability: Can test forward dynamics without learning logic
- Maintainability: Oscillator/neuromodulator logic isolated
- Reusability: Could be reused for different striatal configurations

**Biological Context:**
- Theta oscillations (4-8 Hz): Time action selection windows
- Beta oscillations (15-30 Hz): Suppress premature actions (NoGo)
- Tonic dopamine: Baseline motivation, higher DA → more D1 (Go) activity
- Norepinephrine: Arousal-dependent gain modulation
- PFC gating: Top-down goal selection (frontostriatal loops)

**Key Methods:**
- `forward()`: Main forward pass orchestration
- `set_oscillator_phases()`: Update theta/beta phase from brain clock
- `set_neuromodulators()`: Update neuromodulator levels (unified API)

Author: Thalia Project
Date: December 9, 2025 (extracted during striatum refactoring)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

from thalia.constants.neuromodulation import compute_ne_gain
from thalia.constants.neuron import (
    BASELINE_EXCITATION_SCALE,
    THETA_BASELINE_MIN,
    THETA_BASELINE_RANGE,
    THETA_CONTRAST_MIN,
    THETA_CONTRAST_RANGE,
    TONIC_D1_GAIN_SCALE,
)
from thalia.neuromodulation.mixin import validate_finite
from thalia.utils.oscillator_utils import compute_theta_encoding_retrieval

if TYPE_CHECKING:
    from thalia.components.neurons.neuron import ConductanceLIF

    from .config import StriatumConfig
    from .d1_pathway import D1Pathway
    from .d2_pathway import D2Pathway
    from .homeostasis_component import StriatumHomeostasisComponent


class ForwardPassCoordinator:
    """Coordinates D1/D2 pathway activations during forward pass.

    Responsibilities:
    - Compute D1/D2 activations from input spikes
    - Apply oscillator modulation (theta, beta)
    - Apply neuromodulator modulation (tonic DA, NE)
    - Apply goal-conditioned modulation (PFC gating)
    - Apply homeostatic excitability modulation
    - Run D1/D2 neurons to get spikes
    """

    def __init__(
        self,
        config: StriatumConfig,
        d1_pathway: D1Pathway,
        d2_pathway: D2Pathway,
        d1_neurons: ConductanceLIF,
        d2_neurons: ConductanceLIF,
        homeostasis_manager: Optional[StriatumHomeostasisComponent],
        pfc_modulation_d1: Optional[nn.Parameter],
        pfc_modulation_d2: Optional[nn.Parameter],
        device: torch.device,
        n_actions: int,
        d1_size: int,
        d2_size: int,
        neurons_per_action: int,
    ):
        """Initialize forward pass coordinator.

        Args:
            config: Striatum configuration
            d1_pathway: D1 pathway (weights, eligibility)
            d2_pathway: D2 pathway (weights, eligibility)
            d1_neurons: D1 neuron population
            d2_neurons: D2 neuron population
            homeostasis_manager: Optional homeostasis component
            pfc_modulation_d1: Optional PFC→D1 modulation weights
            pfc_modulation_d2: Optional PFC→D2 modulation weights
            device: Torch device
        """
        self.config = config
        self.d1_pathway = d1_pathway
        self.d2_pathway = d2_pathway
        self.d1_neurons = d1_neurons
        self.d2_neurons = d2_neurons
        self.homeostasis_manager = homeostasis_manager
        self.pfc_modulation_d1 = pfc_modulation_d1
        self.pfc_modulation_d2 = pfc_modulation_d2
        self.device = device

        # Size parameters (no longer in config)
        self.n_actions = n_actions
        self.d1_size = d1_size
        self.d2_size = d2_size
        self.neurons_per_action = neurons_per_action

        # Oscillator state (set by set_oscillator_phases)
        self._theta_phase: float = 0.0
        self._beta_phase: float = 0.0
        self._beta_amplitude: float = 1.0

        # Neuromodulator levels (set by set_neuromodulators)
        self._ne_level: float = 0.0
        self._tonic_dopamine: float = 0.3

    def set_oscillator_phases(
        self,
        theta_phase: float,
        beta_phase: float,
        beta_amplitude: float,
    ) -> None:
        """Set oscillator phases and amplitudes.

        Args:
            theta_phase: Theta phase in radians
            beta_phase: Beta phase in radians
            beta_amplitude: Effective beta amplitude (with coupling)
        """
        self._theta_phase = theta_phase
        self._beta_phase = beta_phase
        self._beta_amplitude = beta_amplitude

    def set_neuromodulators(
        self,
        dopamine: Optional[float] = None,
        norepinephrine: Optional[float] = None,
        acetylcholine: Optional[float] = None,
    ) -> None:
        """Set neuromodulator levels (unified API).

        Args:
            dopamine: Dopamine level [0, 1] (tonic DA)
            norepinephrine: Norepinephrine level [0, 1]
            acetylcholine: Acetylcholine level [0, 1] (stored but not used by striatum)

        Raises:
            ValueError: If any neuromodulator value is NaN or Inf
        """
        # Validate and set dopamine
        if dopamine is not None:
            validate_finite(dopamine, "dopamine", valid_range=(-2.0, 2.0))
            self._tonic_dopamine = dopamine
        # Validate and set norepinephrine
        if norepinephrine is not None:
            validate_finite(norepinephrine, "norepinephrine", valid_range=(0.0, 2.0))
            self._ne_level = norepinephrine
        # Validate and set acetylcholine
        if acetylcholine is not None:
            validate_finite(acetylcholine, "acetylcholine", valid_range=(0.0, 2.0))
            # Store acetylcholine for get_state() even though striatum doesn't use it
            self._ach_level = acetylcholine

    def compute_theta_modulation(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute theta modulation factors.

        Uses standard theta encoding/retrieval utility function for consistency
        with other regions (hippocampus, prefrontal, cortex).

        Returns:
            (theta_baseline_mod, theta_contrast_mod, baseline_exc) - all tensors on self.device
        """
        # Use centralized utility function for consistency
        encoding_mod, retrieval_mod = compute_theta_encoding_retrieval(self._theta_phase)

        # Convert to tensors on device
        encoding_tensor = torch.tensor(encoding_mod, device=self.device, dtype=torch.float32)
        retrieval_tensor = torch.tensor(retrieval_mod, device=self.device, dtype=torch.float32)

        # Baseline and contrast modulation
        theta_baseline_mod = THETA_BASELINE_MIN + THETA_BASELINE_RANGE * encoding_tensor
        theta_contrast_mod = THETA_CONTRAST_MIN + THETA_CONTRAST_RANGE * retrieval_tensor

        return (
            theta_baseline_mod,
            theta_contrast_mod,
            BASELINE_EXCITATION_SCALE * theta_baseline_mod,
        )

    def compute_gain_modulation(self) -> tuple[float, float]:
        """Compute D1/D2 gain modulation from tonic DA, NE, and beta.

        Returns:
            (d1_gain, d2_gain)
        """
        d1_gain = 1.0
        d2_gain = 1.0

        # Tonic dopamine modulation of D1 gain
        if self.config.tonic_modulates_d1_gain:
            tonic_factor = self._tonic_dopamine * TONIC_D1_GAIN_SCALE
            d1_gain = 1.0 + tonic_factor

        # Beta oscillation modulation (action maintenance vs switching)
        # High beta: D1 up, D2 down → action maintenance
        # Low beta: D1 down, D2 up → action flexibility
        beta_mod = self.config.beta_modulation_strength
        d1_gain = d1_gain * (1.0 + beta_mod * (self._beta_amplitude - 0.5))
        d2_gain = d2_gain * (1.0 - beta_mod * (self._beta_amplitude - 0.5))

        # Norepinephrine gain modulation (arousal/uncertainty)
        # High NE increases gain for both pathways (more exploration)
        ne_gain = compute_ne_gain(self._ne_level)

        return d1_gain * ne_gain, d2_gain

    def compute_goal_modulation(
        self,
        d1_activation: torch.Tensor,
        d2_activation: torch.Tensor,
        pfc_goal_context: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compute goal-conditioned modulation of D1/D2 activations.

        Args:
            d1_activation: D1 activations [n_actions] (action-level)
            d2_activation: D2 activations [n_actions] (action-level)
            pfc_goal_context: PFC goal context [pfc_size] or None

        Returns:
            (modulated_d1_activation, modulated_d2_activation, pfc_goal_context)
        """
        if not self.config.use_goal_conditioning or self.pfc_modulation_d1 is None:
            return d1_activation, d2_activation, None

        if pfc_goal_context is None:
            return d1_activation, d2_activation, None

        # Convert bool to float if needed
        if pfc_goal_context.dtype == torch.bool:
            pfc_goal_context = pfc_goal_context.float()

        # Compute per-neuron goal modulation via learned PFC → striatum weights
        # pfc_modulation_d1: [d1_size, pfc_size]
        # pfc_goal_context: [pfc_size]
        # goal_mod_d1_neurons: [d1_size] (per-neuron)
        goal_mod_d1_neurons = torch.sigmoid(torch.matmul(self.pfc_modulation_d1, pfc_goal_context))
        goal_mod_d2_neurons = torch.sigmoid(torch.matmul(self.pfc_modulation_d2, pfc_goal_context))

        # Pool per-neuron modulation to per-action level
        # Each action has neurons_per_pathway MSNs in each pathway
        # d1_size = n_actions * neurons_per_pathway
        if self.neurons_per_action > 1:
            neurons_per_pathway = self.neurons_per_action // 2
            goal_mod_d1 = goal_mod_d1_neurons.view(self.n_actions, neurons_per_pathway).mean(dim=1)
            goal_mod_d2 = goal_mod_d2_neurons.view(self.n_actions, neurons_per_pathway).mean(dim=1)
        else:
            # neurons_per_action == 1: MSN level = action level (1:1)
            goal_mod_d1 = goal_mod_d1_neurons
            goal_mod_d2 = goal_mod_d2_neurons

        # Modulate activations by goal context (both at action level now)
        strength = self.config.goal_modulation_strength
        d1_activation = d1_activation * (1.0 + strength * (goal_mod_d1 - 0.5))
        d2_activation = d2_activation * (1.0 + strength * (goal_mod_d2 - 0.5))

        return d1_activation, d2_activation, pfc_goal_context

    def compute_homeostatic_modulation(
        self,
        d1_gain: float,
        d2_gain: float,
    ) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        """Compute homeostatic excitability modulation.

        Args:
            d1_gain: Current D1 gain
            d2_gain: Current D2 gain

        Returns:
            (modulated_d1_gain, modulated_d2_gain) - may be tensors if homeostasis enabled
        """
        if self.homeostasis_manager is None:
            return d1_gain, d2_gain

        d1_exc_gain, d2_exc_gain = self.homeostasis_manager.compute_excitability()
        return d1_gain * d1_exc_gain, d2_gain * d2_exc_gain

    def extract_pfc_context(self, input_spikes: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract PFC goal context from concatenated input.

        Input format: [cortex_l5 | hippocampus | pfc]
        PFC is at the end, size determined by config.pfc_size.

        Args:
            input_spikes: Full input spikes [n_input]

        Returns:
            PFC context [pfc_size] or None if not available
        """
        if not self.config.use_goal_conditioning:
            return None

        pfc_size = self.config.pfc_size
        if input_spikes.shape[0] < pfc_size:
            return None

        return input_spikes[-pfc_size:]

    def forward(
        self,
        input_spikes: torch.Tensor,
        recent_spikes: torch.Tensor,
        fsi_inhibition: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compute D1/D2 spikes from input.

        Args:
            input_spikes: Input spikes [n_input] (1D)
            recent_spikes: Recent spike history for lateral inhibition [n_output]
            fsi_inhibition: FSI feedforward inhibition [n_output] or scalar (optional)

        Returns:
            (d1_spikes, d2_spikes, pfc_goal_context)
        """
        # Ensure neurons are initialized
        if self.d1_neurons.membrane is None:
            self.d1_neurons.reset_state()
        if self.d2_neurons.membrane is None:
            self.d2_neurons.reset_state()

        # Convert bool spikes to float for matmul
        input_float = input_spikes.float() if input_spikes.dtype == torch.bool else input_spikes

        # NOTE: STP is applied per-source in Striatum.forward()
        # This allows different STP parameters for different input sources
        d1_weights_effective = self.d1_pathway.weights
        d2_weights_effective = self.d2_pathway.weights

        # Compute MSN-level activations: [d1_size] and [d2_size]
        d1_msn_activation = torch.matmul(d1_weights_effective, input_float)
        d2_msn_activation = torch.matmul(d2_weights_effective, input_float)

        # Apply gain modulation (tonic DA, beta, NE) at MSN level
        d1_gain, d2_gain = self.compute_gain_modulation()

        # Apply homeostatic excitability modulation at MSN level
        d1_gain, d2_gain = self.compute_homeostatic_modulation(d1_gain, d2_gain)

        # Apply gains to MSN activations (biological: gains modulate individual neurons)
        d1_msn_activation = d1_msn_activation * d1_gain
        d2_msn_activation = d2_msn_activation * d2_gain

        # Pool MSN activations to action level
        # When neurons_per_action > 1: average over neurons_per_action to get action strength
        # When neurons_per_action == 1: MSN level = action level (1:1 mapping)
        if self.neurons_per_action > 1:
            # Reshape to [n_actions, neurons_per_action/2] and average
            # Note: d1_size = n_actions * neurons_per_action / 2 (D1 and D2 split the pool)
            neurons_per_pathway = self.neurons_per_action // 2
            d1_activation = d1_msn_activation.view(self.n_actions, neurons_per_pathway).mean(dim=1)
            d2_activation = d2_msn_activation.view(self.n_actions, neurons_per_pathway).mean(dim=1)
        else:
            # No pooling needed
            d1_activation = d1_msn_activation
            d2_activation = d2_msn_activation

        # Extract PFC goal context if enabled
        pfc_goal_context = self.extract_pfc_context(input_spikes)

        # Apply theta modulation
        _theta_baseline_mod, theta_contrast_mod, baseline_exc = self.compute_theta_modulation()

        # Apply goal modulation (PFC gating)
        d1_activation, d2_activation, pfc_goal_context = self.compute_goal_modulation(
            d1_activation, d2_activation, pfc_goal_context
        )

        # Compute D1 conductances (action-level, gains already applied)
        d1_g_exc_action = (d1_activation * theta_contrast_mod + baseline_exc).clamp(min=0)

        # Expand action-level conductances to MSN-level
        # Each action has neurons_per_pathway MSNs in D1 pathway
        neurons_per_pathway = self.d1_size // self.n_actions
        d1_g_exc = d1_g_exc_action.repeat_interleave(neurons_per_pathway)  # [d1_size]
        d1_g_inh = torch.zeros_like(d1_g_exc)

        # Add lateral inhibition if enabled (use D1 portion of recent_spikes)
        if self.config.lateral_inhibition:
            d1_recent_spikes = recent_spikes[: self.d1_size]
            d1_g_inh = d1_g_inh + d1_recent_spikes * self.config.inhibition_strength * 0.5

        # Add FSI feedforward inhibition (sharpens action selection timing)
        if fsi_inhibition is not None:
            # FSI inhibition is broadcast to all MSNs (scalar or tensor)
            if isinstance(fsi_inhibition, torch.Tensor):
                if fsi_inhibition.numel() == 1:
                    d1_g_inh = d1_g_inh + fsi_inhibition.item()
                else:
                    d1_g_inh = d1_g_inh + fsi_inhibition
            else:
                d1_g_inh = d1_g_inh + fsi_inhibition

        # Run D1 neurons
        d1_spikes, _ = self.d1_neurons(d1_g_exc, d1_g_inh)

        # Compute D2 conductances (action-level, gains already applied)
        d2_g_exc_action = (d2_activation * theta_contrast_mod + baseline_exc).clamp(min=0)

        # Expand action-level conductances to MSN-level
        # Each action has neurons_per_pathway MSNs in D2 pathway
        neurons_per_pathway = self.d2_size // self.n_actions
        d2_g_exc = d2_g_exc_action.repeat_interleave(neurons_per_pathway)  # [d2_size]
        d2_g_inh = torch.zeros_like(d2_g_exc)

        # Add lateral inhibition if enabled (use D2 portion of recent_spikes)
        if self.config.lateral_inhibition:
            d2_recent_spikes = recent_spikes[self.d1_size :]
            d2_g_inh = d2_g_inh + d2_recent_spikes * self.config.inhibition_strength * 0.5

        # Add FSI feedforward inhibition to D2 (same as D1)
        if fsi_inhibition is not None:
            if isinstance(fsi_inhibition, torch.Tensor):
                if fsi_inhibition.numel() == 1:
                    d2_g_inh = d2_g_inh + fsi_inhibition.item()
                else:
                    d2_g_inh = d2_g_inh + fsi_inhibition
            else:
                d2_g_inh = d2_g_inh + fsi_inhibition

        # Run D2 neurons
        d2_spikes, _ = self.d2_neurons(d2_g_exc, d2_g_inh)

        # Update homeostasis activity tracking
        if self.homeostasis_manager is not None:
            self.homeostasis_manager.update_activity(
                d1_spikes, d2_spikes, decay=self.config.activity_decay
            )

        return d1_spikes, d2_spikes, pfc_goal_context
