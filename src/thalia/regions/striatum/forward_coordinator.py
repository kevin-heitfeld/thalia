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
- `set_norepinephrine()`: Update NE level from neuromodulator system
- `set_tonic_dopamine()`: Update baseline DA level

Author: Thalia Project
Date: December 9, 2025 (extracted during striatum refactoring)
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn

from thalia.components.neurons.neuron_constants import (
    THETA_BASELINE_MIN,
    THETA_BASELINE_RANGE,
    THETA_CONTRAST_MIN,
    THETA_CONTRAST_RANGE,
    BASELINE_EXCITATION_SCALE,
    TONIC_D1_GAIN_SCALE,
    NE_GAIN_RANGE,
)

if TYPE_CHECKING:
    from .config import StriatumConfig
    from .d1_pathway import D1Pathway
    from .d2_pathway import D2Pathway
    from .homeostasis_component import StriatumHomeostasisComponent
    from thalia.components.neurons.neuron import ConductanceLIF


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

        # Oscillator state (set by set_oscillator_phases)
        self._theta_phase: float = 0.0
        self._beta_phase: float = 0.0
        self._beta_amplitude: float = 1.0

        # Norepinephrine level (set by set_norepinephrine)
        self._ne_level: float = 0.0

        # Tonic dopamine level (set externally)
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
            acetylcholine: Acetylcholine level [0, 1] (not used by striatum)
        """
        if dopamine is not None:
            self._tonic_dopamine = dopamine
        if norepinephrine is not None:
            self._ne_level = norepinephrine
        # acetylcholine not used by striatum

    def set_norepinephrine(self, ne_level: float) -> None:
        """Set norepinephrine level (backward compatibility).

        Args:
            ne_level: NE level [0, 1]
        """
        self._ne_level = ne_level

    def set_tonic_dopamine(self, tonic_da: float) -> None:
        """Set tonic dopamine level (backward compatibility).

        Args:
            tonic_da: Tonic dopamine level [0, 1]
        """
        self._tonic_dopamine = tonic_da

    def compute_theta_modulation(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute theta modulation factors.

        Returns:
            (theta_baseline_mod, theta_contrast_mod, baseline_exc) - all tensors on self.device
        """
        # Encoding peak at 0°, retrieval peak at 180°
        encoding_mod = (1 + torch.cos(torch.tensor(self._theta_phase, device=self.device))) / 2
        retrieval_mod = (1 - torch.cos(torch.tensor(self._theta_phase, device=self.device))) / 2

        # Baseline and contrast modulation
        theta_baseline_mod = THETA_BASELINE_MIN + THETA_BASELINE_RANGE * encoding_mod
        theta_contrast_mod = THETA_CONTRAST_MIN + THETA_CONTRAST_RANGE * retrieval_mod

        return (
            theta_baseline_mod,
            theta_contrast_mod,
            BASELINE_EXCITATION_SCALE * theta_baseline_mod
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
        ne_gain = 1.0 + NE_GAIN_RANGE * self._ne_level

        return d1_gain * ne_gain, d2_gain

    def compute_goal_modulation(
        self,
        d1_activation: torch.Tensor,
        d2_activation: torch.Tensor,
        pfc_goal_context: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compute goal-conditioned modulation of D1/D2 activations.

        Args:
            d1_activation: D1 activations [n_output]
            d2_activation: D2 activations [n_output]
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

        # Compute goal modulation via learned PFC → striatum weights
        goal_mod_d1 = torch.sigmoid(
            torch.matmul(self.pfc_modulation_d1, pfc_goal_context)
        )
        goal_mod_d2 = torch.sigmoid(
            torch.matmul(self.pfc_modulation_d2, pfc_goal_context)
        )

        # Modulate activations by goal context
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
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compute D1/D2 spikes from input.

        Args:
            input_spikes: Input spikes [n_input] (1D)
            recent_spikes: Recent spike history for lateral inhibition [n_output]

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

        # Compute base activations from weights
        d1_activation = torch.matmul(self.d1_pathway.weights, input_float)
        d2_activation = torch.matmul(self.d2_pathway.weights, input_float)

        # Extract PFC goal context if enabled
        pfc_goal_context = self.extract_pfc_context(input_spikes)

        # Apply theta modulation
        _theta_baseline_mod, theta_contrast_mod, baseline_exc = self.compute_theta_modulation()

        # Apply gain modulation (tonic DA, beta, NE)
        d1_gain, d2_gain = self.compute_gain_modulation()

        # Apply goal modulation (PFC gating)
        d1_activation, d2_activation, pfc_goal_context = self.compute_goal_modulation(
            d1_activation, d2_activation, pfc_goal_context
        )

        # Apply homeostatic excitability modulation
        d1_gain, d2_gain = self.compute_homeostatic_modulation(d1_gain, d2_gain)

        # Compute D1 conductances
        d1_g_exc = (d1_activation * theta_contrast_mod * d1_gain + baseline_exc).clamp(min=0)
        d1_g_inh = torch.zeros_like(d1_g_exc)

        # Add lateral inhibition if enabled
        if self.config.lateral_inhibition:
            d1_g_inh = d1_g_inh + recent_spikes * self.config.inhibition_strength * 0.5

        # Run D1 neurons
        d1_spikes, _ = self.d1_neurons(d1_g_exc, d1_g_inh)

        # Compute D2 conductances
        d2_g_exc = (d2_activation * theta_contrast_mod * d2_gain + baseline_exc).clamp(min=0)
        d2_g_inh = torch.zeros_like(d2_g_exc)

        # Add lateral inhibition if enabled
        if self.config.lateral_inhibition:
            d2_g_inh = d2_g_inh + recent_spikes * self.config.inhibition_strength * 0.5

        # Run D2 neurons
        d2_spikes, _ = self.d2_neurons(d2_g_exc, d2_g_inh)

        # Update homeostasis activity tracking
        if self.homeostasis_manager is not None:
            self.homeostasis_manager.update_activity(
                d1_spikes, d2_spikes, decay=self.config.activity_decay
            )

        return d1_spikes, d2_spikes, pfc_goal_context
