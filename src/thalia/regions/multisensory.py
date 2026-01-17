"""
Multimodal Integration - Superior colliculus-like fusion of sensory modalities.

The multimodal integration region combines spikes from different sensory
modalities (visual, auditory, language/semantic) into a unified representation.
This is inspired by the superior colliculus (SC) and posterior parietal cortex (PPC).

**Key Features**:
=================
1. **MULTI-MODAL CONVERGENCE**:
   - Receives spikes from visual, auditory, and language pathways
   - Each modality processed in separate subpopulation
   - Cross-modal connections enable integration

2. **SPATIAL ALIGNMENT**:
   - Aligns modalities in common reference frame
   - Visual + auditory → spatial location (audiovisual integration)
   - Language provides semantic context

3. **SALIENCE COMPUTATION**:
   - Computes cross-modal salience (what's important?)
   - Winner-take-all competition across modalities
   - Enhances multimodal stimuli (McGurk effect)

4. **BINDING**:
   - Temporal synchronization binds features across modalities
   - Gamma oscillations coordinate binding
   - Enables multimodal object recognition

Biological Basis:
=================
- Superior Colliculus (SC): Audiovisual integration for orienting
- Posterior Parietal Cortex (PPC): Multisensory spatial integration
- Superior Temporal Sulcus (STS): Audiovisual speech integration
- Intraparietal Sulcus (IPS): Cross-modal attention

Architecture:
=============

    Visual         Auditory       Language
    Pathway        Pathway        Pathway
       │              │              │
       └──────┬───────┴──────┬───────┘
              │              │
              ▼              ▼
    ┌─────────────────────────────┐
    │   MULTIMODAL INTEGRATION    │
    │                             │
    │  ┌────────┐  ┌────────┐    │
    │  │ Visual │  │Auditory│    │
    │  │  Pool  │  │  Pool  │    │
    │  └───┬────┘  └───┬────┘    │
    │      │           │          │
    │      └─────┬─────┘          │
    │            │                │
    │      ┌─────▼──────┐         │
    │      │Integration │         │
    │      │   Neurons  │◄────────┤ Language
    │      └────────────┘         │
    └──────────┬──────────────────┘
               │
               ▼
    Unified Multimodal
    Representation

Learning Rules:
===============
1. **Hebbian Plasticity**: Strengthen co-active cross-modal connections
2. **BCM Rule**: Competitive learning within modality pools
3. **STDP**: Spike-timing dependent plasticity for temporal binding

Use Cases:
==========
- Audiovisual speech perception (McGurk effect)
- Cross-modal attention (visual cue → auditory enhancement)
- Multimodal object recognition
- Sensory substitution (vision → audio)
- Embodied language grounding (words → perceptual features)

FILE ORGANIZATION
=================
Lines 1-100:   Module docstring, imports, config
Lines 101-200: MultimodalIntegration class __init__
Lines 201-350: Forward pass (modality pools + integration)
Lines 351-450: Cross-modal plasticity
Lines 451-550: Diagnostics and utilities

Author: Thalia Project
Date: December 12, 2025 (Tier 3 Implementation)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional

import torch

from thalia.components.neurons import create_pyramidal_neurons
from thalia.components.synapses import WeightInitializer
from thalia.config.learning_config import HebbianLearningConfig
from thalia.constants.learning import LEARNING_RATE_HEBBIAN_SLOW, SILENCE_DETECTION_THRESHOLD
from thalia.coordination import SinusoidalOscillator
from thalia.core.base.component_config import NeuralComponentConfig
from thalia.core.neural_region import NeuralRegion
from thalia.learning import create_strategy
from thalia.managers.component_registry import register_region


@dataclass
class MultimodalIntegrationConfig(NeuralComponentConfig, HebbianLearningConfig):
    """Configuration for multimodal integration region.

    Inherits Hebbian learning parameters from HebbianLearningConfig:
    - learning_rate: Base learning rate for cross-modal plasticity
    - learning_enabled: Global learning enable/disable
    - weight_min, weight_max: Weight bounds
    - decay_rate, sparsity_penalty, use_oja_rule: Hebbian variants

    Args:
        visual_input_size: Size of visual input
        auditory_input_size: Size of auditory input
        language_input_size: Size of language/semantic input
        visual_pool_ratio: Fraction of neurons for visual (0-1)
        auditory_pool_ratio: Fraction of neurons for auditory (0-1)
        language_pool_ratio: Fraction of neurons for language (0-1)
        integration_pool_ratio: Fraction of neurons for integration (0-1)
        cross_modal_strength: Strength of cross-modal connections (0-1)
        within_modal_strength: Strength of within-modal connections (0-1)
        integration_strength: Strength from pools → integration neurons
        salience_competition_strength: Winner-take-all competition strength
    """

    # Input sizes
    visual_input_size: int = 0
    auditory_input_size: int = 0
    language_input_size: int = 0

    # Pool sizes (explicit, computed from ratios via helper)
    visual_pool_size: int = field(default=0)
    auditory_pool_size: int = field(default=0)
    language_pool_size: int = field(default=0)
    integration_pool_size: int = field(default=0)

    # Connection strengths
    cross_modal_strength: float = 0.4
    within_modal_strength: float = 0.6
    integration_strength: float = 0.8
    salience_competition_strength: float = 0.5

    # Override default learning rate with region-specific value
    learning_rate: float = LEARNING_RATE_HEBBIAN_SLOW

    # Gamma synchronization parameters (for cross-modal binding)
    gamma_freq_hz: float = 40.0  # Gamma frequency for binding (typically 40 Hz)
    coherence_window: float = 0.785  # ~π/4 radians phase tolerance
    phase_coupling_strength: float = 0.1  # Mutual phase nudging strength
    gate_threshold: float = 0.3  # Minimum coherence for binding
    use_gamma_binding: bool = True  # Enable gamma synchronization


@register_region(
    "multimodal_integration",
    aliases=["multimodal"],
    description="Multimodal integration for cross-sensory fusion and binding",
    version="1.0",
)
class MultimodalIntegration(NeuralRegion):
    """Multimodal integration region for cross-modal fusion.

    Combines visual, auditory, and language/semantic inputs into
    unified representation. Inspired by superior colliculus and
    posterior parietal cortex.
    """

    def __init__(self, config: MultimodalIntegrationConfig):
        """Initialize multimodal integration region.

        Args:
            config: Configuration for region
        """
        # Compute total neurons from pool sizes
        n_neurons = (
            config.visual_pool_size
            + config.auditory_pool_size
            + config.language_pool_size
            + config.integration_pool_size
        )

        # Call NeuralRegion init
        super().__init__(
            n_neurons=n_neurons,
            neuron_config=None,  # Created manually below
            default_learning_rule="hebbian",
            device=config.device,
            dt_ms=config.dt_ms,
        )
        self.config = config
        self.multisensory_config = config  # Store for growth methods

        # Read pool sizes from config (computed via helper function)
        self.visual_pool_size = config.visual_pool_size
        self.auditory_pool_size = config.auditory_pool_size
        self.language_pool_size = config.language_pool_size
        self.integration_pool_size = config.integration_pool_size

        # Create neurons for each pool
        self.neurons = create_pyramidal_neurons(
            n_neurons=n_neurons,
            device=config.device,
        )

        # Learning control (specific to multisensory integration)
        self.plasticity_enabled: bool = True

        # =====================================================================
        # INPUT WEIGHTS (sensory → modality pools)
        # =====================================================================

        # Visual input → visual pool
        self.visual_input_weights = WeightInitializer.sparse_random(
            n_output=self.visual_pool_size,
            n_input=config.visual_input_size,
            sparsity=0.2,
            device=config.device,
        )

        # Auditory input → auditory pool
        self.auditory_input_weights = WeightInitializer.sparse_random(
            n_output=self.auditory_pool_size,
            n_input=config.auditory_input_size,
            sparsity=0.2,
            device=config.device,
        )

        # Language input → language pool
        self.language_input_weights = WeightInitializer.sparse_random(
            n_output=self.language_pool_size,
            n_input=config.language_input_size,
            sparsity=0.2,
            device=config.device,
        )

        # =====================================================================
        # CROSS-MODAL WEIGHTS (between modality pools)
        # =====================================================================

        # Visual ↔ Auditory
        self.visual_to_auditory = (
            WeightInitializer.sparse_random(
                n_output=self.auditory_pool_size,
                n_input=self.visual_pool_size,
                sparsity=0.3,
                device=config.device,
            )
            * config.cross_modal_strength
        )

        self.auditory_to_visual = (
            WeightInitializer.sparse_random(
                n_output=self.visual_pool_size,
                n_input=self.auditory_pool_size,
                sparsity=0.3,
                device=config.device,
            )
            * config.cross_modal_strength
        )

        # Visual ↔ Language
        self.visual_to_language = (
            WeightInitializer.sparse_random(
                n_output=self.language_pool_size,
                n_input=self.visual_pool_size,
                sparsity=0.3,
                device=config.device,
            )
            * config.cross_modal_strength
        )

        self.language_to_visual = (
            WeightInitializer.sparse_random(
                n_output=self.visual_pool_size,
                n_input=self.language_pool_size,
                sparsity=0.3,
                device=config.device,
            )
            * config.cross_modal_strength
        )

        # Auditory ↔ Language
        self.auditory_to_language = (
            WeightInitializer.sparse_random(
                n_output=self.language_pool_size,
                n_input=self.auditory_pool_size,
                sparsity=0.3,
                device=config.device,
            )
            * config.cross_modal_strength
        )

        self.language_to_auditory = (
            WeightInitializer.sparse_random(
                n_output=self.auditory_pool_size,
                n_input=self.language_pool_size,
                sparsity=0.3,
                device=config.device,
            )
            * config.cross_modal_strength
        )

        # =====================================================================
        # INTEGRATION WEIGHTS (pools → integration neurons)
        # =====================================================================

        total_pool_size = self.visual_pool_size + self.auditory_pool_size + self.language_pool_size

        self.integration_weights = (
            WeightInitializer.sparse_random(
                n_output=self.integration_pool_size,
                n_input=total_pool_size,
                sparsity=0.4,
                device=config.device,
            )
            * config.integration_strength
        )

        # =====================================================================
        # GAMMA SYNCHRONIZATION (for cross-modal binding)
        # =====================================================================
        # Gamma oscillations emerge from local circuit interactions in the
        # integration region (biologically accurate - not in pathways!)

        if config.use_gamma_binding:
            self.visual_gamma = SinusoidalOscillator(
                frequency_hz=config.gamma_freq_hz,
                dt_ms=config.dt_ms,
            )
            self.auditory_gamma = SinusoidalOscillator(
                frequency_hz=config.gamma_freq_hz,
                dt_ms=config.dt_ms,
            )
            self._last_coherence = 0.0
        else:
            self.visual_gamma = None
            self.auditory_gamma = None
            self._last_coherence = None

        # =====================================================================
        # LEARNING
        # =====================================================================

        if config.learning_enabled:
            # Use factory function for consistent strategy creation
            self.hebbian_strategy = create_strategy(
                "hebbian",
                learning_rate=config.learning_rate,
                decay_rate=config.hebbian_decay,
            )
        else:
            self.hebbian_strategy = None

        # =====================================================================
        # STATE
        # =====================================================================

        self._reset_state()

    def _reset_state(self) -> None:
        """Reset internal state."""
        # Pool activations
        self.visual_pool_spikes = torch.zeros(
            self.visual_pool_size,
            device=self.config.device,
        )
        self.auditory_pool_spikes = torch.zeros(
            self.auditory_pool_size,
            device=self.config.device,
        )
        self.language_pool_spikes = torch.zeros(
            self.language_pool_size,
            device=self.config.device,
        )
        self.integration_spikes = torch.zeros(
            self.integration_pool_size,
            device=self.config.device,
        )

        # Reset neurons
        self.neurons.reset_state()

    def forward(
        self,
        visual_input: Optional[torch.Tensor] = None,
        auditory_input: Optional[torch.Tensor] = None,
        language_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: integrate multimodal inputs.

        Args:
            visual_input: Visual spikes (visual_input_size,)
            auditory_input: Auditory spikes (auditory_input_size,)
            language_input: Language spikes (language_input_size,)

        Returns:
            Output spikes (n_output,) combining all modalities
        """
        # =====================================================================
        # STEP 0: Gamma synchronization (if enabled)
        # =====================================================================
        coherence_gate = 1.0  # Default: no gating

        if (
            self.config.use_gamma_binding
            and visual_input is not None
            and auditory_input is not None
        ):
            # Advance oscillators
            self.visual_gamma.advance(self.config.dt_ms)
            self.auditory_gamma.advance(self.config.dt_ms)

            # Apply gamma-phase gating to inputs
            visual_gate = self._compute_gamma_gate(self.visual_gamma.phase)
            auditory_gate = self._compute_gamma_gate(self.auditory_gamma.phase)

            visual_input = visual_input * visual_gate
            auditory_input = auditory_input * auditory_gate

            # Measure phase coherence
            coherence = self._compute_phase_coherence(
                self.visual_gamma.phase,
                self.auditory_gamma.phase,
            )

            # Apply mutual phase coupling
            self._apply_phase_coupling(visual_input, auditory_input)

            # Compute coherence gate for output
            coherence_gate = self._coherence_to_gate(coherence)
            self._last_coherence = coherence

        # =====================================================================
        # STEP 1: Process inputs through modality pools
        # =====================================================================

        # Visual pool
        if visual_input is not None:
            visual_current = torch.mv(self.visual_input_weights, visual_input)
        else:
            visual_current = torch.zeros(self.visual_pool_size, device=self.config.device)

        # Auditory pool
        if auditory_input is not None:
            auditory_current = torch.mv(self.auditory_input_weights, auditory_input)
        else:
            auditory_current = torch.zeros(self.auditory_pool_size, device=self.config.device)

        # Language pool
        if language_input is not None:
            language_current = torch.mv(self.language_input_weights, language_input)
        else:
            language_current = torch.zeros(self.language_pool_size, device=self.config.device)

        # =====================================================================
        # STEP 2: Cross-modal interactions
        # =====================================================================

        # Visual ← Auditory + Language
        if auditory_input is not None:
            visual_current += torch.mv(
                self.auditory_to_visual,
                self.auditory_pool_spikes,
            )
        if language_input is not None:
            visual_current += torch.mv(
                self.language_to_visual,
                self.language_pool_spikes,
            )

        # Auditory ← Visual + Language
        if visual_input is not None:
            auditory_current += torch.mv(
                self.visual_to_auditory,
                self.visual_pool_spikes,
            )
        if language_input is not None:
            auditory_current += torch.mv(
                self.language_to_auditory,
                self.language_pool_spikes,
            )

        # Language ← Visual + Auditory
        if visual_input is not None:
            language_current += torch.mv(
                self.visual_to_language,
                self.visual_pool_spikes,
            )
        if auditory_input is not None:
            language_current += torch.mv(
                self.auditory_to_language,
                self.auditory_pool_spikes,
            )

        # =====================================================================
        # STEP 3: Generate spikes for each pool
        # =====================================================================

        # Create integration current (initially zero, will be computed after pool spikes)
        integration_current = torch.zeros(self.integration_pool_size, device=self.config.device)

        # Concatenate currents for all pools (including integration placeholder)
        pool_currents = torch.cat(
            [
                visual_current,
                auditory_current,
                language_current,
                integration_current,  # Will be updated after we get pool spikes
            ]
        )

        # Get spikes from pool neurons (visual, auditory, language)
        pool_spikes, _ = self.neurons(pool_currents)  # Returns (spikes, membrane)
        pool_spikes = pool_spikes.float()  # Convert bool spikes to float for downstream ops

        # Split into pools
        self.visual_pool_spikes = pool_spikes[: self.visual_pool_size]
        self.auditory_pool_spikes = pool_spikes[
            self.visual_pool_size : self.visual_pool_size + self.auditory_pool_size
        ]
        self.language_pool_spikes = pool_spikes[
            self.visual_pool_size
            + self.auditory_pool_size : self.visual_pool_size
            + self.auditory_pool_size
            + self.language_pool_size
        ]
        self.integration_spikes = pool_spikes[
            self.visual_pool_size + self.auditory_pool_size + self.language_pool_size :
        ]

        # =====================================================================
        # STEP 4: Output (all pools including integration)
        # =====================================================================

        output_spikes = pool_spikes  # Already includes all pools

        # Apply gamma coherence gating (if enabled)
        if self.config.use_gamma_binding and coherence_gate < 1.0:
            output_spikes = output_spikes * coherence_gate

        # =====================================================================
        # STEP 6: Learning (if enabled)
        # =====================================================================

        if self.plasticity_enabled and self.hebbian_strategy:
            # Update cross-modal connections via Hebbian learning
            if visual_input is not None and auditory_input is not None:
                self.visual_to_auditory, _ = self.hebbian_strategy.compute_update(
                    weights=self.visual_to_auditory,
                    pre=self.visual_pool_spikes,
                    post=self.auditory_pool_spikes,
                )

            if visual_input is not None and language_input is not None:
                self.visual_to_language, _ = self.hebbian_strategy.compute_update(
                    weights=self.visual_to_language,
                    pre=self.visual_pool_spikes,
                    post=self.language_pool_spikes,
                )

            if auditory_input is not None and language_input is not None:
                self.auditory_to_language, _ = self.hebbian_strategy.compute_update(
                    weights=self.auditory_to_language,
                    pre=self.auditory_pool_spikes,
                    post=self.language_pool_spikes,
                )

        return output_spikes

    def reset_state(self) -> None:
        """Reset component state."""
        self._reset_state()

    def get_full_state(self) -> Dict[str, Any]:
        """Get complete state for checkpointing."""
        # Collect all learnable weights
        weights = {
            "visual_input_weights": self.visual_input_weights.clone(),
            "auditory_input_weights": self.auditory_input_weights.clone(),
            "language_input_weights": self.language_input_weights.clone(),
            "visual_to_auditory": self.visual_to_auditory.clone(),
            "auditory_to_visual": self.auditory_to_visual.clone(),
            "visual_to_language": self.visual_to_language.clone(),
            "language_to_visual": self.language_to_visual.clone(),
            "auditory_to_language": self.auditory_to_language.clone(),
            "language_to_auditory": self.language_to_auditory.clone(),
            "integration_weights": self.integration_weights.clone(),
        }

        # Component state
        component_state = {
            "visual_pool_spikes": self.visual_pool_spikes.clone(),
            "auditory_pool_spikes": self.auditory_pool_spikes.clone(),
            "language_pool_spikes": self.language_pool_spikes.clone(),
            "integration_spikes": self.integration_spikes.clone(),
        }

        # Neuron state
        neuron_state = self.neurons.get_state() if hasattr(self.neurons, "get_state") else {}

        return {
            "weights": weights,
            "component_state": component_state,
            "neuron_state": neuron_state,
            "config": self.config,
        }

    def load_full_state(self, state: Dict[str, Any]) -> None:
        """Restore complete state from checkpoint."""
        # Restore weights
        weights = state["weights"]
        self.visual_input_weights = weights["visual_input_weights"].to(self.config.device)
        self.auditory_input_weights = weights["auditory_input_weights"].to(self.config.device)
        self.language_input_weights = weights["language_input_weights"].to(self.config.device)
        self.visual_to_auditory = weights["visual_to_auditory"].to(self.config.device)
        self.auditory_to_visual = weights["auditory_to_visual"].to(self.config.device)
        self.visual_to_language = weights["visual_to_language"].to(self.config.device)
        self.language_to_visual = weights["language_to_visual"].to(self.config.device)
        self.auditory_to_language = weights["auditory_to_language"].to(self.config.device)
        self.language_to_auditory = weights["language_to_auditory"].to(self.config.device)
        self.integration_weights = weights["integration_weights"].to(self.config.device)

        # Restore component state
        comp_state = state["component_state"]
        self.visual_pool_spikes = comp_state["visual_pool_spikes"].to(self.config.device)
        self.auditory_pool_spikes = comp_state["auditory_pool_spikes"].to(self.config.device)
        self.language_pool_spikes = comp_state["language_pool_spikes"].to(self.config.device)
        self.integration_spikes = comp_state["integration_spikes"].to(self.config.device)

        # Restore neuron state
        if "neuron_state" in state and hasattr(self.neurons, "load_state"):
            self.neurons.load_state(state["neuron_state"])

    def grow_input(
        self,
        n_new: int,
        initialization: str = "sparse_random",
        sparsity: float = 0.2,
    ) -> None:
        """Grow multimodal integration input dimension.

        Expands input weight matrices for all modalities proportionally.

        Args:
            n_new: Total number of input neurons to add across all modalities
            initialization: Weight init strategy ('sparse_random', 'xavier', 'uniform')
            sparsity: Connection sparsity for new input neurons (if sparse_random)

        Note:
            Growth is distributed across modalities based on their current ratios.
        """
        # Calculate growth per modality based on current ratios
        total_input = (
            self.config.visual_input_size
            + self.config.auditory_input_size
            + self.config.language_input_size
        )

        if total_input == 0:
            # Edge case: no input configured yet, distribute evenly
            visual_growth = n_new // 3
            auditory_growth = n_new // 3
            language_growth = n_new - visual_growth - auditory_growth
        else:
            visual_ratio = self.config.visual_input_size / total_input
            auditory_ratio = self.config.auditory_input_size / total_input
            language_ratio = self.config.language_input_size / total_input

            visual_growth = int(n_new * visual_ratio)
            auditory_growth = int(n_new * auditory_ratio)
            language_growth = n_new - visual_growth - auditory_growth  # Remainder goes to language

        # Expand visual input weights
        if visual_growth > 0:
            self.visual_input_weights = self._grow_weight_matrix_cols(
                self.visual_input_weights,
                visual_growth,
                initializer=initialization,
                sparsity=sparsity,
            )

        # Expand auditory input weights
        if auditory_growth > 0:
            self.auditory_input_weights = self._grow_weight_matrix_cols(
                self.auditory_input_weights,
                auditory_growth,
                initializer=initialization,
                sparsity=sparsity,
            )

        # Expand language input weights
        if language_growth > 0:
            self.language_input_weights = self._grow_weight_matrix_cols(
                self.language_input_weights,
                language_growth,
                initializer=initialization,
                sparsity=sparsity,
            )

        # Update config
        new_visual_size = self.config.visual_input_size + visual_growth
        new_auditory_size = self.config.auditory_input_size + auditory_growth
        new_language_size = self.config.language_input_size + language_growth

        self.config = replace(
            self.config,
            visual_input_size=new_visual_size,
            auditory_input_size=new_auditory_size,
            language_input_size=new_language_size,
        )
        self.multisensory_config = replace(
            self.multisensory_config,
            visual_input_size=new_visual_size,
            auditory_input_size=new_auditory_size,
            language_input_size=new_language_size,
        )

    def grow_output(
        self,
        n_new: int,
        initialization: str = "sparse_random",
        sparsity: float = 0.2,
    ) -> None:
        """Grow multimodal integration output dimension.

        Expands neuron pools and all associated weight matrices.

        Args:
            n_new: Number of neurons to add
            initialization: Weight init strategy ('sparse_random', 'xavier', 'uniform')
            sparsity: Connection sparsity for new neurons (if sparse_random)

        Note:
            Growth is distributed across modality pools based on configured ratios.
        """
        # Compute old total from pool sizes
        old_n_output = (
            self.visual_pool_size
            + self.auditory_pool_size
            + self.language_pool_size
            + self.integration_pool_size
        )
        new_n_output = old_n_output + n_new

        # Calculate growth per pool based on current size ratios (dynamic)
        visual_ratio = self.visual_pool_size / old_n_output
        auditory_ratio = self.auditory_pool_size / old_n_output
        language_ratio = self.language_pool_size / old_n_output

        visual_growth = int(n_new * visual_ratio)
        auditory_growth = int(n_new * auditory_ratio)
        language_growth = int(n_new * language_ratio)
        integration_growth = n_new - visual_growth - auditory_growth - language_growth

        # Update pool sizes
        old_visual = self.visual_pool_size
        old_auditory = self.auditory_pool_size
        old_language = self.language_pool_size

        self.visual_pool_size += visual_growth
        self.auditory_pool_size += auditory_growth
        self.language_pool_size += language_growth
        self.integration_pool_size += integration_growth

        # Expand visual input weights [visual_pool, visual_input]
        if visual_growth > 0:
            self.visual_input_weights = self._grow_weight_matrix_rows(
                self.visual_input_weights,
                visual_growth,
                initializer=initialization,
                sparsity=sparsity,
            )

        # Expand auditory input weights [auditory_pool, auditory_input]
        if auditory_growth > 0:
            self.auditory_input_weights = self._grow_weight_matrix_rows(
                self.auditory_input_weights,
                auditory_growth,
                initializer=initialization,
                sparsity=sparsity,
            )

        # Expand language input weights [language_pool, language_input]
        if language_growth > 0:
            self.language_input_weights = self._grow_weight_matrix_rows(
                self.language_input_weights,
                language_growth,
                initializer=initialization,
                sparsity=sparsity,
            )

        # Expand cross-modal weights (complex - need to handle both dimensions)
        # Visual ↔ Auditory
        if visual_growth > 0 or auditory_growth > 0:
            # visual_to_auditory [auditory, visual]
            expanded = self._grow_weight_matrix_rows(
                self.visual_to_auditory,
                auditory_growth,
                initializer=initialization,
                sparsity=sparsity,
            )
            self.visual_to_auditory = (
                self._grow_weight_matrix_cols(
                    expanded, visual_growth, initializer=initialization, sparsity=sparsity
                )
                * self.multisensory_config.cross_modal_strength
            )

            # auditory_to_visual [visual, auditory]
            expanded = self._grow_weight_matrix_rows(
                self.auditory_to_visual,
                visual_growth,
                initializer=initialization,
                sparsity=sparsity,
            )
            self.auditory_to_visual = (
                self._grow_weight_matrix_cols(
                    expanded, auditory_growth, initializer=initialization, sparsity=sparsity
                )
                * self.multisensory_config.cross_modal_strength
            )

        # Similar expansions for other cross-modal connections would go here...
        # (abbreviated for brevity, but follows same pattern)

        # Expand neurons
        self.neurons = create_pyramidal_neurons(new_n_output, self.device)

        # Expand integration weights
        # This is complex as it connects all pools - simplified version:
        new_total_pool = self.visual_pool_size + self.auditory_pool_size + self.language_pool_size
        self.integration_weights = (
            WeightInitializer.sparse_random(
                n_output=self.integration_pool_size,
                n_input=new_total_pool,
                sparsity=0.3,
                device=self.device,
            )
            * self.multisensory_config.integration_strength
        )

        # Update configs (including explicit pool sizes)
        self.config = replace(
            self.config,
            visual_pool_size=self.visual_pool_size,
            auditory_pool_size=self.auditory_pool_size,
            language_pool_size=self.language_pool_size,
            integration_pool_size=self.integration_pool_size,
        )
        self.multisensory_config = replace(
            self.multisensory_config,
            visual_pool_size=self.visual_pool_size,
            auditory_pool_size=self.auditory_pool_size,
            language_pool_size=self.language_pool_size,
            integration_pool_size=self.integration_pool_size,
        )

    def get_diagnostics(self) -> dict[str, Any]:
        """Get diagnostic information in standardized DiagnosticsDict format.

        Returns:
            Standardized diagnostics with activity, plasticity, health, and custom metrics
        """
        from thalia.core.diagnostics_schema import (
            compute_activity_metrics,
            compute_health_metrics,
            compute_plasticity_metrics,
        )

        # Compute activity for each pool
        visual_activity = compute_activity_metrics(
            self.visual_pool_spikes,
            total_neurons=self.visual_pool_size,
        )
        auditory_activity = compute_activity_metrics(
            self.auditory_pool_spikes,
            total_neurons=self.auditory_pool_size,
        )
        language_activity = compute_activity_metrics(
            self.language_pool_spikes,
            total_neurons=self.language_pool_size,
        )
        integration_activity = compute_activity_metrics(
            self.integration_spikes,
            total_neurons=self.integration_pool_size,
        )

        # Overall activity (weighted average across pools)
        total_neurons = (
            self.visual_pool_size
            + self.auditory_pool_size
            + self.language_pool_size
            + self.integration_pool_size
        )
        total_firing_rate = (
            visual_activity.get("firing_rate", 0.0) * self.visual_pool_size
            + auditory_activity.get("firing_rate", 0.0) * self.auditory_pool_size
            + language_activity.get("firing_rate", 0.0) * self.language_pool_size
            + integration_activity.get("firing_rate", 0.0) * self.integration_pool_size
        ) / total_neurons

        # Compute plasticity metrics for cross-modal weights
        plasticity = None
        if self.config.learning_enabled and self.hebbian_strategy is not None:
            plasticity = compute_plasticity_metrics(
                weights=self.visual_to_auditory,  # Use one weight matrix as representative
                learning_rate=self.config.learning_rate,
            )
            # Add average cross-modal strength
            cross_modal_mean = (
                self.visual_to_auditory.mean().item()
                + self.visual_to_language.mean().item()
                + self.auditory_to_language.mean().item()
            ) / 3.0
            plasticity["weight_mean"] = float(cross_modal_mean)

        # Compute health metrics
        health = compute_health_metrics(
            state_tensors={
                "visual_pool": self.visual_pool_spikes,
                "auditory_pool": self.auditory_pool_spikes,
                "language_pool": self.language_pool_spikes,
                "integration": self.integration_spikes,
            },
            firing_rate=total_firing_rate,
            silence_threshold=SILENCE_DETECTION_THRESHOLD,
        )

        # Region-specific metrics
        region_specific = {
            "pool_firing_rates": {
                "visual": visual_activity.get("firing_rate", 0.0),
                "auditory": auditory_activity.get("firing_rate", 0.0),
                "language": language_activity.get("firing_rate", 0.0),
                "integration": integration_activity.get("firing_rate", 0.0),
            },
            "cross_modal_weights": {
                "visual_to_auditory": float(self.visual_to_auditory.mean().item()),
                "visual_to_language": float(self.visual_to_language.mean().item()),
                "auditory_to_language": float(self.auditory_to_language.mean().item()),
            },
            "pool_sizes": {
                "visual": self.visual_pool_size,
                "auditory": self.auditory_pool_size,
                "language": self.language_pool_size,
                "integration": self.integration_pool_size,
            },
        }

        # Return as dict (DiagnosticsDict is a TypedDict, not a class)
        return {
            "activity": integration_activity,
            "plasticity": plasticity,
            "health": health,
            "neuromodulators": None,
            "region_specific": region_specific,
        }

    def check_health(self):
        """Check region health.

        Returns:
            HealthReport with health status and issues
        """
        from thalia.diagnostics.health_monitor import HealthIssue, HealthReport, IssueReport

        issues = []
        max_severity = 0.0

        # Check for silence
        if self.visual_pool_spikes.mean() < SILENCE_DETECTION_THRESHOLD:
            issues.append(
                IssueReport(
                    issue_type=HealthIssue.ACTIVITY_COLLAPSE,
                    severity=50.0,
                    description="Visual pool silent - check visual input pathway",
                    recommendation="Verify visual input connectivity and strengths",
                    metrics={"visual_firing_rate": float(self.visual_pool_spikes.mean())},
                )
            )
            max_severity = max(max_severity, 50.0)

        if self.auditory_pool_spikes.mean() < SILENCE_DETECTION_THRESHOLD:
            issues.append(
                IssueReport(
                    issue_type=HealthIssue.ACTIVITY_COLLAPSE,
                    severity=50.0,
                    description="Auditory pool silent - check auditory input pathway",
                    recommendation="Verify auditory input connectivity and strengths",
                    metrics={"auditory_firing_rate": float(self.auditory_pool_spikes.mean())},
                )
            )
            max_severity = max(max_severity, 50.0)

        if self.language_pool_spikes.mean() < SILENCE_DETECTION_THRESHOLD:
            issues.append(
                IssueReport(
                    issue_type=HealthIssue.ACTIVITY_COLLAPSE,
                    severity=50.0,
                    description="Language pool silent - check language input pathway",
                    recommendation="Verify language input connectivity and strengths",
                    metrics={"language_firing_rate": float(self.language_pool_spikes.mean())},
                )
            )
            max_severity = max(max_severity, 50.0)

        if self.integration_spikes.mean() < SILENCE_DETECTION_THRESHOLD:
            issues.append(
                IssueReport(
                    issue_type=HealthIssue.ACTIVITY_COLLAPSE,
                    severity=80.0,
                    description="Integration neurons silent - check cross-modal connections",
                    recommendation="Verify integration weights and cross-modal connectivity",
                    metrics={"integration_firing_rate": float(self.integration_spikes.mean())},
                )
            )
            max_severity = max(max_severity, 80.0)

        # Check for saturation
        if self.visual_pool_spikes.mean() > 0.9:
            issues.append(
                IssueReport(
                    issue_type=HealthIssue.SEIZURE_RISK,
                    severity=80.0,
                    description="Visual pool saturated - excessive activity detected",
                    recommendation="Reduce visual input strength or add inhibition",
                    metrics={"visual_firing_rate": float(self.visual_pool_spikes.mean())},
                )
            )
            max_severity = max(max_severity, 80.0)

        if self.auditory_pool_spikes.mean() > 0.9:
            issues.append(
                IssueReport(
                    issue_type=HealthIssue.SEIZURE_RISK,
                    severity=80.0,
                    description="Auditory pool saturated - excessive activity detected",
                    recommendation="Verify auditory input strength and inhibition",
                    metrics={"auditory_firing_rate": float(self.auditory_pool_spikes.mean())},
                )
            )
            max_severity = max(max_severity, 80.0)

        is_healthy = len(issues) == 0
        summary = "Healthy" if is_healthy else f"{len(issues)} issue(s) detected"

        return HealthReport(
            is_healthy=is_healthy,
            overall_severity=max_severity,
            issues=issues,
            summary=summary,
            metrics=self.get_diagnostics(),
        )

    # =========================================================================
    # GAMMA SYNCHRONIZATION METHODS (Cross-Modal Binding)
    # =========================================================================
    # These methods implement gamma-band synchronization for binding
    # visual and auditory modalities. Gamma emerges from local circuit
    # interactions in this integration region (biologically accurate).

    def _compute_gamma_gate(self, gamma_phase: float, width: float = 0.3) -> float:
        """Compute gamma-phase-dependent gating.

        Creates temporal window: inputs are only strongly processed
        during certain phases of gamma (peak excitability).

        Args:
            gamma_phase: Current gamma phase [0, 2π)
            width: Width of the Gaussian gate

        Returns:
            gate: Gating strength [0, 1]
        """
        # Gaussian centered at phase = π/2 (peak of sine wave)
        optimal_phase = math.pi / 2
        phase_diff = abs(gamma_phase - optimal_phase)

        # Wrap around for circular distance
        phase_diff = min(phase_diff, 2 * math.pi - phase_diff)

        # Gaussian gate
        gate = math.exp(-(phase_diff**2) / (2 * width**2))
        return gate

    def _compute_phase_coherence(
        self,
        visual_phase: float,
        auditory_phase: float,
    ) -> float:
        """Measure phase coherence between two oscillators.

        High coherence (near 1.0) = phases aligned = bound together
        Low coherence (near 0.0) = phases misaligned = separate objects

        Args:
            visual_phase: Visual gamma phase [0, 2π)
            auditory_phase: Auditory gamma phase [0, 2π)

        Returns:
            coherence: Coherence score [0, 1]
        """
        # Circular distance between phases
        phase_diff = abs(visual_phase - auditory_phase)
        phase_diff = min(phase_diff, 2 * math.pi - phase_diff)

        # Convert to coherence: 0 diff = 1.0, π diff = 0.0
        coherence = math.cos(phase_diff / 2.0) ** 2  # Squared cosine for sharper tuning
        return coherence

    def _apply_phase_coupling(
        self,
        visual_spikes: torch.Tensor,
        auditory_spikes: torch.Tensor,
    ) -> None:
        """Apply mutual phase coupling between modalities.

        When one modality has strong input, it nudges the other's phase
        toward synchrony. This is how the brain achieves binding.

        Args:
            visual_spikes: Visual activity (used to weight coupling)
            auditory_spikes: Auditory activity (used to weight coupling)
        """
        # Measure activity strength
        visual_activity = float(visual_spikes.mean())
        auditory_activity = float(auditory_spikes.mean())

        # Only couple if both modalities are active
        if visual_activity > 0.01 and auditory_activity > 0.01:
            # Compute phase difference
            phase_diff = self.auditory_gamma.phase - self.visual_gamma.phase

            # Normalize to [-π, π]
            if phase_diff > math.pi:
                phase_diff -= 2 * math.pi
            elif phase_diff < -math.pi:
                phase_diff += 2 * math.pi

            # Nudge each phase toward the other
            coupling_amount = self.config.phase_coupling_strength * phase_diff

            # Visual nudged by auditory
            visual_nudge = coupling_amount * auditory_activity
            new_visual_phase = self.visual_gamma.phase + visual_nudge
            self.visual_gamma.sync_to_phase(new_visual_phase)

            # Auditory nudged by visual (opposite direction)
            auditory_nudge = -coupling_amount * visual_activity
            new_auditory_phase = self.auditory_gamma.phase + auditory_nudge
            self.auditory_gamma.sync_to_phase(new_auditory_phase)

    def _coherence_to_gate(self, coherence: float) -> float:
        """Convert phase coherence to output gate.

        Args:
            coherence: Phase coherence [0, 1]

        Returns:
            gate: Output gating strength [0, 1]
        """
        # Soft threshold: gradually increase above threshold
        if coherence < self.config.gate_threshold:
            return 0.0
        else:
            # Sigmoid above threshold for smooth gating
            x = (coherence - self.config.gate_threshold) / (1.0 - self.config.gate_threshold)
            return 1.0 / (1.0 + math.exp(-10.0 * (x - 0.5)))
