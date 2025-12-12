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

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch

from thalia.components.neurons.neuron_factory import create_pyramidal_neurons
from thalia.regions.base import NeuralComponent, NeuralComponentConfig
from thalia.managers.component_registry import register_region
from thalia.components.synapses.weight_init import WeightInitializer
from thalia.learning.rules.strategies import HebbianStrategy, HebbianConfig


@dataclass
class MultimodalIntegrationConfig(NeuralComponentConfig):
    """Configuration for multimodal integration region.

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
        enable_hebbian: Enable Hebbian cross-modal plasticity
        hebbian_lr: Learning rate for Hebbian plasticity
    """
    # Input sizes
    visual_input_size: int = 0
    auditory_input_size: int = 0
    language_input_size: int = 0

    # Pool sizes (as fractions of n_output)
    visual_pool_ratio: float = 0.3
    auditory_pool_ratio: float = 0.3
    language_pool_ratio: float = 0.2
    integration_pool_ratio: float = 0.2

    # Connection strengths
    cross_modal_strength: float = 0.4
    within_modal_strength: float = 0.6
    integration_strength: float = 0.8
    salience_competition_strength: float = 0.5

    # Plasticity
    enable_hebbian: bool = True
    hebbian_lr: float = 0.001


@register_region("multimodal_integration")
class MultimodalIntegration(NeuralComponent):
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
        super().__init__(config)
        self.config = config

        # Validate pool ratios sum to ~1.0
        total_ratio = (
            config.visual_pool_ratio +
            config.auditory_pool_ratio +
            config.language_pool_ratio +
            config.integration_pool_ratio
        )
        if not (0.95 <= total_ratio <= 1.05):
            raise ValueError(
                f"Pool ratios must sum to ~1.0, got {total_ratio:.3f}"
            )

        # Calculate pool sizes
        self.visual_pool_size = int(config.n_output * config.visual_pool_ratio)
        self.auditory_pool_size = int(config.n_output * config.auditory_pool_ratio)
        self.language_pool_size = int(config.n_output * config.language_pool_ratio)
        self.integration_pool_size = (
            config.n_output -
            self.visual_pool_size -
            self.auditory_pool_size -
            self.language_pool_size
        )

        # Create neurons for each pool
        self.neurons = create_pyramidal_neurons(
            n_neurons=config.n_output,
            config=config,
            device=config.device,
        )

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
        self.visual_to_auditory = WeightInitializer.sparse_random(
            n_output=self.auditory_pool_size,
            n_input=self.visual_pool_size,
            sparsity=0.3,
            device=config.device,
        ) * config.cross_modal_strength

        self.auditory_to_visual = WeightInitializer.sparse_random(
            n_output=self.visual_pool_size,
            n_input=self.auditory_pool_size,
            sparsity=0.3,
            device=config.device,
        ) * config.cross_modal_strength

        # Visual ↔ Language
        self.visual_to_language = WeightInitializer.sparse_random(
            n_output=self.language_pool_size,
            n_input=self.visual_pool_size,
            sparsity=0.3,
            device=config.device,
        ) * config.cross_modal_strength

        self.language_to_visual = WeightInitializer.sparse_random(
            n_output=self.visual_pool_size,
            n_input=self.language_pool_size,
            sparsity=0.3,
            device=config.device,
        ) * config.cross_modal_strength

        # Auditory ↔ Language
        self.auditory_to_language = WeightInitializer.sparse_random(
            n_output=self.language_pool_size,
            n_input=self.auditory_pool_size,
            sparsity=0.3,
            device=config.device,
        ) * config.cross_modal_strength

        self.language_to_auditory = WeightInitializer.sparse_random(
            n_output=self.auditory_pool_size,
            n_input=self.language_pool_size,
            sparsity=0.3,
            device=config.device,
        ) * config.cross_modal_strength

        # =====================================================================
        # INTEGRATION WEIGHTS (pools → integration neurons)
        # =====================================================================

        total_pool_size = (
            self.visual_pool_size +
            self.auditory_pool_size +
            self.language_pool_size
        )

        self.integration_weights = WeightInitializer.sparse_random(
            n_output=self.integration_pool_size,
            n_input=total_pool_size,
            sparsity=0.4,
            device=config.device,
        ) * config.integration_strength

        # =====================================================================
        # LEARNING
        # =====================================================================

        if config.enable_hebbian:
            hebbian_config = HebbianConfig(
                learning_rate=config.hebbian_lr,
                weight_decay=0.0001,
            )
            self.hebbian_strategy = HebbianStrategy(hebbian_config)
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

        # Concatenate currents for all pools
        pool_currents = torch.cat([
            visual_current,
            auditory_current,
            language_current,
        ])

        # Get spikes from pool neurons
        pool_spikes = self.neurons(pool_currents)

        # Split into pools
        self.visual_pool_spikes = pool_spikes[:self.visual_pool_size]
        self.auditory_pool_spikes = pool_spikes[
            self.visual_pool_size:
            self.visual_pool_size + self.auditory_pool_size
        ]
        self.language_pool_spikes = pool_spikes[
            self.visual_pool_size + self.auditory_pool_size:
            self.visual_pool_size + self.auditory_pool_size + self.language_pool_size
        ]

        # =====================================================================
        # STEP 4: Integration neurons (bind modalities)
        # =====================================================================

        # All pool spikes → integration neurons
        integration_current = torch.mv(
            self.integration_weights,
            pool_spikes[:-self.integration_pool_size],  # Exclude integration neurons
        )

        # Get integration spikes
        self.integration_spikes = self.neurons.generate_spikes(
            integration_current,
            neuron_indices=torch.arange(
                self.config.n_output - self.integration_pool_size,
                self.config.n_output,
                device=self.config.device,
            ),
        )

        # =====================================================================
        # STEP 5: Output (all pools + integration)
        # =====================================================================

        output_spikes = torch.cat([
            pool_spikes[:-self.integration_pool_size],
            self.integration_spikes,
        ])

        # =====================================================================
        # STEP 6: Learning (if enabled)
        # =====================================================================

        if self.plasticity_enabled and self.hebbian_strategy:
            # Update cross-modal connections via Hebbian learning
            if visual_input is not None and auditory_input is not None:
                self.visual_to_auditory = self.hebbian_strategy.apply(
                    weights=self.visual_to_auditory,
                    pre_spikes=self.visual_pool_spikes,
                    post_spikes=self.auditory_pool_spikes,
                )

            if visual_input is not None and language_input is not None:
                self.visual_to_language = self.hebbian_strategy.apply(
                    weights=self.visual_to_language,
                    pre_spikes=self.visual_pool_spikes,
                    post_spikes=self.language_pool_spikes,
                )

            if auditory_input is not None and language_input is not None:
                self.auditory_to_language = self.hebbian_strategy.apply(
                    weights=self.auditory_to_language,
                    pre_spikes=self.auditory_pool_spikes,
                    post_spikes=self.language_pool_spikes,
                )

        return output_spikes

    def _initialize_weights(self) -> Optional[torch.Tensor]:
        """Initialize weights (handled in __init__)."""
        return None  # Weights created explicitly in __init__

    def _create_neurons(self) -> Optional[Any]:
        """Create neurons (handled in __init__)."""
        return None  # Neurons created explicitly in __init__

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

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information.

        Returns:
            Dict with visual/auditory/language firing rates, integration activity
        """
        return {
            "visual_pool_firing_rate": float(self.visual_pool_spikes.mean()),
            "auditory_pool_firing_rate": float(self.auditory_pool_spikes.mean()),
            "language_pool_firing_rate": float(self.language_pool_spikes.mean()),
            "integration_firing_rate": float(self.integration_spikes.mean()),
            "visual_pool_active_fraction": float((self.visual_pool_spikes > 0).float().mean()),
            "auditory_pool_active_fraction": float((self.auditory_pool_spikes > 0).float().mean()),
            "language_pool_active_fraction": float((self.language_pool_spikes > 0).float().mean()),
            "cross_modal_weight_mean": float(
                (self.visual_to_auditory.mean() +
                 self.visual_to_language.mean() +
                 self.auditory_to_language.mean()) / 3.0
            ),
        }

    def check_health(self):
        """Check region health.

        Returns:
            HealthReport with health status and issues
        """
        from thalia.diagnostics.health_monitor import HealthReport, IssueReport

        issues = []
        max_severity = 0.0

        # Check for silence
        if self.visual_pool_spikes.mean() < 0.001:
            issues.append(IssueReport(
                component="multimodal_integration.visual_pool",
                issue="Visual pool silent",
                severity="medium",
                recommendation="Check visual input pathway",
                metrics={"firing_rate": float(self.visual_pool_spikes.mean())},
            ))
            max_severity = max(max_severity, 0.5)

        if self.auditory_pool_spikes.mean() < 0.001:
            issues.append(IssueReport(
                component="multimodal_integration.auditory_pool",
                issue="Auditory pool silent",
                severity="medium",
                recommendation="Check auditory input pathway",
                metrics={"firing_rate": float(self.auditory_pool_spikes.mean())},
            ))
            max_severity = max(max_severity, 0.5)

        if self.language_pool_spikes.mean() < 0.001:
            issues.append(IssueReport(
                component="multimodal_integration.language_pool",
                issue="Language pool silent",
                severity="medium",
                recommendation="Check language input pathway",
                metrics={"firing_rate": float(self.language_pool_spikes.mean())},
            ))
            max_severity = max(max_severity, 0.5)

        if self.integration_spikes.mean() < 0.001:
            issues.append(IssueReport(
                component="multimodal_integration.integration_neurons",
                issue="Integration neurons silent",
                severity="high",
                recommendation="Check cross-modal connections and integration weights",
                metrics={"firing_rate": float(self.integration_spikes.mean())},
            ))
            max_severity = max(max_severity, 0.8)

        # Check for saturation
        if self.visual_pool_spikes.mean() > 0.9:
            issues.append(IssueReport(
                component="multimodal_integration.visual_pool",
                issue="Visual pool saturated",
                severity="high",
                recommendation="Reduce visual input strength or add inhibition",
                metrics={"firing_rate": float(self.visual_pool_spikes.mean())},
            ))
            max_severity = max(max_severity, 0.8)

        if self.auditory_pool_spikes.mean() > 0.9:
            issues.append(IssueReport(
                component="multimodal_integration.auditory_pool",
                issue="Auditory pool saturated",
                severity="high",
                recommendation="Reduce auditory input strength or add inhibition",
                metrics={"firing_rate": float(self.auditory_pool_spikes.mean())},
            ))
            max_severity = max(max_severity, 0.8)

        is_healthy = len(issues) == 0
        summary = "Healthy" if is_healthy else f"{len(issues)} issue(s) detected"

        return HealthReport(
            is_healthy=is_healthy,
            overall_severity=max_severity,
            issues=issues,
            summary=summary,
            metrics=self.get_diagnostics(),
        )
