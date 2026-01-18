"""
Afferent Synapses - Synaptic integration layer for neural regions.

This module implements synaptic layers that receive and integrate afferent inputs
from other regions. Unlike pathways (which route spikes), afferent synapses:
- OWN the synaptic weights (connection strengths)
- APPLY learning rules (STDP, three-factor, Hebbian, etc.)
- INTEGRATE short-term plasticity (STP)
- BELONG to the target region (at dendrites)

Architecture v2.0: Synapses are properties of the POST-synaptic neuron, not
the axonal pathway. This matches biological reality where synapses are located
at the dendrites of the receiving neuron.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from thalia.learning.rules.strategies import LearningStrategy


@dataclass
class AfferentSynapsesConfig:
    """Configuration for afferent synaptic layer.

    Attributes:
        n_neurons: Number of post-synaptic neurons
        n_inputs: Number of pre-synaptic inputs (axons)
        learning_rule: Learning strategy name (e.g., 'stdp', 'three_factor')
        learning_rate: Learning rate for weight updates
        use_stp: Whether to apply short-term plasticity
        stp_config: Configuration for STP (if enabled)
        device: Torch device
    """

    n_neurons: int
    n_inputs: int
    learning_rule: str = "hebbian"
    learning_rate: float = 0.001
    use_stp: bool = False
    stp_config: Optional[Any] = None  # Will be STPConfig if use_stp=True
    device: str = "cpu"


class AfferentSynapses(nn.Module):
    """Synaptic integration layer for receiving region inputs.

    Located at the dendrites of the post-synaptic region. Handles:
    1. Synaptic weights [n_neurons, n_inputs]
    2. Learning rules (STDP, BCM, Hebbian, three-factor, etc.)
    3. Short-term plasticity (facilitation/depression)

    Biological Principle:
    Synapses are properties of the POST-synaptic neuron's dendrites, not
    the PRE-synaptic axon. The weights represent connection strengths from
    incoming axons to this neuron's dendrites.

    Example:
        # Striatum receives from cortex L5 (128) + hippocampus (64) + pfc (32)
        # = 224 total afferent axons
        synapses = AfferentSynapses(
            config=AfferentSynapsesConfig(
                n_neurons=70,      # 70 MSN neurons
                n_inputs=224,      # 224 afferent axons
                learning_rule="three_factor",  # Dopamine-gated
                learning_rate=0.001,
            )
        )

        # Integrate synaptic input
        synaptic_current = synapses(input_spikes)  # [224] → [70]

        # Apply learning
        synapses.apply_learning(
            pre_spikes=input_spikes,
            post_spikes=neuron_spikes,
            modulator=dopamine_level,
        )

    Args:
        config: AfferentSynapsesConfig with all parameters
        learning_strategy: Optional pre-constructed learning strategy
    """

    def __init__(
        self,
        config: AfferentSynapsesConfig,
        learning_strategy: Optional[LearningStrategy] = None,
    ):
        super().__init__()

        self.config = config
        self.device = torch.device(config.device)

        # Synaptic weights [n_neurons, n_inputs]
        # Initialize with small random values (Gaussian)
        from thalia.components.synapses.weight_init import WeightInitializer

        self.weights = nn.Parameter(
            WeightInitializer.gaussian(
                n_output=config.n_neurons,
                n_input=config.n_inputs,
                mean=0.3,
                std=0.1,
                device=self.device,
            )
        )

        # Learning strategy (STDP, three-factor, Hebbian, etc.)
        if learning_strategy is not None:
            self.learning_strategy = learning_strategy
        else:
            from thalia.learning.rules.strategies import create_strategy

            self.learning_strategy = create_strategy(
                rule_name=config.learning_rule,
                learning_rate=config.learning_rate,
            )

        # Short-term plasticity (optional)
        self.stp: Optional[Any] = None  # Will be ShortTermPlasticity if enabled
        if config.use_stp:
            try:
                from thalia.components.plasticity.stp import (
                    ShortTermPlasticity,
                )
                from thalia.components.plasticity.stp import STPConfig as STPConfigClass

                stp_config = config.stp_config or STPConfigClass()
                self.stp = ShortTermPlasticity(
                    n_synapses=config.n_inputs,
                    config=stp_config,
                    device=config.device,
                )
            except ImportError:
                # STP not available, skip
                pass

        # Track if learning is enabled
        self.learning_enabled = True

    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """Integrate synaptic input from afferent axons.

        Args:
            input_spikes: Pre-synaptic spikes [n_inputs]

        Returns:
            Synaptic current [n_neurons] = weights @ input_spikes

        Note:
            Does NOT apply learning here - call apply_learning() separately
            after neuron dynamics to get post-synaptic spikes.
        """
        # Apply short-term plasticity if enabled
        effective_weights = self.weights
        if self.stp is not None:
            # STP modulates synaptic efficacy
            stp_modulation = self.stp(input_spikes)  # [n_inputs]
            effective_weights = self.weights * stp_modulation.unsqueeze(0)

        # Synaptic integration: weights @ input → current
        synaptic_current = effective_weights @ input_spikes.float()

        return synaptic_current

    def apply_learning(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        modulator: Optional[float] = None,
    ) -> Dict[str, float]:
        """Apply synaptic learning rule.

        Args:
            pre_spikes: Pre-synaptic spikes [n_inputs]
            post_spikes: Post-synaptic spikes [n_neurons]
            modulator: Neuromodulator level (dopamine, ACh, etc.) for gating

        Returns:
            Dict with learning metrics (e.g., mean weight change)

        Example:
            # Three-factor learning (eligibility × dopamine)
            metrics = synapses.apply_learning(
                pre_spikes=input_spikes,
                post_spikes=neuron_output,
                modulator=dopamine_level,
            )
        """
        if not self.learning_enabled:
            return {"mean_weight_change": 0.0}

        # Compute weight update using learning strategy
        new_weights, metrics = self.learning_strategy.compute_update(
            weights=self.weights,
            pre=pre_spikes,
            post=post_spikes,
            modulator=modulator,
        )

        # Apply update
        self.weights.data = new_weights

        return metrics

    def grow_output(self, n_new: int) -> None:
        """Grow synaptic layer for more post-synaptic neurons.

        When this region grows neurons, it needs more synaptic connections
        for the new neurons to receive inputs.

        Args:
            n_new: Number of new neurons to add

        Example:
            # Striatum grows from 70 to 90 neurons (+20)
            striatum.afferent_synapses.grow_output(n_new=20)
            # weights: [70, 224] → [90, 224]
        """
        old_n_neurons = self.config.n_neurons
        new_n_neurons = old_n_neurons + n_new

        # Create expanded weight matrix
        from thalia.components.synapses.weight_init import WeightInitializer

        new_weights = torch.zeros(
            new_n_neurons,
            self.config.n_inputs,
            device=self.device,
        )

        # Copy old weights (preserve existing neurons)
        new_weights[:old_n_neurons, :] = self.weights.data

        # Initialize new weights (for new neurons)
        new_weights[old_n_neurons:, :] = WeightInitializer.gaussian(
            n_output=n_new,
            n_input=self.config.n_inputs,
            mean=0.3,
            std=0.1,
            device=self.device,
        )

        # Replace parameter
        self.weights = nn.Parameter(new_weights)

        # Update config
        self.config.n_neurons = new_n_neurons

    def reset(self) -> None:
        """Reset synaptic state (STP dynamics, but not weights)."""
        if self.stp is not None:
            self.stp.reset()

    def get_state(self) -> Dict[str, Any]:
        """Get state for checkpointing.

        Returns:
            Dict with weights and STP state
        """
        state = {
            "weights": self.weights.data.cpu().clone(),  # MUST clone() to avoid reference issues
            "config": {
                "n_neurons": self.config.n_neurons,
                "n_inputs": self.config.n_inputs,
                "learning_rule": self.config.learning_rule,
                "learning_rate": self.config.learning_rate,
            },
        }

        if self.stp is not None:
            state["stp"] = self.stp.get_state()

        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from checkpoint.

        Args:
            state: State dict from get_state()
        """
        # Load weights - use copy_() to properly update in-place
        self.weights.data.copy_(state["weights"].to(self.device))

        # Load STP state if present
        if "stp" in state and self.stp is not None:
            self.stp.load_state(state["stp"])

    def __repr__(self) -> str:
        """Human-readable representation."""
        stp_str = " + STP" if self.stp is not None else ""
        return (
            f"AfferentSynapses({self.config.n_inputs} → {self.config.n_neurons}, "
            f"{self.config.learning_rule}{stp_str})"
        )
