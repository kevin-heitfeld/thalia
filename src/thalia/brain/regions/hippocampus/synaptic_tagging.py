"""
Synaptic Tagging and Capture for Hippocampal Memory Consolidation.

Biologically-accurate implementation of synaptic tagging for hippocampal memory.
Tags mark recently-active synapses for consolidation. Dopamine converts tags
to persistent weight changes.

This replaces explicit Episode.priority values with emergent priority
based on synaptic tag strength.
"""

from __future__ import annotations

import torch


class SynapticTagging:
    """Frey-Morris synaptic tagging and capture.

    Biological mechanism:
    1. Spike coincidence creates weak tags at active synapses
    2. Tags decay over ~20-50 timesteps (seconds in biology)
    3. Dopamine converts tags to persistent weight changes
    4. Strong tags = high replay probability (biological priority)

    No explicit priority values - tag strength IS the priority.

    This implements the "synaptic tagging and capture" hypothesis where:
    - Tags: Set by local activity (Hebbian learning)
    - Capture: Neuromodulators (dopamine) convert tags to lasting changes

    Usage:
        tagging = SynapticTagging(n_neurons=100, device="cuda")

        # During encoding
        tagging.update_tags(pre_spikes, post_spikes)

        # After reward
        if dopamine > 0.5:
            weights = tagging.consolidate_tagged_synapses(
                weights, dopamine, learning_rate=0.01
            )

        # During sleep/replay
        replay_probs = tagging.get_replay_probabilities()
    """

    def __init__(
        self,
        n_neurons: int,
        device: str,
        tag_decay: float = 0.95,
        tag_threshold: float = 0.1,
    ):
        """Initialize synaptic tagging system.

        Args:
            n_neurons: Number of neurons in the recurrent layer
            device: Device to place tensors on
            tag_decay: Decay factor per timestep (0.95 = ~20 timestep lifetime)
            tag_threshold: Minimum spike coincidence to create tag
        """
        self.n_neurons = n_neurons
        self.device = torch.device(device)

        # Synaptic tags [post, pre] - matches weight matrix shape
        self.tags = torch.zeros(n_neurons, n_neurons, device=self.device)

        # Tag decay: ~95% per timestep = ~20 timestep lifetime (~20ms @ 1ms dt)
        # In biology: tags last seconds to minutes
        self.tag_decay = tag_decay

        # Tag creation threshold (minimum spike coincidence)
        self.tag_threshold = tag_threshold

    def update_tags(
        self,
        pre_spikes: torch.Tensor,  # [n_pre] binary or float
        post_spikes: torch.Tensor,  # [n_post] binary or float
    ):
        """Mark recently-active synapses with tags.

        Tags are created by spike coincidence (Hebbian rule).
        Existing tags decay naturally.

        Biological: Weak synaptic activity creates protein synthesis tags.
        These tags mark synapses as "eligible" for consolidation.

        Args:
            pre_spikes: Presynaptic spikes (binary or float)
            post_spikes: Postsynaptic spikes (binary or float)
        """
        # Decay existing tags
        self.tags *= self.tag_decay

        # Create new tags at active synapses (outer product)
        # Only create if both pre and post fire (spike coincidence)
        new_tags = torch.outer(post_spikes.float(), pre_spikes.float())

        # Update: take maximum (tags don't subtract, only add/decay)
        self.tags = torch.maximum(self.tags, new_tags)

    def consolidate_tagged_synapses(
        self,
        weights: torch.Tensor,  # [post, pre]
        dopamine: float,  # 0.0 to 1.0
        learning_rate: float = 0.01,
    ) -> torch.Tensor:
        """Strengthen tagged synapses proportional to dopamine.

        Biological: Dopamine (and other neuromodulators) converts tags
        to persistent plasticity by enabling protein synthesis.

        High dopamine (reward) → strong consolidation.
        Low dopamine → weak/no consolidation.

        This is the "capture" part of "tagging and capture" - dopamine
        captures the tags and makes them permanent.

        Args:
            weights: Current synaptic weights [post, pre]
            dopamine: Dopamine level (0-1), gates consolidation strength
            learning_rate: Base learning rate

        Returns:
            Updated weights with tagged synapses strengthened
        """
        if dopamine < 0.01:
            return weights  # No consolidation without dopamine

        # Weight change proportional to: tag_strength × dopamine × learning_rate
        # Biological: Dopamine enables protein synthesis at tagged synapses
        consolidation = learning_rate * dopamine * self.tags

        # Update weights (with bounds)
        new_weights = weights + consolidation
        new_weights = torch.clamp(new_weights, min=0.0, max=1.0)

        return new_weights

    def get_replay_probabilities(self) -> torch.Tensor:
        """Compute probability of replaying each pattern.

        Patterns with strong tags are more likely to replay during
        spontaneous reactivation (sharp-wave ripples).

        This replaces explicit Episode.priority sampling with emergent
        priority based on biological synaptic state.

        Returns:
            [n_neurons] probability distribution over patterns to replay
        """
        # Sum tags per postsynaptic neuron (attractor strength)
        # Neurons with many tagged incoming synapses = recently active patterns
        pattern_strength = self.tags.sum(dim=1)

        # Normalize to probability distribution
        if pattern_strength.sum() > 0:
            probs = pattern_strength / pattern_strength.sum()
        else:
            # Uniform if no tags (shouldn't happen normally)
            probs = torch.ones(self.n_neurons, device=self.device) / self.n_neurons

        return probs

    def grow(self, n_new: int) -> None:
        """Grow tag matrix when neurons are added.

        Args:
            n_new: Number of new neurons to add (grows both dimensions)
        """
        old_size = self.n_neurons
        new_size = old_size + n_new

        # Expand tag matrix [n_neurons, n_neurons]
        new_tags = torch.zeros(new_size, new_size, device=self.device)
        # Copy existing tags to top-left corner
        new_tags[:old_size, :old_size] = self.tags
        self.tags = new_tags

        self.n_neurons = new_size
