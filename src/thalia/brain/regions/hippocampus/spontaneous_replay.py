"""Spontaneous Replay Generator for Hippocampus.

Implements biologically-realistic sharp-wave ripple (SWR) generation in CA3.
Ripples occur probabilistically during low acetylcholine (sleep/rest), replaying
recently-tagged experiences for consolidation.
"""

import torch


class SpontaneousReplayGenerator:
    """Generates spontaneous sharp-wave ripples in CA3.

    Biological sharp-wave ripples (SWRs):
    - Occur during low acetylcholine (sleep, quiet rest, immobility)
    - Rate: ~1-3 Hz during sleep, 0 Hz during active encoding
    - Duration: ~50-150ms (we use ~5 timesteps at 1ms resolution)
    - Refractory period: ~200ms minimum gap between ripples
    - Pattern selection: Biased toward recently-active and rewarded patterns

    Implementation:
    - Probabilistic ripple trigger based on ACh level and time since last ripple
    - Pattern selection weighted by synaptic tags (60%), weight strength (30%), noise (10%)
    - No explicit coordination - just CA3 attractor dynamics + biological gating
    """

    def __init__(
        self,
        ripple_rate_hz: float = 2.0,
        ach_threshold: float = 0.3,
        ripple_refractory_ms: float = 200.0,
        tag_weight: float = 0.6,
        strength_weight: float = 0.3,
        noise_weight: float = 0.1,
        device: str = "cpu",
    ):
        """Initialize spontaneous replay generator.

        Args:
            ripple_rate_hz: Target ripple rate during low ACh (default 2.0 Hz)
            ach_threshold: ACh level below which ripples can occur (default 0.3)
            ripple_refractory_ms: Minimum time between ripples (default 200ms)
            tag_weight: Weight for synaptic tag strength in pattern selection (default 0.6)
            strength_weight: Weight for connection strength in pattern selection (default 0.3)
            noise_weight: Weight for random exploration in pattern selection (default 0.1)
            device: Device for computations ("cpu" or "cuda")
        """
        self.ripple_rate_hz = ripple_rate_hz
        self.ach_threshold = ach_threshold
        self.ripple_refractory_ms = ripple_refractory_ms
        self.tag_weight = tag_weight
        self.strength_weight = strength_weight
        self.noise_weight = noise_weight
        self.device = device

        # Time since last ripple (for refractory period)
        # Initialize to refractory period so first ripple can occur immediately
        self.time_since_ripple_ms = ripple_refractory_ms

        # Validate weights sum to 1.0
        total_weight = tag_weight + strength_weight + noise_weight
        assert abs(total_weight - 1.0) < 1e-6, (
            f"Pattern selection weights must sum to 1.0, got {total_weight}"
        )

    def should_trigger_ripple(self, acetylcholine: float, dt_ms: float) -> bool:
        """Determine if a ripple should occur this timestep.

        Ripples are biologically gated by acetylcholine:
        - High ACh (encoding): No ripples, strong feedforward processing
        - Low ACh (consolidation): Frequent ripples, CA3 attractor dynamics dominate

        Ripple generation is probabilistic with refractory period:
        - P(ripple) = rate × dt (e.g., 2 Hz → 0.002 per ms)
        - Minimum 200ms gap between ripples (biological refractory period)

        Args:
            acetylcholine: Current ACh level (0-1, 1=high encoding mode)
            dt_ms: Timestep duration in milliseconds

        Returns:
            True if ripple should occur this timestep
        """
        # Update time since last ripple
        self.time_since_ripple_ms += dt_ms

        # No ripples during encoding (high ACh)
        if acetylcholine > self.ach_threshold:
            return False

        # Refractory period after last ripple
        if self.time_since_ripple_ms < self.ripple_refractory_ms:
            return False

        # Probabilistic ripple generation
        # P(ripple) = rate × dt
        prob = self.ripple_rate_hz * (dt_ms / 1000.0)

        if torch.rand(1).item() < prob:
            self.time_since_ripple_ms = 0.0  # Reset refractory timer
            return True

        return False

    def select_pattern_to_replay(
        self,
        synaptic_tags: torch.Tensor,
        ca3_weights: torch.Tensor,
        seed_fraction: float,
    ) -> torch.Tensor:
        """Select which pattern to spontaneously reactivate.

        Biological pattern selection is influenced by:
        1. **Synaptic tags (60%)**: Recently active and rewarded experiences
           - Tags created by spike coincidence (Hebbian)
           - Strengthened by dopamine (reward-related)
           - Decay over ~20 timesteps (~20ms)

        2. **Weight strength (30%)**: Well-learned attractor states
           - Strong recurrent connections form stable attractors
           - Reflects consolidated long-term memory

        3. **Random noise (10%)**: Exploration of weak patterns
           - Prevents replay from becoming too deterministic
           - Allows consolidation of less-frequent experiences

        Args:
            synaptic_tags: Tag strength matrix [n_neurons, n_neurons]
            ca3_weights: CA3 recurrent weights [n_neurons, n_neurons]
            seed_fraction: Fraction of neurons to use as seed (default 0.15)

        Returns:
            Sparse binary seed pattern [n_neurons]
        """
        n_neurons = ca3_weights.shape[0]

        # Compute attractor strength for each neuron
        # Strong attractors = many strong incoming connections
        weight_strength = ca3_weights.sum(dim=1)  # [n_neurons]
        if weight_strength.sum() > 0:
            weight_strength = weight_strength / weight_strength.sum()
        else:
            weight_strength = torch.ones(n_neurons, device=self.device) / n_neurons

        # Compute tag strength for each neuron
        # Strong tags = recent activity + dopamine gating
        tag_strength = synaptic_tags.sum(dim=1)  # [n_neurons]
        tag_total = tag_strength.sum()

        # Adaptive weighting: when tags are absent, rely more on weights
        # This is biologically correct: no recent activity → use consolidated memory
        if tag_total > 1e-6:
            # Tags present: use configured weights
            tag_strength = tag_strength / tag_total
            effective_tag_weight = self.tag_weight
            effective_strength_weight = self.strength_weight
            effective_noise_weight = self.noise_weight
        else:
            # No tags: add tag weight to strength weight (rely on consolidated memory)
            # This is biologically correct: no recent activity → use well-learned patterns
            # Noise remains constant for exploration (unrelated to memory)
            # e.g., if tag=0.6, strength=0.3, noise=0.1:
            #   new_strength = 0.3 + 0.6 = 0.9 (consolidated memory)
            #   new_noise = 0.1 (constant exploration)
            tag_strength = torch.zeros(n_neurons, device=self.device)
            effective_tag_weight = 0.0
            effective_strength_weight = self.strength_weight + self.tag_weight
            effective_noise_weight = self.noise_weight

        # Random noise (uniform exploration)
        noise = torch.ones(n_neurons, device=self.device) / n_neurons

        # Combined probability: weighted mixture
        probs = (
            effective_tag_weight * tag_strength
            + effective_strength_weight * weight_strength
            + effective_noise_weight * noise
        )

        # Sample seed neurons for pattern initialization
        # Biological ripples start with partial pattern activation (~10-20% of CA3)
        n_seed = max(10, int(n_neurons * seed_fraction))
        seed_indices = torch.multinomial(probs, num_samples=n_seed, replacement=False)

        # Create sparse seed pattern (binary)
        seed_pattern = torch.zeros(n_neurons, device=self.device, dtype=torch.bool)
        seed_pattern[seed_indices] = True

        return seed_pattern
