"""
Daydream Mode - Spontaneous cognition without external input.

This module enables "thinking without sensory input" - the network
generates thoughts by randomly walking through its attractor landscape.
This is crucial for:

- Creativity: Novel combinations of concepts emerge
- Consolidation: Replaying and strengthening associations
- Problem-solving: Incubation effects during rest
- Imagination: Simulating scenarios not currently perceived

The daydream process works by:
1. Amplifying internal noise to drive transitions
2. Letting attractor dynamics guide the trajectory
3. Using recency and association strength to bias transitions
4. Optionally applying soft constraints (themes, goals)

Key features:
- Free association: Pure random walk through concepts
- Themed daydreaming: Biased toward certain concept clusters
- Goal-directed reverie: Searching for solutions
- Dream-like mode: Higher noise, stranger associations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Tuple, Any
from enum import Enum, auto

import torch
import torch.nn as nn

from thalia.core.neuron import LIFNeuron, LIFConfig
from thalia.dynamics.attractor import AttractorNetwork, AttractorConfig, ConceptNetwork
from thalia.dynamics.manifold import ThoughtTrajectory


class DaydreamMode(Enum):
    """Different modes of spontaneous cognition."""
    FREE = auto()          # Pure random walk
    THEMED = auto()        # Biased toward theme concepts
    GOAL_DIRECTED = auto() # Searching for target
    DREAM = auto()         # High noise, loose associations


@dataclass
class DaydreamConfig:
    """Configuration for daydream dynamics.

    Attributes:
        n_neurons: Number of neurons in daydream network
        base_noise: Baseline noise level for transitions
        noise_amplification: How much to amplify noise in dream mode
        transition_threshold: How much activity needed to trigger transition
        association_strength: Weight of learned associations
        recency_decay: How fast recency bias decays
        dwell_time_mean: Average timesteps spent at each attractor
        dwell_time_std: Variation in dwell time
        tau_mem: Membrane time constant
        dt: Simulation timestep
    """
    n_neurons: int = 128
    base_noise: float = 0.1
    noise_amplification: float = 3.0  # For dream mode
    transition_threshold: float = 0.3
    association_strength: float = 0.5
    recency_decay: float = 0.95
    dwell_time_mean: float = 50.0  # timesteps
    dwell_time_std: float = 20.0
    tau_mem: float = 30.0  # Slower dynamics for daydreaming
    dt: float = 1.0

    # Goal-directed parameters
    goal_attraction: float = 0.2  # How strongly goals bias transitions
    exploration_rate: float = 0.3  # Probability of random jump even with goal


@dataclass
class DaydreamState:
    """State at each daydream timestep.

    Attributes:
        timestep: Current time
        current_concept: Active concept index
        concept_name: Name of active concept
        spikes: Neural spikes
        energy: Network energy
        mode: Current daydream mode
        concepts_visited: List of concepts visited so far
        transition_occurred: Whether a transition happened this step
        novelty: How novel/unexpected this transition was
    """
    timestep: int
    current_concept: int
    concept_name: str
    spikes: torch.Tensor
    energy: float
    mode: DaydreamMode
    concepts_visited: List[str]
    transition_occurred: bool
    novelty: float


class DaydreamNetwork(nn.Module):
    """Network for spontaneous thought generation.

    This network "daydreams" - generating sequences of thoughts
    without external input. It uses an attractor network as its
    substrate but with modified dynamics:

    - Higher noise drives spontaneous transitions
    - Association weights guide likely transitions
    - Recency effects prevent repetition
    - Optional goals/themes bias the trajectory

    Example:
        >>> daydreamer = DaydreamNetwork(DaydreamConfig())
        >>>
        >>> # Store some concepts
        >>> daydreamer.store_concept(pattern_apple, "apple")
        >>> daydreamer.store_concept(pattern_tree, "tree")
        >>> daydreamer.store_concept(pattern_fruit, "fruit")
        >>>
        >>> # Create associations
        >>> daydreamer.associate("apple", "fruit", 0.8)
        >>> daydreamer.associate("apple", "tree", 0.6)
        >>>
        >>> # Let the network daydream
        >>> daydreamer.start_daydream(mode=DaydreamMode.FREE)
        >>> for _ in range(1000):
        ...     state = daydreamer.step()
        ...     if state.transition_occurred:
        ...         print(f"Now thinking about: {state.concept_name}")
    """

    def __init__(self, config: Optional[DaydreamConfig] = None):
        super().__init__()
        self.config = config or DaydreamConfig()
        n = self.config.n_neurons

        # Create attractor network with elevated noise
        attractor_config = AttractorConfig(
            n_neurons=n,
            tau_mem=self.config.tau_mem,
            noise_std=self.config.base_noise,
            dt=self.config.dt,
        )
        self.attractors = ConceptNetwork(attractor_config)

        # Transition probability matrix (learned associations)
        # Higher values = more likely to transition from i to j
        self.register_buffer(
            "associations",
            torch.eye(n) * 0.1  # Start with weak self-association
        )

        # Recency buffer - recently visited concepts less likely
        self.register_buffer(
            "recency",
            torch.zeros(n)
        )

        # Theme/goal biases
        self.register_buffer(
            "theme_bias",
            torch.zeros(n)
        )
        self.register_buffer(
            "goal_pattern",
            torch.zeros(n)
        )

        # State
        self._mode = DaydreamMode.FREE
        self._timestep = 0
        self._current_concept = -1
        self._dwell_counter = 0
        self._target_dwell_time = 0
        self._concepts_visited: List[str] = []
        self._trajectory = ThoughtTrajectory()

        # Concept name lookup (index -> name)
        self._concept_names: Dict[int, str] = {}
        self._name_to_index: Dict[str, int] = {}

        # Callbacks
        self._on_transition: List[Callable[[int, int, str], None]] = []

    def reset_state(self, batch_size: int = 1) -> None:
        """Reset daydream state."""
        self.attractors.reset_state(batch_size)
        self.recency.zero_()
        self._timestep = 0
        self._current_concept = -1
        self._dwell_counter = 0
        self._target_dwell_time = self._sample_dwell_time()
        self._concepts_visited = []
        self._trajectory = ThoughtTrajectory()

    def _sample_dwell_time(self) -> int:
        """Sample how long to stay at current attractor."""
        mean = self.config.dwell_time_mean
        std = self.config.dwell_time_std
        return max(1, int(torch.normal(torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)).item()))

    def store_concept(self, pattern: torch.Tensor, name: str) -> int:
        """Store a concept for daydreaming.

        Args:
            pattern: Activity pattern for concept
            name: Human-readable name

        Returns:
            Concept index
        """
        idx = self.attractors.store_concept(pattern, name)
        self._concept_names[idx] = name
        self._name_to_index[name] = idx
        return idx

    def associate(
        self,
        concept_a: str | int,
        concept_b: str | int,
        strength: float = 0.5,
        bidirectional: bool = True
    ) -> None:
        """Create association between concepts.

        When daydreaming at concept_a, there's higher probability
        of transitioning to concept_b.

        Args:
            concept_a: Source concept (name or index)
            concept_b: Target concept (name or index)
            strength: Association strength (0-1)
            bidirectional: Whether to create reverse association too
        """
        # Convert names to indices
        if isinstance(concept_a, str):
            concept_a = self._name_to_index.get(concept_a, -1)
        if isinstance(concept_b, str):
            concept_b = self._name_to_index.get(concept_b, -1)

        if concept_a < 0 or concept_b < 0:
            return

        # Update association matrix
        n_concepts = len(self._concept_names)
        if concept_a < n_concepts and concept_b < n_concepts:
            self.associations[concept_a, concept_b] = strength
            if bidirectional:
                self.associations[concept_b, concept_a] = strength

        # Also update attractor associations
        self.attractors.associate(concept_a, concept_b, strength)

    def set_theme(self, theme_concepts: List[str | int], strength: float = 0.3) -> None:
        """Set a theme that biases daydreaming toward certain concepts.

        Args:
            theme_concepts: List of concept names/indices to favor
            strength: How strongly to bias toward theme
        """
        self.theme_bias.zero_()
        for concept in theme_concepts:
            if isinstance(concept, str):
                concept = self._name_to_index.get(concept, -1)
            if 0 <= concept < len(self._concept_names):
                self.theme_bias[concept] = strength

    def clear_theme(self) -> None:
        """Remove theme bias."""
        self.theme_bias.zero_()

    def set_goal(self, goal_pattern: torch.Tensor) -> None:
        """Set a goal pattern to search for during daydreaming.

        The network will be biased toward concepts similar to this pattern.
        """
        self.goal_pattern = goal_pattern.to(self.associations.device)

    def clear_goal(self) -> None:
        """Remove goal bias."""
        self.goal_pattern.zero_()

    def start_daydream(
        self,
        mode: DaydreamMode = DaydreamMode.FREE,
        start_concept: Optional[str | int] = None
    ) -> None:
        """Begin a daydream session.

        Args:
            mode: Type of daydreaming
            start_concept: Optional concept to start from
        """
        self._mode = mode
        self.reset_state(batch_size=1)

        # Initialize at starting concept if provided
        if start_concept is not None:
            if isinstance(start_concept, str):
                start_concept = self._name_to_index.get(start_concept, 0)
            if 0 <= start_concept < len(self.attractors.patterns):
                # Inject starting pattern
                pattern = self.attractors.patterns[start_concept]
                self._cue_concept(pattern, steps=20)
                self._current_concept = start_concept
                name = self._concept_names.get(start_concept, f"concept_{start_concept}")
                self._concepts_visited.append(name)

    def _cue_concept(self, pattern: torch.Tensor, steps: int = 20) -> None:
        """Initialize network at a concept."""
        for _ in range(steps):
            self.attractors(pattern.unsqueeze(0))

    def _get_noise_level(self) -> float:
        """Get current noise level based on mode."""
        base = self.config.base_noise
        if self._mode == DaydreamMode.DREAM:
            return base * self.config.noise_amplification
        elif self._mode == DaydreamMode.GOAL_DIRECTED:
            return base * 0.7  # Less noise when focused
        else:
            return base

    def _compute_transition_bias(self) -> torch.Tensor:
        """Compute bias for next concept based on mode and state."""
        n_concepts = len(self._concept_names)
        if n_concepts == 0:
            return torch.zeros(self.config.n_neurons, device=self.associations.device)

        bias = torch.zeros(n_concepts, device=self.associations.device)

        # Association-based bias (from current concept)
        if self._current_concept >= 0:
            assoc_bias = self.associations[self._current_concept, :n_concepts]
            bias += assoc_bias * self.config.association_strength

        # Recency penalty (avoid repetition)
        bias -= self.recency[:n_concepts] * 0.5

        # Theme bias
        if self._mode == DaydreamMode.THEMED:
            bias += self.theme_bias[:n_concepts]

        # Goal attraction
        if self._mode == DaydreamMode.GOAL_DIRECTED and self.goal_pattern.sum() > 0:
            # Bias toward concepts similar to goal
            for i in range(n_concepts):
                if i < len(self.attractors.patterns):
                    similarity = torch.cosine_similarity(
                        self.attractors.patterns[i].unsqueeze(0),
                        self.goal_pattern.unsqueeze(0)
                    )
                    bias[i] += similarity.item() * self.config.goal_attraction

        return bias

    def _should_transition(self) -> bool:
        """Determine if it's time to transition to new concept."""
        self._dwell_counter += 1

        # Dwell time elapsed?
        if self._dwell_counter >= self._target_dwell_time:
            return True

        # Random jump in dream mode?
        if self._mode == DaydreamMode.DREAM:
            if torch.rand(1).item() < 0.02:  # 2% chance per step
                return True

        return False

    def _select_next_concept(self) -> int:
        """Select next concept based on biases and randomness."""
        n_concepts = len(self._concept_names)
        if n_concepts == 0:
            return -1

        # Compute transition probabilities
        bias = self._compute_transition_bias()

        # Add randomness
        noise = torch.randn_like(bias) * self._get_noise_level()

        # In goal-directed mode, occasionally explore randomly
        if self._mode == DaydreamMode.GOAL_DIRECTED:
            if torch.rand(1).item() < self.config.exploration_rate:
                # Pure random choice
                return torch.randint(0, n_concepts, (1,)).item()

        # Softmax to get probabilities
        logits = bias + noise
        probs = torch.softmax(logits, dim=0)

        # Sample
        next_concept = torch.multinomial(probs, 1).item()

        return next_concept

    def _compute_novelty(self, from_concept: int, to_concept: int) -> float:
        """Compute how novel/surprising a transition is."""
        if from_concept < 0 or to_concept < 0:
            return 0.5  # Neutral novelty

        n_concepts = len(self._concept_names)
        if from_concept >= n_concepts or to_concept >= n_concepts:
            return 0.5

        # Low association = high novelty
        assoc = self.associations[from_concept, to_concept].item()

        # Recency also affects novelty (visiting old memory is less novel)
        recency = self.recency[to_concept].item() if to_concept < self.recency.shape[0] else 0

        novelty = 1.0 - assoc * 0.5 - recency * 0.3
        return max(0, min(1, novelty))

    def step(self) -> DaydreamState:
        """Execute one timestep of daydreaming.

        Returns:
            Current daydream state
        """
        self._timestep += 1
        transition_occurred = False
        novelty = 0.0
        old_concept = self._current_concept

        # Check for transition
        if self._should_transition():
            new_concept = self._select_next_concept()

            if new_concept != self._current_concept and new_concept >= 0:
                transition_occurred = True
                novelty = self._compute_novelty(self._current_concept, new_concept)

                # Update state
                self._current_concept = new_concept
                self._dwell_counter = 0
                self._target_dwell_time = self._sample_dwell_time()

                # Update recency
                self.recency *= self.config.recency_decay
                if new_concept < self.recency.shape[0]:
                    self.recency[new_concept] = 1.0

                # Track trajectory
                name = self._concept_names.get(new_concept, f"concept_{new_concept}")
                self._concepts_visited.append(name)
                self._trajectory.add_state(new_concept, self._timestep)

                # Cue the new concept
                if new_concept < len(self.attractors.patterns):
                    pattern = self.attractors.patterns[new_concept]
                    self.attractors(pattern.unsqueeze(0) * 2.0)  # Strong cue

                # Callbacks
                for callback in self._on_transition:
                    callback(old_concept, new_concept, name)

        # Step the attractor network
        noise = torch.randn(1, self.config.n_neurons, device=self.associations.device)
        noise *= self._get_noise_level()
        spikes, membrane = self.attractors(noise)

        # Compute energy
        energy = self.attractors.energy(spikes.squeeze()).item()

        # Build state
        name = self._concept_names.get(self._current_concept, "")

        return DaydreamState(
            timestep=self._timestep,
            current_concept=self._current_concept,
            concept_name=name,
            spikes=spikes,
            energy=energy,
            mode=self._mode,
            concepts_visited=list(self._concepts_visited),
            transition_occurred=transition_occurred,
            novelty=novelty,
        )

    def daydream(
        self,
        steps: int,
        mode: DaydreamMode = DaydreamMode.FREE,
        start_concept: Optional[str | int] = None,
    ) -> List[DaydreamState]:
        """Run a complete daydream session.

        Convenience method that runs multiple steps and returns
        the full trajectory.

        Args:
            steps: Number of timesteps
            mode: Daydream mode
            start_concept: Optional starting concept

        Returns:
            List of states at each timestep
        """
        self.start_daydream(mode=mode, start_concept=start_concept)

        states = []
        for _ in range(steps):
            state = self.step()
            states.append(state)

        return states

    def get_trajectory(self) -> ThoughtTrajectory:
        """Get the thought trajectory from current session."""
        return self._trajectory

    def get_concepts_visited(self) -> List[str]:
        """Get list of concepts visited in current session."""
        return list(self._concepts_visited)

    def get_transition_matrix(self) -> torch.Tensor:
        """Get the learned transition/association matrix."""
        return self.associations.clone()

    def on_transition(self, callback: Callable[[int, int, str], None]) -> None:
        """Register callback for concept transitions.

        Callback receives (old_concept, new_concept, new_name).
        """
        self._on_transition.append(callback)


class DaydreamIntegration:
    """Integration helper to add daydream capability to ThinkingSNN.

    This class wraps a ThinkingSNN and adds daydream functionality
    without modifying the original class.

    Example:
        >>> from thalia.cognition import ThinkingSNN
        >>> thinker = ThinkingSNN(ThinkingConfig())
        >>> daydreamer = DaydreamIntegration(thinker)
        >>>
        >>> # Normal thinking
        >>> thinker.think()
        >>>
        >>> # Daydream mode
        >>> thoughts = daydreamer.daydream(steps=500)
    """

    def __init__(self, thinker: Any):  # Any to avoid circular import
        """Wrap a ThinkingSNN with daydream capabilities.

        Args:
            thinker: ThinkingSNN instance to wrap
        """
        self.thinker = thinker
        self._daydream_active = False
        self._original_noise = None
        self._daydream_noise_multiplier = 3.0

    def enter_daydream(self, noise_multiplier: float = 3.0) -> None:
        """Enter daydream mode.

        Increases noise to drive spontaneous transitions.
        """
        if not self._daydream_active:
            self._original_noise = self.thinker.concepts.neurons.config.noise_std
            self.thinker.concepts.neurons.config.noise_std *= noise_multiplier
            self._daydream_active = True
            self._daydream_noise_multiplier = noise_multiplier

    def exit_daydream(self) -> None:
        """Exit daydream mode, restore normal noise levels."""
        if self._daydream_active and self._original_noise is not None:
            self.thinker.concepts.neurons.config.noise_std = self._original_noise
            self._daydream_active = False

    def daydream(
        self,
        steps: int,
        return_on_loop: bool = False,
        max_loop_length: int = 3,
    ) -> List[str]:
        """Run daydream session and return concepts visited.

        Args:
            steps: Number of timesteps
            return_on_loop: Whether to stop if caught in a loop
            max_loop_length: Max concepts to check for loop detection

        Returns:
            List of concept names visited
        """
        self.enter_daydream()

        try:
            concepts_visited = []
            recent_concepts = []

            for _ in range(steps):
                state = self.thinker.think()

                if state.concept_changed:
                    concepts_visited.append(state.concept_name)
                    recent_concepts.append(state.current_concept)

                    # Keep only recent for loop detection
                    if len(recent_concepts) > max_loop_length * 2:
                        recent_concepts = recent_concepts[-max_loop_length * 2:]

                    # Check for loops
                    if return_on_loop and len(recent_concepts) >= max_loop_length * 2:
                        # Check if last N concepts repeat
                        half = len(recent_concepts) // 2
                        if recent_concepts[:half] == recent_concepts[half:]:
                            break

            return concepts_visited

        finally:
            self.exit_daydream()

    def free_associate(
        self,
        start_concept: str,
        max_hops: int = 10,
        min_transitions: int = 3,
    ) -> List[str]:
        """Free associate starting from a concept.

        Args:
            start_concept: Concept to start from
            max_hops: Maximum number of concept transitions
            min_transitions: Minimum transitions before stopping

        Returns:
            Chain of associated concepts
        """
        # Find and cue starting concept
        start_idx = -1
        for i, name in enumerate(self.thinker.concepts.concept_names):
            if name == start_concept:
                start_idx = i
                break

        if start_idx >= 0 and start_idx < len(self.thinker.concepts.patterns):
            pattern = self.thinker.concepts.patterns[start_idx]
            self.thinker.attend_to(pattern.unsqueeze(0))

        # Daydream until we have enough transitions
        self.enter_daydream()

        try:
            chain = [start_concept]
            steps = 0
            max_steps = max_hops * 100  # Give time for transitions

            while len(chain) <= max_hops and steps < max_steps:
                state = self.thinker.think()
                steps += 1

                if state.concept_changed and state.concept_name:
                    if state.concept_name != chain[-1]:  # Avoid duplicates
                        chain.append(state.concept_name)

                    # Check if we have minimum transitions
                    if len(chain) >= min_transitions + 1:
                        # Random chance to stop (diminishing)
                        stop_prob = (len(chain) - min_transitions) / (max_hops - min_transitions + 1)
                        if torch.rand(1).item() < stop_prob * 0.3:
                            break

            # Clear attention
            self.thinker._attention_target = None
            return chain

        finally:
            self.exit_daydream()
