"""
ThinkingSNN - The core thinking architecture.

This is the main integration module that combines all THALIA components
into a unified "thinking machine":

- Attractor dynamics for concept representation
- Working memory for holding thoughts
- STDP and homeostatic plasticity for learning
- Reward modulation for goal-directed behavior
- Spontaneous thought generation through noise

A thought is represented as a trajectory through attractor space,
with the system naturally transitioning between stable states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Callable

import torch
import torch.nn as nn

from thalia.core.neuron import LIFNeuron, LIFConfig
from thalia.dynamics.attractor import AttractorNetwork, AttractorConfig, ConceptNetwork
from thalia.dynamics.manifold import ActivityTracker, ThoughtTrajectory
from thalia.memory.working_memory import WorkingMemory, WorkingMemoryConfig
from thalia.learning.stdp import STDP
from thalia.learning.homeostatic import IntrinsicPlasticity, SynapticScaling
from thalia.learning.reward import RewardModulatedSTDP


@dataclass
class ThinkingConfig:
    """Configuration for ThinkingSNN.

    Attributes:
        n_concepts: Number of concept neurons in attractor network
        n_wm_slots: Working memory capacity
        wm_slot_size: Neurons per WM slot
        noise_std: Noise for spontaneous thought transitions
        tau_mem: Membrane time constant
        enable_learning: Whether to enable online learning
        enable_homeostasis: Whether to enable homeostatic regulation
        dt: Simulation timestep in ms
    """
    n_concepts: int = 256
    n_wm_slots: int = 7
    wm_slot_size: int = 64
    noise_std: float = 0.05
    tau_mem: float = 20.0
    enable_learning: bool = True
    enable_homeostasis: bool = True
    dt: float = 1.0

    # Learning parameters
    stdp_lr: float = 0.01
    homeostatic_lr: float = 0.001

    # Thinking parameters
    spontaneous_rate: float = 0.01  # Rate of random thought initiations
    attention_strength: float = 0.5  # How strongly attention guides thought


class ThinkingSNN(nn.Module):
    """A spiking neural network that thinks.

    This is the main class that integrates all THALIA components into
    a unified cognitive architecture. It can:

    - Store and recall concepts as attractor states
    - Maintain information in working memory
    - Learn from experience via STDP
    - Regulate activity via homeostatic mechanisms
    - Generate spontaneous thoughts through noise-driven dynamics
    - Follow goals through reward modulation

    Example:
        >>> thinker = ThinkingSNN(ThinkingConfig())
        >>> thinker.reset_state(batch_size=1)
        >>>
        >>> # Store some concepts
        >>> thinker.store_concept(pattern1, "apple")
        >>> thinker.store_concept(pattern2, "banana")
        >>>
        >>> # Think for a while
        >>> for t in range(1000):
        ...     thought = thinker.think()
        ...     if thought.concept_changed:
        ...         print(f"Now thinking about: {thought.current_concept}")
    """

    def __init__(self, config: Optional[ThinkingConfig] = None):
        super().__init__()
        self.config = config or ThinkingConfig()

        # === Core Components ===

        # Concept network (attractor dynamics)
        attractor_config = AttractorConfig(
            n_neurons=self.config.n_concepts,
            tau_mem=self.config.tau_mem,
            noise_std=self.config.noise_std,
            dt=self.config.dt,
        )
        self.concepts = ConceptNetwork(attractor_config)

        # Working memory
        wm_config = WorkingMemoryConfig(
            n_slots=self.config.n_wm_slots,
            slot_size=self.config.wm_slot_size,
            tau_mem=self.config.tau_mem * 1.5,  # Longer for WM
            noise_std=self.config.noise_std * 0.5,  # Less noise
            dt=self.config.dt,
        )
        self.working_memory = WorkingMemory(wm_config)

        # Activity tracking
        self.tracker = ActivityTracker(max_history=10000)
        self.trajectory = ThoughtTrajectory()

        # === Learning Components ===

        if self.config.enable_learning:
            from thalia.learning.stdp import STDPConfig
            from thalia.learning.reward import RSTDPConfig

            stdp_config = STDPConfig(
                a_plus=self.config.stdp_lr,
                a_minus=self.config.stdp_lr,
            )
            self.stdp = STDP(
                n_pre=self.config.n_concepts,
                n_post=self.config.n_concepts,
                config=stdp_config,
            )

            rstdp_config = RSTDPConfig(
                learning_rate=self.config.stdp_lr,
            )
            self.reward_stdp = RewardModulatedSTDP(
                n_pre=self.config.n_concepts,
                n_post=self.config.n_concepts,
                config=rstdp_config,
            )
        else:
            self.stdp = None
            self.reward_stdp = None

        if self.config.enable_homeostasis:
            from thalia.learning.homeostatic import IntrinsicPlasticityConfig, SynapticScalingConfig

            ip_config = IntrinsicPlasticityConfig(
                learning_rate=self.config.homeostatic_lr,
            )
            self.intrinsic_plasticity = IntrinsicPlasticity(
                n_neurons=self.config.n_concepts,
                config=ip_config,
            )

            ss_config = SynapticScalingConfig(
                learning_rate=self.config.homeostatic_lr,
            )
            self.synaptic_scaling = SynapticScaling(
                n_neurons=self.config.n_concepts,
                config=ss_config,
            )
        else:
            self.intrinsic_plasticity = None
            self.synaptic_scaling = None

        # === State ===

        self._timestep: int = 0
        self._current_concept: int = -1
        self._attention_target: Optional[torch.Tensor] = None
        self._goal_signal: float = 0.0
        self._last_spikes: Optional[torch.Tensor] = None

        # Callbacks for external monitoring
        self._on_concept_change: List[Callable[[int, str], None]] = []

    def reset_state(self, batch_size: int = 1) -> None:
        """Reset all internal state."""
        self.concepts.reset_state(batch_size)
        self.working_memory.reset_state(batch_size)
        self.tracker.reset()
        self.trajectory = ThoughtTrajectory()

        self._timestep = 0
        self._current_concept = -1
        self._attention_target = None
        self._last_spikes = None

        if self.intrinsic_plasticity:
            self.intrinsic_plasticity.reset()

    def store_concept(self, pattern: torch.Tensor, name: str) -> int:
        """Store a concept in the concept network.

        Args:
            pattern: Activity pattern representing the concept
            name: Human-readable name

        Returns:
            Index of the stored concept
        """
        return self.concepts.store_concept(pattern, name)

    def associate_concepts(self, concept_a: int, concept_b: int, strength: float = 1.0) -> None:
        """Create association between concepts.

        When one concept is active, it will tend to activate the other.
        """
        self.concepts.associate(concept_a, concept_b, strength)

    def attend_to(self, target: torch.Tensor) -> None:
        """Direct attention to a specific pattern.

        This biases the dynamics toward the target pattern.
        """
        self._attention_target = target

    def set_goal(self, reward: float) -> None:
        """Set goal signal for reward-modulated learning.

        Positive values encourage current patterns, negative discourages.
        """
        self._goal_signal = reward

    def think(self) -> ThoughtState:
        """Execute one timestep of thinking.

        This is the main entry point - call repeatedly to generate
        a stream of thoughts.

        Returns:
            ThoughtState with current thinking status
        """
        self._timestep += 1

        # === Process Working Memory ===
        wm_output = self.working_memory()

        # === Compute Input to Concept Network ===

        # Attention signal
        if self._attention_target is not None:
            attention_input = self._attention_target * self.config.attention_strength
        else:
            attention_input = None

        # Spontaneous activation
        if torch.rand(1).item() < self.config.spontaneous_rate:
            # Random nudge
            batch_size = 1
            if self.concepts.neurons.membrane is not None:
                batch_size = self.concepts.neurons.membrane.shape[0]
            device = self.concepts.weights.device
            spontaneous = torch.randn(batch_size, self.config.n_concepts, device=device) * 0.5
        else:
            spontaneous = None

        # Combine inputs
        external_input = None
        if attention_input is not None:
            external_input = attention_input
        if spontaneous is not None:
            if external_input is not None:
                external_input = external_input + spontaneous
            else:
                external_input = spontaneous

        # === Step Concept Network ===
        spikes, membrane = self.concepts(external_input)

        # === Track Activity ===
        self.tracker.record(spikes)

        # Check for concept change
        new_concept = self.concepts.get_attractor_state()
        concept_changed = (new_concept != self._current_concept and new_concept >= 0)

        if concept_changed:
            old_concept = self._current_concept
            self._current_concept = new_concept
            self.trajectory.add_state(new_concept, self._timestep)

            # Notify callbacks
            name = self.concepts.get_concept_name(new_concept)
            for callback in self._on_concept_change:
                callback(new_concept, name)

        # === Apply Learning ===

        if self.config.enable_learning and self._last_spikes is not None:
            # STDP on concept network weights
            if self.stdp is not None:
                # STDP uses forward() - pre_spikes, post_spikes
                weight_delta = self.stdp(self._last_spikes, spikes)
                # Apply small fraction of update
                with torch.no_grad():
                    self.concepts.weights.data += weight_delta * 0.001

            # Reward modulation
            if self.reward_stdp is not None and self._goal_signal != 0:
                self.reward_stdp.update_traces(self._last_spikes, spikes)
                weight_delta = self.reward_stdp.apply_reward(self._goal_signal)
                with torch.no_grad():
                    self.concepts.weights.data += weight_delta

        if self.config.enable_homeostasis:
            # Intrinsic plasticity
            if self.intrinsic_plasticity is not None:
                _ = self.intrinsic_plasticity(spikes)  # Returns threshold deltas
                # Could apply to neuron thresholds if exposed

            # Synaptic scaling
            if self.synaptic_scaling is not None and self._timestep % 100 == 0:
                scale = self.synaptic_scaling(spikes)
                with torch.no_grad():
                    # Scale incoming weights per neuron: weight[i,j] scaled by scale[j]
                    # scale is (batch, n_neurons), squeeze batch dim for weight multiply
                    self.concepts.weights.data *= scale.squeeze(0).unsqueeze(0)

        # === Store State ===
        self._last_spikes = spikes

        # === Compute Energy ===
        energy = self.concepts.energy(spikes.squeeze())

        # === Build Result ===
        return ThoughtState(
            timestep=self._timestep,
            spikes=spikes,
            membrane=membrane,
            current_concept=self._current_concept,
            concept_name=self.concepts.get_concept_name(self._current_concept),
            concept_changed=concept_changed,
            energy=energy.item(),
            wm_status=self.working_memory.get_status(),
        )

    def load_to_memory(self, slot: int, pattern: torch.Tensor, label: Optional[str] = None) -> None:
        """Load a pattern into working memory.

        Args:
            slot: Which WM slot to use
            pattern: Pattern to load
            label: Optional label
        """
        self.working_memory.load(slot, pattern, label)

    def read_from_memory(self, slot: int) -> Optional[torch.Tensor]:
        """Read pattern from working memory slot."""
        return self.working_memory.read(slot)

    def get_trajectory(self) -> ThoughtTrajectory:
        """Get the thought trajectory so far."""
        return self.trajectory

    def get_activity_history(self) -> torch.Tensor:
        """Get neural activity history."""
        return self.tracker.get_trajectory()

    def project_activity(self, n_components: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project activity history to low dimensions via PCA."""
        return self.tracker.project_pca(n_components)

    def on_concept_change(self, callback: Callable[[int, str], None]) -> None:
        """Register callback for concept changes."""
        self._on_concept_change.append(callback)

    def think_until_stable(self, max_steps: int = 500, stability_window: int = 50) -> int:
        """Think until activity stabilizes.

        Returns number of steps taken.
        """
        stable_count = 0
        last_concept = -1

        for step in range(max_steps):
            state = self.think()

            if state.current_concept == last_concept:
                stable_count += 1
            else:
                stable_count = 0
                last_concept = state.current_concept

            if stable_count >= stability_window:
                return step + 1

        return max_steps

    def generate_thought_chain(self, steps: int, start_concept: Optional[int] = None) -> List[str]:
        """Generate a chain of thoughts.

        Args:
            steps: How many timesteps to run
            start_concept: Optional concept to start from

        Returns:
            List of concept names visited
        """
        if start_concept is not None and len(self.concepts.patterns) > start_concept:
            # Cue the starting concept
            pattern = self.concepts.patterns[start_concept]
            self.attend_to(pattern.unsqueeze(0))

        concepts_visited = []
        last_concept = -1

        for _ in range(steps):
            state = self.think()

            if state.current_concept != last_concept and state.current_concept >= 0:
                concepts_visited.append(state.concept_name)
                last_concept = state.current_concept

        # Clear attention
        self._attention_target = None

        return concepts_visited


@dataclass
class ThoughtState:
    """State returned by think() at each timestep.

    Attributes:
        timestep: Current simulation time
        spikes: Spike output from concept network
        membrane: Membrane potentials
        current_concept: Index of active concept (-1 if none)
        concept_name: Name of active concept
        concept_changed: Whether concept changed this step
        energy: Network energy (lower = more stable)
        wm_status: Working memory status dict
    """
    timestep: int
    spikes: torch.Tensor
    membrane: torch.Tensor
    current_concept: int
    concept_name: str
    concept_changed: bool
    energy: float
    wm_status: Dict[str, Any]
