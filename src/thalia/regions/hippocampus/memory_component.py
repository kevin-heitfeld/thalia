"""
Hippocampus Memory Component - DEPRECATED (Phase 4: Emergent RL Migration)

**THIS MODULE IS DEPRECATED AND WILL BE REMOVED**

As of Phase 4 of the emergent RL migration (January 19, 2026), explicit episode
buffers are being removed in favor of biologically-accurate memory storage:

**OLD (Explicit Episode Buffer)**:
- Episodes stored in Python list
- Explicit Episode dataclass with metadata
- Similarity-based retrieval

**NEW (Emergent Memory)**:
- Memory IS the synaptic weights (CA3 recurrent connections)
- Pattern storage via Hebbian learning during forward()
- Pattern retrieval via CA3 attractor dynamics
- Priority via synaptic tagging (Phase 1)

**Migration Path**:
If your code uses:
- `Episode` dataclass → Remove (no longer needed)
- `hippocampus.memory.episode_buffer` → Use CA3 weights directly
- `store_episode()` → Just run `forward()`, Hebbian learning stores patterns
- `retrieve_memories()` → Use `forward()` with partial cue, CA3 completes pattern

See: `temp/emergent_rl_migration.md` Phase 4 for details.

Standardized component following the region_components pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import warnings

import torch

from thalia.config.region_configs import HippocampusConfig
from thalia.core.region_components import MemoryComponent
from thalia.managers.base_manager import ManagerContext
from thalia.utils.core_utils import cosine_similarity_safe

# Emit deprecation warning when module is imported
warnings.warn(
    "HippocampusMemoryComponent is deprecated as of Phase 4 (Emergent RL Migration). "
    "Memory storage now uses CA3 synaptic weights (Hebbian learning), not explicit episodes. "
    "See temp/emergent_rl_migration.md for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)


@dataclass
class Episode:
    """An episode stored in episodic memory for replay.

    **DEPRECATED**: This dataclass is deprecated as of Phase 4 (Emergent RL Migration).
    Memory storage now uses CA3 synaptic weights, not explicit Episode objects.

    See temp/emergent_rl_migration.md Phase 4 for migration details.

    Episodes are stored with priority for experience replay,
    where more important episodes (high reward, correct trials)
    are replayed more frequently.

    Episodes can store either:
    - A single state (traditional): Just the activity pattern at decision time
    - A sequence (extended): List of states from each gamma slot during encoding

    During sleep replay, sequences are replayed time-compressed using
    the gamma oscillator to drive slot-by-slot reactivation.

    **Phase 1 Enhancement (Biologically-Accurate Consolidation):**
    Sharp-wave ripples originate in CA3, not CA1. We now explicitly store
    the CA3 attractor pattern so consolidation replay can accurately
    reproduce the CA3→CA1 pathway dynamics during sleep.
    """

    state: torch.Tensor  # Activity pattern at decision time (or final state)
    action: int  # Selected action
    reward: float  # Received reward
    correct: bool  # Whether the action was correct
    context: Optional[torch.Tensor] = None  # Context/cue pattern
    metadata: Optional[Dict[str, Any]] = None  # Additional info
    priority: float = 1.0  # Replay priority
    timestamp: int = 0  # When this episode occurred
    sequence: Optional[List[torch.Tensor]] = None  # Sequence of states for gamma-driven replay
    ca3_pattern: Optional[torch.Tensor] = None  # CA3 attractor pattern (for consolidation replay)


class HippocampusMemoryComponent(MemoryComponent):
    """Manages episodic memory buffer for hippocampus.

    **DEPRECATED**: This class is deprecated as of Phase 4 (Emergent RL Migration).

    Memory storage now uses CA3 synaptic weights (Hebbian learning during forward()),
    not explicit episode buffers. Pattern retrieval uses CA3 attractor dynamics,
    not similarity search.

    **Migration Guide**:
    - Pattern storage: Just run `hippocampus.forward()` - Hebbian learning stores patterns
    - Pattern retrieval: Run `forward()` with partial cue - CA3 attractor completes it
    - Priority: Use synaptic tagging (Phase 1) - tags mark recently-active synapses

    See `temp/emergent_rl_migration.md` Phase 4 for detailed migration steps.

    This component implements the episodic memory buffer that stores experiences
    for offline replay, pattern completion, and consolidation to cortex.

    Responsibilities:
    =================
    1. **Episode Storage**: Store experiences with metadata and priority
    2. **Prioritized Sampling**: Sample important/surprising episodes for replay
    3. **Pattern Completion**: Retrieve similar experiences given partial cue
    4. **Buffer Management**: Enforce capacity limits, replace old episodes

    Key Features:
    =============
    - **Priority-Based Storage**: High-priority episodes (surprising, rewarding)
      are less likely to be overwritten

    - **Similarity-Based Retrieval**: Pattern completion via cosine similarity
      between query and stored patterns (CA3 autoassociation model)

    - **Sequence Support**: Store temporal sequences, not just single states
      (models sequential replay during sharp-wave ripples)

    - **Metadata Tracking**: Store task context, timestamps, confidence

    Biological Motivation:
    =====================
    The hippocampus rapidly stores episodic memories:
    - **CA3**: Autoassociative network for pattern completion
    - **CA1**: Comparator for novelty detection
    - **Replay**: Offline reactivation during sleep/rest (SWRs)
    - **Consolidation**: Gradual transfer to cortex over days/weeks

    This component models the episode buffer and retrieval mechanisms.

    Usage:
    ======
        memory = HippocampusMemoryComponent(config, context)

        # Store episode after trial
        memory.store_memory(
            state=ca3_pattern,
            action=action_taken,
            reward=reward_received,
            correct=was_correct,
            priority_boost=surprise_value,
        )

        # Retrieve similar episodes (pattern completion)
        similar_episodes = memory.retrieve_memories(
            query_state=partial_cue,       # Partial cue pattern for retrieval
            query_action=optional_action,  # if filtering by action
            k=5,                           # Top 5 matches
            similarity_threshold=0.2,      # Minimum similarity
        )

        # Sample for replay
        replay_batch = memory.sample_batch(batch_size=32, prioritized=True)

    See Also:
    =========
    - `thalia.regions.hippocampus.replay_engine` for replay mechanisms
    - `thalia.regions.hippocampus.hindsight_relabeling` for goal relabeling
    - `docs/design/memory_consolidation.md` (if exists)
    """

    def __init__(self, config: HippocampusConfig, context: ManagerContext):
        """Initialize hippocampus memory component.

        **DEPRECATED**: Memory component is deprecated (Phase 4).
        Use CA3 synaptic weights for storage instead.

        Args:
            config: Hippocampus configuration
            context: Manager context (device, dimensions, etc.)
        """
        super().__init__(config, context)
        self.episode_buffer: List[Episode] = []

        # Emit runtime warning
        warnings.warn(
            "HippocampusMemoryComponent.__init__() called - this component is deprecated. "
            "Memory storage now uses CA3 synaptic weights (Hebbian learning).",
            DeprecationWarning,
            stacklevel=2,
        )

    def store_memory(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        correct: bool,
        context: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority_boost: float = 0.0,
        sequence: Optional[List[torch.Tensor]] = None,
        ca3_pattern: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Store an episode in memory buffer.

        Args:
            state: Final activity pattern
            action: Selected action
            reward: Received reward
            correct: Whether action was correct
            context: Optional context/cue pattern
            metadata: Optional additional info
            priority_boost: Extra priority
            sequence: Optional CA3 pattern sequence
            ca3_pattern: CA3 attractor pattern (for consolidation replay)
            **kwargs: Additional parameters
        """
        cfg = self.config

        # Compute priority
        base_priority = 1.0 + abs(reward)
        if correct:
            base_priority += 0.5
        base_priority += priority_boost

        # Clone sequence tensors if provided
        sequence_cloned = None
        if sequence is not None:
            sequence_cloned = [s.clone().detach() for s in sequence]

        episode = Episode(
            state=state.clone().detach(),
            context=context.clone().detach() if context is not None else None,
            action=action,
            reward=reward,
            correct=correct,
            metadata=metadata,
            priority=base_priority,
            timestamp=len(self.episode_buffer),
            sequence=sequence_cloned,
            ca3_pattern=ca3_pattern.clone().detach() if ca3_pattern is not None else None,
        )

        # Buffer management
        max_episodes = getattr(cfg, "max_episodes", 100)
        if len(self.episode_buffer) >= max_episodes:
            min_idx = min(
                range(len(self.episode_buffer)), key=lambda i: self.episode_buffer[i].priority
            )
            self.episode_buffer.pop(min_idx)

        self.episode_buffer.append(episode)

    def retrieve_memories(
        self,
        query_state: torch.Tensor,
        query_action: Optional[int] = None,
        k: int = 5,
        similarity_threshold: float = 0.0,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Retrieve K most similar episodes from memory.

        Args:
            query_state: State to find similar experiences for
            k: Number of similar episodes to retrieve
            similarity_threshold: Minimum similarity to return
            query_action: Optional action to filter by
            **kwargs: Additional parameters

        Returns:
            List of similar episodes with similarity scores
        """
        if not self.episode_buffer:
            return []

        similarities = []
        for episode in self.episode_buffer:
            # Compute cosine similarity
            sim = cosine_similarity_safe(query_state, episode.state)

            # Boost similarity if actions match
            if query_action is not None and episode.action == query_action:
                sim = min(1.0, sim * 1.2)

            if sim >= similarity_threshold:
                similarities.append(
                    {
                        "state": episode.state,
                        "action": episode.action,
                        "reward": episode.reward,
                        "correct": episode.correct,
                        "similarity": sim,
                        "context": episode.context,
                        "metadata": episode.metadata,
                        "sequence": episode.sequence,
                    }
                )

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:k]

    def sample_episodes_prioritized(self, n: int) -> List[Episode]:
        """Sample episodes with probability proportional to priority.

        Args:
            n: Number of episodes to sample

        Returns:
            Sampled episodes
        """
        if not self.episode_buffer:
            return []

        n = min(n, len(self.episode_buffer))
        priorities = torch.tensor([ep.priority for ep in self.episode_buffer])
        probs = priorities / priorities.sum()

        indices = torch.multinomial(probs, n, replacement=False)
        return [self.episode_buffer[i] for i in indices]

    def get_memory_diagnostics(self) -> Dict[str, Any]:
        """Get memory-specific diagnostics."""
        diag = super().get_memory_diagnostics()
        diag.update(
            {
                "buffer_size": len(self.episode_buffer),
                "max_episodes": getattr(self.config, "max_episodes", 100),
                "total_priority": sum(ep.priority for ep in self.episode_buffer),
            }
        )
        return diag
