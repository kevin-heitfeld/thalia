"""
Hippocampus Memory Component

Manages episodic memory buffer for experience storage, retrieval, and prioritized sampling.
Standardized component following the region_components pattern.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any, TYPE_CHECKING

import torch

from thalia.core.region_components import MemoryComponent
from thalia.core.base_manager import ManagerContext
from thalia.core.utils import cosine_similarity_safe

if TYPE_CHECKING:
    from thalia.regions.hippocampus.config import Episode, HippocampusConfig


class HippocampusMemoryComponent(MemoryComponent):
    """Manages episodic memory buffer for hippocampus.

    Handles:
    1. Episode storage with priority
    2. Prioritized sampling for replay
    3. Pattern completion (retrieve similar experiences)
    4. Buffer management (capacity limits)
    """

    def __init__(self, config: HippocampusConfig, context: ManagerContext):
        """Initialize hippocampus memory component.

        Args:
            config: Hippocampus configuration
            context: Manager context (device, dimensions, etc.)
        """
        super().__init__(config, context)
        self.episode_buffer: List[Episode] = []

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
        **kwargs
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
            **kwargs: Additional parameters
        """
        from thalia.regions.hippocampus.config import Episode

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
        )

        # Buffer management
        max_episodes = getattr(cfg, 'max_episodes', 100)
        if len(self.episode_buffer) >= max_episodes:
            min_idx = min(range(len(self.episode_buffer)),
                         key=lambda i: self.episode_buffer[i].priority)
            self.episode_buffer.pop(min_idx)

        self.episode_buffer.append(episode)

    def retrieve_memories(
        self,
        query_state: torch.Tensor,
        k: int = 5,
        similarity_threshold: float = 0.0,
        query_action: Optional[int] = None,
        **kwargs
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
                similarities.append({
                    'state': episode.state,
                    'action': episode.action,
                    'reward': episode.reward,
                    'correct': episode.correct,
                    'similarity': sim,
                    'context': episode.context,
                    'metadata': episode.metadata,
                    'sequence': episode.sequence,
                })

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:k]

    def sample_prioritized(self, n: int) -> List[Episode]:
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
        diag.update({
            "buffer_size": len(self.episode_buffer),
            "max_episodes": getattr(self.config, 'max_episodes', 100),
            "total_priority": sum(ep.priority for ep in self.episode_buffer),
        })
        return diag


# Backwards compatibility alias
EpisodeManager = HippocampusMemoryComponent
