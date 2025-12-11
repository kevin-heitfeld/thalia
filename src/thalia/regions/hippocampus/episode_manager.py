"""
Episode Manager for Trisynaptic Hippocampus.

Handles episodic memory storage, retrieval, and prioritized sampling.
"""

from typing import List, Optional, Dict, Any

import torch

from thalia.core.base_manager import BaseManager, ManagerContext
from thalia.core.utils import cosine_similarity_safe
from thalia.regions.hippocampus.config import Episode, TrisynapticConfig


class EpisodeManager(BaseManager[TrisynapticConfig]):
    """Manages episodic memory buffer for experience replay.
    
    Responsibilities:
    - Store episodes with priority
    - Sample prioritized episodes
    - Retrieve similar experiences (pattern completion)
    - Buffer management (capacity limits)
    """
    
    def __init__(self, config: TrisynapticConfig, context: ManagerContext):
        """Initialize episode manager.
        
        Args:
            config: Hippocampus configuration
            context: Manager context (device, dimensions, etc.)
        """
        super().__init__(config, context)
        self.episode_buffer: List[Episode] = []
        
    def store_episode(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        correct: bool,
        context: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority_boost: float = 0.0,
        sequence: Optional[List[torch.Tensor]] = None,
    ) -> None:
        """Store an episode in episodic memory for later replay.

        Priority is computed based on reward magnitude and correctness.

        Args:
            state: Final activity pattern at decision time
            action: Selected action
            reward: Received reward
            correct: Whether the action was correct
            context: Optional context/cue pattern
            metadata: Optional additional info
            priority_boost: Extra priority for this episode
            sequence: Optional list of CA3 patterns from each gamma slot
                      during encoding. Enables gamma-driven replay.
        """
        cfg = self.config

        # Compute priority based on reward and correctness
        base_priority = 1.0 + abs(reward)
        if correct:
            base_priority += 0.5  # Boost for correct trials
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

        # Buffer management: keep limited episodes
        max_episodes = getattr(cfg, 'max_episodes', 100)
        if len(self.episode_buffer) >= max_episodes:
            # Remove lowest priority
            min_idx = min(range(len(self.episode_buffer)),
                         key=lambda i: self.episode_buffer[i].priority)
            self.episode_buffer.pop(min_idx)

        self.episode_buffer.append(episode)
        
    def sample_episodes_prioritized(self, n: int) -> List[Episode]:
        """Sample episodes with probability proportional to priority.
        
        Args:
            n: Number of episodes to sample
            
        Returns:
            Sampled episodes (up to n, or fewer if buffer is small)
        """
        if not self.episode_buffer:
            return []

        n = min(n, len(self.episode_buffer))
        priorities = torch.tensor([ep.priority for ep in self.episode_buffer])
        probs = priorities / priorities.sum()

        indices = torch.multinomial(probs, n, replacement=False)
        return [self.episode_buffer[i] for i in indices]
        
    def retrieve_similar(
        self,
        query_state: torch.Tensor,
        query_action: Optional[int] = None,
        k: int = 5,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Retrieve K most similar past experiences from episodic memory.

        For Phase 2 model-based planning: provides outcome predictions based
        on similar past experiences. Uses pattern completion capability of
        hippocampus to predict what will happen next.

        Biology: Hippocampus retrieves similar past episodes during planning
        and decision-making (Johnson & Redish, 2007). CA3 pattern completion
        allows partial cues to retrieve full memories.

        Args:
            query_state: State to find similar experiences for [n] (1D, ADR-005)
            query_action: Optional action to filter by (boosts similarity)
            k: Number of similar experiences to retrieve
            similarity_threshold: Minimum similarity to return (0.0-1.0)

        Returns:
            similar_episodes: List of dicts with keys:
                - 'state': Episode state tensor
                - 'action': Action taken
                - 'next_state': Resulting state (approximated)
                - 'reward': Reward received
                - 'similarity': Cosine similarity score (0.0-1.0)
                - 'context': Optional context tensor
                - 'metadata': Optional metadata dict

        Note:
            Uses cosine similarity in state space. For more sophisticated
            retrieval, could use CA3 recurrent dynamics or DG-CA3-CA1 circuit.
        """
        if not self.episode_buffer:
            return []

        k = min(k, len(self.episode_buffer))

        # Compute similarity to all stored episodes
        similarities = []
        for episode in self.episode_buffer:
            # Cosine similarity between query and episode state
            sim = cosine_similarity_safe(
                query_state.unsqueeze(0),
                episode.state.unsqueeze(0)
            ).item()

            # If action provided, boost similarity for matching actions
            if query_action is not None and episode.action == query_action:
                sim = min(1.0, sim * 1.2)  # 20% boost for matching action

            similarities.append((sim, episode))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Build result list with top-K above threshold
        similar = []
        for sim, episode in similarities[:k]:
            if sim >= similarity_threshold:
                # Approximate next_state (we don't explicitly store it)
                # For now, use the state itself as a proxy
                # In full implementation, would track state transitions
                next_state_approx = episode.state  # Could be improved

                similar.append({
                    'state': episode.state,
                    'action': episode.action,
                    'next_state': next_state_approx,
                    'reward': episode.reward,
                    'similarity': sim,
                    'context': episode.context,
                    'metadata': episode.metadata,
                    'correct': episode.correct,
                    'priority': episode.priority,
                })

        return similar
        
    def clear_buffer(self) -> None:
        """Clear all episodes from buffer."""
        self.episode_buffer.clear()
    
    def reset_state(self) -> None:
        """Reset episode manager state (trial boundaries)."""
        # Episode buffer persists across trials (long-term memory)
        pass
    
    def to(self, device: torch.device) -> "EpisodeManager":
        """Move all tensors to specified device.
        
        Args:
            device: Target device
            
        Returns:
            Self for chaining
        """
        self.context.device = device
        # Move episode tensors to device
        for episode in self.episode_buffer:
            episode.state = episode.state.to(device)
            if episode.context is not None:
                episode.context = episode.context.to(device)
            if episode.sequence is not None:
                episode.sequence = [s.to(device) for s in episode.sequence]
        return self
        
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get episode buffer diagnostics."""
        if not self.episode_buffer:
            return {
                "buffer_size": 0,
                "total_rewards": 0.0,
                "correct_ratio": 0.0,
            }
            
        return {
            "buffer_size": len(self.episode_buffer),
            "total_rewards": sum(ep.reward for ep in self.episode_buffer),
            "correct_ratio": sum(ep.correct for ep in self.episode_buffer) / len(self.episode_buffer),
            "mean_priority": sum(ep.priority for ep in self.episode_buffer) / len(self.episode_buffer),
        }
