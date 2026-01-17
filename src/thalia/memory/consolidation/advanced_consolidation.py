"""
Advanced Consolidation - Schema extraction, semantic reorganization, interference resolution.

This module extends the basic consolidation system (consolidation.py) with higher-order
memory processing that matches human memory consolidation:

1. SCHEMA EXTRACTION (REM phase)
   - Cluster similar episodes by semantic content
   - Extract prototypical "average" patterns
   - Replay with noise for generalization
   - Store schemas for future use

2. SEMANTIC REORGANIZATION
   - Organize memories by meaning, not temporal order
   - K-means clustering in semantic feature space
   - Reorder replay buffer by semantic similarity
   - Enable semantic-based sequential replay

3. INTERFERENCE RESOLUTION
   - Detect interfering memory pairs (similar input, different output)
   - Apply contrastive learning to orthogonalize representations
   - Strengthen unique features, suppress shared features
   - Prevent catastrophic forgetting

4. INTEGRATION
   - Extends existing NREM/REM consolidation
   - Adds schema-based replay during REM phase
   - Periodic semantic reorganization of memory buffer
   - On-demand interference resolution when detected

Biology:
========
- REM sleep creates generalized schemas from episodic memories (Stickgold, 2005)
- Long-term memories reorganized by semantic similarity (Bauer & Larkina, 2014)
- Consolidation separates overlapping memories (McClelland et al., 1995)
- Prototypical averaging enables generalization (Rosch, 1978)

Usage:
======

    from thalia.memory.consolidation.advanced_consolidation import (
        SchemaExtractionConsolidation,
        SemanticReorganization,
        InterferenceResolution,
        run_advanced_consolidation,
    )

    # Schema extraction during REM
    schema_system = SchemaExtractionConsolidation(
        similarity_threshold=0.7,
        cluster_size=5,
    )
    schema_system.rem_schema_extraction(
        brain=brain,
        episodes=replay_buffer,
        n_steps=3000,
    )

    # Semantic reorganization
    semantic_system = SemanticReorganization(n_clusters=10)
    semantic_system.reorganize_episodes(episodes, extract_features_fn)

    # Interference resolution
    interference_system = InterferenceResolution()
    interfering = interference_system.detect_interference(
        episodes,
        similarity_threshold=0.8,
    )
    if len(interfering) > 10:
        interference_system.resolve_interference(
            brain=brain,
            interfering_pairs=interfering,
            n_steps=1000,
        )

    # Or use integrated system
    run_advanced_consolidation(
        brain=brain,
        episodes=replay_buffer,
        extract_features_fn=lambda x: brain.cortex(x),
        n_steps=6000,
    )

References:
===========
- McClelland et al. (1995): Complementary learning systems
- Stickgold (2005): Sleep-dependent memory consolidation
- Rosch (1978): Principles of categorization
- Bauer & Larkina (2014): Childhood amnesia and semantic memory

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from thalia.regions.hippocampus.config import Episode

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class SchemaExtractionConfig:
    """Configuration for schema extraction during REM."""

    similarity_threshold: float = 0.7  # Cosine similarity threshold for clustering
    cluster_size: int = 5  # Minimum episodes per schema
    noise_std: float = 0.2  # Noise for generalization
    learning_signal: float = 0.5  # Moderate learning (between awake and NREM)
    max_schemas: int = 100  # Maximum stored schemas
    min_exemplars: int = 2  # Minimum exemplars to form schema


@dataclass
class SemanticReorganizationConfig:
    """Configuration for semantic memory reorganization."""

    n_clusters: int = 10  # Number of semantic clusters
    max_iterations: int = 100  # K-means max iterations
    tolerance: float = 1e-4  # K-means convergence tolerance
    same_cluster_prob: float = 0.8  # Probability of sampling from same cluster
    device: str = "cpu"  # Device for computation


@dataclass
class InterferenceResolutionConfig:
    """Configuration for interference resolution."""

    similarity_threshold: float = 0.8  # Input similarity threshold for interference
    dissimilarity_threshold: float = 0.3  # Output dissimilarity threshold
    learning_signal: float = 0.3  # Learning rate for contrastive STDP
    max_pairs_to_resolve: int = 50  # Maximum interfering pairs to process
    contrastive_strength: float = 0.5  # Strength of contrastive push


# ============================================================================
# Schema Extraction (REM Phase)
# ============================================================================


@dataclass
class Schema:
    """A learned schema representing a category of experiences."""

    prototype_input: torch.Tensor  # Prototypical input pattern
    prototype_target: torch.Tensor  # Prototypical target pattern
    n_exemplars: int  # Number of episodes in this schema
    last_updated: int  # Last consolidation step
    cluster_members: List[int] = field(default_factory=list)  # Episode indices


class SchemaExtractionConsolidation:
    """Extract abstract schemas during REM consolidation.

    Biology: REM sleep creates generalized schemas from episodic memories
    by replaying similar episodes and extracting common structure.

    Algorithm:
    1. Cluster replay buffer by similarity (cosine > threshold)
    2. For each cluster, compute prototypical average
    3. Replay prototypes with noise (generalization)
    4. Store schemas for future retrieval

    Example:
        >>> schema_system = SchemaExtractionConsolidation()
        >>> schema_system.rem_schema_extraction(
        ...     brain=brain,
        ...     episodes=replay_buffer,
        ...     n_steps=3000,
        ... )
        >>> print(f"Extracted {len(schema_system.schemas)} schemas")
    """

    def __init__(self, config: Optional[SchemaExtractionConfig] = None):
        """Initialize schema extraction system.

        Args:
            config: Configuration for schema extraction
        """
        self.config = config or SchemaExtractionConfig()
        self.schemas: Dict[int, Schema] = {}  # schema_id → Schema
        self._schema_counter = 0

    def rem_schema_extraction(
        self,
        brain: Any,  # Brain instance
        episodes: List[Episode],
        n_steps: int,
        extract_features_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """Extract schemas during REM phase.

        Args:
            brain: Brain instance with forward() method
            episodes: List of episodes to process
            n_steps: Number of REM steps
            extract_features_fn: Function to extract semantic features
                                If None, uses input directly

        Returns:
            metrics: Dictionary with extraction statistics
        """
        if len(episodes) < self.config.min_exemplars:
            return {
                "n_schemas_extracted": 0,
                "avg_cluster_size": 0.0,
                "replay_steps": 0,
            }

        n_schemas_extracted = 0
        total_cluster_size = 0
        replay_steps = 0

        for step in range(n_steps):
            # Sample cluster of similar episodes
            cluster = self._sample_similar_cluster(
                episodes,
                k=self.config.cluster_size,
                similarity_threshold=self.config.similarity_threshold,
                extract_features_fn=extract_features_fn,
            )

            if len(cluster) < self.config.min_exemplars:
                # Not enough similar episodes, use random replay
                episode = episodes[torch.randint(len(episodes), (1,)).item()]
                brain.forward(episode.state, learning_signal=0.0)
                replay_steps += 1
                continue

            # Extract prototypical pattern
            prototypical_input = torch.stack([ep.state for ep in cluster]).mean(dim=0)

            # Target: Average reward-weighted
            weights = torch.tensor([ep.reward for ep in cluster], device=prototypical_input.device)
            weights = F.softmax(weights / 0.1, dim=0)  # Temperature = 0.1

            if cluster[0].context is not None:
                prototypical_target = torch.stack([ep.context for ep in cluster])
                prototypical_target = (prototypical_target * weights.view(-1, 1)).sum(dim=0)
            else:
                prototypical_target = prototypical_input.clone()

            # Add noise for generalization
            noisy_input = (
                prototypical_input + torch.randn_like(prototypical_input) * self.config.noise_std
            )
            noisy_input = noisy_input.clamp(0, 1)

            # Replay prototype with moderate learning signal
            brain.forward(noisy_input, learning_signal=self.config.learning_signal)
            replay_steps += 1

            # Store schema
            if len(self.schemas) < self.config.max_schemas:
                schema_id = self._schema_counter
                self._schema_counter += 1

                self.schemas[schema_id] = Schema(
                    prototype_input=prototypical_input,
                    prototype_target=prototypical_target,
                    n_exemplars=len(cluster),
                    last_updated=step,
                    cluster_members=[id(ep) for ep in cluster],
                )

                n_schemas_extracted += 1
                total_cluster_size += len(cluster)

        return {
            "n_schemas_extracted": n_schemas_extracted,
            "avg_cluster_size": total_cluster_size / max(1, n_schemas_extracted),
            "replay_steps": replay_steps,
            "total_schemas": len(self.schemas),
        }

    def _sample_similar_cluster(
        self,
        episodes: List[Episode],
        k: int,
        similarity_threshold: float,
        extract_features_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
    ) -> List[Episode]:
        """Sample cluster of k similar episodes.

        Args:
            episodes: List of episodes
            k: Desired cluster size
            similarity_threshold: Minimum cosine similarity
            extract_features_fn: Feature extraction function

        Returns:
            cluster: List of similar episodes
        """
        if len(episodes) == 0:
            return []

        # Sample anchor episode randomly
        anchor_idx = torch.randint(len(episodes), (1,)).item()
        anchor = episodes[anchor_idx]

        # Extract features
        if extract_features_fn is not None:
            anchor_features = extract_features_fn(anchor.state.unsqueeze(0)).squeeze(0)
        else:
            anchor_features = anchor.state

        # Compute similarity to all other episodes
        similarities: List[Tuple[Episode, float]] = []

        for i, episode in enumerate(episodes):
            if i == anchor_idx:
                continue

            # Extract features
            if extract_features_fn is not None:
                ep_features = extract_features_fn(episode.state.unsqueeze(0)).squeeze(0)
            else:
                ep_features = episode.state

            # Cosine similarity
            sim = self._cosine_similarity(anchor_features, ep_features)

            if sim > similarity_threshold:
                similarities.append((episode, sim))

        # Take top k most similar
        similarities.sort(key=lambda x: x[1], reverse=True)
        cluster = [anchor] + [ep for ep, _ in similarities[: k - 1]]

        return cluster

    def _cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute cosine similarity between two tensors.

        Args:
            a: First tensor
            b: Second tensor

        Returns:
            similarity: Cosine similarity (0-1)
        """
        a_norm = a / (a.norm() + 1e-8)
        b_norm = b / (b.norm() + 1e-8)
        return (a_norm * b_norm).sum().item()

    def get_schema(self, schema_id: int) -> Optional[Schema]:
        """Retrieve a stored schema.

        Args:
            schema_id: Schema identifier

        Returns:
            schema: Schema object or None if not found
        """
        return self.schemas.get(schema_id)

    def clear_old_schemas(self, max_age: int, current_step: int) -> int:
        """Remove schemas that haven't been updated recently.

        Args:
            max_age: Maximum age in steps
            current_step: Current consolidation step

        Returns:
            n_removed: Number of schemas removed
        """
        to_remove = []

        for schema_id, schema in self.schemas.items():
            age = current_step - schema.last_updated
            if age > max_age:
                to_remove.append(schema_id)

        for schema_id in to_remove:
            del self.schemas[schema_id]

        return len(to_remove)


# ============================================================================
# Semantic Reorganization
# ============================================================================


class SemanticReorganization:
    """Reorganize memories by semantic similarity.

    Biology: Long-term memories are organized by semantic content,
    not temporal order. This enables generalization and transfer.

    Algorithm:
    1. Extract semantic features from all episodes
    2. K-means clustering by feature similarity
    3. Reorder buffer: Similar episodes adjacent
    4. Update sampling to prefer within-cluster transitions

    Example:
        >>> semantic_system = SemanticReorganization(n_clusters=10)
        >>> semantic_system.reorganize_episodes(
        ...     episodes=replay_buffer,
        ...     extract_features_fn=lambda x: brain.cortex(x),
        ... )
        >>> sequence = semantic_system.sample_semantic_sequence(
        ...     episodes=replay_buffer,
        ...     n_samples=10,
        ... )
    """

    def __init__(self, config: Optional[SemanticReorganizationConfig] = None):
        """Initialize semantic reorganization system.

        Args:
            config: Configuration for reorganization
        """
        self.config = config or SemanticReorganizationConfig()
        self.cluster_centers: Optional[torch.Tensor] = None
        self.cluster_assignments: Optional[torch.Tensor] = None

    def reorganize_episodes(
        self,
        episodes: List[Episode],
        extract_features_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> Dict[str, Any]:
        """Reorganize episodes by semantic clusters.

        Args:
            episodes: List of episodes to reorganize (modified in-place)
            extract_features_fn: Function to extract semantic features

        Returns:
            metrics: Dictionary with reorganization statistics
        """
        if len(episodes) < self.config.n_clusters:
            return {
                "n_episodes": len(episodes),
                "n_clusters": 0,
                "avg_cluster_size": 0.0,
            }

        # Extract semantic features
        features_list = []
        for episode in episodes:
            semantic_features = extract_features_fn(episode.state.unsqueeze(0))
            features_list.append(semantic_features.squeeze(0))

        features = torch.stack(features_list).to(self.config.device)

        # K-means clustering
        self.cluster_centers, self.cluster_assignments = self._kmeans(
            features,
            n_clusters=self.config.n_clusters,
            max_iterations=self.config.max_iterations,
            tolerance=self.config.tolerance,
        )

        # Reorganize episodes by cluster
        reorganized_episodes = []
        cluster_sizes = []

        for cluster_id in range(self.config.n_clusters):
            cluster_mask = self.cluster_assignments == cluster_id
            cluster_indices = torch.where(cluster_mask)[0]

            cluster_episodes = [episodes[i] for i in cluster_indices]
            reorganized_episodes.extend(cluster_episodes)
            cluster_sizes.append(len(cluster_episodes))

            # Store cluster_id in episode metadata
            for ep in cluster_episodes:
                if ep.metadata is None:
                    ep.metadata = {}
                ep.metadata["cluster_id"] = cluster_id

        # Replace original list contents
        episodes[:] = reorganized_episodes

        return {
            "n_episodes": len(episodes),
            "n_clusters": self.config.n_clusters,
            "avg_cluster_size": sum(cluster_sizes) / len(cluster_sizes),
            "min_cluster_size": min(cluster_sizes),
            "max_cluster_size": max(cluster_sizes),
        }

    def sample_semantic_sequence(
        self,
        episodes: List[Episode],
        n_samples: int = 10,
    ) -> List[Episode]:
        """Sample sequence following semantic similarity.

        Args:
            episodes: List of episodes (must be reorganized first)
            n_samples: Number of episodes to sample

        Returns:
            sequence: List of semantically related episodes
        """
        if len(episodes) == 0 or self.cluster_assignments is None:
            return []

        sequence = []

        # Start with random episode
        current_episode = episodes[torch.randint(len(episodes), (1,)).item()]
        sequence.append(current_episode)

        # Sample next episodes by semantic proximity
        for _ in range(n_samples - 1):
            current_cluster = current_episode.metadata.get("cluster_id", 0)

            # Same cluster (80% prob) or adjacent cluster (20% prob)
            if torch.rand(1).item() < self.config.same_cluster_prob:
                next_cluster = current_cluster
            else:
                # Random walk to adjacent cluster
                next_cluster = (
                    current_cluster + torch.randint(-1, 2, (1,)).item()
                ) % self.config.n_clusters

            # Sample from target cluster
            next_episode = self._sample_from_cluster(episodes, next_cluster)
            if next_episode is not None:
                sequence.append(next_episode)
                current_episode = next_episode

        return sequence

    def _sample_from_cluster(
        self,
        episodes: List[Episode],
        cluster_id: int,
    ) -> Optional[Episode]:
        """Sample random episode from specified cluster.

        Args:
            episodes: List of episodes
            cluster_id: Target cluster

        Returns:
            episode: Random episode from cluster or None
        """
        cluster_episodes = [
            ep
            for ep in episodes
            if ep.metadata is not None and ep.metadata.get("cluster_id") == cluster_id
        ]

        if not cluster_episodes:
            return None

        return cluster_episodes[torch.randint(len(cluster_episodes), (1,)).item()]

    def _kmeans(
        self,
        features: torch.Tensor,
        n_clusters: int,
        max_iterations: int,
        tolerance: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """K-means clustering.

        Args:
            features: [n_samples, feature_dim] tensor
            n_clusters: Number of clusters
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            centers: [n_clusters, feature_dim] cluster centers
            assignments: [n_samples] cluster assignments
        """
        n_samples = features.shape[0]

        # Initialize centers randomly
        indices = torch.randperm(n_samples)[:n_clusters]
        centers = features[indices].clone()

        for iteration in range(max_iterations):
            # Assign to nearest center
            distances = torch.cdist(features, centers)
            assignments = distances.argmin(dim=1)

            # Update centers
            new_centers = []
            for k in range(n_clusters):
                cluster_mask = assignments == k
                if cluster_mask.sum() > 0:
                    new_centers.append(features[cluster_mask].mean(dim=0))
                else:
                    # Empty cluster, keep old center
                    new_centers.append(centers[k])

            new_centers = torch.stack(new_centers)

            # Check convergence
            center_shift = (new_centers - centers).norm(dim=1).max()
            centers = new_centers

            if center_shift < tolerance:
                break

        return centers, assignments


# ============================================================================
# Interference Resolution
# ============================================================================


class InterferenceResolution:
    """Resolve interference between overlapping memories.

    Biology: Consolidation separates overlapping memories by
    orthogonalizing their representations (pattern separation).

    Algorithm:
    1. Detect interfering pairs: high input similarity + low output similarity
    2. Replay pairs in alternation with contrastive objective
    3. Push representations apart (maximize distance)
    4. Strengthen unique features, suppress shared features

    Example:
        >>> interference_system = InterferenceResolution()
        >>> interfering = interference_system.detect_interference(
        ...     episodes=replay_buffer,
        ...     similarity_threshold=0.8,
        ... )
        >>> if len(interfering) > 10:
        ...     interference_system.resolve_interference(
        ...         brain=brain,
        ...         interfering_pairs=interfering,
        ...         n_steps=1000,
        ...     )
    """

    def __init__(self, config: Optional[InterferenceResolutionConfig] = None):
        """Initialize interference resolution system.

        Args:
            config: Configuration for interference resolution
        """
        self.config = config or InterferenceResolutionConfig()

    def detect_interference(
        self,
        episodes: List[Episode],
        extract_features_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> List[Tuple[Episode, Episode, float]]:
        """Detect interfering memory pairs.

        Interference = high input similarity + low output similarity
        (Same stimulus should produce different responses)

        Args:
            episodes: List of episodes to check
            extract_features_fn: Feature extraction function (optional)

        Returns:
            interfering_pairs: List of (ep1, ep2, input_similarity)
        """
        interfering_pairs = []

        for i, ep1 in enumerate(episodes):
            for j in range(i + 1, min(i + 100, len(episodes))):  # Limit search
                ep2 = episodes[j]

                # Compute input similarity
                if extract_features_fn is not None:
                    feat1 = extract_features_fn(ep1.state.unsqueeze(0)).squeeze(0)
                    feat2 = extract_features_fn(ep2.state.unsqueeze(0)).squeeze(0)
                    input_sim = self._cosine_similarity(feat1, feat2)
                else:
                    input_sim = self._cosine_similarity(ep1.state, ep2.state)

                # Check if inputs are similar
                if input_sim < self.config.similarity_threshold:
                    continue

                # Compute output dissimilarity
                if ep1.context is not None and ep2.context is not None:
                    output_sim = self._cosine_similarity(ep1.context, ep2.context)
                else:
                    # Use action dissimilarity as proxy
                    output_sim = 1.0 if ep1.action == ep2.action else 0.0

                # Interference: high input sim + low output sim
                if output_sim < self.config.dissimilarity_threshold:
                    interfering_pairs.append((ep1, ep2, input_sim))

        # Sort by input similarity (most interfering first)
        interfering_pairs.sort(key=lambda x: x[2], reverse=True)

        return interfering_pairs[: self.config.max_pairs_to_resolve]

    def resolve_interference(
        self,
        brain: Any,
        interfering_pairs: List[Tuple[Episode, Episode, float]],
        n_steps: int,
    ) -> Dict[str, float]:
        """Resolve interference via contrastive learning.

        Args:
            brain: Brain instance with forward() method
            interfering_pairs: List of interfering pairs
            n_steps: Number of resolution steps

        Returns:
            metrics: Dictionary with resolution statistics
        """
        if not interfering_pairs:
            return {
                "pairs_resolved": 0,
                "avg_input_similarity": 0.0,
                "resolution_steps": 0,
            }

        total_input_sim = 0.0

        for step in range(n_steps):
            # Sample interfering pair
            pair_idx = torch.randint(len(interfering_pairs), (1,)).item()
            ep1, ep2, input_sim = interfering_pairs[pair_idx]

            total_input_sim += input_sim

            # Replay both with contrastive objective
            # Forward ep1
            repr1 = brain.forward(ep1.state, learning_signal=self.config.learning_signal)

            # Forward ep2
            repr2 = brain.forward(ep2.state, learning_signal=self.config.learning_signal)

            # Contrastive push: Apply anti-Hebbian learning
            # (Implemented by inverting the learning signal for shared activations)
            self._apply_contrastive_push(brain, repr1, repr2)

        return {
            "pairs_resolved": len(interfering_pairs),
            "avg_input_similarity": total_input_sim / n_steps,
            "resolution_steps": n_steps,
        }

    def _cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute cosine similarity.

        Args:
            a: First tensor
            b: Second tensor

        Returns:
            similarity: Cosine similarity (0-1)
        """
        a_norm = a / (a.norm() + 1e-8)
        b_norm = b / (b.norm() + 1e-8)
        return (a_norm * b_norm).sum().item()

    def _apply_contrastive_push(
        self,
        brain: Any,
        repr1: torch.Tensor,
        repr2: torch.Tensor,
    ) -> None:
        """Apply contrastive learning to push representations apart.

        This is a simplified implementation that sets a flag in the brain
        to invert the learning signal for shared activations.

        In practice, this would be implemented via:
        - Anti-Hebbian STDP for co-active neurons
        - Heterosynaptic plasticity
        - Homeostatic scaling

        Args:
            brain: Brain instance
            repr1: First representation
            repr2: Second representation
        """
        # Identify shared activations (both > 0)
        shared_mask = (repr1 > 0) & (repr2 > 0)

        # In a full implementation, we would:
        # 1. Weaken connections that drive shared activations
        # 2. Strengthen connections that drive unique activations
        # 3. Use contrastive loss: max(0, similarity - margin)

        # For now, set a flag for the learning system
        if hasattr(brain, "contrastive_mode"):
            brain.contrastive_mode = True
            brain.contrastive_mask = shared_mask

        # Note: The actual weight updates happen in the learning rules
        # (STDP, BCM, etc.) which check for contrastive_mode


# ============================================================================
# Integrated Advanced Consolidation
# ============================================================================


def run_advanced_consolidation(
    brain: Any,
    episodes: List[Episode],
    extract_features_fn: Callable[[torch.Tensor], torch.Tensor],
    n_steps: int = 6000,
    config_schema: Optional[SchemaExtractionConfig] = None,
    config_semantic: Optional[SemanticReorganizationConfig] = None,
    config_interference: Optional[InterferenceResolutionConfig] = None,
) -> Dict[str, Any]:
    """Run complete advanced consolidation cycle.

    This integrates:
    1. Interference resolution (if needed)
    2. REM schema extraction
    3. Semantic reorganization

    Args:
        brain: Brain instance
        episodes: List of episodes to consolidate
        extract_features_fn: Function to extract semantic features
        n_steps: Total consolidation steps
        config_schema: Schema extraction config
        config_semantic: Semantic reorganization config
        config_interference: Interference resolution config

    Returns:
        metrics: Combined metrics from all systems
    """
    metrics = {}

    # Phase 1: Interference Resolution (if needed)
    interference_system = InterferenceResolution(config_interference)
    interfering = interference_system.detect_interference(
        episodes,
        extract_features_fn=extract_features_fn,
    )

    if len(interfering) > 10:
        interference_metrics = interference_system.resolve_interference(
            brain=brain,
            interfering_pairs=interfering,
            n_steps=n_steps // 6,  # ~1000 steps
        )
        metrics["interference"] = interference_metrics
    else:
        metrics["interference"] = {"pairs_resolved": 0}

    # Phase 2: REM Schema Extraction
    schema_system = SchemaExtractionConsolidation(config_schema)
    schema_metrics = schema_system.rem_schema_extraction(
        brain=brain,
        episodes=episodes,
        n_steps=n_steps // 2,  # ~3000 steps
        extract_features_fn=extract_features_fn,
    )
    metrics["schema_extraction"] = schema_metrics

    # Phase 3: Semantic Reorganization
    semantic_system = SemanticReorganization(config_semantic)
    semantic_metrics = semantic_system.reorganize_episodes(
        episodes=episodes,
        extract_features_fn=extract_features_fn,
    )
    metrics["semantic_reorganization"] = semantic_metrics

    # Phase 4: Random Replay (for generalization)
    if len(episodes) > 0:
        for step in range(n_steps // 3):  # ~2000 steps
            episode = episodes[torch.randint(len(episodes), (1,)).item()]
            brain.forward(episode.state, learning_signal=0.0)

    metrics["total_steps"] = n_steps

    return metrics
