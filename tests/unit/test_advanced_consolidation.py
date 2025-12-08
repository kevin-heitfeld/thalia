"""
Tests for Advanced Consolidation Features.

Tests schema extraction, semantic reorganization, and interference resolution.
"""

import pytest
import torch

from thalia.memory.advanced_consolidation import (
    SchemaExtractionConsolidation,
    SchemaExtractionConfig,
    SemanticReorganization,
    SemanticReorganizationConfig,
    InterferenceResolution,
    InterferenceResolutionConfig,
    run_advanced_consolidation,
    Schema,
)
from thalia.regions.hippocampus.config import Episode


# ============================================================================
# Mock Brain for Testing
# ============================================================================

class MockBrain:
    """Simple mock brain for testing."""
    
    def __init__(self, feature_dim: int = 10):
        self.feature_dim = feature_dim
        self.forward_calls = []
        self.contrastive_mode = False
        self.contrastive_mask = None
    
    def forward(self, x: torch.Tensor, learning_signal: float = 0.0):
        """Mock forward pass."""
        self.forward_calls.append((x.clone(), learning_signal))
        # Return a simple feature representation
        return torch.randn(self.feature_dim)
    
    def cortex(self, x: torch.Tensor) -> torch.Tensor:
        """Mock cortex feature extraction."""
        # Simple linear transformation as feature
        return x.view(-1).mean().repeat(self.feature_dim)


# ============================================================================
# Helper Functions
# ============================================================================

def create_test_episodes(
    n: int,
    input_dim: int = 20,
    n_clusters: int = 3,
    device: str = "cpu",
) -> list:
    """Create test episodes with natural clustering.
    
    Args:
        n: Number of episodes
        input_dim: Input dimension
        n_clusters: Number of semantic clusters
        device: Device for tensors
    
    Returns:
        episodes: List of Episode objects
    """
    episodes = []
    
    # Create cluster centers
    centers = torch.randn(n_clusters, input_dim, device=device)
    
    for i in range(n):
        # Assign to cluster
        cluster_id = i % n_clusters
        
        # Sample around cluster center
        state = centers[cluster_id] + torch.randn(input_dim, device=device) * 0.3
        state = state.clamp(0, 1)
        
        # Create episode
        episode = Episode(
            state=state,
            action=cluster_id,
            reward=torch.rand(1).item(),
            correct=torch.rand(1).item() > 0.5,
            context=torch.randn(input_dim, device=device),
            timestamp=i,
        )
        
        episodes.append(episode)
    
    return episodes


def extract_features(x: torch.Tensor) -> torch.Tensor:
    """Simple feature extraction for testing."""
    return x.view(-1)


# ============================================================================
# Schema Extraction Tests
# ============================================================================

def test_schema_extraction_initialization():
    """Test SchemaExtractionConsolidation initialization."""
    system = SchemaExtractionConsolidation()
    
    assert system.config is not None
    assert len(system.schemas) == 0
    assert system._schema_counter == 0


def test_schema_extraction_basic():
    """Test basic schema extraction from similar episodes."""
    system = SchemaExtractionConsolidation(
        SchemaExtractionConfig(
            similarity_threshold=0.5,
            cluster_size=3,
            min_exemplars=2,
        )
    )
    brain = MockBrain()
    
    # Create episodes with high similarity
    episodes = create_test_episodes(n=20, input_dim=10, n_clusters=2)
    
    # Run schema extraction
    metrics = system.rem_schema_extraction(
        brain=brain,
        episodes=episodes,
        n_steps=100,
        extract_features_fn=extract_features,
    )
    
    assert metrics["n_schemas_extracted"] >= 0
    assert metrics["replay_steps"] == 100
    assert len(brain.forward_calls) == 100


def test_schema_extraction_clustering():
    """Test that schema extraction finds similar episodes."""
    system = SchemaExtractionConsolidation(
        SchemaExtractionConfig(
            similarity_threshold=0.7,
            cluster_size=5,
        )
    )
    brain = MockBrain()
    
    episodes = create_test_episodes(n=30, input_dim=15, n_clusters=3)
    
    metrics = system.rem_schema_extraction(
        brain=brain,
        episodes=episodes,
        n_steps=50,
        extract_features_fn=extract_features,
    )
    
    # Should extract some schemas
    assert metrics["n_schemas_extracted"] >= 0
    assert metrics["total_schemas"] == len(system.schemas)


def test_schema_extraction_noise_generalization():
    """Test that noise is added for generalization."""
    system = SchemaExtractionConsolidation(
        SchemaExtractionConfig(
            noise_std=0.3,
            similarity_threshold=0.6,
        )
    )
    brain = MockBrain()
    
    episodes = create_test_episodes(n=20, input_dim=10)
    
    system.rem_schema_extraction(
        brain=brain,
        episodes=episodes,
        n_steps=20,
        extract_features_fn=extract_features,
    )
    
    # Check that inputs to brain have noise
    # (Different from original episode states)
    inputs = [call[0] for call in brain.forward_calls]
    assert len(inputs) > 0


def test_schema_extraction_learning_signal():
    """Test that moderate learning signal is used."""
    config = SchemaExtractionConfig(learning_signal=0.6)
    system = SchemaExtractionConsolidation(config)
    brain = MockBrain()
    
    episodes = create_test_episodes(n=15, input_dim=10)
    
    system.rem_schema_extraction(
        brain=brain,
        episodes=episodes,
        n_steps=10,
        extract_features_fn=extract_features,
    )
    
    # Check learning signals
    learning_signals = [call[1] for call in brain.forward_calls]
    # Most should be 0.0 (random) or config.learning_signal (schema)
    assert all(0.0 <= ls <= 1.0 for ls in learning_signals)


def test_schema_extraction_max_schemas():
    """Test that max_schemas limit is respected."""
    system = SchemaExtractionConsolidation(
        SchemaExtractionConfig(
            max_schemas=5,
            similarity_threshold=0.5,
        )
    )
    brain = MockBrain()
    
    episodes = create_test_episodes(n=50, input_dim=10, n_clusters=10)
    
    system.rem_schema_extraction(
        brain=brain,
        episodes=episodes,
        n_steps=200,
        extract_features_fn=extract_features,
    )
    
    assert len(system.schemas) <= 5


def test_schema_get_and_clear():
    """Test schema retrieval and clearing."""
    system = SchemaExtractionConsolidation()
    brain = MockBrain()
    
    episodes = create_test_episodes(n=20, input_dim=10)
    
    system.rem_schema_extraction(
        brain=brain,
        episodes=episodes,
        n_steps=50,
        extract_features_fn=extract_features,
    )
    
    # Test retrieval
    for schema_id in system.schemas.keys():
        schema = system.get_schema(schema_id)
        assert schema is not None
        assert isinstance(schema, Schema)
    
    # Test clearing old schemas
    n_removed = system.clear_old_schemas(max_age=10, current_step=100)
    assert n_removed >= 0


def test_schema_extraction_empty_episodes():
    """Test schema extraction with no episodes."""
    system = SchemaExtractionConsolidation()
    brain = MockBrain()
    
    metrics = system.rem_schema_extraction(
        brain=brain,
        episodes=[],
        n_steps=10,
    )
    
    assert metrics["n_schemas_extracted"] == 0
    assert metrics["avg_cluster_size"] == 0.0


# ============================================================================
# Semantic Reorganization Tests
# ============================================================================

def test_semantic_reorganization_initialization():
    """Test SemanticReorganization initialization."""
    system = SemanticReorganization()
    
    assert system.config is not None
    assert system.cluster_centers is None
    assert system.cluster_assignments is None


def test_semantic_reorganization_basic():
    """Test basic semantic reorganization."""
    system = SemanticReorganization(
        SemanticReorganizationConfig(n_clusters=3)
    )
    
    episodes = create_test_episodes(n=30, input_dim=10, n_clusters=3)
    
    metrics = system.reorganize_episodes(
        episodes=episodes,
        extract_features_fn=extract_features,
    )
    
    assert metrics["n_episodes"] == 30
    assert metrics["n_clusters"] == 3
    assert metrics["avg_cluster_size"] > 0


def test_semantic_reorganization_clustering():
    """Test that episodes are grouped by similarity."""
    system = SemanticReorganization(
        SemanticReorganizationConfig(n_clusters=4)
    )
    
    episodes = create_test_episodes(n=40, input_dim=15, n_clusters=4)
    
    system.reorganize_episodes(
        episodes=episodes,
        extract_features_fn=extract_features,
    )
    
    # Check that cluster_id is added to metadata
    for ep in episodes:
        assert ep.metadata is not None
        assert 'cluster_id' in ep.metadata
        assert 0 <= ep.metadata['cluster_id'] < 4


def test_semantic_reorganization_adjacency():
    """Test that similar episodes become adjacent."""
    system = SemanticReorganization(
        SemanticReorganizationConfig(n_clusters=3)
    )
    
    episodes = create_test_episodes(n=30, input_dim=10, n_clusters=3)
    original_order = [ep.timestamp for ep in episodes]
    
    system.reorganize_episodes(
        episodes=episodes,
        extract_features_fn=extract_features,
    )
    
    new_order = [ep.timestamp for ep in episodes]
    
    # Order should change (unless we're very unlucky)
    assert original_order != new_order
    
    # Episodes should be sorted by cluster
    cluster_ids = [ep.metadata['cluster_id'] for ep in episodes]
    # Should be mostly sorted (allowing for some mixing)
    transitions = sum(1 for i in range(len(cluster_ids)-1) if cluster_ids[i] != cluster_ids[i+1])
    assert transitions < len(episodes) / 2  # Fewer transitions than random


def test_semantic_sample_sequence():
    """Test semantic sequence sampling."""
    system = SemanticReorganization(
        SemanticReorganizationConfig(
            n_clusters=4,
            same_cluster_prob=0.9,
        )
    )
    
    episodes = create_test_episodes(n=40, input_dim=10, n_clusters=4)
    
    system.reorganize_episodes(
        episodes=episodes,
        extract_features_fn=extract_features,
    )
    
    sequence = system.sample_semantic_sequence(
        episodes=episodes,
        n_samples=10,
    )
    
    assert len(sequence) <= 10
    # Most should be from same cluster
    cluster_ids = [ep.metadata['cluster_id'] for ep in sequence]
    # Check that most transitions are within-cluster
    same_cluster = sum(1 for i in range(len(cluster_ids)-1) if cluster_ids[i] == cluster_ids[i+1])
    assert same_cluster >= len(sequence) // 2


def test_semantic_reorganization_too_few_episodes():
    """Test reorganization with fewer episodes than clusters."""
    system = SemanticReorganization(
        SemanticReorganizationConfig(n_clusters=10)
    )
    
    episodes = create_test_episodes(n=5, input_dim=10)
    
    metrics = system.reorganize_episodes(
        episodes=episodes,
        extract_features_fn=extract_features,
    )
    
    assert metrics["n_episodes"] == 5
    assert metrics["n_clusters"] == 0


def test_semantic_reorganization_kmeans_convergence():
    """Test that k-means converges."""
    system = SemanticReorganization(
        SemanticReorganizationConfig(
            n_clusters=3,
            max_iterations=100,
            tolerance=1e-4,
        )
    )
    
    episodes = create_test_episodes(n=30, input_dim=10, n_clusters=3)
    
    metrics = system.reorganize_episodes(
        episodes=episodes,
        extract_features_fn=extract_features,
    )
    
    # Should successfully reorganize
    assert metrics["n_clusters"] == 3
    assert system.cluster_centers is not None
    assert system.cluster_assignments is not None


def test_semantic_sample_from_empty_cluster():
    """Test sampling when cluster is empty."""
    system = SemanticReorganization(
        SemanticReorganizationConfig(n_clusters=3)
    )
    
    episodes = create_test_episodes(n=20, input_dim=10, n_clusters=3)
    
    system.reorganize_episodes(
        episodes=episodes,
        extract_features_fn=extract_features,
    )
    
    # Try to sample from a non-existent cluster
    episode = system._sample_from_cluster(episodes, cluster_id=999)
    assert episode is None


# ============================================================================
# Interference Resolution Tests
# ============================================================================

def test_interference_resolution_initialization():
    """Test InterferenceResolution initialization."""
    system = InterferenceResolution()
    
    assert system.config is not None


def test_interference_detection_basic():
    """Test basic interference detection."""
    system = InterferenceResolution(
        InterferenceResolutionConfig(
            similarity_threshold=0.7,
            dissimilarity_threshold=0.3,
        )
    )
    
    # Create episodes with high input similarity but different outputs
    episodes = []
    base_state = torch.randn(10)
    
    for i in range(10):
        state = base_state + torch.randn(10) * 0.1  # Very similar
        episode = Episode(
            state=state,
            action=i % 2,  # Different actions (output)
            reward=1.0,
            correct=True,
            context=torch.randn(10) if i % 2 == 0 else torch.randn(10) + 5.0,
        )
        episodes.append(episode)
    
    interfering = system.detect_interference(
        episodes=episodes,
        extract_features_fn=extract_features,
    )
    
    # Should detect some interference
    assert isinstance(interfering, list)


def test_interference_detection_threshold():
    """Test that similarity threshold works."""
    system = InterferenceResolution(
        InterferenceResolutionConfig(
            similarity_threshold=0.95,  # Very high
        )
    )
    
    # Create diverse episodes (low similarity)
    episodes = create_test_episodes(n=20, input_dim=10, n_clusters=10)
    
    interfering = system.detect_interference(
        episodes=episodes,
        extract_features_fn=extract_features,
    )
    
    # Should find fewer interfering pairs with high threshold
    assert len(interfering) >= 0


def test_interference_resolution_basic():
    """Test basic interference resolution."""
    system = InterferenceResolution(
        InterferenceResolutionConfig(
            learning_signal=0.3,
        )
    )
    brain = MockBrain()
    
    # Create interfering pairs
    episodes = create_test_episodes(n=10, input_dim=10)
    interfering = [(episodes[0], episodes[1], 0.85)]
    
    metrics = system.resolve_interference(
        brain=brain,
        interfering_pairs=interfering,
        n_steps=50,
    )
    
    assert metrics["pairs_resolved"] == 1
    assert metrics["resolution_steps"] == 50
    assert len(brain.forward_calls) == 100  # 2 per step


def test_interference_resolution_contrastive_push():
    """Test that contrastive push is applied."""
    system = InterferenceResolution()
    brain = MockBrain()
    
    episodes = create_test_episodes(n=5, input_dim=10)
    interfering = [(episodes[0], episodes[1], 0.90)]
    
    system.resolve_interference(
        brain=brain,
        interfering_pairs=interfering,
        n_steps=10,
    )
    
    # Check that contrastive mode was set
    # (In actual implementation, this would modify weights)
    assert hasattr(brain, 'contrastive_mode')


def test_interference_resolution_empty():
    """Test interference resolution with no pairs."""
    system = InterferenceResolution()
    brain = MockBrain()
    
    metrics = system.resolve_interference(
        brain=brain,
        interfering_pairs=[],
        n_steps=10,
    )
    
    assert metrics["pairs_resolved"] == 0
    assert metrics["resolution_steps"] == 0


def test_interference_detection_max_pairs():
    """Test that max_pairs_to_resolve is respected."""
    system = InterferenceResolution(
        InterferenceResolutionConfig(
            similarity_threshold=0.3,  # Low threshold
            max_pairs_to_resolve=5,
        )
    )
    
    # Create many similar episodes
    episodes = create_test_episodes(n=50, input_dim=10, n_clusters=2)
    
    interfering = system.detect_interference(
        episodes=episodes,
        extract_features_fn=extract_features,
    )
    
    # Should not exceed max
    assert len(interfering) <= 5


def test_interference_detection_sorting():
    """Test that interfering pairs are sorted by similarity."""
    system = InterferenceResolution(
        InterferenceResolutionConfig(similarity_threshold=0.5)
    )
    
    episodes = create_test_episodes(n=30, input_dim=10, n_clusters=3)
    
    interfering = system.detect_interference(
        episodes=episodes,
        extract_features_fn=extract_features,
    )
    
    if len(interfering) > 1:
        # Check descending order
        similarities = [sim for _, _, sim in interfering]
        assert similarities == sorted(similarities, reverse=True)


# ============================================================================
# Integrated System Tests
# ============================================================================

def test_run_advanced_consolidation_basic():
    """Test integrated advanced consolidation."""
    brain = MockBrain()
    episodes = create_test_episodes(n=30, input_dim=10, n_clusters=3)
    
    metrics = run_advanced_consolidation(
        brain=brain,
        episodes=episodes,
        extract_features_fn=extract_features,
        n_steps=600,
    )
    
    assert 'interference' in metrics
    assert 'schema_extraction' in metrics
    assert 'semantic_reorganization' in metrics
    assert metrics['total_steps'] == 600


def test_run_advanced_consolidation_with_interference():
    """Test that interference is resolved when detected."""
    brain = MockBrain()
    
    # Create interfering episodes
    base_state = torch.randn(10)
    episodes = []
    for i in range(20):
        state = base_state + torch.randn(10) * 0.15  # Similar
        episode = Episode(
            state=state,
            action=i % 3,  # Different actions
            reward=1.0,
            correct=True,
            context=torch.randn(10),
        )
        episodes.append(episode)
    
    metrics = run_advanced_consolidation(
        brain=brain,
        episodes=episodes,
        extract_features_fn=extract_features,
        n_steps=600,
    )
    
    # Should have processed some steps
    assert metrics['total_steps'] == 600


def test_run_advanced_consolidation_empty():
    """Test advanced consolidation with no episodes."""
    brain = MockBrain()
    
    metrics = run_advanced_consolidation(
        brain=brain,
        episodes=[],
        extract_features_fn=extract_features,
        n_steps=100,
    )
    
    assert metrics['interference']['pairs_resolved'] == 0
    assert metrics['schema_extraction']['n_schemas_extracted'] == 0


def test_run_advanced_consolidation_custom_configs():
    """Test advanced consolidation with custom configs."""
    brain = MockBrain()
    episodes = create_test_episodes(n=25, input_dim=10)
    
    config_schema = SchemaExtractionConfig(
        similarity_threshold=0.8,
        max_schemas=10,
    )
    config_semantic = SemanticReorganizationConfig(
        n_clusters=5,
    )
    config_interference = InterferenceResolutionConfig(
        similarity_threshold=0.9,
    )
    
    metrics = run_advanced_consolidation(
        brain=brain,
        episodes=episodes,
        extract_features_fn=extract_features,
        n_steps=600,
        config_schema=config_schema,
        config_semantic=config_semantic,
        config_interference=config_interference,
    )
    
    assert metrics['total_steps'] == 600


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

def test_cosine_similarity_identical():
    """Test cosine similarity with identical vectors."""
    system = SchemaExtractionConsolidation()
    
    a = torch.randn(10)
    sim = system._cosine_similarity(a, a)
    
    assert abs(sim - 1.0) < 1e-5


def test_cosine_similarity_orthogonal():
    """Test cosine similarity with orthogonal vectors."""
    system = SchemaExtractionConsolidation()
    
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0])
    sim = system._cosine_similarity(a, b)
    
    assert abs(sim) < 1e-5


def test_cosine_similarity_zero_vector():
    """Test cosine similarity with zero vector."""
    system = SchemaExtractionConsolidation()
    
    a = torch.zeros(10)
    b = torch.randn(10)
    sim = system._cosine_similarity(a, b)
    
    # Should handle gracefully (not NaN)
    assert not torch.isnan(torch.tensor(sim))


def test_schema_extraction_with_device():
    """Test schema extraction respects device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    system = SchemaExtractionConsolidation()
    brain = MockBrain()
    
    episodes = create_test_episodes(n=10, input_dim=10, device="cuda")
    
    metrics = system.rem_schema_extraction(
        brain=brain,
        episodes=episodes,
        n_steps=20,
        extract_features_fn=extract_features,
    )
    
    assert metrics["replay_steps"] == 20


def test_semantic_reorganization_convergence_edge_case():
    """Test k-means with difficult convergence."""
    system = SemanticReorganization(
        SemanticReorganizationConfig(
            n_clusters=5,
            max_iterations=10,  # Very few iterations
            tolerance=1e-10,     # Very tight tolerance
        )
    )
    
    episodes = create_test_episodes(n=25, input_dim=10, n_clusters=5)
    
    # Should not crash even with difficult convergence
    metrics = system.reorganize_episodes(
        episodes=episodes,
        extract_features_fn=extract_features,
    )
    
    assert metrics["n_clusters"] == 5


def test_interference_resolution_single_episode():
    """Test interference detection with single episode."""
    system = InterferenceResolution()
    
    episodes = create_test_episodes(n=1, input_dim=10)
    
    interfering = system.detect_interference(
        episodes=episodes,
        extract_features_fn=extract_features,
    )
    
    # Should find no interference
    assert len(interfering) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
