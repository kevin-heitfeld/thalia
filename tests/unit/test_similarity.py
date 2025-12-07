"""Tests for similarity computation utilities.

Tests the consolidated similarity methods:
- cosine_similarity_safe() in core.utils
- similarity_diagnostics() in DiagnosticsMixin
- compute_spike_similarity() in spike_coding
"""

import pytest
import torch
from thalia.core.utils import cosine_similarity_safe
from thalia.core.diagnostics_mixin import DiagnosticsMixin
from thalia.core.spike_coding import compute_spike_similarity


class TestCosineSimilaritySafe:
    """Test the canonical cosine similarity implementation."""
    
    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0."""
        a = torch.randn(10)
        sim = cosine_similarity_safe(a, a)
        assert torch.isclose(sim, torch.tensor(1.0), atol=1e-6)
    
    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0.0."""
        a = torch.tensor([1.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0])
        sim = cosine_similarity_safe(a, b)
        assert torch.isclose(sim, torch.tensor(0.0), atol=1e-6)
    
    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1.0."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = -a
        sim = cosine_similarity_safe(a, b)
        assert torch.isclose(sim, torch.tensor(-1.0), atol=1e-6)
    
    def test_zero_vector(self):
        """Zero vectors should not crash (use epsilon)."""
        a = torch.zeros(10)
        b = torch.randn(10)
        sim = cosine_similarity_safe(a, b)
        assert torch.isfinite(sim)
    
    def test_batched_vectors(self):
        """Should work with batched tensors."""
        a = torch.randn(5, 10)
        b = torch.randn(5, 10)
        sim = cosine_similarity_safe(a, b, dim=-1)
        assert sim.shape == (5,)
        assert torch.all(torch.isfinite(sim))
    
    def test_custom_epsilon(self):
        """Should respect custom epsilon value."""
        a = torch.zeros(10)
        b = torch.zeros(10)
        # With large epsilon, should still work
        sim = cosine_similarity_safe(a, b, eps=1.0)
        assert torch.isfinite(sim)
    
    def test_custom_dim(self):
        """Should work with custom dimension."""
        a = torch.randn(3, 4, 5)
        b = torch.randn(3, 4, 5)
        # Compute similarity along dim 1
        sim = cosine_similarity_safe(a, b, dim=1)
        assert sim.shape == (3, 5)
    
    def test_numerical_stability(self):
        """Should be numerically stable with small values."""
        a = torch.randn(10) * 1e-10
        b = torch.randn(10) * 1e-10
        sim = cosine_similarity_safe(a, b)
        assert torch.isfinite(sim)
        assert -1.0 <= sim.item() <= 1.0


class TestSimilarityDiagnostics:
    """Test the similarity_diagnostics method from DiagnosticsMixin."""
    
    @pytest.fixture
    def mixin(self):
        """Create a DiagnosticsMixin instance."""
        return DiagnosticsMixin()
    
    def test_cosine_similarity(self, mixin):
        """Should compute cosine similarity correctly."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([2.0, 4.0, 6.0])  # Parallel to a
        
        result = mixin.similarity_diagnostics(a, b)
        
        assert "cosine_similarity" in result
        assert torch.isclose(
            torch.tensor(result["cosine_similarity"]),
            torch.tensor(1.0),
            atol=1e-6
        )
    
    def test_binary_patterns(self, mixin):
        """Should compute overlap and jaccard for binary patterns."""
        a = torch.tensor([1.0, 1.0, 0.0, 0.0])
        b = torch.tensor([1.0, 0.0, 1.0, 0.0])
        
        result = mixin.similarity_diagnostics(a, b)
        
        # Overlap: 1 position (index 0)
        assert result["overlap"] == 1.0
        
        # Jaccard: intersection/union = 1/3
        expected_jaccard = 1.0 / 3.0
        assert abs(result["jaccard"] - expected_jaccard) < 1e-6
    
    def test_non_binary_patterns(self, mixin):
        """Should return zero overlap/jaccard for non-binary patterns."""
        a = torch.randn(10) + 5.0  # Values > 1
        b = torch.randn(10) + 5.0
        
        result = mixin.similarity_diagnostics(a, b)
        
        assert result["overlap"] == 0.0
        assert result["jaccard"] == 0.0
        assert "cosine_similarity" in result
    
    def test_prefix(self, mixin):
        """Should use custom prefix for metric names."""
        a = torch.randn(10)
        b = torch.randn(10)
        
        result = mixin.similarity_diagnostics(a, b, prefix="test")
        
        assert "test_cosine_similarity" in result
        assert "test_overlap" in result
        assert "test_jaccard" in result
    
    def test_custom_epsilon(self, mixin):
        """Should respect custom epsilon value."""
        a = torch.zeros(10)
        b = torch.zeros(10)
        
        result = mixin.similarity_diagnostics(a, b, eps=1.0)
        
        assert "cosine_similarity" in result
        assert torch.isfinite(torch.tensor(result["cosine_similarity"]))


class TestComputeSpikeSimilarity:
    """Test spike pattern similarity computation."""
    
    def test_cosine_method(self):
        """Should compute cosine similarity for spikes."""
        batch, seq, time, neurons = 2, 3, 10, 5
        spikes1 = torch.rand(batch, seq, time, neurons)
        spikes2 = torch.rand(batch, seq, time, neurons)
        
        sim = compute_spike_similarity(spikes1, spikes2, method="cosine")
        
        assert sim.shape == (batch, seq)
        assert torch.all(torch.isfinite(sim))
        # Cosine similarity should be in [-1, 1]
        assert torch.all(sim >= -1.0)
        assert torch.all(sim <= 1.0)
    
    def test_identical_spikes_cosine(self):
        """Identical spikes should have cosine similarity 1.0."""
        spikes = torch.rand(2, 3, 10, 5)
        sim = compute_spike_similarity(spikes, spikes, method="cosine")
        
        assert torch.allclose(sim, torch.ones_like(sim), atol=1e-5)
    
    def test_correlation_method(self):
        """Should compute correlation for spikes."""
        batch, seq, time, neurons = 2, 3, 10, 5
        spikes1 = torch.rand(batch, seq, time, neurons)
        spikes2 = torch.rand(batch, seq, time, neurons)
        
        sim = compute_spike_similarity(spikes1, spikes2, method="correlation")
        
        assert sim.shape == (batch, seq)
        assert torch.all(torch.isfinite(sim))
    
    def test_overlap_method(self):
        """Should compute Jaccard overlap for binary spikes."""
        batch, seq, time, neurons = 2, 3, 10, 5
        # Binary spikes
        spikes1 = (torch.rand(batch, seq, time, neurons) > 0.5).float()
        spikes2 = (torch.rand(batch, seq, time, neurons) > 0.5).float()
        
        sim = compute_spike_similarity(spikes1, spikes2, method="overlap")
        
        assert sim.shape == (batch, seq)
        assert torch.all(torch.isfinite(sim))
        # Jaccard should be in [0, 1]
        assert torch.all(sim >= 0.0)
        assert torch.all(sim <= 1.0)
    
    def test_invalid_method(self):
        """Should raise error for invalid method."""
        spikes = torch.rand(2, 3, 10, 5)
        
        with pytest.raises(ValueError, match="Unknown similarity method"):
            compute_spike_similarity(spikes, spikes, method="invalid")
    
    def test_zero_spikes(self):
        """Should handle all-zero spikes gracefully."""
        batch, seq, time, neurons = 2, 3, 10, 5
        spikes1 = torch.zeros(batch, seq, time, neurons)
        spikes2 = torch.rand(batch, seq, time, neurons)
        
        # All methods should not crash
        for method in ["cosine", "correlation", "overlap"]:
            sim = compute_spike_similarity(spikes1, spikes2, method=method)
            assert torch.all(torch.isfinite(sim))


class TestIntegration:
    """Test integration between different similarity methods."""
    
    def test_consistency_1d_vectors(self):
        """DiagnosticsMixin should use cosine_similarity_safe internally."""
        mixin = DiagnosticsMixin()
        a = torch.randn(20)
        b = torch.randn(20)
        
        # Compute using both methods
        direct = cosine_similarity_safe(a, b)
        via_diagnostics = mixin.similarity_diagnostics(a, b)["cosine_similarity"]
        
        # Should be identical
        assert abs(direct.item() - via_diagnostics) < 1e-6
    
    def test_consistency_spike_patterns(self):
        """compute_spike_similarity should match cosine_similarity_safe for cosine method."""
        batch, seq, time, neurons = 4, 5, 8, 10
        spikes1 = torch.rand(batch, seq, time, neurons)
        spikes2 = torch.rand(batch, seq, time, neurons)
        
        # Via spike similarity
        spike_sim = compute_spike_similarity(spikes1, spikes2, method="cosine")
        
        # Direct computation
        flat1 = spikes1.reshape(batch, seq, -1)
        flat2 = spikes2.reshape(batch, seq, -1)
        direct_sim = cosine_similarity_safe(flat1, flat2, eps=1e-6, dim=-1)
        
        # Should match
        assert torch.allclose(spike_sim, direct_sim, atol=1e-5)
    
    def test_all_methods_with_same_input(self):
        """Test that all three implementations work with compatible inputs."""
        # Create test vectors
        a = torch.randn(20)
        b = torch.randn(20)
        
        # Test cosine_similarity_safe
        sim1 = cosine_similarity_safe(a, b)
        assert torch.isfinite(sim1)
        
        # Test via DiagnosticsMixin
        mixin = DiagnosticsMixin()
        result = mixin.similarity_diagnostics(a, b)
        sim2 = result["cosine_similarity"]
        assert abs(sim1.item() - sim2) < 1e-6
        
        # Test via spike similarity (reshape to spike format)
        spikes_a = a.view(1, 1, 4, 5)  # batch=1, seq=1, time=4, neurons=5
        spikes_b = b.view(1, 1, 4, 5)
        sim3 = compute_spike_similarity(spikes_a, spikes_b, method="cosine")
        assert torch.isclose(sim1, sim3.squeeze(), atol=1e-5)
