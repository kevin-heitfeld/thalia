"""Tests for evaluation and diagnostics module."""

import pytest
import torch
import numpy as np

from thalia.evaluation.weights import (
    compute_diagonal_score,
    analyze_recurrent_structure,
    compute_weight_statistics,
    RecurrentAnalysis,
    WeightStatistics,
)


class TestComputeDiagonalScore:
    """Tests for compute_diagonal_score function."""

    def test_perfect_diagonal(self):
        """Test perfect diagonal mapping gives full score."""
        # Create weight matrix where output i has max at input i*2
        # With linear mapping, expected inputs are [0, 2, 5, 7, 9] for n=5, n_input=10
        weights = torch.zeros(5, 10)
        expected = np.linspace(0, 9, 5).round().astype(int)  # [0, 2, 5, 7, 9]
        for i in range(5):
            weights[i, expected[i]] = 1.0
        
        correct, total, mapping = compute_diagonal_score(weights)
        
        assert correct == total
        assert total == 5
        assert mapping == expected.tolist()

    def test_random_weights(self):
        """Test that random weights give low diagonal score."""
        torch.manual_seed(42)
        weights = torch.rand(10, 20)

        correct, total, mapping = compute_diagonal_score(weights)

        # Random should have low score (not all correct)
        assert correct < total
        assert total == 10

    def test_custom_expected_mapping(self):
        """Test with custom expected mapping."""
        weights = torch.zeros(3, 6)
        weights[0, 1] = 1.0
        weights[1, 3] = 1.0
        weights[2, 5] = 1.0

        expected = np.array([1, 3, 5])
        correct, total, mapping = compute_diagonal_score(weights, expected)

        assert correct == 3
        assert mapping == [1, 3, 5]


class TestAnalyzeRecurrentStructure:
    """Tests for analyze_recurrent_structure function."""

    def test_perfect_chain_gapped(self):
        """Test perfect chain detection for gapped pattern."""
        # Create weight matrix with i→i+1 connections
        weights = torch.zeros(5, 5)
        for i in range(4):
            weights[i, i + 1] = 1.0

        analysis = analyze_recurrent_structure(weights, pattern_type="gapped")

        assert analysis.correct_count == 4  # First 4 neurons correct
        assert analysis.pattern_type == "gapped"

    def test_perfect_chain_circular(self):
        """Test perfect chain detection for circular pattern."""
        weights = torch.zeros(5, 5)
        for i in range(5):
            weights[i, (i + 1) % 5] = 1.0

        analysis = analyze_recurrent_structure(weights, pattern_type="circular")

        assert analysis.correct_count == 5
        assert analysis.pattern_type == "circular"

    def test_broken_chain(self):
        """Test detection of broken chain."""
        weights = torch.zeros(5, 5)
        weights[0, 1] = 1.0  # 0→1 correct
        weights[1, 3] = 1.0  # 1→3 wrong (should be 2)
        weights[2, 3] = 1.0  # 2→3 correct

        analysis = analyze_recurrent_structure(weights, pattern_type="gapped",
                                               n_analyze=3)

        # Only 0→1 and 2→3 are correct
        assert analysis.correct_count == 2

    def test_analysis_connections(self):
        """Test that connection details are recorded."""
        weights = torch.zeros(3, 3)
        weights[0, 1] = 0.8
        weights[1, 2] = 0.5

        analysis = analyze_recurrent_structure(weights, n_analyze=2)

        assert len(analysis.connections) == 2
        assert analysis.connections[0]["source"] == 0
        assert analysis.connections[0]["target"] == 1
        assert abs(analysis.connections[0]["weight"] - 0.8) < 0.01


class TestComputeWeightStatistics:
    """Tests for compute_weight_statistics function."""

    def test_basic_statistics(self):
        """Test basic statistical computation."""
        weights = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        stats = compute_weight_statistics(weights)

        assert stats.min == 1.0
        assert stats.max == 4.0
        assert stats.mean == 2.5

    def test_sparsity_calculation(self):
        """Test sparsity is calculated correctly."""
        # 2 out of 4 weights are below 0.01
        weights = torch.tensor([[0.001, 0.002], [0.5, 0.8]])

        stats = compute_weight_statistics(weights, sparsity_threshold=0.01)

        assert stats.sparsity == 0.5  # 2/4 are sparse

    def test_statistics_str(self):
        """Test string representation."""
        weights = torch.ones(3, 3)
        stats = compute_weight_statistics(weights)

        s = str(stats)
        assert "WeightStats" in s
        assert "min" in s
        assert "sparsity" in s


class TestRecurrentAnalysis:
    """Tests for RecurrentAnalysis dataclass."""

    def test_accuracy_property(self):
        """Test accuracy calculation."""
        analysis = RecurrentAnalysis(
            pattern_type="gapped",
            correct_count=4,
            total_analyzed=5,
            connections=[],
        )

        assert analysis.accuracy == 0.8

    def test_zero_total_accuracy(self):
        """Test accuracy with zero total."""
        analysis = RecurrentAnalysis(
            pattern_type="gapped",
            correct_count=0,
            total_analyzed=0,
            connections=[],
        )

        assert analysis.accuracy == 0.0


class TestWeightStatistics:
    """Tests for WeightStatistics dataclass."""

    def test_str_method(self):
        """Test string representation."""
        stats = WeightStatistics(
            min=0.0, max=1.0, mean=0.5, std=0.2, sparsity=0.1
        )

        s = str(stats)
        assert "0.5" in s  # mean
        assert "10.00%" in s  # sparsity as percentage
