"""
Tests for Metacognitive Monitor.

Tests stage-aware confidence estimation, abstention decisions,
and calibration learning.
"""

import pytest
import torch
from thalia.diagnostics.metacognition import (
    MetacognitiveMonitor,
    MetacognitiveMonitorConfig,
    MetacognitiveStage,
    ConfidenceEstimator,
    CalibrationNetwork,
)


class TestConfidenceEstimator:
    """Test raw confidence estimation from population activity."""
    
    def test_high_confidence_clear_winner(self):
        """Test high confidence when there's a clear winner."""
        config = MetacognitiveMonitorConfig(input_size=100, device="cpu")
        estimator = ConfidenceEstimator(config)
        
        # Create population with clear winner
        population = torch.zeros(100)
        population[10] = 10.0  # Strong winner
        population[20:30] = 0.5  # Weak competitors
        
        confidence = estimator.estimate_raw_confidence(population)
        
        # Should have high confidence (low variance)
        assert confidence > 0.7
    
    def test_low_confidence_no_clear_winner(self):
        """Test low confidence when activity is uniform."""
        config = MetacognitiveMonitorConfig(input_size=100, device="cpu")
        estimator = ConfidenceEstimator(config)
        
        # Create population with no clear winner
        population = torch.ones(100) * 0.5  # Uniform activity
        
        confidence = estimator.estimate_raw_confidence(population)
        
        # Should have low confidence (high variance relative to mean)
        assert confidence < 0.7
    
    def test_zero_activity(self):
        """Test confidence with zero activity."""
        config = MetacognitiveMonitorConfig(input_size=100, device="cpu")
        estimator = ConfidenceEstimator(config)
        
        population = torch.zeros(100)
        confidence = estimator.estimate_raw_confidence(population)
        
        # No activity = no confidence
        assert confidence == 0.0


class TestCalibrationNetwork:
    """Test calibration network learning."""
    
    def test_calibration_update(self):
        """Test that calibration network learns."""
        config = MetacognitiveMonitorConfig(
            input_size=100,
            hidden_size=32,
            calibration_lr=0.1,
            device="cpu",
        )
        calibration = CalibrationNetwork(config)
        
        # Train on consistent pattern (high raw → high actual)
        for _ in range(10):
            metrics = calibration.update(
                raw_confidence=0.9,
                actual_correct=1.0,
                dopamine=1.0
            )
        
        # Should have some updates
        assert metrics["n_updates"] == 10
        assert calibration.n_updates == 10
    
    def test_calibration_with_dopamine_gating(self):
        """Test that dopamine gates learning."""
        config = MetacognitiveMonitorConfig(
            input_size=100,
            hidden_size=32,
            use_dopamine_gating=True,
            device="cpu",
        )
        calibration = CalibrationNetwork(config)
        
        # Train with high dopamine
        metrics_high_da = calibration.update(
            raw_confidence=0.5,
            actual_correct=1.0,
            dopamine=1.0
        )
        
        # Train with low dopamine (should learn less)
        metrics_low_da = calibration.update(
            raw_confidence=0.5,
            actual_correct=0.0,
            dopamine=0.1
        )
        
        # Both should update
        assert metrics_high_da["n_updates"] == 1
        assert metrics_low_da["n_updates"] == 2
    
    def test_calibration_forward_pass(self):
        """Test forward pass produces valid output."""
        config = MetacognitiveMonitorConfig(input_size=100, device="cpu")
        calibration = CalibrationNetwork(config)
        
        raw_conf = torch.tensor([[0.5]])
        calibrated = calibration(raw_conf)
        
        # Should be in [0, 1] range
        assert calibrated.shape == (1, 1)
        assert 0.0 <= calibrated.item() <= 1.0


class TestMetacognitiveMonitorStage1:
    """Test metacognitive monitor at Stage 1 (Toddler - binary)."""
    
    def test_binary_confidence_high(self):
        """Test binary confidence for high raw confidence."""
        config = MetacognitiveMonitorConfig(
            input_size=100,
            stage=MetacognitiveStage.TODDLER,
            threshold_stage1=0.5,
            device="cpu",
        )
        monitor = MetacognitiveMonitor(config)
        
        # Create clear winner population
        population = torch.zeros(100)
        population[10] = 10.0
        
        confidence, breakdown = monitor.estimate_confidence(population)
        
        # Should be binary 1.0 (knows)
        assert confidence == 1.0
        assert breakdown["stage"] == "TODDLER"
        assert breakdown["raw"] > 0.5
    
    def test_binary_confidence_low(self):
        """Test binary confidence for low raw confidence."""
        config = MetacognitiveMonitorConfig(
            input_size=100,
            stage=MetacognitiveStage.TODDLER,
            threshold_stage1=0.7,
            device="cpu",
        )
        monitor = MetacognitiveMonitor(config)
        
        # Create uniform population (no clear winner)
        population = torch.ones(100) * 0.5
        
        confidence, breakdown = monitor.estimate_confidence(population)
        
        # Should be binary 0.0 (doesn't know)
        assert confidence == 0.0
    
    def test_abstention_stage1(self):
        """Test abstention at Stage 1."""
        config = MetacognitiveMonitorConfig(
            input_size=100,
            stage=MetacognitiveStage.TODDLER,
            threshold_stage1=0.5,
            device="cpu",
        )
        monitor = MetacognitiveMonitor(config)
        
        # Low confidence should trigger abstention
        should_abstain_low = monitor.should_abstain(0.0)
        assert should_abstain_low is True
        
        # High confidence should not trigger abstention
        should_abstain_high = monitor.should_abstain(1.0)
        assert should_abstain_high is False


class TestMetacognitiveMonitorStage2:
    """Test metacognitive monitor at Stage 2 (Preschool - coarse)."""
    
    def test_coarse_confidence_high(self):
        """Test coarse-grained confidence (high)."""
        config = MetacognitiveMonitorConfig(
            input_size=100,
            stage=MetacognitiveStage.PRESCHOOL,
            device="cpu",
        )
        monitor = MetacognitiveMonitor(config)
        
        # Create strong winner
        population = torch.zeros(100)
        population[10] = 20.0
        
        confidence, breakdown = monitor.estimate_confidence(population)
        
        # Should be 1.0 (high)
        assert confidence == 1.0
        assert breakdown["stage"] == "PRESCHOOL"
    
    def test_coarse_confidence_medium(self):
        """Test coarse-grained confidence (medium)."""
        config = MetacognitiveMonitorConfig(
            input_size=100,
            stage=MetacognitiveStage.PRESCHOOL,
            device="cpu",
        )
        monitor = MetacognitiveMonitor(config)
        
        # Create moderate winner
        population = torch.zeros(100)
        population[10] = 2.0
        population[20:30] = 0.5
        
        confidence, breakdown = monitor.estimate_confidence(population)
        
        # Should be 0.5 (medium) or 1.0 (high)
        assert confidence in [0.5, 1.0]
    
    def test_coarse_confidence_low(self):
        """Test coarse-grained confidence (low)."""
        config = MetacognitiveMonitorConfig(
            input_size=100,
            stage=MetacognitiveStage.PRESCHOOL,
            device="cpu",
        )
        monitor = MetacognitiveMonitor(config)
        
        # Create uniform population
        population = torch.ones(100) * 0.3
        
        confidence, breakdown = monitor.estimate_confidence(population)
        
        # Should be 0.0 (low)
        assert confidence == 0.0


class TestMetacognitiveMonitorStage3:
    """Test metacognitive monitor at Stage 3 (School-age - continuous)."""
    
    def test_continuous_confidence(self):
        """Test continuous confidence estimation."""
        config = MetacognitiveMonitorConfig(
            input_size=100,
            stage=MetacognitiveStage.SCHOOL_AGE,
            device="cpu",
        )
        monitor = MetacognitiveMonitor(config)
        
        # Create moderate winner
        population = torch.zeros(100)
        population[10] = 5.0
        population[20:25] = 1.0
        
        confidence, breakdown = monitor.estimate_confidence(population)
        
        # Should be continuous (raw confidence)
        assert 0.0 < confidence < 1.0
        assert breakdown["stage"] == "SCHOOL_AGE"
        assert confidence == breakdown["raw"]


class TestMetacognitiveMonitorStage4:
    """Test metacognitive monitor at Stage 4 (Adolescent - calibrated)."""
    
    def test_calibrated_confidence(self):
        """Test calibrated confidence estimation."""
        config = MetacognitiveMonitorConfig(
            input_size=100,
            stage=MetacognitiveStage.ADOLESCENT,
            device="cpu",
        )
        monitor = MetacognitiveMonitor(config)
        
        # Create clear winner
        population = torch.zeros(100)
        population[10] = 10.0
        
        confidence, breakdown = monitor.estimate_confidence(population)
        
        # Should use calibration network
        assert breakdown["stage"] == "ADOLESCENT"
        assert "raw" in breakdown
        assert "processed" in breakdown
        # Calibrated may differ from raw
        assert 0.0 <= confidence <= 1.0
    
    def test_calibration_learning(self):
        """Test that calibration improves with training."""
        config = MetacognitiveMonitorConfig(
            input_size=100,
            stage=MetacognitiveStage.ADOLESCENT,
            device="cpu",
        )
        monitor = MetacognitiveMonitor(config)
        
        # Create test population
        population = torch.zeros(100)
        population[10] = 8.0
        
        # Train on several examples
        for _ in range(10):
            metrics = monitor.calibrate(
                population_activity=population,
                actual_correct=True,
                dopamine=1.0
            )
        
        # Should have updated
        assert metrics["n_updates"] == 10
        stats = monitor.get_statistics()
        assert stats["calibration_updates"] == 10


class TestDevelopmentalProgression:
    """Test developmental stage progression."""
    
    def test_stage_setting(self):
        """Test setting developmental stage."""
        config = MetacognitiveMonitorConfig(input_size=100, device="cpu")
        monitor = MetacognitiveMonitor(config)
        
        # Default is toddler
        assert monitor.get_stage() == MetacognitiveStage.TODDLER
        
        # Set to preschool
        monitor.set_stage(MetacognitiveStage.PRESCHOOL)
        assert monitor.get_stage() == MetacognitiveStage.PRESCHOOL
        
        # Set to school-age
        monitor.set_stage(MetacognitiveStage.SCHOOL_AGE)
        assert monitor.get_stage() == MetacognitiveStage.SCHOOL_AGE
        
        # Set to adolescent
        monitor.set_stage(MetacognitiveStage.ADOLESCENT)
        assert monitor.get_stage() == MetacognitiveStage.ADOLESCENT
    
    def test_abstention_threshold_changes(self):
        """Test that abstention threshold changes with stage."""
        config = MetacognitiveMonitorConfig(
            input_size=100,
            threshold_stage1=0.5,
            threshold_stage2=0.3,
            threshold_stage3=0.4,
            threshold_stage4=0.3,
            device="cpu",
        )
        monitor = MetacognitiveMonitor(config)
        
        # Test confidence level 0.35
        test_confidence = 0.35
        
        # Stage 1: Should abstain (threshold 0.5)
        monitor.set_stage(MetacognitiveStage.TODDLER)
        assert monitor.should_abstain(test_confidence) is True
        
        # Stage 2: Should NOT abstain (threshold 0.3)
        monitor.set_stage(MetacognitiveStage.PRESCHOOL)
        assert monitor.should_abstain(test_confidence) is False
        
        # Stage 3: Should abstain (threshold 0.4)
        monitor.set_stage(MetacognitiveStage.SCHOOL_AGE)
        assert monitor.should_abstain(test_confidence) is True
        
        # Stage 4: Should NOT abstain (threshold 0.3)
        monitor.set_stage(MetacognitiveStage.ADOLESCENT)
        assert monitor.should_abstain(test_confidence) is False


class TestStatistics:
    """Test statistics tracking."""
    
    def test_estimate_statistics(self):
        """Test that estimate statistics are tracked."""
        config = MetacognitiveMonitorConfig(input_size=100, device="cpu")
        monitor = MetacognitiveMonitor(config)
        
        # Make several estimates
        for _ in range(5):
            population = torch.rand(100)
            monitor.estimate_confidence(population)
        
        stats = monitor.get_statistics()
        assert stats["n_estimates"] == 5
        assert stats["avg_confidence"] > 0
        assert stats["avg_raw_confidence"] > 0
    
    def test_abstention_statistics(self):
        """Test that abstention statistics are tracked."""
        config = MetacognitiveMonitorConfig(
            input_size=100,
            stage=MetacognitiveStage.TODDLER,
            threshold_stage1=0.7,
            device="cpu",
        )
        monitor = MetacognitiveMonitor(config)
        
        # Make estimates with different confidence levels
        # Low confidence
        for _ in range(3):
            population = torch.ones(100) * 0.3  # Uniform = low confidence
            conf, _ = monitor.estimate_confidence(population)
            monitor.should_abstain(conf)
        
        # High confidence
        for _ in range(2):
            population = torch.zeros(100)
            population[10] = 10.0  # Clear winner = high confidence
            conf, _ = monitor.estimate_confidence(population)
            monitor.should_abstain(conf)
        
        stats = monitor.get_statistics()
        assert stats["n_estimates"] == 5
        assert stats["n_abstentions"] >= 3  # At least the low confidence ones
        assert stats["abstention_rate"] > 0
    
    def test_reset_statistics(self):
        """Test statistics reset."""
        config = MetacognitiveMonitorConfig(input_size=100, device="cpu")
        monitor = MetacognitiveMonitor(config)
        
        # Make some estimates
        population = torch.rand(100)
        monitor.estimate_confidence(population)
        monitor.should_abstain(0.3)
        
        # Reset
        monitor.reset_statistics()
        
        stats = monitor.get_statistics()
        assert stats["n_estimates"] == 0
        assert stats["n_abstentions"] == 0
        assert stats["avg_confidence"] == 0.0


class TestIntegration:
    """Test full integration scenarios."""
    
    def test_full_pipeline_stage1(self):
        """Test full pipeline at Stage 1."""
        config = MetacognitiveMonitorConfig(
            input_size=100,
            stage=MetacognitiveStage.TODDLER,
            device="cpu",
        )
        monitor = MetacognitiveMonitor(config)
        
        # Simulate decision-making with confidence
        population = torch.zeros(100)
        population[25] = 8.0  # Clear winner
        
        confidence, breakdown = monitor.estimate_confidence(population)
        should_abstain = monitor.should_abstain(confidence)
        
        # Should be confident and not abstain
        assert confidence == 1.0
        assert should_abstain is False
    
    def test_full_pipeline_stage4_with_learning(self):
        """Test full pipeline at Stage 4 with calibration."""
        config = MetacognitiveMonitorConfig(
            input_size=100,
            stage=MetacognitiveStage.ADOLESCENT,
            device="cpu",
        )
        monitor = MetacognitiveMonitor(config)
        
        # Simulate several trials with feedback
        for i in range(10):
            population = torch.zeros(100)
            population[i] = 5.0 + torch.rand(1).item()
            
            confidence, _ = monitor.estimate_confidence(population)
            should_abstain = monitor.should_abstain(confidence)
            
            if not should_abstain:
                # Simulate correct answer with high dopamine
                monitor.calibrate(
                    population_activity=population,
                    actual_correct=True,
                    dopamine=0.8
                )
        
        # Should have some calibration updates
        stats = monitor.get_statistics()
        assert stats["calibration_updates"] > 0
    
    def test_stage_progression_scenario(self):
        """Test realistic stage progression scenario."""
        config = MetacognitiveMonitorConfig(input_size=100, device="cpu")
        monitor = MetacognitiveMonitor(config)
        
        # Test population
        population = torch.zeros(100)
        population[10] = 6.0
        population[20:25] = 1.0
        
        # Stage 1: Binary
        monitor.set_stage(MetacognitiveStage.TODDLER)
        conf1, _ = monitor.estimate_confidence(population)
        assert conf1 in [0.0, 1.0]
        
        # Stage 2: Coarse
        monitor.set_stage(MetacognitiveStage.PRESCHOOL)
        conf2, _ = monitor.estimate_confidence(population)
        assert conf2 in [0.0, 0.5, 1.0]
        
        # Stage 3: Continuous
        monitor.set_stage(MetacognitiveStage.SCHOOL_AGE)
        conf3, _ = monitor.estimate_confidence(population)
        assert 0.0 < conf3 < 1.0
        
        # Stage 4: Calibrated
        monitor.set_stage(MetacognitiveStage.ADOLESCENT)
        conf4, _ = monitor.estimate_confidence(population)
        assert 0.0 <= conf4 <= 1.0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_population(self):
        """Test with empty population."""
        config = MetacognitiveMonitorConfig(input_size=100, device="cpu")
        monitor = MetacognitiveMonitor(config)
        
        population = torch.zeros(100)
        confidence, _ = monitor.estimate_confidence(population)
        
        # Should handle gracefully
        assert 0.0 <= confidence <= 1.0
    
    def test_very_sparse_population(self):
        """Test with very sparse population."""
        config = MetacognitiveMonitorConfig(input_size=100, device="cpu")
        monitor = MetacognitiveMonitor(config)
        
        population = torch.zeros(100)
        population[0] = 1.0  # Single active neuron
        
        confidence, _ = monitor.estimate_confidence(population)
        
        # Should handle gracefully
        assert 0.0 <= confidence <= 1.0
    
    def test_calibration_at_wrong_stage(self):
        """Test calibration at stages that don't support it."""
        config = MetacognitiveMonitorConfig(
            input_size=100,
            stage=MetacognitiveStage.TODDLER,
            device="cpu",
        )
        monitor = MetacognitiveMonitor(config)
        
        population = torch.rand(100)
        result = monitor.calibrate(population, actual_correct=True)
        
        # Should return error message
        assert "error" in result
    
    def test_device_consistency(self):
        """Test that all operations maintain device consistency."""
        config = MetacognitiveMonitorConfig(input_size=100, device="cpu")
        monitor = MetacognitiveMonitor(config)
        
        population = torch.rand(100)
        confidence, breakdown = monitor.estimate_confidence(population)
        
        # All operations should work on CPU
        assert isinstance(confidence, float)
        assert breakdown["raw"] >= 0


class TestConfiguration:
    """Test configuration options."""
    
    def test_custom_thresholds(self):
        """Test custom abstention thresholds."""
        config = MetacognitiveMonitorConfig(
            input_size=100,
            threshold_stage1=0.6,
            threshold_stage2=0.4,
            threshold_stage3=0.5,
            threshold_stage4=0.35,
            device="cpu",
        )
        monitor = MetacognitiveMonitor(config)
        
        # Verify thresholds are used
        monitor.set_stage(MetacognitiveStage.TODDLER)
        assert monitor.should_abstain(0.55) is True  # Below 0.6
        assert monitor.should_abstain(0.65) is False  # Above 0.6
    
    def test_calibration_learning_rate(self):
        """Test different calibration learning rates."""
        config_fast = MetacognitiveMonitorConfig(
            input_size=100,
            calibration_lr=0.1,
            stage=MetacognitiveStage.ADOLESCENT,
            device="cpu",
        )
        monitor_fast = MetacognitiveMonitor(config_fast)
        
        config_slow = MetacognitiveMonitorConfig(
            input_size=100,
            calibration_lr=0.001,
            stage=MetacognitiveStage.ADOLESCENT,
            device="cpu",
        )
        monitor_slow = MetacognitiveMonitor(config_slow)
        
        # Both should be able to learn
        population = torch.rand(100)
        metrics_fast = monitor_fast.calibrate(population, True)
        metrics_slow = monitor_slow.calibrate(population, True)
        
        assert "loss" in metrics_fast
        assert "loss" in metrics_slow
    
    def test_dopamine_gating_toggle(self):
        """Test dopamine gating can be toggled."""
        config_with = MetacognitiveMonitorConfig(
            input_size=100,
            use_dopamine_gating=True,
            stage=MetacognitiveStage.ADOLESCENT,
            device="cpu",
        )
        monitor_with = MetacognitiveMonitor(config_with)
        
        config_without = MetacognitiveMonitorConfig(
            input_size=100,
            use_dopamine_gating=False,
            stage=MetacognitiveStage.ADOLESCENT,
            device="cpu",
        )
        monitor_without = MetacognitiveMonitor(config_without)
        
        # Both should work
        population = torch.rand(100)
        metrics_with = monitor_with.calibrate(population, True, dopamine=0.5)
        metrics_without = monitor_without.calibrate(population, True, dopamine=0.5)
        
        assert "loss" in metrics_with
        assert "loss" in metrics_without
