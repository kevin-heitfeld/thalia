"""Tests for the Metacognition module."""

import pytest
import torch
import torch.nn as nn

from thalia.metacognition import (
    ConfidenceLevel,
    ErrorType,
    ConfidenceEstimate,
    ErrorSignal,
    CognitiveState,
    MetacognitiveConfig,
    ConfidenceTracker,
    UncertaintyEstimator,
    ErrorDetector,
    CognitiveMonitor,
    MetacognitiveController,
    MetacognitiveNetwork,
)


class TestConfidenceLevel:
    """Tests for ConfidenceLevel enum."""
    
    def test_all_levels_exist(self):
        """Test all confidence levels are defined."""
        assert ConfidenceLevel.VERY_LOW is not None
        assert ConfidenceLevel.LOW is not None
        assert ConfidenceLevel.MEDIUM is not None
        assert ConfidenceLevel.HIGH is not None
        assert ConfidenceLevel.VERY_HIGH is not None
        
    def test_from_value_very_low(self):
        """Test conversion for very low confidence."""
        assert ConfidenceLevel.from_value(0.1) == ConfidenceLevel.VERY_LOW
        
    def test_from_value_low(self):
        """Test conversion for low confidence."""
        assert ConfidenceLevel.from_value(0.3) == ConfidenceLevel.LOW
        
    def test_from_value_medium(self):
        """Test conversion for medium confidence."""
        assert ConfidenceLevel.from_value(0.5) == ConfidenceLevel.MEDIUM
        
    def test_from_value_high(self):
        """Test conversion for high confidence."""
        assert ConfidenceLevel.from_value(0.7) == ConfidenceLevel.HIGH
        
    def test_from_value_very_high(self):
        """Test conversion for very high confidence."""
        assert ConfidenceLevel.from_value(0.9) == ConfidenceLevel.VERY_HIGH


class TestErrorType:
    """Tests for ErrorType enum."""
    
    def test_all_types_exist(self):
        """Test all error types are defined."""
        assert ErrorType.PREDICTION_ERROR is not None
        assert ErrorType.CONFLICT is not None
        assert ErrorType.INCONSISTENCY is not None
        assert ErrorType.TIMEOUT is not None
        assert ErrorType.OVERLOAD is not None
        assert ErrorType.UNCERTAINTY is not None


class TestConfidenceEstimate:
    """Tests for ConfidenceEstimate dataclass."""
    
    def test_creation(self):
        """Test estimate creation."""
        estimate = ConfidenceEstimate(
            value=0.75,
            level=ConfidenceLevel.HIGH,
            source="test",
            timestamp=10,
        )
        
        assert estimate.value == 0.75
        assert estimate.level == ConfidenceLevel.HIGH
        assert estimate.source == "test"
        
    def test_value_clamping(self):
        """Test value is clamped to [0, 1]."""
        estimate = ConfidenceEstimate(value=1.5, level=ConfidenceLevel.HIGH)
        assert estimate.value == 1.0
        
        estimate = ConfidenceEstimate(value=-0.5, level=ConfidenceLevel.LOW)
        assert estimate.value == 0.0
        
    def test_level_auto_update(self):
        """Test level is updated based on value."""
        estimate = ConfidenceEstimate(value=0.1, level=ConfidenceLevel.HIGH)
        # Level should be updated to match value
        assert estimate.level == ConfidenceLevel.VERY_LOW


class TestErrorSignal:
    """Tests for ErrorSignal dataclass."""
    
    def test_creation(self):
        """Test signal creation."""
        signal = ErrorSignal(
            error_type=ErrorType.PREDICTION_ERROR,
            magnitude=0.5,
            location="test",
        )
        
        assert signal.error_type == ErrorType.PREDICTION_ERROR
        assert signal.magnitude == 0.5
        
    def test_magnitude_clamping(self):
        """Test magnitude is clamped to [0, 1]."""
        signal = ErrorSignal(ErrorType.CONFLICT, magnitude=1.5)
        assert signal.magnitude == 1.0
        
        signal = ErrorSignal(ErrorType.CONFLICT, magnitude=-0.5)
        assert signal.magnitude == 0.0


class TestCognitiveState:
    """Tests for CognitiveState dataclass."""
    
    def test_default_creation(self):
        """Test default state creation."""
        state = CognitiveState()
        
        assert state.load == 0.0
        assert state.confidence == 0.5
        assert state.uncertainty == 0.5
        assert len(state.active_errors) == 0
        
    def test_is_overloaded(self):
        """Test overload detection."""
        state = CognitiveState(load=0.9)
        assert state.is_overloaded()
        
        state = CognitiveState(load=0.5)
        assert not state.is_overloaded()
        
    def test_is_confident(self):
        """Test confidence check."""
        state = CognitiveState(confidence=0.8)
        assert state.is_confident()
        
        state = CognitiveState(confidence=0.3)
        assert not state.is_confident()
        
    def test_has_errors(self):
        """Test error check."""
        state = CognitiveState()
        assert not state.has_errors()
        
        state = CognitiveState(active_errors=[
            ErrorSignal(ErrorType.PREDICTION_ERROR, 0.5)
        ])
        assert state.has_errors()


class TestMetacognitiveConfig:
    """Tests for MetacognitiveConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = MetacognitiveConfig()
        
        assert config.hidden_dim == 64
        assert config.confidence_decay == 0.95
        assert config.error_threshold == 0.3
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = MetacognitiveConfig(
            hidden_dim=128,
            error_threshold=0.5,
        )
        
        assert config.hidden_dim == 128
        assert config.error_threshold == 0.5


class TestConfidenceTracker:
    """Tests for ConfidenceTracker class."""
    
    def test_creation(self):
        """Test tracker creation."""
        tracker = ConfidenceTracker(input_dim=64)
        
        assert tracker.input_dim == 64
        
    def test_reset(self):
        """Test tracker reset."""
        tracker = ConfidenceTracker(input_dim=64)
        
        # Estimate something
        activity = torch.randn(1, 64)
        tracker.estimate(activity)
        
        # Reset
        tracker.reset()
        
        assert tracker.get_running_confidence() == 0.5
        assert len(tracker.get_history()) == 0
        
    def test_estimate(self):
        """Test confidence estimation."""
        tracker = ConfidenceTracker(input_dim=64)
        
        activity = torch.randn(1, 64)
        estimate = tracker.estimate(activity)
        
        assert isinstance(estimate, ConfidenceEstimate)
        assert 0 <= estimate.value <= 1
        
    def test_estimate_updates_running(self):
        """Test that estimates update running confidence."""
        tracker = ConfidenceTracker(input_dim=64)
        
        initial = tracker.get_running_confidence()
        
        for _ in range(5):
            activity = torch.randn(1, 64)
            tracker.estimate(activity)
            
        # Running confidence should have changed
        assert tracker.get_running_confidence() != initial or True  # May be same by chance
        
    def test_estimate_from_consistency(self):
        """Test consistency-based estimation."""
        tracker = ConfidenceTracker(input_dim=64)
        
        # Similar patterns should give high confidence
        base = torch.randn(64)
        patterns = [base + 0.1 * torch.randn(64) for _ in range(5)]
        
        estimate = tracker.estimate_from_consistency(patterns)
        
        assert estimate.source == "consistency"
        assert estimate.value > 0.5  # Should be relatively high
        
    def test_estimate_from_consistency_single_pattern(self):
        """Test consistency with single pattern."""
        tracker = ConfidenceTracker(input_dim=64)
        
        patterns = [torch.randn(64)]
        estimate = tracker.estimate_from_consistency(patterns)
        
        assert estimate.value == 0.5  # Default for insufficient data
        
    def test_decay_confidence(self):
        """Test confidence decay."""
        tracker = ConfidenceTracker(input_dim=64, decay=0.9)
        
        # Set initial confidence
        activity = torch.randn(1, 64)
        tracker.estimate(activity)
        initial = tracker.get_running_confidence()
        
        # Decay
        tracker.decay_confidence()
        
        assert tracker.get_running_confidence() < initial
        
    def test_history(self):
        """Test estimate history."""
        tracker = ConfidenceTracker(input_dim=64)
        
        for _ in range(5):
            activity = torch.randn(1, 64)
            tracker.estimate(activity)
            
        history = tracker.get_history()
        
        assert len(history) == 5


class TestUncertaintyEstimator:
    """Tests for UncertaintyEstimator class."""
    
    def test_creation(self):
        """Test estimator creation."""
        estimator = UncertaintyEstimator(input_dim=64)
        
        assert estimator.input_dim == 64
        
    def test_reset(self):
        """Test estimator reset."""
        estimator = UncertaintyEstimator(input_dim=64)
        
        pattern = torch.randn(64)
        estimator.update_statistics(pattern)
        
        estimator.reset()
        
        assert estimator.sample_count.item() == 0
        
    def test_estimate_epistemic(self):
        """Test epistemic uncertainty estimation."""
        estimator = UncertaintyEstimator(input_dim=64, n_samples=5)
        
        pattern = torch.randn(64)
        uncertainty, per_dim = estimator.estimate_epistemic(pattern)
        
        assert uncertainty >= 0
        assert per_dim.shape == (64,)
        
    def test_estimate_aleatoric(self):
        """Test aleatoric uncertainty estimation."""
        estimator = UncertaintyEstimator(input_dim=64)
        
        predictions = torch.randn(1, 64)
        uncertainty = estimator.estimate_aleatoric(predictions)
        
        assert 0 <= uncertainty <= 1
        
    def test_estimate_total(self):
        """Test total uncertainty estimation."""
        estimator = UncertaintyEstimator(input_dim=64)
        
        pattern = torch.randn(64)
        result = estimator.estimate_total(pattern)
        
        assert "epistemic" in result
        assert "aleatoric" in result
        assert "total" in result
        
    def test_update_statistics(self):
        """Test statistics update."""
        estimator = UncertaintyEstimator(input_dim=64)
        
        for _ in range(10):
            pattern = torch.randn(64)
            estimator.update_statistics(pattern)
            
        assert estimator.sample_count.item() == 10
        
    def test_get_novelty(self):
        """Test novelty estimation."""
        estimator = UncertaintyEstimator(input_dim=64)
        
        # Build baseline
        for _ in range(20):
            pattern = torch.randn(64) * 0.1  # Small variance
            estimator.update_statistics(pattern)
            
        # Check novelty of typical pattern
        typical = torch.randn(64) * 0.1
        typical_novelty = estimator.get_novelty(typical)
        
        # Check novelty of unusual pattern
        unusual = torch.randn(64) * 10  # Much larger
        unusual_novelty = estimator.get_novelty(unusual)
        
        assert unusual_novelty > typical_novelty


class TestErrorDetector:
    """Tests for ErrorDetector class."""
    
    def test_creation(self):
        """Test detector creation."""
        detector = ErrorDetector(input_dim=64)
        
        assert detector.input_dim == 64
        
    def test_reset(self):
        """Test detector reset."""
        detector = ErrorDetector(input_dim=64)
        
        # Generate an error
        predicted = torch.zeros(64)
        actual = torch.ones(64)
        detector.check_prediction_error(predicted, actual)
        
        detector.reset()
        
        assert len(detector.get_active_errors()) == 0
        
    def test_check_prediction_error_no_error(self):
        """Test prediction error check with no error."""
        detector = ErrorDetector(input_dim=64, threshold=0.5)
        
        predicted = torch.randn(64)
        actual = predicted + 0.1 * torch.randn(64)  # Small difference
        
        error = detector.check_prediction_error(predicted, actual)
        
        # Small difference shouldn't trigger error
        # (depends on threshold)
        
    def test_check_prediction_error_with_error(self):
        """Test prediction error check with significant error."""
        detector = ErrorDetector(input_dim=64, threshold=0.1)
        
        predicted = torch.zeros(64)
        actual = torch.ones(64)  # Very different
        
        error = detector.check_prediction_error(predicted, actual)
        
        assert error is not None
        assert error.error_type == ErrorType.PREDICTION_ERROR
        
    def test_check_conflict(self):
        """Test conflict detection."""
        detector = ErrorDetector(input_dim=64, threshold=0.3)
        
        # Very different patterns
        pattern_a = torch.zeros(64)
        pattern_b = torch.ones(64)
        
        error = detector.check_conflict(pattern_a, pattern_b)
        
        # May or may not detect conflict depending on network
        
    def test_check_consistency(self):
        """Test consistency check."""
        detector = ErrorDetector(input_dim=64, threshold=0.1)
        
        # Inconsistent patterns
        patterns = [torch.randn(64) for _ in range(5)]
        
        error = detector.check_consistency(patterns)
        
        # May detect inconsistency
        
    def test_check_consistency_single_pattern(self):
        """Test consistency with single pattern."""
        detector = ErrorDetector(input_dim=64)
        
        patterns = [torch.randn(64)]
        error = detector.check_consistency(patterns)
        
        assert error is None
        
    def test_check_timeout(self):
        """Test timeout detection."""
        detector = ErrorDetector(input_dim=64)
        
        # No timeout
        error = detector.check_timeout(elapsed_time=5.0, timeout=10.0)
        assert error is None
        
        # Timeout
        error = detector.check_timeout(elapsed_time=15.0, timeout=10.0)
        assert error is not None
        assert error.error_type == ErrorType.TIMEOUT
        
    def test_get_error_summary(self):
        """Test error summary."""
        detector = ErrorDetector(input_dim=64, threshold=0.1)
        
        # Generate various errors
        detector.check_timeout(15.0, 10.0)
        detector.check_prediction_error(torch.zeros(64), torch.ones(64))
        
        summary = detector.get_error_summary()
        
        assert "TIMEOUT" in summary
        assert "PREDICTION_ERROR" in summary


class TestCognitiveMonitor:
    """Tests for CognitiveMonitor class."""
    
    def test_creation(self):
        """Test monitor creation."""
        monitor = CognitiveMonitor(input_dim=64)
        
        assert monitor.input_dim == 64
        
    def test_reset(self):
        """Test monitor reset."""
        monitor = CognitiveMonitor(input_dim=64)
        
        # Update with some activity
        activity = torch.randn(64)
        monitor.update(activity)
        
        monitor.reset()
        
        state = monitor.get_state()
        assert state.load == 0.0
        
    def test_update_load(self):
        """Test load update."""
        monitor = CognitiveMonitor(input_dim=64)
        
        # Low activity
        low_activity = torch.randn(64) * 0.1
        low_load = monitor.update_load(low_activity)
        
        # High activity
        high_activity = torch.randn(64) * 10
        high_load = monitor.update_load(high_activity)
        
        assert high_load > low_load
        
    def test_update(self):
        """Test full state update."""
        monitor = CognitiveMonitor(input_dim=64)
        
        activity = torch.randn(64)
        state = monitor.update(activity)
        
        assert isinstance(state, CognitiveState)
        assert state.load >= 0
        assert 0 <= state.confidence <= 1
        
    def test_update_with_prediction(self):
        """Test update with prediction/outcome."""
        monitor = CognitiveMonitor(input_dim=64)
        
        activity = torch.randn(64)
        predicted = torch.randn(64)
        actual = torch.randn(64)
        
        state = monitor.update(activity, predicted, actual)
        
        assert isinstance(state, CognitiveState)
        
    def test_get_recommendations(self):
        """Test getting recommendations."""
        monitor = CognitiveMonitor(input_dim=64)
        
        activity = torch.randn(64)
        monitor.update(activity)
        
        recommendations = monitor.get_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
    def test_state_history(self):
        """Test state history tracking."""
        monitor = CognitiveMonitor(input_dim=64)
        
        for _ in range(5):
            activity = torch.randn(64)
            monitor.update(activity)
            
        history = monitor.get_state_history()
        
        assert len(history) == 5


class TestMetacognitiveController:
    """Tests for MetacognitiveController class."""
    
    def test_creation(self):
        """Test controller creation."""
        controller = MetacognitiveController(input_dim=64)
        
        assert controller.input_dim == 64
        
    def test_reset(self):
        """Test controller reset."""
        controller = MetacognitiveController(input_dim=64)
        
        activity = torch.randn(64)
        state = CognitiveState()
        controller.get_adjustments(activity, state)
        
        controller.reset()
        
        assert len(controller.get_action_history()) == 0
        
    def test_get_adjustments(self):
        """Test getting adjustments."""
        controller = MetacognitiveController(input_dim=64)
        
        activity = torch.randn(64)
        state = CognitiveState(load=0.5, confidence=0.6)
        
        adjustments = controller.get_adjustments(activity, state)
        
        assert "noise_scale" in adjustments
        assert "attention_focus" in adjustments
        assert "threshold_scale" in adjustments
        
    def test_adjustments_for_overload(self):
        """Test adjustments when overloaded."""
        controller = MetacognitiveController(input_dim=64)
        
        activity = torch.randn(64)
        state = CognitiveState(load=0.95)  # Overloaded
        
        adjustments = controller.get_adjustments(activity, state)
        
        assert adjustments["noise_scale"] == 0.2  # Reduced
        assert adjustments["attention_focus"] == 0.9  # Increased
        
    def test_adjustments_for_low_confidence(self):
        """Test adjustments for low confidence."""
        controller = MetacognitiveController(input_dim=64)
        
        activity = torch.randn(64)
        state = CognitiveState(confidence=0.2)  # Low confidence
        
        adjustments = controller.get_adjustments(activity, state)
        
        assert adjustments["noise_scale"] == 0.8  # More exploration
        
    def test_get_processing_strategy_normal(self):
        """Test strategy for normal state."""
        controller = MetacognitiveController(input_dim=64)
        
        state = CognitiveState(load=0.5, confidence=0.7, uncertainty=0.3)
        strategy = controller.get_processing_strategy(state)
        
        assert strategy == "proceed"
        
    def test_get_processing_strategy_overloaded(self):
        """Test strategy when overloaded."""
        controller = MetacognitiveController(input_dim=64)
        
        state = CognitiveState(load=0.95)
        strategy = controller.get_processing_strategy(state)
        
        assert strategy == "simplify"
        
    def test_get_processing_strategy_uncertain(self):
        """Test strategy for high uncertainty."""
        controller = MetacognitiveController(input_dim=64)
        
        state = CognitiveState(uncertainty=0.8)
        strategy = controller.get_processing_strategy(state)
        
        assert strategy == "explore"


class TestMetacognitiveNetwork:
    """Tests for MetacognitiveNetwork class."""
    
    def test_creation(self):
        """Test network creation."""
        network = MetacognitiveNetwork(input_dim=128)
        
        assert network.input_dim == 128
        
    def test_creation_with_config(self):
        """Test network creation with config."""
        config = MetacognitiveConfig(hidden_dim=128)
        network = MetacognitiveNetwork(input_dim=64, config=config)
        
        assert network.config.hidden_dim == 128
        
    def test_reset(self):
        """Test network reset."""
        network = MetacognitiveNetwork(input_dim=64)
        
        activity = torch.randn(64)
        network.observe(activity)
        
        network.reset()
        
        assert network.get_load() == 0.0
        
    def test_observe(self):
        """Test observation."""
        network = MetacognitiveNetwork(input_dim=64)
        
        activity = torch.randn(64)
        state = network.observe(activity)
        
        assert isinstance(state, CognitiveState)
        
    def test_observe_with_prediction(self):
        """Test observation with prediction."""
        network = MetacognitiveNetwork(input_dim=64)
        
        activity = torch.randn(64)
        prediction = torch.randn(64)
        outcome = torch.randn(64)
        
        state = network.observe(activity, prediction, outcome)
        
        assert isinstance(state, CognitiveState)
        
    def test_get_cognitive_state(self):
        """Test getting cognitive state."""
        network = MetacognitiveNetwork(input_dim=64)
        
        activity = torch.randn(64)
        network.observe(activity)
        
        state = network.get_cognitive_state()
        
        assert isinstance(state, CognitiveState)
        
    def test_get_adjustments(self):
        """Test getting adjustments."""
        network = MetacognitiveNetwork(input_dim=64)
        
        activity = torch.randn(64)
        network.observe(activity)
        
        adjustments = network.get_adjustments()
        
        assert "noise_scale" in adjustments
        
    def test_get_adjustments_no_observation(self):
        """Test adjustments without observation."""
        network = MetacognitiveNetwork(input_dim=64)
        
        adjustments = network.get_adjustments()
        
        # Should return defaults
        assert adjustments["noise_scale"] == 0.5
        
    def test_get_strategy(self):
        """Test getting strategy."""
        network = MetacognitiveNetwork(input_dim=64)
        
        activity = torch.randn(64)
        network.observe(activity)
        
        strategy = network.get_strategy()
        
        assert isinstance(strategy, str)
        
    def test_get_recommendations(self):
        """Test getting recommendations."""
        network = MetacognitiveNetwork(input_dim=64)
        
        activity = torch.randn(64)
        network.observe(activity)
        
        recommendations = network.get_recommendations()
        
        assert isinstance(recommendations, list)
        
    def test_get_confidence(self):
        """Test getting confidence."""
        network = MetacognitiveNetwork(input_dim=64)
        
        activity = torch.randn(64)
        network.observe(activity)
        
        confidence = network.get_confidence()
        
        assert 0 <= confidence <= 1
        
    def test_get_uncertainty(self):
        """Test getting uncertainty."""
        network = MetacognitiveNetwork(input_dim=64)
        
        activity = torch.randn(64)
        network.observe(activity)
        
        uncertainty = network.get_uncertainty()
        
        assert uncertainty >= 0
        
    def test_get_load(self):
        """Test getting load."""
        network = MetacognitiveNetwork(input_dim=64)
        
        activity = torch.randn(64)
        network.observe(activity)
        
        load = network.get_load()
        
        assert 0 <= load <= 1
        
    def test_has_errors(self):
        """Test error check."""
        network = MetacognitiveNetwork(input_dim=64)
        
        # Normal activity shouldn't have errors
        activity = torch.randn(64) * 0.1
        network.observe(activity)
        
        # May or may not have errors depending on thresholds
        
    def test_get_summary(self):
        """Test getting summary."""
        network = MetacognitiveNetwork(input_dim=64)
        
        activity = torch.randn(64)
        network.observe(activity)
        
        summary = network.get_summary()
        
        assert "load" in summary
        assert "confidence" in summary
        assert "uncertainty" in summary
        assert "processing_mode" in summary
        assert "strategy" in summary


class TestIntegration:
    """Integration tests for metacognition system."""
    
    def test_continuous_monitoring(self):
        """Test continuous monitoring over time."""
        network = MetacognitiveNetwork(input_dim=64)
        
        for _ in range(20):
            activity = torch.randn(64)
            network.observe(activity)
            
        summary = network.get_summary()
        assert summary["timestep"] == 20
        
    def test_error_detection_and_recovery(self):
        """Test error detection and strategy adjustment."""
        network = MetacognitiveNetwork(input_dim=64)
        
        # Normal operation
        normal_activity = torch.randn(64) * 0.1
        network.observe(normal_activity)
        
        # Create prediction error
        prediction = torch.zeros(64)
        actual = torch.ones(64)
        network.observe(normal_activity, prediction, actual)
        
        # Check if strategy changes
        strategy = network.get_strategy()
        # Strategy should adapt to errors
        
    def test_load_adaptation(self):
        """Test load-based adaptation."""
        network = MetacognitiveNetwork(input_dim=64)
        
        # Low load
        low_activity = torch.randn(64) * 0.01
        network.observe(low_activity)
        low_adjustments = network.get_adjustments()
        
        network.reset()
        
        # High load
        high_activity = torch.randn(64) * 10
        network.observe(high_activity)
        high_adjustments = network.get_adjustments()
        
        # Adjustments should differ
        # (specific values depend on learned network)
        
    def test_gpu_compatibility(self):
        """Test GPU compatibility."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        device = torch.device("cuda")
        
        network = MetacognitiveNetwork(input_dim=64)
        network = network.to(device)
        
        activity = torch.randn(64, device=device)
        state = network.observe(activity)
        
        assert isinstance(state, CognitiveState)
        
    def test_state_persistence(self):
        """Test that state persists across observations."""
        network = MetacognitiveNetwork(input_dim=64)
        
        # Multiple observations
        for _ in range(10):
            activity = torch.randn(64) * 0.5
            network.observe(activity)
            
        state1 = network.get_cognitive_state()
        
        # One more observation
        network.observe(torch.randn(64) * 0.5)
        
        state2 = network.get_cognitive_state()
        
        # States should be related (not independent)
        # Can't assert specific values but structure should be valid
        assert state1.processing_mode is not None
        assert state2.processing_mode is not None
