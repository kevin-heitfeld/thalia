"""
Tests for streaming trainer on DynamicBrain.

Tests continuous online learning without epochs or train/eval splits.
This is an exact copy of test_streaming_trainer.py but using DynamicBrain instead of EventDrivenBrain.
"""

import pytest
import torch

from tests.utils import create_test_brain
from thalia.training.streaming import (
    DriftDetector,
    ExperienceBuffer,
    StreamConfig,
    StreamingTrainer,
)

CHECKPOINT_DIR = "temp/test_checkpoints/streaming_trainer"


@pytest.fixture
def mock_brain():
    """Create minimal brain for testing."""
    return create_test_brain(
        input_size=20,  # Match thalamus relay size for 1:1 relay
        thalamus_size=20,
        cortex_size=30,
        hippocampus_size=40,
        pfc_size=20,
        n_actions=5,
    )


def test_stream_config_defaults():
    """Test StreamConfig has valid default values."""
    config = StreamConfig()
    # Test contract: defaults should be reasonable
    assert (
        0 < config.eval_frequency <= 100000
    ), f"Eval frequency should be positive and reasonable, got {config.eval_frequency}"
    assert isinstance(config.enable_replay, bool), "Replay flag should be boolean"
    assert (
        0 < config.replay_buffer_size <= 1000000
    ), f"Buffer size should be positive and reasonable, got {config.replay_buffer_size}"


def test_experience_buffer_add_and_sample():
    """Test experience buffer storage and sampling."""
    buffer = ExperienceBuffer(capacity=5, max_age=None)

    # Add samples
    n_samples_added = 3
    for i in range(n_samples_added):
        buffer.add({"input": torch.randn(10), "label": i})

    # Test contract: buffer should contain added samples
    assert len(buffer) == n_samples_added, f"Buffer should contain {n_samples_added} samples"

    # Sample
    sample = buffer.sample_random()
    # Test sample structure (type system guarantees non-None if method succeeds)
    assert "input" in sample, "Sample should contain input"
    assert "label" in sample, "Sample should contain label"
    assert "_timestamp" not in sample, "Internal timestamp should be removed"


def test_experience_buffer_max_capacity():
    """Test buffer respects capacity limit."""
    buffer = ExperienceBuffer(capacity=3, max_age=None)

    # Add more than capacity
    capacity = 3
    for i in range(5):
        buffer.add({"value": i})

    # Test contract: buffer should not exceed capacity
    assert len(buffer) == capacity, f"Buffer should be capped at capacity ({capacity})"

    # Check oldest samples were dropped
    samples = [buffer.buffer[i]["value"] for i in range(len(buffer))]
    assert 0 not in samples  # First samples dropped
    assert 1 not in samples


def test_experience_buffer_max_age():
    """Test buffer filters by age."""
    buffer = ExperienceBuffer(capacity=10, max_age=3)

    # Add samples
    for i in range(5):
        buffer.add({"value": i})

    # Sample should only get recent ones
    sampled_values = set()
    for _ in range(20):
        sample = buffer.sample_random()
        if sample:
            sampled_values.add(sample["value"])

    # Only recent samples (within max_age=3) should appear
    assert 0 not in sampled_values  # Too old
    assert 1 not in sampled_values  # Too old
    assert 4 in sampled_values  # Recent


def test_drift_detector_no_drift():
    """Test drift detector with stable performance."""
    detector = DriftDetector(window_size=5, threshold=0.1)

    # Add stable performance
    for _ in range(10):
        alert = detector.update(0.9)

    assert alert is None  # No drift detected


def test_drift_detector_detects_drop():
    """Test drift detector alerts on performance drop."""
    detector = DriftDetector(window_size=5, threshold=0.1)

    # Establish baseline
    for _ in range(5):
        detector.update(0.9)

    # Sudden drop
    alert = detector.update(0.7)  # 22% drop

    assert alert is not None
    assert alert["type"] == "catastrophic_forgetting"
    assert alert["drop_fraction"] > 0.1


def test_streaming_trainer_init(mock_brain):
    """Test StreamingTrainer initialization and config."""
    config = StreamConfig(
        eval_frequency=100,
        enable_replay=True,
    )

    # Explicitly set eval_frequency to test configuration handling
    config.eval_frequency = 100

    trainer = StreamingTrainer(
        brain=mock_brain,
        config=config,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    # Test contract: trainer should use explicitly configured values
    assert trainer.config.eval_frequency == 100, "Should use explicitly set eval frequency"
    assert trainer.config.enable_replay is True, "Should enable replay as configured"


def test_streaming_trainer_process_sample(mock_brain):
    """Test processing single sample."""
    config = StreamConfig()
    trainer = StreamingTrainer(
        brain=mock_brain,
        config=config,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    # Create sample
    sample = {
        "input": torch.randn(128),
        "reward": 1.0,
    }

    # Process sample through public API (train_online with single-item generator)
    def single_sample_stream():
        yield sample

    # Contract: Should process sample without raising
    stats = trainer.train_online(data_stream=single_sample_stream(), max_samples=1)

    # Verify sample was processed
    assert stats.samples_processed == 1, "Should process one sample"


def test_streaming_trainer_train_online(mock_brain):
    """Test continuous training from stream."""
    config = StreamConfig(
        eval_frequency=50,
        checkpoint_frequency=100,
        health_check_frequency=50,
    )

    def simple_evaluator(brain):
        return {"accuracy": 0.8}

    trainer = StreamingTrainer(
        brain=mock_brain,
        config=config,
        checkpoint_dir=CHECKPOINT_DIR,
        evaluator=simple_evaluator,
    )

    # Create simple stream
    max_samples = 150

    def data_stream():
        for i in range(max_samples):
            yield {"input": torch.randn(128), "label": i % 5}

    # Train
    stats = trainer.train_online(
        data_stream=data_stream(),
        max_samples=max_samples,
        verbose=False,
    )

    # Test contract: should process requested number of samples
    assert stats.samples_processed == max_samples, f"Should process {max_samples} samples"
    assert stats.duration_seconds > 0, "Should track training duration"
    assert len(stats.performance_history) > 0, "Should have evaluation history"


def test_streaming_trainer_with_replay(mock_brain):
    """Test replay during training."""
    config = StreamConfig(
        enable_replay=True,
        replay_frequency=5,  # Replay every 5 samples
        replay_buffer_size=10,
    )

    trainer = StreamingTrainer(
        brain=mock_brain,
        config=config,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    # Create stream
    def data_stream():
        for i in range(20):
            yield {"input": torch.randn(128), "reward": 1.0}

    # Train
    _stats = trainer.train_online(
        data_stream=data_stream(),
        max_samples=20,
        verbose=False,
    )

    # Check replay buffer has samples
    assert len(trainer.replay_buffer) > 0


def test_streaming_trainer_early_stop(mock_brain):
    """Test training stops at max_samples."""
    config = StreamConfig()
    trainer = StreamingTrainer(
        brain=mock_brain,
        config=config,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    # Infinite stream
    def infinite_stream():
        i = 0
        while True:
            yield {"input": torch.randn(128)}
            i += 1

    # Should stop at max_samples
    max_samples = 50
    stats = trainer.train_online(
        data_stream=infinite_stream(),
        max_samples=max_samples,
        verbose=False,
    )

    # Test contract: should process exactly max_samples
    assert (
        stats.samples_processed == max_samples
    ), f"Should stop at {max_samples} samples (not process indefinitely)"


def test_streaming_trainer_performance_summary(mock_brain):
    """Test performance summary aggregation."""
    config = StreamConfig(eval_frequency=10)

    def mock_evaluator(brain):
        return {"acc": 0.85, "loss": 0.2}

    trainer = StreamingTrainer(
        brain=mock_brain,
        config=config,
        checkpoint_dir=CHECKPOINT_DIR,
        evaluator=mock_evaluator,
    )

    # Create stream
    def data_stream():
        for _ in range(30):
            yield {"input": torch.randn(128)}

    # Train
    trainer.train_online(
        data_stream=data_stream(),
        max_samples=30,
        verbose=False,
    )

    # Get summary
    summary = trainer.get_performance_summary()

    assert "acc_mean" in summary
    assert "acc_std" in summary
    assert "loss_mean" in summary
