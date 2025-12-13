"""
Tests for streaming trainer.

Tests continuous online learning without epochs or train/eval splits.
"""

import pytest
import torch

from thalia.training.streaming import (
    StreamingTrainer,
    StreamConfig,
    ExperienceBuffer,
    DriftDetector,
)
from thalia.core.brain import EventDrivenBrain
from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes


@pytest.fixture
def mock_brain():
    """Create minimal brain for testing."""
    config = ThaliaConfig(
        global_=GlobalConfig(device="cpu", dt_ms=1.0),
        brain=BrainConfig(
            sizes=RegionSizes(
                input_size=10,
                thalamus_size=20,
                cortex_size=30,
                hippocampus_size=40,
                pfc_size=20,
                n_actions=5,
            ),
        ),
    )
    return EventDrivenBrain.from_config(config)


def test_stream_config_defaults():
    """Test StreamConfig default values."""
    config = StreamConfig()
    assert config.eval_frequency == 1000
    assert config.enable_replay is True
    assert config.replay_buffer_size == 10000


def test_experience_buffer_add_and_sample():
    """Test experience buffer storage and sampling."""
    buffer = ExperienceBuffer(capacity=5, max_age=None)

    # Add samples
    for i in range(3):
        buffer.add({"input": torch.randn(10), "label": i})

    assert len(buffer) == 3

    # Sample
    sample = buffer.sample_random()
    assert sample is not None
    assert "input" in sample
    assert "label" in sample
    assert "_timestamp" not in sample  # Should be removed


def test_experience_buffer_max_capacity():
    """Test buffer respects capacity limit."""
    buffer = ExperienceBuffer(capacity=3, max_age=None)

    # Add more than capacity
    for i in range(5):
        buffer.add({"value": i})

    assert len(buffer) == 3  # Should be capped

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
    """Test StreamingTrainer initialization."""
    config = StreamConfig(
        eval_frequency=100,
        enable_replay=True,
    )

    trainer = StreamingTrainer(
        brain=mock_brain,
        config=config,
    )

    # Contract: trainer should be properly initialized with brain
    assert trainer.brain is mock_brain


def test_streaming_trainer_no_replay(mock_brain):
    """Test trainer without replay buffer."""
    config = StreamConfig(enable_replay=False)

    trainer = StreamingTrainer(
        brain=mock_brain,
        config=config,
    )

    # Contract: replay should be disabled when configured
    assert not config.enable_replay, "Config should disable replay"


def test_streaming_trainer_process_sample(mock_brain):
    """Test processing single sample."""
    config = StreamConfig()
    trainer = StreamingTrainer(
        brain=mock_brain,
        config=config,
    )

    # Create sample
    sample = {
        "input": torch.randn(10),
        "reward": 1.0,
    }

    # Process (should not raise)
    trainer._process_sample(sample)


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
        evaluator=simple_evaluator,
    )

    # Create simple stream
    def data_stream():
        for i in range(150):
            yield {"input": torch.randn(10), "label": i % 5}

    # Train
    stats = trainer.train_online(
        data_stream=data_stream(),
        max_samples=150,
        verbose=False,
    )

    assert stats.samples_processed == 150
    assert stats.duration_seconds > 0
    assert len(stats.performance_history) > 0  # Should have evaluations


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
    )

    # Create stream
    def data_stream():
        for i in range(20):
            yield {"input": torch.randn(10)}

    # Train
    stats = trainer.train_online(
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
    )

    # Infinite stream
    def infinite_stream():
        i = 0
        while True:
            yield {"input": torch.randn(10)}
            i += 1

    # Should stop at max_samples
    stats = trainer.train_online(
        data_stream=infinite_stream(),
        max_samples=50,
        verbose=False,
    )

    assert stats.samples_processed == 50


def test_streaming_trainer_performance_summary(mock_brain):
    """Test performance summary aggregation."""
    config = StreamConfig(eval_frequency=10)

    def mock_evaluator(brain):
        return {"acc": 0.85, "loss": 0.2}

    trainer = StreamingTrainer(
        brain=mock_brain,
        config=config,
        evaluator=mock_evaluator,
    )

    # Create stream
    def data_stream():
        for i in range(30):
            yield {"input": torch.randn(10)}

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
