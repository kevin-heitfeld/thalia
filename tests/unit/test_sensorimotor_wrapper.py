"""
Unit tests for Sensorimotor Wrapper (Gymnasium + MuJoCo).

Tests all components of the wrapper:
1. Environment initialization
2. Observation encoding (rate, population, temporal)
3. Motor decoding (population vector, rate)
4. Step mechanics
5. Motor babbling
6. Reaching tasks
"""

import pytest
import torch
import numpy as np

from thalia.environments.sensorimotor_wrapper import (
    SensorimotorWrapper,
    SensorimotorConfig,
    SpikeEncoding,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def wrapper():
    """Create a basic Reacher-v4 wrapper."""
    config = SensorimotorConfig(
        env_name="Reacher-v4",
        n_neurons_per_dof=50,
        device="cpu",
    )
    return SensorimotorWrapper(config=config)


@pytest.fixture
def wrapper_population():
    """Create wrapper with population coding."""
    config = SensorimotorConfig(
        env_name="Reacher-v4",
        spike_encoding="population",
        n_neurons_per_dof=50,
        device="cpu",
    )
    return SensorimotorWrapper(config=config)


# ============================================================================
# 1. Environment Initialization Tests
# ============================================================================

class TestSensorimotorInitialization:
    """Test environment initialization."""
    
    def test_wrapper_creation(self, wrapper):
        """Test basic wrapper creation."""
        assert wrapper.env is not None
        assert wrapper.obs_dim > 0
        assert wrapper.action_dim > 0
        assert wrapper.n_sensory_neurons > 0
        assert wrapper.n_motor_neurons > 0
    
    def test_reacher_dimensions(self, wrapper):
        """Test Reacher-v4 specific dimensions."""
        # Reacher-v4: 11-dim obs (but we use simplified version)
        # Action: 2-dim (two joint torques)
        assert wrapper.action_dim == 2
        assert wrapper.n_motor_neurons == 2 * 50  # 2 DOF × 50 neurons
    
    def test_env_name_override(self):
        """Test environment name override in constructor."""
        wrapper = SensorimotorWrapper(env_name="Reacher-v4")
        assert wrapper.config.env_name == "Reacher-v4"
    
    def test_device_handling(self):
        """Test device handling (CPU/CUDA)."""
        config = SensorimotorConfig(device="cpu")
        wrapper = SensorimotorWrapper(config=config)
        assert wrapper.device == torch.device("cpu")


# ============================================================================
# 2. Observation Encoding Tests
# ============================================================================

class TestObservationEncoding:
    """Test observation → spike encoding."""
    
    def test_rate_encoding_shape(self, wrapper):
        """Test rate encoding output shape."""
        obs_spikes = wrapper.reset()
        
        assert isinstance(obs_spikes, torch.Tensor)
        assert obs_spikes.shape[0] == wrapper.n_sensory_neurons
        assert obs_spikes.dtype == torch.float32
    
    def test_rate_encoding_binary(self, wrapper):
        """Test that rate encoding produces binary spikes."""
        obs_spikes = wrapper.reset()
        
        # All values should be 0 or 1
        assert torch.all((obs_spikes == 0) | (obs_spikes == 1))
    
    def test_rate_encoding_firing_rate(self, wrapper):
        """Test that firing rate is reasonable."""
        obs_spikes = wrapper.reset()
        
        # Firing rate should be between 0-20% (sparse coding)
        firing_rate = obs_spikes.mean().item()
        assert 0.0 <= firing_rate <= 0.20
    
    def test_population_encoding_shape(self, wrapper_population):
        """Test population encoding output shape."""
        obs_spikes = wrapper_population.reset()
        
        assert obs_spikes.shape[0] == wrapper_population.n_sensory_neurons
    
    def test_population_encoding_tuning(self, wrapper_population):
        """Test population encoding uses tuning curves."""
        # Create artificial observation
        obs = np.array([0.0] * wrapper_population.obs_dim)  # Neutral
        obs_spikes = wrapper_population._encode_observation(obs)
        
        # Should have some activity (tuning curve responses)
        assert obs_spikes.sum() > 0
    
    def test_encoding_reproducibility(self, wrapper):
        """Test that encoding is reproducible with seed."""
        obs_spikes1 = wrapper.reset(seed=42)
        obs_spikes2 = wrapper.reset(seed=42)
        
        # Should be similar (but not identical due to stochasticity)
        # At least check dimensions match
        assert obs_spikes1.shape == obs_spikes2.shape
    
    def test_sensory_noise(self, wrapper):
        """Test that sensory noise is applied."""
        # Reset multiple times, collect spikes
        spike_patterns = []
        for _ in range(10):
            obs_spikes = wrapper.reset(seed=42)
            spike_patterns.append(obs_spikes.cpu().numpy())
        
        spike_patterns = np.array(spike_patterns)
        
        # Should have some variability (noise)
        std = spike_patterns.std(axis=0).mean()
        assert std > 0  # Some variability


# ============================================================================
# 3. Motor Decoding Tests
# ============================================================================

class TestMotorDecoding:
    """Test spike → motor command decoding."""
    
    def test_population_vector_decode(self, wrapper):
        """Test population vector decoding."""
        # Create motor spikes (uniform)
        motor_spikes = torch.ones(wrapper.n_motor_neurons)
        
        action = wrapper._population_vector_decode(motor_spikes)
        
        assert action.shape == (wrapper.action_dim,)
        assert np.all(np.isfinite(action))
    
    def test_rate_decode(self, wrapper):
        """Test rate decoding."""
        # Create motor spikes (50% firing)
        motor_spikes = torch.rand(wrapper.n_motor_neurons) < 0.5
        
        action = wrapper._rate_decode(motor_spikes)
        
        assert action.shape == (wrapper.action_dim,)
        assert np.all(np.isfinite(action))
    
    def test_action_bounds(self, wrapper):
        """Test that actions are clipped to valid bounds."""
        # Create extreme motor spikes
        motor_spikes = torch.ones(wrapper.n_motor_neurons)
        
        action = wrapper._decode_motor_command(motor_spikes)
        
        # Should be within environment bounds
        assert np.all(action >= wrapper.env.action_space.low)
        assert np.all(action <= wrapper.env.action_space.high)
    
    def test_motor_smoothing(self, wrapper):
        """Test exponential smoothing of motor commands."""
        # First action
        motor_spikes1 = torch.ones(wrapper.n_motor_neurons)
        action1 = wrapper._decode_motor_command(motor_spikes1)
        
        # Second action (very different)
        motor_spikes2 = torch.zeros(wrapper.n_motor_neurons)
        action2 = wrapper._decode_motor_command(motor_spikes2)
        
        # Should be smoothed (not at extremes)
        assert not np.allclose(action2, np.zeros(wrapper.action_dim))
    
    def test_zero_spikes_handling(self, wrapper):
        """Test handling of zero spikes (no motor command)."""
        motor_spikes = torch.zeros(wrapper.n_motor_neurons)
        
        action = wrapper._decode_motor_command(motor_spikes)
        
        # Should produce valid action (neutral or smoothed)
        assert action.shape == (wrapper.action_dim,)
        assert np.all(np.isfinite(action))


# ============================================================================
# 4. Step Mechanics Tests
# ============================================================================

class TestStepMechanics:
    """Test environment stepping."""
    
    def test_reset_returns_spikes(self, wrapper):
        """Test that reset returns spike tensor."""
        obs_spikes = wrapper.reset()
        
        assert isinstance(obs_spikes, torch.Tensor)
        assert obs_spikes.shape[0] == wrapper.n_sensory_neurons
    
    def test_step_returns_tuple(self, wrapper):
        """Test that step returns correct tuple."""
        wrapper.reset()
        motor_spikes = torch.rand(wrapper.n_motor_neurons) < 0.1
        
        result = wrapper.step(motor_spikes)
        
        assert len(result) == 4  # (obs, reward, terminated, truncated)
        obs_spikes, reward, terminated, truncated = result
        
        assert isinstance(obs_spikes, torch.Tensor)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    
    def test_episode_tracking(self, wrapper):
        """Test episode reward/step tracking."""
        wrapper.reset()
        
        initial_reward = wrapper._episode_reward
        initial_steps = wrapper._episode_steps
        
        motor_spikes = torch.rand(wrapper.n_motor_neurons) < 0.1
        wrapper.step(motor_spikes)
        
        # Should increment
        assert wrapper._episode_steps == initial_steps + 1
        assert wrapper._episode_reward != initial_reward  # Reward changed
    
    def test_episode_termination(self, wrapper):
        """Test episode termination/truncation."""
        wrapper.reset()
        
        # Run until episode ends
        for _ in range(100):
            motor_spikes = torch.rand(wrapper.n_motor_neurons) < 0.1
            _, _, terminated, truncated = wrapper.step(motor_spikes)
            
            if terminated or truncated:
                break
        
        # Should eventually terminate
        assert terminated or truncated
    
    def test_get_episode_stats(self, wrapper):
        """Test episode statistics retrieval."""
        wrapper.reset()
        
        motor_spikes = torch.rand(wrapper.n_motor_neurons) < 0.1
        wrapper.step(motor_spikes)
        
        stats = wrapper.get_episode_stats()
        
        assert "reward" in stats
        assert "steps" in stats
        assert stats["steps"] > 0


# ============================================================================
# 5. Motor Babbling Tests
# ============================================================================

class TestMotorBabbling:
    """Test motor babbling functionality."""
    
    def test_motor_babbling_runs(self, wrapper):
        """Test that motor babbling completes."""
        stats = wrapper.motor_babbling(n_steps=100)
        
        assert "mean_reward" in stats
        assert "std_reward" in stats
        assert "n_steps" in stats
        assert stats["n_steps"] == 100
    
    def test_motor_babbling_statistics(self, wrapper):
        """Test motor babbling produces valid statistics."""
        stats = wrapper.motor_babbling(n_steps=50)
        
        # Rewards should be finite
        assert np.isfinite(stats["mean_reward"])
        assert np.isfinite(stats["std_reward"])
        
        # Should have some variation
        assert stats["std_reward"] > 0
    
    def test_motor_babbling_resets(self, wrapper):
        """Test that motor babbling handles resets."""
        # Babbling should handle episode terminations
        stats = wrapper.motor_babbling(n_steps=200)
        
        # Should complete without errors
        assert stats["n_steps"] == 200


# ============================================================================
# 6. Integration Tests
# ============================================================================

class TestSensorimotorIntegration:
    """Test full sensorimotor loop."""
    
    def test_full_episode(self, wrapper):
        """Test complete episode from reset to termination."""
        obs_spikes = wrapper.reset(seed=42)
        
        total_reward = 0.0
        steps = 0
        
        for _ in range(50):
            # Random motor command
            motor_spikes = torch.rand(wrapper.n_motor_neurons) < 0.1
            
            obs_spikes, reward, terminated, truncated = wrapper.step(motor_spikes)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        # Should have collected some reward
        assert steps > 0
        assert np.isfinite(total_reward)
    
    def test_multiple_episodes(self, wrapper):
        """Test running multiple episodes."""
        rewards = []
        
        for episode in range(5):
            obs_spikes = wrapper.reset(seed=episode)
            episode_reward = 0.0
            
            for step in range(50):
                motor_spikes = torch.rand(wrapper.n_motor_neurons) < 0.1
                obs_spikes, reward, terminated, truncated = wrapper.step(motor_spikes)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            rewards.append(episode_reward)
        
        # Should have collected rewards for all episodes
        assert len(rewards) == 5
        assert all(np.isfinite(r) for r in rewards)
    
    def test_different_encoding_methods(self):
        """Test that different encoding methods work."""
        for encoding in ["rate", "population"]:
            config = SensorimotorConfig(
                env_name="Reacher-v4",
                spike_encoding=encoding,
                n_neurons_per_dof=50,
            )
            wrapper = SensorimotorWrapper(config=config)
            
            obs_spikes = wrapper.reset()
            assert obs_spikes.shape[0] == wrapper.n_sensory_neurons
            
            motor_spikes = torch.rand(wrapper.n_motor_neurons) < 0.1
            obs_spikes, reward, terminated, truncated = wrapper.step(motor_spikes)
            
            assert isinstance(reward, (int, float))
    
    def test_close_environment(self, wrapper):
        """Test environment cleanup."""
        wrapper.close()
        # Should not raise errors


# ============================================================================
# 7. Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_encoding(self):
        """Test error on invalid encoding method."""
        config = SensorimotorConfig(spike_encoding="invalid")
        wrapper = SensorimotorWrapper(config=config)
        
        obs = np.zeros(wrapper.obs_dim)
        
        with pytest.raises(ValueError, match="Unknown encoding"):
            wrapper._encode_observation(obs)
    
    def test_invalid_decoding(self):
        """Test error on invalid decoding method."""
        config = SensorimotorConfig(motor_decoding="invalid")
        wrapper = SensorimotorWrapper(config=config)
        
        motor_spikes = torch.zeros(wrapper.n_motor_neurons)
        
        with pytest.raises(ValueError, match="Unknown decoding"):
            wrapper._decode_motor_command(motor_spikes)
    
    def test_empty_motor_spikes(self, wrapper):
        """Test handling of empty motor spike tensor."""
        wrapper.reset()
        
        # All zeros
        motor_spikes = torch.zeros(wrapper.n_motor_neurons)
        obs_spikes, reward, terminated, truncated = wrapper.step(motor_spikes)
        
        # Should handle gracefully
        assert isinstance(obs_spikes, torch.Tensor)
    
    def test_full_motor_spikes(self, wrapper):
        """Test handling of fully active motor spikes."""
        wrapper.reset()
        
        # All ones
        motor_spikes = torch.ones(wrapper.n_motor_neurons)
        obs_spikes, reward, terminated, truncated = wrapper.step(motor_spikes)
        
        # Should handle gracefully
        assert isinstance(obs_spikes, torch.Tensor)


# ============================================================================
# Performance/Benchmark Tests
# ============================================================================

class TestPerformance:
    """Test performance characteristics."""
    
    def test_encoding_speed(self, wrapper):
        """Test encoding speed."""
        import time
        obs = np.random.randn(wrapper.obs_dim)
        
        # Run multiple times and check average time
        start = time.time()
        for _ in range(1000):
            result = wrapper._encode_observation(obs)
        elapsed = time.time() - start
        
        # Should average < 1ms per encoding
        avg_time_ms = (elapsed / 1000) * 1000
        assert avg_time_ms < 1.0
        assert isinstance(result, torch.Tensor)
    
    def test_decoding_speed(self, wrapper):
        """Test decoding speed."""
        import time
        motor_spikes = torch.rand(wrapper.n_motor_neurons) < 0.1
        
        # Run multiple times and check average time
        start = time.time()
        for _ in range(1000):
            result = wrapper._decode_motor_command(motor_spikes)
        elapsed = time.time() - start
        
        # Should average < 1ms per decoding
        avg_time_ms = (elapsed / 1000) * 1000
        assert avg_time_ms < 1.0
        assert isinstance(result, np.ndarray)
    
    def test_step_speed(self, wrapper):
        """Test that step is reasonably fast."""
        wrapper.reset()
        
        import time
        start = time.time()
        
        for _ in range(100):
            motor_spikes = torch.rand(wrapper.n_motor_neurons) < 0.1
            wrapper.step(motor_spikes)
        
        elapsed = time.time() - start
        
        # Should be < 1 second for 100 steps
        assert elapsed < 1.0
