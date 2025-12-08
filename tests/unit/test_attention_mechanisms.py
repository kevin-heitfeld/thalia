"""
Tests for Enhanced Attention Mechanisms.

Tests bottom-up salience detection, top-down modulation,
developmental stage progression, and combined attention.
"""

import pytest
import torch
from thalia.integration.pathways.attention import (
    AttentionMechanisms,
    AttentionMechanismsConfig,
    AttentionStage,
)


class TestBottomUpSalience:
    """Test bottom-up salience detection components."""
    
    def test_brightness_contrast_detection(self):
        """Test that brightness contrast is detected correctly."""
        config = AttentionMechanismsConfig(
            input_size=100,
            device="cpu",
        )
        attention = AttentionMechanisms(config)
        
        # Create input with bright spot in center
        input_spikes = torch.zeros(100)
        input_spikes[45:55] = 1.0  # Bright region
        
        contrast = attention.brightness_contrast(input_spikes)
        
        # Contrast should be detected
        assert contrast.shape == (100,)
        assert contrast.sum() > 0
    
    def test_motion_saliency_first_frame(self):
        """Test motion saliency returns zeros on first frame."""
        config = AttentionMechanismsConfig(input_size=100, device="cpu")
        attention = AttentionMechanisms(config)
        
        input_spikes = torch.rand(100)
        motion = attention.motion_saliency(input_spikes)
        
        # First frame should have zero motion
        assert motion.shape == (100,)
        assert torch.allclose(motion, torch.zeros_like(motion))
    
    def test_motion_saliency_detects_change(self):
        """Test motion saliency detects frame-to-frame changes."""
        config = AttentionMechanismsConfig(input_size=100, device="cpu")
        attention = AttentionMechanisms(config)
        
        # First frame
        input1 = torch.zeros(100)
        motion1 = attention.motion_saliency(input1)
        assert torch.allclose(motion1, torch.zeros_like(motion1))
        
        # Second frame with change
        input2 = torch.zeros(100)
        input2[40:60] = 1.0  # Motion in center
        motion2 = attention.motion_saliency(input2)
        
        # Motion should be detected in changed region
        assert motion2[40:60].sum() > 0
    
    def test_novelty_detector_first_frame(self):
        """Test novelty detector returns ones on first frame (all novel)."""
        config = AttentionMechanismsConfig(input_size=100, device="cpu")
        attention = AttentionMechanisms(config)
        
        input_spikes = torch.rand(100)
        novelty = attention.novelty_detector(input_spikes)
        
        # First frame should have ones (all novel)
        assert novelty.shape == (100,)
        assert torch.allclose(novelty, torch.ones_like(novelty))
    
    def test_novelty_detector_detects_deviation(self):
        """Test novelty detector detects deviation from history."""
        config = AttentionMechanismsConfig(
            input_size=100,
            device="cpu",
        )
        attention = AttentionMechanisms(config)
        
        # Build history with consistent input
        for _ in range(10):
            input_consistent = torch.zeros(100)
            input_consistent[:50] = 0.5
            attention.novelty_detector(input_consistent)
        
        # Now send novel input
        input_novel = torch.zeros(100)
        input_novel[50:] = 1.0  # Novel activity in second half
        novelty = attention.novelty_detector(input_novel)
        
        # Novelty should be higher in novel region
        assert novelty[50:].mean() > novelty[:50].mean()
    
    def test_bottom_up_salience_normalization(self):
        """Test bottom-up salience is normalized."""
        config = AttentionMechanismsConfig(
            input_size=100,
            brightness_contrast_weight=0.5,
            motion_weight=0.3,
            novelty_weight=0.2,
            device="cpu",
        )
        attention = AttentionMechanisms(config)
        
        # Create input with strong contrast
        input_spikes = torch.zeros(100)
        input_spikes[40:60] = 1.0
        
        salience = attention.bottom_up_salience(input_spikes)
        
        # Salience should be present and normalized to [0, 1]
        assert salience.shape == (100,)
        assert salience.min() >= 0
        assert salience.max() <= 1.0
    
    def test_bottom_up_statistics(self):
        """Test bottom-up salience statistics tracking."""
        config = AttentionMechanismsConfig(input_size=100, device="cpu")
        attention = AttentionMechanisms(config)
        
        # Generate several salience maps
        for _ in range(5):
            input_spikes = torch.rand(100)
            attention.bottom_up_salience(input_spikes)
        
        stats = attention.get_statistics()
        assert stats["n_bottom_up"] == 5
        assert stats["avg_bottom_up_strength"] > 0


class TestTopDownModulation:
    """Test top-down attention modulation."""
    
    def test_top_down_modulation_without_pathway(self):
        """Test that top-down returns uniform without pathway."""
        config = AttentionMechanismsConfig(
            input_size=100,
            use_top_down=False,
            device="cpu",
        )
        attention = AttentionMechanisms(config)
        
        goal_spikes = torch.rand(50) > 0.5
        input_spikes = torch.rand(100)
        
        modulation = attention.top_down_modulation(input_spikes, goal_spikes)
        
        # Should return uniform
        assert modulation.shape == (100,)
        expected_uniform = 1.0 / 100
        assert torch.allclose(modulation, torch.full((100,), expected_uniform), atol=1e-5)
    
    def test_top_down_statistics(self):
        """Test top-down modulation statistics tracking."""
        config = AttentionMechanismsConfig(
            input_size=100,
            use_top_down=False,
            device="cpu",
        )
        attention = AttentionMechanisms(config)
        
        # Generate several modulations
        for _ in range(3):
            goal_spikes = torch.rand(50) > 0.5
            input_spikes = torch.rand(100)
            attention.top_down_modulation(input_spikes, goal_spikes)
        
        stats = attention.get_statistics()
        assert stats["n_top_down"] == 3


class TestCombinedAttention:
    """Test combined bottom-up and top-down attention."""
    
    def test_combined_attention_infant_stage(self):
        """Test combined attention at infant stage (100% bottom-up)."""
        config = AttentionMechanismsConfig(
            input_size=100,
            stage=AttentionStage.INFANT,
            device="cpu",
        )
        attention = AttentionMechanisms(config)
        
        # Create inputs
        input_spikes = torch.rand(100)
        goal_spikes = torch.rand(50) > 0.5
        
        combined, components = attention.combined_attention(input_spikes, goal_spikes)
        
        # Should be pure bottom-up
        assert combined.shape == (100,)
        assert components["weights"] == (1.0, 0.0)
    
    def test_combined_attention_school_age_stage(self):
        """Test combined attention at school-age stage (70% top-down)."""
        config = AttentionMechanismsConfig(
            input_size=100,
            stage=AttentionStage.SCHOOL_AGE,
            device="cpu",
        )
        attention = AttentionMechanisms(config)
        
        # Create inputs
        input_spikes = torch.rand(100)
        goal_spikes = torch.rand(50) > 0.5
        
        combined, components = attention.combined_attention(input_spikes, goal_spikes)
        
        # Should be mostly top-down
        assert combined.shape == (100,)
        assert components["weights"] == (0.3, 0.7)
    
    def test_combined_attention_balanced_stage(self):
        """Test combined attention at preschool stage (50/50)."""
        config = AttentionMechanismsConfig(
            input_size=100,
            stage=AttentionStage.PRESCHOOL,
            device="cpu",
        )
        attention = AttentionMechanisms(config)
        
        # Create inputs
        input_spikes = torch.rand(100)
        goal_spikes = torch.rand(50) > 0.5
        
        combined, components = attention.combined_attention(input_spikes, goal_spikes)
        
        # Should be balanced
        assert combined.shape == (100,)
        assert components["weights"] == (0.5, 0.5)
    
    def test_combined_attention_without_goal(self):
        """Test combined attention without goal spikes (bottom-up only)."""
        config = AttentionMechanismsConfig(
            input_size=100,
            stage=AttentionStage.SCHOOL_AGE,
            device="cpu",
        )
        attention = AttentionMechanisms(config)
        
        input_spikes = torch.rand(100)
        combined, components = attention.combined_attention(input_spikes, goal=None)
        
        # Should fall back to bottom-up weighted
        assert combined.shape == (100,)
        assert "bottom_up" in components
        assert "top_down" in components
    
    def test_combined_statistics(self):
        """Test combined attention statistics tracking."""
        config = AttentionMechanismsConfig(input_size=100, device="cpu")
        attention = AttentionMechanisms(config)
        
        # Generate several combined attentions
        for _ in range(4):
            input_spikes = torch.rand(100)
            goal_spikes = torch.rand(50) > 0.5
            attention.combined_attention(input_spikes, goal_spikes)
        
        stats = attention.get_statistics()
        assert stats["n_combined"] == 4


class TestDevelopmentalProgression:
    """Test developmental stage progression."""
    
    def test_stage_setting(self):
        """Test setting developmental stage."""
        config = AttentionMechanismsConfig(
            input_size=100,
            stage=AttentionStage.INFANT,
            device="cpu",
        )
        attention = AttentionMechanisms(config)
        
        # Default should be infant
        assert attention.config.stage == AttentionStage.INFANT
        
        # Set to toddler
        attention.set_stage(AttentionStage.TODDLER)
        assert attention.config.stage == AttentionStage.TODDLER
        
        # Set to preschool
        attention.set_stage(AttentionStage.PRESCHOOL)
        assert attention.config.stage == AttentionStage.PRESCHOOL
        
        # Set to school-age
        attention.set_stage(AttentionStage.SCHOOL_AGE)
        assert attention.config.stage == AttentionStage.SCHOOL_AGE
    
    def test_stage_weights_infant(self):
        """Test stage weights for infant stage."""
        config = AttentionMechanismsConfig(input_size=100, device="cpu")
        attention = AttentionMechanisms(config)
        
        weights = attention.get_stage_weights(AttentionStage.INFANT)
        assert weights == (1.0, 0.0)
    
    def test_stage_weights_toddler(self):
        """Test stage weights for toddler stage."""
        config = AttentionMechanismsConfig(input_size=100, device="cpu")
        attention = AttentionMechanisms(config)
        
        weights = attention.get_stage_weights(AttentionStage.TODDLER)
        assert weights == (0.7, 0.3)
    
    def test_stage_weights_preschool(self):
        """Test stage weights for preschool stage."""
        config = AttentionMechanismsConfig(input_size=100, device="cpu")
        attention = AttentionMechanisms(config)
        
        weights = attention.get_stage_weights(AttentionStage.PRESCHOOL)
        assert weights == (0.5, 0.5)
    
    def test_stage_weights_school_age(self):
        """Test stage weights for school-age stage."""
        config = AttentionMechanismsConfig(input_size=100, device="cpu")
        attention = AttentionMechanisms(config)
        
        weights = attention.get_stage_weights(AttentionStage.SCHOOL_AGE)
        assert weights == (0.3, 0.7)
    
    def test_stage_progression_affects_attention(self):
        """Test that stage progression changes attention weighting."""
        config = AttentionMechanismsConfig(input_size=100, device="cpu")
        attention = AttentionMechanisms(config)
        
        # Create consistent inputs
        input_spikes = torch.rand(100)
        goal_spikes = torch.rand(50) > 0.5
        
        # Get attention at infant stage
        attention.set_stage(AttentionStage.INFANT)
        infant_attention, infant_comp = attention.combined_attention(input_spikes, goal_spikes)
        
        # Get attention at school-age stage
        attention.set_stage(AttentionStage.SCHOOL_AGE)
        school_attention, school_comp = attention.combined_attention(input_spikes, goal_spikes)
        
        # Weights should differ
        assert infant_comp["weights"] != school_comp["weights"]


class TestMemoryTracking:
    """Test memory tracking for motion and novelty."""
    
    def test_motion_memory_updates(self):
        """Test that motion memory (prev_input) updates correctly."""
        config = AttentionMechanismsConfig(input_size=100, device="cpu")
        attention = AttentionMechanisms(config)
        
        # First input
        input1 = torch.rand(100)
        attention.motion_saliency(input1)
        
        # Memory should be set
        assert attention.prev_input is not None
        assert torch.allclose(attention.prev_input, input1)
        
        # Second input
        input2 = torch.rand(100)
        attention.motion_saliency(input2)
        
        # Memory should update
        assert torch.allclose(attention.prev_input, input2)
    
    def test_novelty_memory_ema(self):
        """Test that novelty memory uses exponential moving average."""
        config = AttentionMechanismsConfig(
            input_size=100,
            device="cpu",
        )
        attention = AttentionMechanisms(config)
        
        # First input
        input1 = torch.ones(100)
        attention.novelty_detector(input1)
        
        # History should be set
        assert attention.input_history is not None
        assert torch.allclose(attention.input_history, input1)
        
        # Second input (zeros)
        input2 = torch.zeros(100)
        attention.novelty_detector(input2)
        
        # History should be EMA: 0.1 * 0 + 0.9 * 1 = 0.9
        expected = torch.full((100,), 0.9)
        assert torch.allclose(attention.input_history, expected)
    
    def test_memory_persistence_across_calls(self):
        """Test that memory persists across multiple calls."""
        config = AttentionMechanismsConfig(input_size=100, device="cpu")
        attention = AttentionMechanisms(config)
        
        # Build up motion history
        for i in range(5):
            input_spikes = torch.full((100,), float(i))
            attention.motion_saliency(input_spikes)
        
        # Memory should be last input
        assert attention.prev_input is not None
        assert torch.allclose(attention.prev_input, torch.full((100,), 4.0))


class TestAttentionApplication:
    """Test attention application to inputs."""
    
    def test_apply_attention_returns_attended_input(self):
        """Test that apply_attention returns attended input."""
        config = AttentionMechanismsConfig(input_size=100, device="cpu")
        attention = AttentionMechanisms(config)
        
        # Create input and goal
        input_spikes = torch.ones(100)
        goal_spikes = torch.rand(50) > 0.5
        
        output, components = attention.apply_attention(input_spikes, goal_spikes)
        
        # Output should be input with attention applied
        assert output.shape == input_spikes.shape
        assert "bottom_up" in components
        assert "top_down" in components
    
    def test_apply_attention_preserves_shape(self):
        """Test that attention application preserves input shape."""
        config = AttentionMechanismsConfig(input_size=100, device="cpu")
        attention = AttentionMechanisms(config)
        
        input_spikes = torch.rand(100)
        goal_spikes = torch.rand(50) > 0.5
        
        output, components = attention.apply_attention(input_spikes, goal_spikes)
        
        assert output.shape == input_spikes.shape


class TestIntegration:
    """Test full attention pipeline integration."""
    
    def test_full_pipeline_infant(self):
        """Test full attention pipeline at infant stage."""
        config = AttentionMechanismsConfig(
            input_size=100,
            stage=AttentionStage.INFANT,
            device="cpu",
        )
        attention = AttentionMechanisms(config)
        
        # Process input
        input_spikes = torch.rand(100)
        goal_spikes = torch.rand(50) > 0.5
        
        # Apply attention
        output, components = attention.apply_attention(input_spikes, goal_spikes)
        
        # Should have valid output
        assert output.shape == (100,)
        assert output.min() >= 0
        assert components["weights"] == (1.0, 0.0)
    
    def test_full_pipeline_school_age(self):
        """Test full attention pipeline at school-age stage."""
        config = AttentionMechanismsConfig(
            input_size=100,
            stage=AttentionStage.SCHOOL_AGE,
            device="cpu",
        )
        attention = AttentionMechanisms(config)
        
        # Process input
        input_spikes = torch.rand(100)
        goal_spikes = torch.rand(50) > 0.5
        
        # Apply attention
        output, components = attention.apply_attention(input_spikes, goal_spikes)
        
        # Should have valid output
        assert output.shape == (100,)
        assert output.min() >= 0
        assert components["weights"] == (0.3, 0.7)
    
    def test_temporal_dynamics(self):
        """Test attention over multiple timesteps."""
        config = AttentionMechanismsConfig(
            input_size=100,
            stage=AttentionStage.TODDLER,
            device="cpu",
        )
        attention = AttentionMechanisms(config)
        
        # Process sequence of inputs
        outputs = []
        for t in range(10):
            input_spikes = torch.rand(100)
            goal_spikes = torch.rand(50) > 0.5
            
            output, components = attention.apply_attention(input_spikes, goal_spikes)
            outputs.append(output)
        
        # Should have outputs for all timesteps
        assert len(outputs) == 10
        for output in outputs:
            assert output.shape == (100,)
            assert output.min() >= 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_input(self):
        """Test handling of zero input."""
        config = AttentionMechanismsConfig(input_size=100, device="cpu")
        attention = AttentionMechanisms(config)
        
        input_spikes = torch.zeros(100)
        salience = attention.bottom_up_salience(input_spikes)
        
        # Should return valid salience
        assert salience.shape == (100,)
        assert salience.min() >= 0
        assert salience.max() <= 1.0
    
    def test_uniform_input(self):
        """Test handling of uniform input."""
        config = AttentionMechanismsConfig(input_size=100, device="cpu")
        attention = AttentionMechanisms(config)
        
        input_spikes = torch.ones(100) * 0.5
        salience = attention.bottom_up_salience(input_spikes)
        
        # Should return valid salience
        assert salience.shape == (100,)
        assert salience.min() >= 0
        assert salience.max() <= 1.0
    
    def test_missing_goal_spikes(self):
        """Test handling when goal spikes are None."""
        config = AttentionMechanismsConfig(
            input_size=100,
            stage=AttentionStage.SCHOOL_AGE,
            device="cpu",
        )
        attention = AttentionMechanisms(config)
        
        input_spikes = torch.rand(100)
        combined, components = attention.combined_attention(input_spikes, goal=None)
        
        # Should fall back to bottom-up weighted with uniform top-down
        assert combined.shape == (100,)
        assert "bottom_up" in components
        assert "top_down" in components
    
    def test_very_sparse_input(self):
        """Test handling of very sparse input."""
        config = AttentionMechanismsConfig(input_size=100, device="cpu")
        attention = AttentionMechanisms(config)
        
        # Only one spike
        input_spikes = torch.zeros(100)
        input_spikes[50] = 1.0
        
        salience = attention.bottom_up_salience(input_spikes)
        
        # Should still produce valid attention
        assert salience.shape == (100,)
        assert salience.min() >= 0
        assert salience.max() <= 1.0
    
    def test_device_consistency(self):
        """Test that all operations maintain device consistency."""
        config = AttentionMechanismsConfig(input_size=100, device="cpu")
        attention = AttentionMechanisms(config)
        
        input_spikes = torch.rand(100)
        goal_spikes = torch.rand(50) > 0.5
        
        # All outputs should be on correct device
        salience = attention.bottom_up_salience(input_spikes)
        assert salience.device == torch.device("cpu")
        
        modulation = attention.top_down_modulation(input_spikes, goal_spikes)
        assert modulation.device == torch.device("cpu")
        
        combined, _ = attention.combined_attention(input_spikes, goal_spikes)
        assert combined.device == torch.device("cpu")


class TestConfiguration:
    """Test configuration and initialization."""
    
    def test_custom_weights(self):
        """Test custom salience component weights."""
        config = AttentionMechanismsConfig(
            input_size=100,
            brightness_contrast_weight=0.6,
            motion_weight=0.3,
            novelty_weight=0.1,
            device="cpu",
        )
        attention = AttentionMechanisms(config)
        
        assert attention.config.brightness_contrast_weight == 0.6
        assert attention.config.motion_weight == 0.3
        assert attention.config.novelty_weight == 0.1
    
    def test_custom_stage(self):
        """Test custom stage initialization."""
        config = AttentionMechanismsConfig(
            input_size=100,
            stage=AttentionStage.SCHOOL_AGE,
            device="cpu",
        )
        attention = AttentionMechanisms(config)
        
        assert attention.config.stage == AttentionStage.SCHOOL_AGE
        assert attention.w_bottom_up == 0.3
        assert attention.w_top_down == 0.7
    
    def test_custom_weight_override(self):
        """Test custom weight override."""
        config = AttentionMechanismsConfig(
            input_size=100,
            stage=AttentionStage.INFANT,  # Default would be 1.0/0.0
            bottom_up_weight=0.6,
            top_down_weight=0.4,
            device="cpu",
        )
        attention = AttentionMechanisms(config)
        
        # Custom weights should override stage defaults
        assert attention.w_bottom_up == 0.6
        assert attention.w_top_down == 0.4
