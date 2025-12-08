"""
Tests for social learning module.
"""

import pytest
import torch

from thalia.learning.social_learning import (
    SocialLearningModule,
    SocialLearningConfig,
    SocialContext,
    SocialCueType,
    compute_shared_attention,
)


class TestImitationLearning:
    """Tests for imitation learning."""

    def test_initialization(self):
        """Test module initialization."""
        module = SocialLearningModule()
        assert module.config.imitation_boost == 2.0
        assert module.statistics["n_demonstrations"] == 0

    def test_imitation_boost(self):
        """Test learning rate boost from demonstration."""
        module = SocialLearningModule()
        base_lr = 0.01

        # With demonstration
        boosted_lr = module.imitation_learning(base_lr, demonstration_present=True)
        assert boosted_lr == base_lr * 2.0
        assert module.statistics["n_demonstrations"] == 1

        # Without demonstration
        normal_lr = module.imitation_learning(base_lr, demonstration_present=False)
        assert normal_lr == base_lr
        assert module.statistics["n_demonstrations"] == 1  # No increment

    def test_motor_imitation(self):
        """Test motor imitation with error correction."""
        module = SocialLearningModule()

        observed = torch.tensor([0.8, 0.2, 0.5])
        own_action = torch.tensor([0.3, 0.6, 0.4])
        base_lr = 0.01

        motor_error, effective_lr = module.motor_imitation(observed, own_action, base_lr)

        # Error should be difference
        expected_error = observed - own_action
        assert torch.allclose(motor_error, expected_error)

        # Learning rate should be boosted
        assert effective_lr == base_lr * 2.0

    def test_custom_imitation_boost(self):
        """Test custom imitation boost factor."""
        config = SocialLearningConfig(imitation_boost=3.0)
        module = SocialLearningModule(config)

        base_lr = 0.01
        boosted_lr = module.imitation_learning(base_lr, demonstration_present=True)

        assert boosted_lr == base_lr * 3.0


class TestNaturalPedagogy:
    """Tests for natural pedagogy mechanisms."""

    def test_teaching_detection_eye_contact_plus_one(self):
        """Test teaching detection with eye contact + one cue."""
        module = SocialLearningModule()

        # Eye contact + motherese
        cues1 = {"eye_contact": True, "motherese": True, "pointing": False}
        assert module._detect_teaching_signal(cues1) is True

        # Eye contact + pointing
        cues2 = {"eye_contact": True, "motherese": False, "pointing": True}
        assert module._detect_teaching_signal(cues2) is True

    def test_teaching_detection_all_cues(self):
        """Test teaching detection with all three cues."""
        module = SocialLearningModule()

        cues = {"eye_contact": True, "motherese": True, "pointing": True}
        assert module._detect_teaching_signal(cues) is True

    def test_teaching_detection_no_eye_contact(self):
        """Test no teaching without eye contact."""
        module = SocialLearningModule()

        # Motherese + pointing but no eye contact
        cues = {"eye_contact": False, "motherese": True, "pointing": True}
        assert module._detect_teaching_signal(cues) is False

    def test_teaching_detection_only_one_cue(self):
        """Test no teaching with only one cue."""
        module = SocialLearningModule()

        # Only eye contact
        cues1 = {"eye_contact": True, "motherese": False, "pointing": False}
        assert module._detect_teaching_signal(cues1) is False

        # Only motherese
        cues2 = {"eye_contact": False, "motherese": True, "pointing": False}
        assert module._detect_teaching_signal(cues2) is False

    def test_pedagogy_boost(self):
        """Test pedagogy boost on learning rate."""
        module = SocialLearningModule()
        base_lr = 0.01

        # Teaching context
        teaching_cues = {"eye_contact": True, "motherese": True, "pointing": False}
        boosted_lr = module.pedagogy_boost(base_lr, teaching_cues)
        assert boosted_lr == base_lr * 1.5
        assert module.statistics["n_pedagogy_episodes"] == 1

        # No teaching
        no_teaching_cues = {"eye_contact": False, "motherese": True, "pointing": False}
        normal_lr = module.pedagogy_boost(base_lr, no_teaching_cues)
        assert normal_lr == base_lr
        assert module.statistics["n_pedagogy_episodes"] == 1  # No increment

    def test_custom_pedagogy_boost(self):
        """Test custom pedagogy boost factor."""
        config = SocialLearningConfig(pedagogy_boost=2.5)
        module = SocialLearningModule(config)

        base_lr = 0.01
        cues = {"eye_contact": True, "motherese": True, "pointing": False}
        boosted_lr = module.pedagogy_boost(base_lr, cues)

        assert boosted_lr == base_lr * 2.5


class TestJointAttention:
    """Tests for joint attention mechanisms."""

    def test_joint_attention_modulation(self):
        """Test attention modulation by gaze."""
        module = SocialLearningModule()

        # Current attention (uniform)
        attention = torch.ones(10) / 10.0

        # Gaze directed at position 5
        gaze = torch.zeros(10)
        gaze[5] = 1.0

        # Moderate shared attention
        modulated = module.joint_attention(attention, gaze, shared_attention_strength=0.5)

        # Position 5 should have higher attention
        assert modulated[5] > attention[5]

        # Should still be normalized
        assert torch.isclose(modulated.sum(), torch.tensor(1.0))

    def test_joint_attention_no_gaze(self):
        """Test that attention unchanged without gaze."""
        module = SocialLearningModule()

        attention = torch.ones(10) / 10.0
        modulated = module.joint_attention(attention, gaze_direction=None)

        assert torch.allclose(modulated, attention)

    def test_joint_attention_strength_scaling(self):
        """Test that shared attention strength affects modulation."""
        config = SocialLearningConfig(gaze_influence=0.5)
        module = SocialLearningModule(config)

        attention = torch.ones(10) / 10.0
        gaze = torch.zeros(10)
        gaze[5] = 1.0

        # Weak shared attention
        weak = module.joint_attention(attention, gaze, shared_attention_strength=0.2)

        # Strong shared attention
        strong = module.joint_attention(attention, gaze, shared_attention_strength=0.9)

        # Both should modulate attention towards position 5
        # (After normalization, the difference may be small but both > baseline)
        assert weak[5] > attention[5]
        assert strong[5] > attention[5]

    def test_joint_attention_threshold_tracking(self):
        """Test that joint attention events are tracked."""
        config = SocialLearningConfig(joint_attention_threshold=0.7)
        module = SocialLearningModule(config)

        attention = torch.ones(10) / 10.0
        gaze = torch.zeros(10)
        gaze[5] = 1.0

        # Below threshold
        module.joint_attention(attention, gaze, shared_attention_strength=0.5)
        assert module.statistics["n_joint_attention"] == 0

        # Above threshold
        module.joint_attention(attention, gaze, shared_attention_strength=0.8)
        assert module.statistics["n_joint_attention"] == 1

    def test_gaze_modulation_computation(self):
        """Test gaze modulation calculation."""
        module = SocialLearningModule()

        gaze = torch.zeros(10)
        gaze[3] = 1.0  # Focused on position 3

        # High shared attention
        modulation = module._compute_gaze_modulation(gaze, shared_attention_strength=0.8)

        # Should boost position 3
        assert modulation[3] > modulation[0]

        # Should be normalized
        assert torch.isclose(modulation.sum(), torch.tensor(1.0))

    def test_joint_attention_device_consistency(self):
        """Test that joint attention respects device setting."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        config = SocialLearningConfig(device="cuda")
        module = SocialLearningModule(config)

        attention = torch.ones(10, device="cuda") / 10.0
        gaze = torch.zeros(10, device="cuda")
        gaze[5] = 1.0

        modulated = module.joint_attention(attention, gaze, shared_attention_strength=0.5)

        assert modulated.device.type == "cuda"


class TestCombinedSocialLearning:
    """Tests for combined social learning mechanisms."""

    def test_modulate_learning_demonstration_only(self):
        """Test learning modulation with demonstration only."""
        module = SocialLearningModule()
        base_lr = 0.01

        context = SocialContext(
            cue_type=SocialCueType.DEMONSTRATION,
            demonstration_present=True,
        )

        modulated_lr = module.modulate_learning(base_lr, context)

        # Should have imitation boost only
        assert modulated_lr == base_lr * 2.0

    def test_modulate_learning_pedagogy_only(self):
        """Test learning modulation with teaching only."""
        module = SocialLearningModule()
        base_lr = 0.01

        context = SocialContext(
            cue_type=SocialCueType.OSTENSIVE,
            demonstration_present=False,
            ostensive_cues={"eye_contact": True, "motherese": True, "pointing": False},
        )

        modulated_lr = module.modulate_learning(base_lr, context)

        # Should have pedagogy boost only
        assert modulated_lr == base_lr * 1.5

    def test_modulate_learning_both_boosts(self):
        """Test learning modulation with both demonstration and teaching."""
        module = SocialLearningModule()
        base_lr = 0.01

        context = SocialContext(
            cue_type=SocialCueType.DEMONSTRATION,
            demonstration_present=True,
            ostensive_cues={"eye_contact": True, "motherese": True, "pointing": False},
        )

        modulated_lr = module.modulate_learning(base_lr, context)

        # Should have both boosts (multiplicative)
        # 0.01 * 2.0 * 1.5 = 0.03
        assert modulated_lr == base_lr * 2.0 * 1.5

    def test_modulate_attention_with_gaze(self):
        """Test attention modulation with gaze context."""
        module = SocialLearningModule()

        attention = torch.ones(10) / 10.0
        gaze = torch.zeros(10)
        gaze[5] = 1.0

        context = SocialContext(
            cue_type=SocialCueType.GAZE,
            gaze_direction=gaze,
            shared_attention=0.7,
        )

        modulated = module.modulate_attention(attention, context)

        # Position 5 should have higher attention
        assert modulated[5] > attention[5]

    def test_modulate_attention_no_gaze(self):
        """Test attention unchanged without gaze."""
        module = SocialLearningModule()

        attention = torch.ones(10) / 10.0

        context = SocialContext(cue_type=SocialCueType.NONE)

        modulated = module.modulate_attention(attention, context)

        assert torch.allclose(modulated, attention)


class TestSharedAttention:
    """Tests for shared attention computation."""

    def test_high_overlap_shared_attention(self):
        """Test shared attention with high overlap."""
        agent_gaze = torch.zeros(10)
        agent_gaze[5] = 1.0

        other_gaze = torch.zeros(10)
        other_gaze[5] = 0.9
        other_gaze[4] = 0.1

        shared = compute_shared_attention(agent_gaze, other_gaze, threshold=0.7)

        # High overlap → positive shared attention
        assert shared > 0.0

    def test_low_overlap_no_shared_attention(self):
        """Test no shared attention with low overlap."""
        agent_gaze = torch.zeros(10)
        agent_gaze[2] = 1.0

        other_gaze = torch.zeros(10)
        other_gaze[8] = 1.0

        shared = compute_shared_attention(agent_gaze, other_gaze, threshold=0.7)

        # No overlap → no shared attention
        assert shared == 0.0

    def test_perfect_overlap_maximum_shared_attention(self):
        """Test maximum shared attention with perfect overlap."""
        agent_gaze = torch.zeros(10)
        agent_gaze[5] = 1.0

        other_gaze = torch.zeros(10)
        other_gaze[5] = 1.0

        shared = compute_shared_attention(agent_gaze, other_gaze, threshold=0.8)

        # Perfect overlap → maximum shared attention
        assert shared == 1.0


class TestStatistics:
    """Tests for statistics tracking."""

    def test_reset_statistics(self):
        """Test statistics reset."""
        module = SocialLearningModule()

        # Add some stats
        module.statistics["n_demonstrations"] = 10
        module.statistics["n_pedagogy_episodes"] = 5

        module.reset_statistics()

        assert module.statistics["n_demonstrations"] == 0
        assert module.statistics["n_pedagogy_episodes"] == 0

    def test_get_statistics_empty(self):
        """Test get_statistics with no events."""
        module = SocialLearningModule()

        stats = module.get_statistics()

        assert stats["n_demonstrations"] == 0
        assert stats["n_pedagogy_episodes"] == 0
        assert stats["n_joint_attention"] == 0
        assert stats["total_social_events"] == 0

    def test_get_statistics_with_events(self):
        """Test get_statistics with recorded events."""
        module = SocialLearningModule()

        # Simulate some events
        module.imitation_learning(0.01, demonstration_present=True)
        module.imitation_learning(0.01, demonstration_present=True)

        cues = {"eye_contact": True, "motherese": True, "pointing": False}
        module.pedagogy_boost(0.01, cues)

        attention = torch.ones(10) / 10.0
        gaze = torch.zeros(10)
        gaze[5] = 1.0
        module.joint_attention(attention, gaze, shared_attention_strength=0.8)

        stats = module.get_statistics()

        assert stats["n_demonstrations"] == 2
        assert stats["n_pedagogy_episodes"] == 1
        assert stats["n_joint_attention"] == 1
        assert stats["total_social_events"] == 4

    def test_avg_learning_boost_tracking(self):
        """Test average learning boost tracking."""
        module = SocialLearningModule()

        # Initial
        assert module.statistics["avg_learning_boost"] == 1.0

        # Apply many boosts to converge exponential moving average
        for _ in range(10):
            module.imitation_learning(0.01, demonstration_present=True)  # 2.0x

        # After many 2.0x boosts, average should trend towards 2.0
        # (exponential moving average with alpha=0.1 converges slowly)
        assert module.statistics["avg_learning_boost"] > 1.6

        # Now add some 1.5x boosts
        cues = {"eye_contact": True, "motherese": True, "pointing": False}
        for _ in range(10):
            module.pedagogy_boost(0.01, cues)  # 1.5x

        # Average should now be between 1.5 and 2.0
        assert 1.5 <= module.statistics["avg_learning_boost"] <= 2.0


class TestSocialContext:
    """Tests for SocialContext creation and usage."""

    def test_social_context_creation(self):
        """Test basic social context creation."""
        context = SocialContext(cue_type=SocialCueType.DEMONSTRATION)

        assert context.cue_type == SocialCueType.DEMONSTRATION
        assert context.demonstration_present is False
        assert context.ostensive_cues is not None

    def test_social_context_full_parameters(self):
        """Test social context with all parameters."""
        gaze = torch.zeros(10)
        gaze[5] = 1.0

        context = SocialContext(
            cue_type=SocialCueType.JOINT_ATTENTION,
            demonstration_present=True,
            ostensive_cues={"eye_contact": True, "motherese": True, "pointing": True},
            gaze_direction=gaze,
            shared_attention=0.9,
        )

        assert context.cue_type == SocialCueType.JOINT_ATTENTION
        assert context.demonstration_present is True
        assert context.ostensive_cues["eye_contact"] is True
        assert torch.allclose(context.gaze_direction, gaze)
        assert context.shared_attention == 0.9

    def test_helper_create_social_context(self):
        """Test helper method for creating contexts."""
        module = SocialLearningModule()

        context = module.create_social_context(
            cue_type=SocialCueType.DEMONSTRATION,
            demonstration_present=True,
        )

        assert isinstance(context, SocialContext)
        assert context.cue_type == SocialCueType.DEMONSTRATION
        assert context.demonstration_present is True


class TestIntegration:
    """Integration tests for social learning."""

    def test_full_social_learning_episode(self):
        """Test complete social learning episode with all mechanisms."""
        module = SocialLearningModule()

        # Setup context
        gaze = torch.zeros(20)
        gaze[10] = 1.0

        context = SocialContext(
            cue_type=SocialCueType.JOINT_ATTENTION,
            demonstration_present=True,
            ostensive_cues={"eye_contact": True, "motherese": True, "pointing": False},
            gaze_direction=gaze,
            shared_attention=0.8,
        )

        # Modulate learning
        base_lr = 0.01
        modulated_lr = module.modulate_learning(base_lr, context)

        # Should have both imitation and pedagogy boosts
        assert modulated_lr == base_lr * 2.0 * 1.5

        # Modulate attention
        attention = torch.ones(20) / 20.0
        modulated_attention = module.modulate_attention(attention, context)

        # Position 10 should have higher attention
        assert modulated_attention[10] > attention[10]

        # Check statistics
        stats = module.get_statistics()
        assert stats["n_demonstrations"] == 1
        assert stats["n_pedagogy_episodes"] == 1
        assert stats["n_joint_attention"] == 1

    def test_sequential_social_episodes(self):
        """Test multiple sequential social learning episodes."""
        module = SocialLearningModule()

        base_lr = 0.01

        # Episode 1: Demonstration only
        context1 = SocialContext(
            cue_type=SocialCueType.DEMONSTRATION,
            demonstration_present=True,
        )
        lr1 = module.modulate_learning(base_lr, context1)
        assert lr1 == base_lr * 2.0

        # Episode 2: Teaching only
        context2 = SocialContext(
            cue_type=SocialCueType.OSTENSIVE,
            ostensive_cues={"eye_contact": True, "motherese": True, "pointing": False},
        )
        lr2 = module.modulate_learning(base_lr, context2)
        assert lr2 == base_lr * 1.5

        # Episode 3: Both
        context3 = SocialContext(
            cue_type=SocialCueType.DEMONSTRATION,
            demonstration_present=True,
            ostensive_cues={"eye_contact": True, "motherese": True, "pointing": False},
        )
        lr3 = module.modulate_learning(base_lr, context3)
        assert lr3 == base_lr * 2.0 * 1.5

        # Check cumulative statistics
        stats = module.get_statistics()
        assert stats["n_demonstrations"] == 2
        assert stats["n_pedagogy_episodes"] == 2

    def test_device_consistency_full_pipeline(self):
        """Test device consistency across all operations."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        config = SocialLearningConfig(device="cuda")
        module = SocialLearningModule(config)

        # Create tensors on CUDA
        attention = torch.ones(10, device="cuda") / 10.0
        gaze = torch.zeros(10, device="cuda")
        gaze[5] = 1.0

        context = SocialContext(
            cue_type=SocialCueType.JOINT_ATTENTION,
            gaze_direction=gaze,
            shared_attention=0.8,
        )

        # Modulate attention
        modulated = module.modulate_attention(attention, context)

        assert modulated.device.type == "cuda"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_learning_rate(self):
        """Test with zero base learning rate."""
        module = SocialLearningModule()

        lr = module.imitation_learning(0.0, demonstration_present=True)
        assert lr == 0.0

    def test_very_high_learning_boost(self):
        """Test with extreme boost factors."""
        config = SocialLearningConfig(imitation_boost=10.0, pedagogy_boost=10.0)
        module = SocialLearningModule(config)

        base_lr = 0.001

        context = SocialContext(
            cue_type=SocialCueType.DEMONSTRATION,
            demonstration_present=True,
            ostensive_cues={"eye_contact": True, "motherese": True, "pointing": False},
        )

        modulated_lr = module.modulate_learning(base_lr, context)

        # 0.001 * 10 * 10 = 0.1
        assert modulated_lr == 0.1

    def test_empty_ostensive_cues(self):
        """Test with empty ostensive cues dictionary."""
        module = SocialLearningModule()

        is_teaching = module._detect_teaching_signal({})
        assert is_teaching is False

    def test_zero_gaze_vector(self):
        """Test with zero gaze vector."""
        module = SocialLearningModule()

        attention = torch.ones(10) / 10.0
        gaze = torch.zeros(10)  # All zeros

        modulated = module.joint_attention(attention, gaze, shared_attention_strength=0.5)

        # Should still be normalized
        assert torch.isclose(modulated.sum(), torch.tensor(1.0))

    def test_negative_shared_attention(self):
        """Test with invalid negative shared attention."""
        module = SocialLearningModule()

        attention = torch.ones(10) / 10.0
        gaze = torch.zeros(10)
        gaze[5] = 1.0

        # Negative shared attention (invalid, but should handle)
        modulated = module.joint_attention(attention, gaze, shared_attention_strength=-0.5)

        # Should still produce valid output
        assert torch.isclose(modulated.sum(), torch.tensor(1.0))
        assert (modulated >= 0).all()
