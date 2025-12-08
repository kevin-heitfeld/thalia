"""
Tests for executive function tasks.
"""

import pytest
import torch
import numpy as np

from thalia.tasks.executive_function import (
    ExecutiveFunctionTasks,
    TaskType,
    StimulusType,
    GoNoGoConfig,
    DelayedGratificationConfig,
    DCCSConfig,
)


class TestGoNoGo:
    """Tests for Go/No-Go task."""
    
    def test_initialization(self):
        """Test task initialization."""
        tasks = ExecutiveFunctionTasks()
        assert tasks.statistics["n_trials"] == 0
        assert tasks.statistics["n_correct"] == 0
    
    def test_generate_trials(self):
        """Test Go/No-Go trial generation."""
        tasks = ExecutiveFunctionTasks()
        config = GoNoGoConfig(n_stimuli=100, target_probability=0.7)
        
        stimuli, stimulus_types, correct_responses = tasks.go_no_go(config)
        
        # Check counts
        assert len(stimuli) == 100
        assert len(stimulus_types) == 100
        assert len(correct_responses) == 100
        
        # Check types
        assert all(isinstance(s, torch.Tensor) for s in stimuli)
        assert all(isinstance(st, StimulusType) for st in stimulus_types)
        assert all(isinstance(cr, bool) for cr in correct_responses)
        
        # Check stimulus dimensions
        assert all(s.shape == (config.stimulus_dim,) for s in stimuli)
    
    def test_target_probability(self):
        """Test target probability is respected."""
        tasks = ExecutiveFunctionTasks()
        config = GoNoGoConfig(n_stimuli=1000, target_probability=0.7)
        
        _, stimulus_types, _ = tasks.go_no_go(config)
        
        n_targets = sum(1 for st in stimulus_types if st == StimulusType.TARGET)
        target_ratio = n_targets / len(stimulus_types)
        
        # Should be close to 0.7 (within 5%)
        assert abs(target_ratio - 0.7) < 0.05
    
    def test_target_stimulus_pattern(self):
        """Test target stimuli have correct pattern."""
        tasks = ExecutiveFunctionTasks()
        config = GoNoGoConfig(stimulus_dim=64)
        
        stimulus = tasks._generate_target_stimulus(config.stimulus_dim, torch.device("cpu"))
        
        # First half should be high
        assert stimulus[:32].mean() > 0.5
        # Second half should be low
        assert stimulus[32:].mean() < 0.3
    
    def test_distractor_stimulus_pattern(self):
        """Test distractor stimuli have correct pattern."""
        tasks = ExecutiveFunctionTasks()
        config = GoNoGoConfig(stimulus_dim=64)
        
        stimulus = tasks._generate_distractor_stimulus(config.stimulus_dim, torch.device("cpu"))
        
        # First half should be low
        assert stimulus[:32].mean() < 0.3
        # Second half should be high
        assert stimulus[32:].mean() > 0.5
    
    def test_correct_response_mapping(self):
        """Test correct responses match stimulus types."""
        tasks = ExecutiveFunctionTasks()
        config = GoNoGoConfig(n_stimuli=100)
        
        _, stimulus_types, correct_responses = tasks.go_no_go(config)
        
        for st, cr in zip(stimulus_types, correct_responses):
            if st == StimulusType.TARGET:
                assert cr is True  # Should respond
            elif st == StimulusType.DISTRACTOR:
                assert cr is False  # Should inhibit
    
    def test_evaluation_perfect_performance(self):
        """Test evaluation with perfect performance."""
        tasks = ExecutiveFunctionTasks()
        config = GoNoGoConfig(n_stimuli=100)
        
        _, stimulus_types, correct_responses = tasks.go_no_go(config)
        
        # Perfect responses
        responses = correct_responses.copy()
        
        metrics = tasks.evaluate_go_no_go(responses, correct_responses, stimulus_types)
        
        assert metrics["overall_accuracy"] == 1.0
        assert metrics["go_accuracy"] == 1.0
        assert metrics["no_go_accuracy"] == 1.0
        assert metrics["hit_rate"] == 1.0
        assert metrics["false_alarm_rate"] == 0.0
        assert metrics["d_prime"] > 3.0  # High sensitivity
    
    def test_evaluation_chance_performance(self):
        """Test evaluation with random responses."""
        tasks = ExecutiveFunctionTasks()
        config = GoNoGoConfig(n_stimuli=100)
        
        _, stimulus_types, correct_responses = tasks.go_no_go(config)
        
        # Random responses
        responses = [bool(np.random.rand() > 0.5) for _ in range(len(correct_responses))]
        
        metrics = tasks.evaluate_go_no_go(responses, correct_responses, stimulus_types)
        
        # Should be around 0.5
        assert 0.3 < metrics["overall_accuracy"] < 0.7
        assert -1.0 < metrics["d_prime"] < 1.0  # Low sensitivity
    
    def test_evaluation_statistics(self):
        """Test that statistics are updated correctly."""
        tasks = ExecutiveFunctionTasks()
        config = GoNoGoConfig(n_stimuli=50)
        
        _, stimulus_types, correct_responses = tasks.go_no_go(config)
        responses = correct_responses.copy()
        
        tasks.evaluate_go_no_go(responses, correct_responses, stimulus_types)
        
        assert tasks.statistics["n_trials"] == 50
        assert tasks.statistics["n_correct"] == 50
        assert "go_no_go" in tasks.statistics["by_task"]
    
    def test_batch_generation(self):
        """Test batch generation for Go/No-Go."""
        tasks = ExecutiveFunctionTasks()
        
        stimuli, labels = tasks.generate_batch(TaskType.GO_NO_GO, batch_size=32)
        
        assert stimuli.shape == (32, 64)  # Default stimulus_dim
        assert labels.shape == (32,)
        assert labels.dtype == torch.long


class TestDelayedGratification:
    """Tests for delayed gratification task."""
    
    def test_task_generation(self):
        """Test delayed gratification task generation."""
        tasks = ExecutiveFunctionTasks()
        config = DelayedGratificationConfig(
            immediate_reward=1.0,
            delayed_reward=3.0,
            delay_steps=50,
            discount_rate=0.95,
        )
        
        task_info = tasks.delayed_gratification(config)
        
        assert "immediate_reward" in task_info
        assert "delayed_reward" in task_info
        assert "delay_steps" in task_info
        assert "present_value_delayed" in task_info
        assert "optimal_choice" in task_info
    
    def test_optimal_choice_delayed(self):
        """Test optimal choice when delayed is better."""
        tasks = ExecutiveFunctionTasks()
        config = DelayedGratificationConfig(
            immediate_reward=1.0,
            delayed_reward=5.0,  # Much larger
            delay_steps=10,  # Short delay
            discount_rate=0.95,
        )
        
        task_info = tasks.delayed_gratification(config)
        
        assert task_info["optimal_choice"] == "delayed"
    
    def test_optimal_choice_immediate(self):
        """Test optimal choice when immediate is better."""
        tasks = ExecutiveFunctionTasks()
        config = DelayedGratificationConfig(
            immediate_reward=1.0,
            delayed_reward=1.2,  # Only slightly larger
            delay_steps=100,  # Long delay
            discount_rate=0.90,  # Higher discounting
        )
        
        task_info = tasks.delayed_gratification(config)
        
        # With high discount and long delay, immediate may be better
        # PV = 1.2 * 0.9^100 ≈ 0.000037 < 1.0
        assert task_info["optimal_choice"] == "immediate"
    
    def test_temporal_discounting(self):
        """Test temporal discounting calculation."""
        tasks = ExecutiveFunctionTasks()
        config = DelayedGratificationConfig(
            immediate_reward=1.0,
            delayed_reward=2.0,
            delay_steps=10,
            discount_rate=0.9,
        )
        
        task_info = tasks.delayed_gratification(config)
        
        expected_pv = 2.0 * (0.9 ** 10)  # ≈ 0.697
        assert abs(task_info["present_value_delayed"] - expected_pv) < 0.01
    
    def test_evaluation_perfect_patience(self):
        """Test evaluation with always choosing delayed."""
        tasks = ExecutiveFunctionTasks()
        
        choices = ["delayed"] * 10
        optimal_choices = ["delayed"] * 10
        
        metrics = tasks.evaluate_delayed_gratification(choices, optimal_choices)
        
        assert metrics["accuracy"] == 1.0
        assert metrics["delayed_proportion"] == 1.0
        assert metrics["immediate_proportion"] == 0.0
    
    def test_evaluation_no_patience(self):
        """Test evaluation with always choosing immediate."""
        tasks = ExecutiveFunctionTasks()
        
        choices = ["immediate"] * 10
        optimal_choices = ["immediate"] * 10
        
        metrics = tasks.evaluate_delayed_gratification(choices, optimal_choices)
        
        assert metrics["accuracy"] == 1.0
        assert metrics["delayed_proportion"] == 0.0
        assert metrics["immediate_proportion"] == 1.0
    
    def test_evaluation_mixed_choices(self):
        """Test evaluation with mixed choices."""
        tasks = ExecutiveFunctionTasks()
        
        choices = ["delayed", "immediate", "delayed", "immediate"]
        optimal_choices = ["delayed", "delayed", "immediate", "immediate"]
        
        metrics = tasks.evaluate_delayed_gratification(choices, optimal_choices)
        
        assert metrics["accuracy"] == 0.5  # 2/4 correct
        assert metrics["delayed_proportion"] == 0.5
        assert metrics["immediate_proportion"] == 0.5
    
    def test_evaluation_statistics(self):
        """Test statistics tracking for delayed gratification."""
        tasks = ExecutiveFunctionTasks()
        
        choices = ["delayed"] * 5
        optimal_choices = ["delayed"] * 5
        
        tasks.evaluate_delayed_gratification(choices, optimal_choices)
        
        assert tasks.statistics["n_trials"] == 5
        assert tasks.statistics["n_correct"] == 5
        assert "delayed_gratification" in tasks.statistics["by_task"]


class TestDCCS:
    """Tests for Dimensional Change Card Sort."""
    
    def test_task_generation(self):
        """Test DCCS task generation."""
        tasks = ExecutiveFunctionTasks()
        config = DCCSConfig(n_trials=24, switch_trial=12)
        
        cards, rules, correct_bins = tasks.dccs(config)
        
        assert len(cards) == 24
        assert len(rules) == 24
        assert len(correct_bins) == 24
    
    def test_rule_switching(self):
        """Test rule switches at correct trial."""
        tasks = ExecutiveFunctionTasks()
        config = DCCSConfig(n_trials=24, switch_trial=12)
        
        _, rules, _ = tasks.dccs(config)
        
        # First 12 trials: color rule
        assert all(r == "color" for r in rules[:12])
        # Last 12 trials: shape rule
        assert all(r == "shape" for r in rules[12:])
    
    def test_card_encoding(self):
        """Test card encoding format."""
        tasks = ExecutiveFunctionTasks()
        config = DCCSConfig(n_dimensions=2, n_features_per_dim=2)
        
        cards, _, _ = tasks.dccs(config)
        
        # Each card should be one-hot encoded
        for card in cards:
            assert card.shape == (4,)  # 2 dims × 2 features
            assert card.sum() == 2  # One feature per dimension
    
    def test_correct_bin_color_rule(self):
        """Test correct bin assignment for color rule."""
        tasks = ExecutiveFunctionTasks()
        config = DCCSConfig(n_trials=12, switch_trial=24)  # Only color rule
        
        cards, rules, correct_bins = tasks.dccs(config)
        
        for card, rule, bin in zip(cards, rules, correct_bins):
            assert rule == "color"
            # Bin should match color (first 2 positions)
            color_idx = torch.argmax(card[:2]).item()
            assert bin == color_idx
    
    def test_correct_bin_shape_rule(self):
        """Test correct bin assignment for shape rule."""
        tasks = ExecutiveFunctionTasks()
        config = DCCSConfig(n_trials=12, switch_trial=0)  # Only shape rule
        
        cards, rules, correct_bins = tasks.dccs(config)
        
        for card, rule, bin in zip(cards, rules, correct_bins):
            assert rule == "shape"
            # Bin should match shape (last 2 positions)
            shape_idx = torch.argmax(card[2:]).item()
            assert bin == shape_idx
    
    def test_evaluation_perfect_performance(self):
        """Test DCCS evaluation with perfect performance."""
        tasks = ExecutiveFunctionTasks()
        config = DCCSConfig(n_trials=24, switch_trial=12)
        
        _, _, correct_bins = tasks.dccs(config)
        
        # Perfect responses
        responses = correct_bins.copy()
        
        metrics = tasks.evaluate_dccs(responses, correct_bins, None, switch_trial=12)
        
        assert metrics["overall_accuracy"] == 1.0
        assert metrics["pre_switch_accuracy"] == 1.0
        assert metrics["post_switch_accuracy"] == 1.0
        assert metrics["switch_cost"] == 0.0
    
    def test_evaluation_perseveration(self):
        """Test DCCS evaluation with perseveration errors."""
        tasks = ExecutiveFunctionTasks()
        config = DCCSConfig(n_trials=24, switch_trial=12)
        
        cards, rules, correct_bins = tasks.dccs(config)
        
        # Perfect pre-switch, but perseverate post-switch
        responses = []
        for i, card in enumerate(cards):
            if i < 12:
                # Pre-switch: correct (color rule)
                responses.append(correct_bins[i])
            else:
                # Post-switch: perseverate with old color rule
                color_idx = torch.argmax(card[:2]).item()
                responses.append(color_idx)
        
        metrics = tasks.evaluate_dccs(responses, correct_bins, rules, switch_trial=12)
        
        assert metrics["pre_switch_accuracy"] == 1.0
        # Post-switch will be poor if color and shape often differ
        assert metrics["post_switch_accuracy"] < 1.0
        assert metrics["switch_cost"] > 0.0
    
    def test_evaluation_statistics(self):
        """Test statistics tracking for DCCS."""
        tasks = ExecutiveFunctionTasks()
        config = DCCSConfig(n_trials=24)
        
        _, _, correct_bins = tasks.dccs(config)
        responses = correct_bins.copy()
        
        tasks.evaluate_dccs(responses, correct_bins, None, switch_trial=12)
        
        assert tasks.statistics["n_trials"] == 24
        assert tasks.statistics["n_correct"] == 24
        assert "dccs" in tasks.statistics["by_task"]
    
    def test_batch_generation(self):
        """Test batch generation for DCCS."""
        tasks = ExecutiveFunctionTasks()
        
        cards, labels = tasks.generate_batch(TaskType.DCCS, batch_size=24)
        
        assert cards.shape == (24, 4)  # 2 dims × 2 features
        assert labels.shape == (24,)
        assert labels.dtype == torch.long


class TestStatistics:
    """Tests for task statistics."""
    
    def test_reset_statistics(self):
        """Test statistics reset."""
        tasks = ExecutiveFunctionTasks()
        
        # Add some stats
        tasks.statistics["n_trials"] = 100
        tasks.statistics["n_correct"] = 80
        
        tasks.reset_statistics()
        
        assert tasks.statistics["n_trials"] == 0
        assert tasks.statistics["n_correct"] == 0
        assert len(tasks.statistics["by_task"]) == 0
    
    def test_get_statistics_empty(self):
        """Test get_statistics with no data."""
        tasks = ExecutiveFunctionTasks()
        
        stats = tasks.get_statistics()
        
        assert stats["overall_accuracy"] == 0.0
        assert stats["n_trials"] == 0
        assert len(stats["by_task"]) == 0
    
    def test_get_statistics_multiple_tasks(self):
        """Test get_statistics with multiple task types."""
        tasks = ExecutiveFunctionTasks()
        
        # Run Go/No-Go
        config_gng = GoNoGoConfig(n_stimuli=50)
        _, stimulus_types_gng, correct_responses_gng = tasks.go_no_go(config_gng)
        tasks.evaluate_go_no_go(correct_responses_gng, correct_responses_gng, stimulus_types_gng)
        
        # Run delayed gratification
        choices = ["delayed"] * 10
        optimal_choices = ["delayed"] * 10
        tasks.evaluate_delayed_gratification(choices, optimal_choices)
        
        # Run DCCS
        config_dccs = DCCSConfig(n_trials=24)
        _, _, correct_bins = tasks.dccs(config_dccs)
        tasks.evaluate_dccs(correct_bins, correct_bins, None, switch_trial=12)
        
        stats = tasks.get_statistics()
        
        assert stats["overall_accuracy"] == 1.0  # All perfect
        assert stats["n_trials"] == 84  # 50 + 10 + 24
        assert "go_no_go" in stats["by_task"]
        assert "delayed_gratification" in stats["by_task"]
        assert "dccs" in stats["by_task"]


class TestIntegration:
    """Integration tests for executive function tasks."""
    
    def test_sequential_tasks(self):
        """Test running multiple tasks sequentially."""
        tasks = ExecutiveFunctionTasks()
        
        # 1. Go/No-Go
        config_gng = GoNoGoConfig(n_stimuli=30)
        stimuli, stimulus_types, correct_responses = tasks.go_no_go(config_gng)
        metrics_gng = tasks.evaluate_go_no_go(correct_responses, correct_responses, stimulus_types)
        
        # 2. Delayed Gratification
        config_dg = DelayedGratificationConfig()
        task_info = tasks.delayed_gratification(config_dg)
        choices = [task_info["optimal_choice"]] * 10
        optimal = [task_info["optimal_choice"]] * 10
        metrics_dg = tasks.evaluate_delayed_gratification(choices, optimal)
        
        # 3. DCCS
        config_dccs = DCCSConfig(n_trials=20)
        cards, rules, bins = tasks.dccs(config_dccs)
        metrics_dccs = tasks.evaluate_dccs(bins, bins, rules, switch_trial=10)
        
        # All tasks should have run successfully
        assert metrics_gng["overall_accuracy"] == 1.0
        assert metrics_dg["accuracy"] == 1.0
        assert metrics_dccs["overall_accuracy"] == 1.0
        
        # Statistics should reflect all tasks
        stats = tasks.get_statistics()
        assert stats["n_trials"] == 60  # 30 + 10 + 20
        assert len(stats["by_task"]) == 3
    
    def test_device_consistency(self):
        """Test that all tasks respect device setting."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        tasks = ExecutiveFunctionTasks()
        device = "cuda"
        
        # Go/No-Go
        config_gng = GoNoGoConfig(n_stimuli=10, device=device)
        stimuli, _, _ = tasks.go_no_go(config_gng)
        assert all(s.device.type == device for s in stimuli)
        
        # DCCS
        config_dccs = DCCSConfig(n_trials=10, device=device)
        cards, _, _ = tasks.dccs(config_dccs)
        assert all(c.device.type == device for c in cards)
    
    def test_reproducibility_with_seed(self):
        """Test that tasks are reproducible with same seed."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        tasks1 = ExecutiveFunctionTasks()
        config1 = GoNoGoConfig(n_stimuli=50)
        _, types1, responses1 = tasks1.go_no_go(config1)
        
        np.random.seed(42)
        torch.manual_seed(42)
        
        tasks2 = ExecutiveFunctionTasks()
        config2 = GoNoGoConfig(n_stimuli=50)
        _, types2, responses2 = tasks2.go_no_go(config2)
        
        assert types1 == types2
        assert responses1 == responses2


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_evaluation(self):
        """Test evaluation with empty lists."""
        tasks = ExecutiveFunctionTasks()
        
        with pytest.raises((ValueError, ZeroDivisionError, IndexError)):
            tasks.evaluate_go_no_go([], [], [])
    
    def test_invalid_task_type_batch(self):
        """Test batch generation with invalid task type."""
        tasks = ExecutiveFunctionTasks()
        
        with pytest.raises(ValueError):
            tasks.generate_batch(TaskType.DELAYED_GRATIFICATION, batch_size=10)
    
    def test_zero_trials(self):
        """Test with zero trials."""
        tasks = ExecutiveFunctionTasks()
        config = GoNoGoConfig(n_stimuli=0)
        
        stimuli, types, responses = tasks.go_no_go(config)
        
        assert len(stimuli) == 0
        assert len(types) == 0
        assert len(responses) == 0
    
    def test_extreme_discount_rate(self):
        """Test delayed gratification with extreme discount rates."""
        tasks = ExecutiveFunctionTasks()
        
        # Very patient (discount rate near 1, short delay)
        config_patient = DelayedGratificationConfig(
            immediate_reward=1.0,
            delayed_reward=1.5,
            delay_steps=10,
            discount_rate=0.99,
        )
        task_info = tasks.delayed_gratification(config_patient)
        # 1.5 * 0.99^10 = 1.357 > 1.0
        assert task_info["optimal_choice"] == "delayed"
        
        # Very impatient (discount rate near 0)
        config_impatient = DelayedGratificationConfig(
            immediate_reward=1.0,
            delayed_reward=10.0,
            delay_steps=10,
            discount_rate=0.1,
        )
        task_info = tasks.delayed_gratification(config_impatient)
        # 10.0 * 0.1^10 = 0.00001 << 1.0
        assert task_info["optimal_choice"] == "immediate"
