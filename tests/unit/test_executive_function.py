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
    TaskSwitchingConfig,
    TowerOfHanoiConfig,
    RavensMatricesConfig,
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


# ============================================================================
# Stage 2: Task Switching Tests
# ============================================================================

class TestTaskSwitching:
    """Tests for task switching paradigm."""
    
    def test_initialization(self):
        """Test task switching initialization."""
        tasks = ExecutiveFunctionTasks()
        config = TaskSwitchingConfig()
        
        stimuli, task_cues, responses, is_switch = tasks.task_switching(config)
        
        assert len(stimuli) == config.n_trials
        assert len(task_cues) == config.n_trials
        assert len(responses) == config.n_trials
        assert len(is_switch) == config.n_trials
    
    def test_stimulus_dimensions(self):
        """Test stimulus dimensions are correct."""
        tasks = ExecutiveFunctionTasks()
        config = TaskSwitchingConfig(stimulus_dim=32)
        
        stimuli, _, _, _ = tasks.task_switching(config)
        
        assert all(s.shape == (32,) for s in stimuli)
    
    def test_task_cues(self):
        """Test task cues are binary."""
        tasks = ExecutiveFunctionTasks()
        config = TaskSwitchingConfig(n_tasks=2)
        
        _, task_cues, _, _ = tasks.task_switching(config)
        
        # All cues should be 0 or 1
        assert all(cue in [0, 1] for cue in task_cues)
    
    def test_switch_probability(self):
        """Test switch probability is approximately respected."""
        tasks = ExecutiveFunctionTasks()
        config = TaskSwitchingConfig(
            n_trials=1000,
            switch_probability=0.3,
        )
        
        _, _, _, is_switch = tasks.task_switching(config)
        
        switch_rate = sum(is_switch) / len(is_switch)
        
        # Should be close to 0.3 (within tolerance)
        assert 0.25 < switch_rate < 0.35
    
    def test_first_trial_not_switch(self):
        """Test first trial is never marked as switch."""
        tasks = ExecutiveFunctionTasks()
        config = TaskSwitchingConfig()
        
        _, _, _, is_switch = tasks.task_switching(config)
        
        assert is_switch[0] == False
    
    def test_correct_responses(self):
        """Test responses are valid."""
        tasks = ExecutiveFunctionTasks()
        config = TaskSwitchingConfig(n_responses=2)
        
        _, _, responses, _ = tasks.task_switching(config)
        
        # All responses should be 0 or 1
        assert all(r in [0, 1] for r in responses)
    
    def test_evaluate_task_switching(self):
        """Test evaluation metrics."""
        tasks = ExecutiveFunctionTasks()
        config = TaskSwitchingConfig(n_trials=40)
        
        _, _, correct_responses, is_switch = tasks.task_switching(config)
        
        # Perfect performance
        metrics = tasks.evaluate_task_switching(
            responses=correct_responses,
            correct_responses=correct_responses,
            is_switch=is_switch,
        )
        
        assert metrics["overall_accuracy"] == 1.0
        assert metrics["switch_accuracy"] == 1.0
        assert metrics["repeat_accuracy"] == 1.0
        assert metrics["switch_cost"] == 0.0
    
    def test_evaluate_switch_cost(self):
        """Test switch cost calculation."""
        tasks = ExecutiveFunctionTasks()
        
        # Create scenario with switch cost
        correct = [1, 1, 0, 0, 1]
        is_switch = [False, False, True, False, True]
        
        # Worse on switch trials: correct on repeat, wrong on switch
        responses = [1, 1, 1, 0, 0]  # Wrong on both switches (indices 2 and 4)
        
        metrics = tasks.evaluate_task_switching(
            responses=responses,
            correct_responses=correct,
            is_switch=is_switch,
        )
        
        # Overall: 3/5 correct
        assert metrics["overall_accuracy"] == pytest.approx(3/5)
        # Repeat trials (indices 0,1,3): all correct = 3/3
        assert metrics["repeat_accuracy"] == pytest.approx(1.0)
        # Switch trials (indices 2,4): all wrong = 0/2
        assert metrics["switch_accuracy"] == pytest.approx(0.0)
        # Switch cost should be positive (1.0 - 0.0)
        assert metrics["switch_cost"] == pytest.approx(1.0)
    
    def test_statistics_update(self):
        """Test statistics are updated correctly."""
        tasks = ExecutiveFunctionTasks()
        config = TaskSwitchingConfig(n_trials=30)
        
        _, _, correct_responses, is_switch = tasks.task_switching(config)
        
        # Evaluate with perfect performance
        tasks.evaluate_task_switching(
            responses=correct_responses,
            correct_responses=correct_responses,
            is_switch=is_switch,
        )
        
        assert tasks.statistics["n_trials"] == 30
        assert tasks.statistics["n_correct"] == 30
        assert "task_switching" in tasks.statistics["by_task"]
        assert tasks.statistics["by_task"]["task_switching"]["total"] == 30
    
    def test_switch_and_repeat_counts(self):
        """Test switch and repeat trial counts."""
        tasks = ExecutiveFunctionTasks()
        
        correct = [1, 0, 1, 0, 1]
        is_switch = [False, True, False, True, False]
        
        metrics = tasks.evaluate_task_switching(
            responses=correct,
            correct_responses=correct,
            is_switch=is_switch,
        )
        
        assert metrics["n_switch_trials"] == 2
        assert metrics["n_repeat_trials"] == 3
    
    def test_with_device(self):
        """Test task switching respects device parameter."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        tasks = ExecutiveFunctionTasks()
        config = TaskSwitchingConfig(device="cuda")
        
        stimuli, _, _, _ = tasks.task_switching(config)
        
        assert all(s.device.type == "cuda" for s in stimuli)
    
    def test_generate_batch_task_switching(self):
        """Test batch generation for task switching."""
        tasks = ExecutiveFunctionTasks()
        
        batch_stimuli, batch_labels = tasks.generate_batch(
            task_type=TaskType.TASK_SWITCHING,
            batch_size=20,
        )
        
        assert batch_stimuli.shape[0] == 20
        assert batch_labels.shape[0] == 20
        # Stimulus includes task cue (extra dimension)
        assert batch_stimuli.shape[1] == 64 + 1  # Default stimulus_dim + cue
    
    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        tasks1 = ExecutiveFunctionTasks()
        tasks2 = ExecutiveFunctionTasks()
        
        np.random.seed(42)
        torch.manual_seed(42)
        _, cues1, _, switches1 = tasks1.task_switching(
            TaskSwitchingConfig(n_trials=20)
        )
        
        np.random.seed(42)
        torch.manual_seed(42)
        _, cues2, _, switches2 = tasks2.task_switching(
            TaskSwitchingConfig(n_trials=20)
        )
        
        assert cues1 == cues2
        assert switches1 == switches2
    
    def test_no_switches_edge_case(self):
        """Test evaluation with no switches."""
        tasks = ExecutiveFunctionTasks()
        
        correct = [1, 0, 1, 0]
        is_switch = [False, False, False, False]
        
        metrics = tasks.evaluate_task_switching(
            responses=correct,
            correct_responses=correct,
            is_switch=is_switch,
        )
        
        assert metrics["repeat_accuracy"] == 1.0
        assert np.isnan(metrics["switch_accuracy"])
        assert np.isnan(metrics["switch_cost"])
    
    def test_all_switches_edge_case(self):
        """Test evaluation with all switches."""
        tasks = ExecutiveFunctionTasks()
        
        correct = [1, 0, 1, 0]
        is_switch = [True, True, True, True]
        
        metrics = tasks.evaluate_task_switching(
            responses=correct,
            correct_responses=correct,
            is_switch=is_switch,
        )
        
        assert metrics["switch_accuracy"] == 1.0
        assert np.isnan(metrics["repeat_accuracy"])
        assert np.isnan(metrics["switch_cost"])


# ============================================================================
# Integration Tests
# ============================================================================

class TestExecutiveFunctionIntegration:
    """Integration tests across multiple executive function tasks."""
    
    def test_stage2_tasks_integration(self):
        """Test Stage 2 tasks (DCCS + task switching) together."""
        tasks = ExecutiveFunctionTasks()
        
        # DCCS
        dccs_cards, dccs_rules, dccs_correct = tasks.dccs()
        dccs_responses = dccs_correct  # Perfect performance
        dccs_metrics = tasks.evaluate_dccs(
            responses=dccs_responses,
            correct_bins=dccs_correct,
            rules=dccs_rules,
        )
        
        # Task Switching
        ts_stimuli, ts_cues, ts_correct, ts_switch = tasks.task_switching()
        ts_metrics = tasks.evaluate_task_switching(
            responses=ts_correct,
            correct_responses=ts_correct,
            is_switch=ts_switch,
        )
        
        # Both should have perfect accuracy
        assert dccs_metrics["overall_accuracy"] == 1.0
        assert ts_metrics["overall_accuracy"] == 1.0
        
        # Statistics should include both tasks
        stats = tasks.get_statistics()
        assert "dccs" in stats["by_task"]
        assert "task_switching" in stats["by_task"]


class TestTowerOfHanoi:
    """Tests for Tower of Hanoi planning task (Stage 3-4)."""
    
    def test_initialization(self):
        """Test Tower of Hanoi task generation."""
        tasks = ExecutiveFunctionTasks()
        config = TowerOfHanoiConfig(n_disks=3)
        
        states, disk_moved, moves, optimal = tasks.tower_of_hanoi(config)
        
        # Check return structure
        assert len(states) > 0
        assert len(disk_moved) == len(moves)
        assert optimal == 7  # 2^3 - 1 = 7 moves
        assert all(isinstance(s, torch.Tensor) for s in states)
        assert all(isinstance(d, int) for d in disk_moved)
        assert all(isinstance(m, tuple) and len(m) == 2 for m in moves)
    
    def test_optimal_solution_3_disks(self):
        """Test that 3-disk solution is optimal."""
        tasks = ExecutiveFunctionTasks()
        config = TowerOfHanoiConfig(n_disks=3)
        
        states, disk_moved, moves, optimal = tasks.tower_of_hanoi(config)
        
        # Should take exactly 7 moves (optimal)
        assert len(moves) == 7
        assert optimal == 7
    
    def test_optimal_solution_4_disks(self):
        """Test that 4-disk solution is optimal."""
        tasks = ExecutiveFunctionTasks()
        config = TowerOfHanoiConfig(n_disks=4, optimal_moves=15)
        
        states, disk_moved, moves, optimal = tasks.tower_of_hanoi(config)
        
        # Should take exactly 15 moves (optimal for 4 disks)
        assert len(moves) == 15
        assert optimal == 15
    
    def test_state_encoding(self):
        """Test state encoding dimensions."""
        tasks = ExecutiveFunctionTasks()
        config = TowerOfHanoiConfig(n_disks=3, encode_dim=64)
        
        states, _, _, _ = tasks.tower_of_hanoi(config)
        
        # States should have correct dimension (3 pegs × encode_dim)
        expected_dim = 3 * config.encode_dim
        assert all(s.shape == (expected_dim,) for s in states)
    
    def test_valid_moves(self):
        """Test that all moves are valid."""
        tasks = ExecutiveFunctionTasks()
        config = TowerOfHanoiConfig(n_disks=3)
        
        _, _, moves, _ = tasks.tower_of_hanoi(config)
        
        # All moves should be between valid pegs (0, 1, 2)
        for from_peg, to_peg in moves:
            assert 0 <= from_peg <= 2
            assert 0 <= to_peg <= 2
            assert from_peg != to_peg
    
    def test_evaluate_perfect_solution(self):
        """Test evaluation of optimal solution."""
        tasks = ExecutiveFunctionTasks()
        config = TowerOfHanoiConfig(n_disks=3)
        
        states, _, moves, optimal = tasks.tower_of_hanoi(config)
        
        # Create target state (all disks on peg 2)
        target_state = tasks._encode_hanoi_state(
            [[], [], [0, 1, 2]],  # All disks on target peg
            config.encode_dim,
            config.n_disks,
            config.device,
        )
        
        metrics = tasks.evaluate_tower_of_hanoi(
            moves,
            optimal,
            states[-1],
            target_state,
        )
        
        assert metrics["success"] == 1.0
        assert metrics["efficiency"] == 1.0  # Optimal
        assert metrics["planning_quality"] == 1.0
        assert metrics["extra_moves"] == 0
    
    def test_evaluate_suboptimal_solution(self):
        """Test evaluation with extra moves."""
        tasks = ExecutiveFunctionTasks()
        config = TowerOfHanoiConfig(n_disks=3)
        
        states, _, _, optimal = tasks.tower_of_hanoi(config)
        
        # Add fake extra moves
        extra_moves = [(0, 1), (1, 2), (0, 1)]  # 3 extra moves
        moves = [(0, 1)] * 10  # 10 moves instead of 7
        
        # Create target state
        target_state = tasks._encode_hanoi_state(
            [[], [], [0, 1, 2]],
            config.encode_dim,
            config.n_disks,
            config.device,
        )
        
        metrics = tasks.evaluate_tower_of_hanoi(
            moves,
            optimal,
            states[-1],  # Assume still solved
            target_state,
        )
        
        # Efficiency should be less than 1.0 due to extra moves
        assert metrics["efficiency"] < 1.0
        assert metrics["n_moves"] == 10
        assert metrics["optimal_moves"] == 7
        assert metrics["extra_moves"] == 3
    
    def test_statistics_update(self):
        """Test that Tower of Hanoi statistics are updated."""
        tasks = ExecutiveFunctionTasks()
        config = TowerOfHanoiConfig(n_disks=3)
        
        states, _, moves, optimal = tasks.tower_of_hanoi(config)
        
        target_state = tasks._encode_hanoi_state(
            [[], [], [0, 1, 2]],
            config.encode_dim,
            config.n_disks,
            config.device,
        )
        
        tasks.evaluate_tower_of_hanoi(moves, optimal, states[-1], target_state)
        
        stats = tasks.get_statistics()
        assert "tower_of_hanoi" in stats["by_task"]
        assert stats["by_task"]["tower_of_hanoi"] == 1.0  # Perfect performance
    
    def test_device_handling(self):
        """Test device handling for Tower of Hanoi."""
        tasks = ExecutiveFunctionTasks()
        config = TowerOfHanoiConfig(n_disks=2, device="cpu")
        
        states, _, _, _ = tasks.tower_of_hanoi(config)
        
        # All states should be on correct device
        assert all(s.device.type == "cpu" for s in states)
    
    def test_max_moves_limit(self):
        """Test that max_moves limit is respected."""
        tasks = ExecutiveFunctionTasks()
        config = TowerOfHanoiConfig(n_disks=5, max_moves=10)
        
        _, _, moves, _ = tasks.tower_of_hanoi(config)
        
        # Should not exceed max_moves
        assert len(moves) <= 10


class TestRavensMatrices:
    """Tests for Raven's Progressive Matrices (Stage 3-4)."""
    
    def test_initialization(self):
        """Test Raven's matrices generation."""
        tasks = ExecutiveFunctionTasks()
        config = RavensMatricesConfig()
        
        matrix, answer_choices, correct_idx, rule_type = tasks.ravens_matrices(config)
        
        # Check structure
        assert matrix.shape == (9, config.stimulus_dim)  # 3x3 grid
        assert answer_choices.shape == (config.n_answer_choices, config.stimulus_dim)
        assert 0 <= correct_idx < config.n_answer_choices
        assert rule_type in config.rule_types
    
    def test_missing_cell(self):
        """Test that last cell is missing (zeros)."""
        tasks = ExecutiveFunctionTasks()
        config = RavensMatricesConfig()
        
        matrix, _, _, _ = tasks.ravens_matrices(config)
        
        # Last cell should be all zeros
        assert torch.allclose(matrix[-1], torch.zeros_like(matrix[-1]))
    
    def test_answer_choices_count(self):
        """Test correct number of answer choices."""
        tasks = ExecutiveFunctionTasks()
        config = RavensMatricesConfig(n_answer_choices=6)
        
        _, answer_choices, _, _ = tasks.ravens_matrices(config)
        
        assert answer_choices.shape[0] == 6
    
    def test_progression_rule(self):
        """Test progression rule generation."""
        tasks = ExecutiveFunctionTasks()
        config = RavensMatricesConfig(rule_types=["progression"])
        
        _, _, _, rule_type = tasks.ravens_matrices(config)
        
        assert rule_type == "progression"
    
    def test_constant_rule(self):
        """Test constant rule generation."""
        tasks = ExecutiveFunctionTasks()
        config = RavensMatricesConfig(rule_types=["constant"])
        
        _, _, _, rule_type = tasks.ravens_matrices(config)
        
        assert rule_type == "constant"
    
    def test_distribution_rule(self):
        """Test distribution rule generation."""
        tasks = ExecutiveFunctionTasks()
        config = RavensMatricesConfig(rule_types=["distribution"])
        
        _, _, _, rule_type = tasks.ravens_matrices(config)
        
        assert rule_type == "distribution"
    
    def test_complexity_levels(self):
        """Test different complexity levels."""
        tasks = ExecutiveFunctionTasks()
        
        for complexity in ["simple", "medium", "hard"]:
            config = RavensMatricesConfig(pattern_complexity=complexity)
            matrix, _, _, _ = tasks.ravens_matrices(config)
            
            # Matrix should be generated without error
            assert matrix.shape == (9, config.stimulus_dim)
    
    def test_correct_answer_in_choices(self):
        """Test that correct answer is actually in the choices."""
        tasks = ExecutiveFunctionTasks()
        config = RavensMatricesConfig()
        
        _, answer_choices, correct_idx, _ = tasks.ravens_matrices(config)
        
        # Correct answer should be at correct_idx
        assert 0 <= correct_idx < len(answer_choices)
    
    def test_evaluate_correct_answer(self):
        """Test evaluation with correct answer."""
        tasks = ExecutiveFunctionTasks()
        config = RavensMatricesConfig()
        
        _, _, correct_idx, rule_type = tasks.ravens_matrices(config)
        
        metrics = tasks.evaluate_ravens_matrices(
            selected_answer=correct_idx,
            correct_answer=correct_idx,
            rule_type=rule_type,
        )
        
        assert metrics["correct"] is True
        assert metrics["selected_answer"] == correct_idx
        assert metrics["correct_answer"] == correct_idx
        assert metrics["rule_type"] == rule_type
    
    def test_evaluate_incorrect_answer(self):
        """Test evaluation with incorrect answer."""
        tasks = ExecutiveFunctionTasks()
        config = RavensMatricesConfig()
        
        _, _, correct_idx, rule_type = tasks.ravens_matrices(config)
        
        # Select wrong answer
        wrong_idx = (correct_idx + 1) % config.n_answer_choices
        
        metrics = tasks.evaluate_ravens_matrices(
            selected_answer=wrong_idx,
            correct_answer=correct_idx,
            rule_type=rule_type,
        )
        
        assert metrics["correct"] is False
        assert metrics["selected_answer"] == wrong_idx
        assert metrics["correct_answer"] == correct_idx
    
    def test_statistics_by_rule_type(self):
        """Test statistics tracking by rule type."""
        tasks = ExecutiveFunctionTasks()
        
        # Test progression
        config1 = RavensMatricesConfig(rule_types=["progression"])
        _, _, correct_idx1, rule_type1 = tasks.ravens_matrices(config1)
        tasks.evaluate_ravens_matrices(correct_idx1, correct_idx1, rule_type1)
        
        # Test constant
        config2 = RavensMatricesConfig(rule_types=["constant"])
        _, _, correct_idx2, rule_type2 = tasks.ravens_matrices(config2)
        tasks.evaluate_ravens_matrices(correct_idx2, correct_idx2, rule_type2)
        
        stats = tasks.get_statistics()
        assert "ravens_matrices" in stats["by_task"]
        assert stats["by_task"]["ravens_matrices"] == 1.0  # Both correct
    
    def test_response_time_tracking(self):
        """Test optional response time tracking."""
        tasks = ExecutiveFunctionTasks()
        config = RavensMatricesConfig()
        
        _, _, correct_idx, rule_type = tasks.ravens_matrices(config)
        
        metrics = tasks.evaluate_ravens_matrices(
            selected_answer=correct_idx,
            correct_answer=correct_idx,
            rule_type=rule_type,
            response_time=2.5,
        )
        
        assert "response_time" in metrics
        assert metrics["response_time"] == 2.5
    
    def test_device_handling(self):
        """Test device handling for Raven's matrices."""
        tasks = ExecutiveFunctionTasks()
        config = RavensMatricesConfig(device="cpu")
        
        matrix, answer_choices, _, _ = tasks.ravens_matrices(config)
        
        assert matrix.device.type == "cpu"
        assert answer_choices.device.type == "cpu"
    
    def test_reproducibility(self):
        """Test reproducibility with fixed seed."""
        tasks1 = ExecutiveFunctionTasks()
        tasks2 = ExecutiveFunctionTasks()
        
        np.random.seed(42)
        torch.manual_seed(42)
        config1 = RavensMatricesConfig()
        matrix1, _, correct_idx1, rule_type1 = tasks1.ravens_matrices(config1)
        
        np.random.seed(42)
        torch.manual_seed(42)
        config2 = RavensMatricesConfig()
        matrix2, _, correct_idx2, rule_type2 = tasks2.ravens_matrices(config2)
        
        # Should generate same matrix (except last cell which is zeros)
        assert torch.allclose(matrix1[:-1], matrix2[:-1])
        assert correct_idx1 == correct_idx2
        assert rule_type1 == rule_type2


class TestStage34Integration:
    """Integration tests for Stage 3-4 tasks."""
    
    def test_all_stage34_tasks(self):
        """Test that all Stage 3-4 tasks can be generated."""
        tasks = ExecutiveFunctionTasks()
        
        # Tower of Hanoi
        hanoi_config = TowerOfHanoiConfig(n_disks=3)
        states, disk_moved, moves, optimal = tasks.tower_of_hanoi(hanoi_config)
        assert len(states) > 0
        
        # Raven's Matrices
        ravens_config = RavensMatricesConfig()
        matrix, answer_choices, correct_idx, rule_type = tasks.ravens_matrices(ravens_config)
        assert matrix.shape[0] == 9
        
        # Both should work without errors
        print(f"Tower of Hanoi: {len(moves)} moves (optimal: {optimal})")
        print(f"Raven's: {rule_type} rule")
    
    def test_combined_statistics(self):
        """Test combined statistics across all stages."""
        tasks = ExecutiveFunctionTasks()
        
        # Stage 1: Go/No-Go
        gng_config = GoNoGoConfig(n_stimuli=10)
        stimuli, stimulus_types, correct_responses = tasks.go_no_go(gng_config)
        tasks.evaluate_go_no_go(
            responses=[True] * 10,
            correct_responses=correct_responses,
            stimulus_types=stimulus_types,
        )
        
        # Stage 3: Tower of Hanoi
        hanoi_config = TowerOfHanoiConfig(n_disks=3)
        states, _, moves, optimal = tasks.tower_of_hanoi(hanoi_config)
        target_state = tasks._encode_hanoi_state(
            [[], [], [0, 1, 2]],
            hanoi_config.encode_dim,
            hanoi_config.n_disks,
            hanoi_config.device,
        )
        tasks.evaluate_tower_of_hanoi(moves, optimal, states[-1], target_state)
        
        # Stage 4: Raven's
        ravens_config = RavensMatricesConfig()
        _, _, correct_idx, rule_type = tasks.ravens_matrices(ravens_config)
        tasks.evaluate_ravens_matrices(correct_idx, correct_idx, rule_type)
        
        # Should track all tasks
        stats = tasks.get_statistics()
        assert "go_no_go" in stats["by_task"]
        assert "tower_of_hanoi" in stats["by_task"]
        assert "ravens_matrices" in stats["by_task"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

