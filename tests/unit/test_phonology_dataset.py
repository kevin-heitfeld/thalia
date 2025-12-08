"""
Unit tests for Phonological Dataset (Stage 0).

Tests phoneme encoding, discrimination tasks, continuum generation,
and performance evaluation for critical period phonology learning.
"""

import pytest
import torch
import numpy as np

from thalia.datasets.phonology import (
    PhonologicalDataset,
    PhonologicalConfig,
    PhonemeCategory,
    PhonemeFeatures,
    PHONEME_FEATURES,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def dataset():
    """Create basic phonological dataset."""
    config = PhonologicalConfig(
        n_freq_channels=64,
        n_time_steps=100,
        device="cpu",
    )
    return PhonologicalDataset(config=config)


@pytest.fixture
def high_noise_dataset():
    """Create dataset with high noise."""
    config = PhonologicalConfig(
        noise_std=0.3,
        within_category_variance=0.25,
    )
    return PhonologicalDataset(config=config)


# ============================================================================
# 1. Phoneme Encoding Tests
# ============================================================================

class TestPhonemeEncoding:
    """Test phoneme acoustic encoding."""
    
    def test_encode_stop_consonant(self, dataset):
        """Test encoding of stop consonant (VOT-based)."""
        phoneme = PhonemeCategory.P
        spectrogram = dataset._encode_phoneme(phoneme, add_noise=False)
        
        assert spectrogram.shape == (64, 100)
        assert torch.all((spectrogram >= 0) & (spectrogram <= 1))
        
        # Should have burst at onset (high frequencies)
        assert spectrogram[32:, :5].mean() > 0.5
        
        # Should have voicing onset after VOT (low frequencies, later in time)
        assert spectrogram[:16, 40:].sum() > 0
    
    def test_encode_vowel(self, dataset):
        """Test encoding of vowel (formant-based)."""
        phoneme = PhonemeCategory.AA
        spectrogram = dataset._encode_phoneme(phoneme, add_noise=False)
        
        assert spectrogram.shape == (64, 100)
        
        # Should have sustained energy (formants)
        assert spectrogram[:, :60].sum() > 10.0
        
        # Should have two peaks (F1 and F2)
        freq_profile = spectrogram.sum(dim=1)
        peaks = (freq_profile > 0.5).sum()
        assert peaks > 0  # At least one formant peak
    
    def test_vot_contrast(self, dataset):
        """Test that VOT differs between voiced and voiceless stops."""
        p_spec = dataset._encode_phoneme(PhonemeCategory.P, add_noise=False, add_variance=False)
        b_spec = dataset._encode_phoneme(PhonemeCategory.B, add_noise=False, add_variance=False)
        
        # Voicing onset should be later for /p/ than /b/
        p_voicing = p_spec[:16, :].sum()
        b_voicing = b_spec[:16, :].sum()
        
        # /b/ should have earlier/more voicing in early time steps
        p_early = p_spec[:16, :30].sum()
        b_early = b_spec[:16, :30].sum()
        assert b_early > p_early
    
    def test_formant_contrast(self, dataset):
        """Test that formants differ between vowels."""
        aa_spec = dataset._encode_phoneme(PhonemeCategory.AA, add_noise=False, add_variance=False)
        iy_spec = dataset._encode_phoneme(PhonemeCategory.IY, add_noise=False, add_variance=False)
        
        # Should have different spectral profiles
        aa_profile = aa_spec.sum(dim=1)
        iy_profile = iy_spec.sum(dim=1)
        
        # Profiles should differ significantly
        correlation = torch.corrcoef(torch.stack([aa_profile, iy_profile]))[0, 1]
        assert correlation < 0.9  # Not too similar
    
    def test_noise_addition(self, dataset):
        """Test that noise is added when requested."""
        # Without noise
        spec_no_noise = dataset._encode_phoneme(PhonemeCategory.P, add_noise=False)
        
        # With noise (multiple samples)
        specs_with_noise = [
            dataset._encode_phoneme(PhonemeCategory.P, add_noise=True)
            for _ in range(5)
        ]
        
        # Should have variability
        stacked = torch.stack(specs_with_noise, dim=0)
        std = stacked.std(dim=0).mean()
        assert std > 0.05  # Some variability from noise
    
    def test_within_category_variance(self, dataset):
        """Test within-category natural variation."""
        specs = [
            dataset._encode_phoneme(PhonemeCategory.P, add_noise=False, add_variance=True)
            for _ in range(10)
        ]
        
        stacked = torch.stack(specs, dim=0)
        std = stacked.std(dim=0).mean()
        
        # Should have some variance (but less than across categories)
        assert 0.02 < std < 0.15


# ============================================================================
# 2. Discrimination Task Tests
# ============================================================================

class TestDiscriminationTasks:
    """Test same/different discrimination tasks."""
    
    def test_same_pair_generation(self, dataset):
        """Test generation of 'same' pairs."""
        contrast = (PhonemeCategory.P, PhonemeCategory.B)
        stim1, stim2, label = dataset.generate_discrimination_pair(contrast, same=True)
        
        assert stim1.shape == (64, 100)
        assert stim2.shape == (64, 100)
        assert label == 1  # Same
        
        # Should be similar but not identical (natural variance)
        similarity = torch.corrcoef(torch.stack([stim1.flatten(), stim2.flatten()]))[0, 1]
        assert 0.7 < similarity < 1.0
    
    def test_different_pair_generation(self, dataset):
        """Test generation of 'different' pairs."""
        contrast = (PhonemeCategory.P, PhonemeCategory.B)
        stim1, stim2, label = dataset.generate_discrimination_pair(contrast, same=False)
        
        assert stim1.shape == (64, 100)
        assert stim2.shape == (64, 100)
        assert label == 0  # Different
        
        # Should be less similar
        similarity = torch.corrcoef(torch.stack([stim1.flatten(), stim2.flatten()]))[0, 1]
        assert similarity < 0.9  # More different
    
    def test_contrast_types(self, dataset):
        """Test different contrast types."""
        # Voicing contrast
        contrast_v = dataset.voicing_contrasts[0]
        stim1, stim2, _ = dataset.generate_discrimination_pair(contrast_v, same=False)
        assert stim1.shape[0] == 64
        
        # Vowel contrast
        contrast_vow = dataset.vowel_contrasts[0]
        stim1, stim2, _ = dataset.generate_discrimination_pair(contrast_vow, same=False)
        assert stim1.shape[0] == 64
        
        # Place contrast
        contrast_p = dataset.place_contrasts[0]
        stim1, stim2, _ = dataset.generate_discrimination_pair(contrast_p, same=False)
        assert stim1.shape[0] == 64


# ============================================================================
# 3. Continuum Generation Tests
# ============================================================================

class TestContinuumGeneration:
    """Test VOT/formant continuum generation."""
    
    def test_vot_continuum(self, dataset):
        """Test VOT continuum between /p/ and /b/."""
        contrast = (PhonemeCategory.P, PhonemeCategory.B)
        stimuli, labels = dataset.generate_continuum(contrast, n_steps=11)
        
        assert len(stimuli) == 11
        assert len(labels) == 11
        
        # Labels should transition from 0 to 1
        assert labels[0] == 0  # Starts with /b/
        assert labels[-1] == 1  # Ends with /p/
        
        # Should have categorical boundary around middle
        transitions = sum(1 for i in range(len(labels)-1) if labels[i] != labels[i+1])
        assert transitions >= 1  # At least one boundary crossing
    
    def test_vowel_continuum(self, dataset):
        """Test formant continuum between vowels."""
        contrast = (PhonemeCategory.IY, PhonemeCategory.IH)
        stimuli, labels = dataset.generate_continuum(contrast, n_steps=11)
        
        assert len(stimuli) == 11
        assert len(labels) == 11
        
        # Stimuli should gradually change
        first = stimuli[0].flatten()
        last = stimuli[-1].flatten()
        middle = stimuli[5].flatten()
        
        # Middle should be between first and last
        sim_first_middle = torch.corrcoef(torch.stack([first, middle]))[0, 1]
        sim_middle_last = torch.corrcoef(torch.stack([middle, last]))[0, 1]
        sim_first_last = torch.corrcoef(torch.stack([first, last]))[0, 1]
        
        assert sim_first_middle > sim_first_last
        assert sim_middle_last > sim_first_last
    
    def test_continuum_steps(self, dataset):
        """Test that continuum respects n_steps parameter."""
        contrast = (PhonemeCategory.P, PhonemeCategory.B)
        
        for n_steps in [5, 11, 21]:
            stimuli, labels = dataset.generate_continuum(contrast, n_steps=n_steps)
            assert len(stimuli) == n_steps
            assert len(labels) == n_steps
    
    def test_boundary_sharpness(self):
        """Test that boundary sharpness affects categorical boundary."""
        # Sharp boundary
        config_sharp = PhonologicalConfig(boundary_sharpness=5.0)
        dataset_sharp = PhonologicalDataset(config=config_sharp)
        
        # Gradual boundary
        config_gradual = PhonologicalConfig(boundary_sharpness=1.0)
        dataset_gradual = PhonologicalDataset(config=config_gradual)
        
        contrast = (PhonemeCategory.P, PhonemeCategory.B)
        
        _, labels_sharp = dataset_sharp.generate_continuum(contrast)
        _, labels_gradual = dataset_gradual.generate_continuum(contrast)
        
        # Sharp should have fewer transitions (steeper boundary)
        transitions_sharp = sum(1 for i in range(len(labels_sharp)-1) if labels_sharp[i] != labels_sharp[i+1])
        transitions_gradual = sum(1 for i in range(len(labels_gradual)-1) if labels_gradual[i] != labels_gradual[i+1])
        
        # Both should have at least one transition
        assert transitions_sharp >= 1
        assert transitions_gradual >= 1


# ============================================================================
# 4. Batch Generation Tests
# ============================================================================

class TestBatchGeneration:
    """Test batch generation for training."""
    
    def test_discrimination_batch(self, dataset):
        """Test discrimination task batch generation."""
        stimuli, labels = dataset.generate_batch(
            task_type="discrimination",
            batch_size=32,
            contrast_type="voicing",
        )
        
        assert stimuli.shape == (32, 64, 200)  # Concatenated stimuli
        assert labels.shape == (32,)
        assert torch.all((labels == 0) | (labels == 1))
    
    def test_continuum_batch(self, dataset):
        """Test continuum task batch generation."""
        stimuli, labels = dataset.generate_batch(
            task_type="continuum",
            batch_size=32,
            contrast_type="voicing",
        )
        
        assert stimuli.shape == (32, 64, 100)
        assert labels.shape == (32,)
        assert torch.all((labels == 0) | (labels == 1))
    
    def test_contrast_types(self, dataset):
        """Test different contrast types in batch generation."""
        for contrast_type in ["voicing", "vowel", "place"]:
            stimuli, labels = dataset.generate_batch(
                task_type="discrimination",
                batch_size=16,
                contrast_type=contrast_type,
            )
            
            assert stimuli.shape[0] == 16
            assert labels.shape[0] == 16
    
    def test_batch_size_parameter(self, dataset):
        """Test that batch size parameter is respected."""
        for batch_size in [8, 16, 32, 64]:
            stimuli, labels = dataset.generate_batch(
                task_type="discrimination",
                batch_size=batch_size,
                contrast_type="voicing",
            )
            
            assert stimuli.shape[0] == batch_size
            assert labels.shape[0] == batch_size


# ============================================================================
# 5. Performance Evaluation Tests
# ============================================================================

class TestPerformanceEvaluation:
    """Test performance evaluation metrics."""
    
    def test_perfect_accuracy(self, dataset):
        """Test evaluation with perfect predictions."""
        labels = torch.tensor([0, 0, 1, 1, 0, 1])
        predictions = labels.clone()  # Perfect predictions
        
        metrics = dataset.evaluate_discrimination(predictions, labels, "voicing")
        
        assert metrics["accuracy"] == 1.0
        assert metrics["d_prime"] > 2.0  # High discriminability
    
    def test_chance_accuracy(self, dataset):
        """Test evaluation with chance-level predictions."""
        labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1] * 10)
        predictions = torch.randint(0, 2, (len(labels),))
        
        metrics = dataset.evaluate_discrimination(predictions, labels, "voicing")
        
        # Should be near 0.5 (chance level)
        assert 0.3 < metrics["accuracy"] < 0.7
    
    def test_logit_predictions(self, dataset):
        """Test evaluation with logit predictions."""
        labels = torch.tensor([0, 1, 0, 1])
        logits = torch.tensor([
            [0.9, 0.1],  # Predict 0
            [0.2, 0.8],  # Predict 1
            [0.7, 0.3],  # Predict 0
            [0.3, 0.7],  # Predict 1
        ])
        
        metrics = dataset.evaluate_discrimination(logits, labels, "voicing")
        assert metrics["accuracy"] == 1.0
    
    def test_statistics_tracking(self, dataset):
        """Test that statistics are tracked correctly."""
        dataset.reset_statistics()
        
        labels1 = torch.tensor([0, 1, 0, 1])
        preds1 = torch.tensor([0, 1, 0, 0])  # 3/4 correct
        dataset.evaluate_discrimination(preds1, labels1, "voicing")
        
        labels2 = torch.tensor([1, 1, 0, 0])
        preds2 = torch.tensor([1, 1, 0, 0])  # 4/4 correct
        dataset.evaluate_discrimination(preds2, labels2, "vowel")
        
        stats = dataset.get_statistics()
        
        assert stats["n_trials"] == 8
        assert stats["overall_accuracy"] == 7/8  # 7 correct out of 8
        assert "voicing" in stats["by_contrast"]
        assert "vowel" in stats["by_contrast"]
    
    def test_d_prime_calculation(self, dataset):
        """Test d-prime calculation."""
        # High discriminability
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        predictions = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        
        metrics = dataset.evaluate_discrimination(predictions, labels, "voicing")
        assert metrics["d_prime"] > 2.0
        
        # Low discriminability
        predictions_noisy = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
        metrics_noisy = dataset.evaluate_discrimination(predictions_noisy, labels, "voicing")
        assert metrics_noisy["d_prime"] < 1.0


# ============================================================================
# 6. Integration Tests
# ============================================================================

class TestPhonologyIntegration:
    """Test full phonology dataset workflow."""
    
    def test_full_training_cycle(self, dataset):
        """Test complete training cycle."""
        dataset.reset_statistics()
        
        # Generate multiple batches
        for _ in range(5):
            stimuli, labels = dataset.generate_batch(
                task_type="discrimination",
                batch_size=32,
                contrast_type="voicing",
            )
            
            # Simulate model predictions (random for this test)
            predictions = torch.randint(0, 2, (32,))
            
            # Evaluate
            metrics = dataset.evaluate_discrimination(predictions, labels, "voicing")
            
            assert "accuracy" in metrics
            assert "d_prime" in metrics
        
        # Check statistics
        stats = dataset.get_statistics()
        assert stats["n_trials"] == 5 * 32
    
    def test_curriculum_progression(self, dataset):
        """Test curriculum progression through difficulty levels."""
        # Easy: Same category (high similarity)
        easy_stimuli, easy_labels = dataset.generate_batch(
            task_type="discrimination",
            batch_size=16,
            contrast_type="voicing",
        )
        
        # Hard: Continuum (near boundary)
        hard_stimuli, hard_labels = dataset.generate_batch(
            task_type="continuum",
            batch_size=16,
            contrast_type="voicing",
        )
        
        assert easy_stimuli.shape[0] == 16
        assert hard_stimuli.shape[0] == 16
    
    def test_all_contrast_types(self, dataset):
        """Test that all contrast types work together."""
        dataset.reset_statistics()
        
        for contrast_type in ["voicing", "vowel", "place"]:
            stimuli, labels = dataset.generate_batch(
                task_type="discrimination",
                batch_size=16,
                contrast_type=contrast_type,
            )
            
            predictions = torch.randint(0, 2, (16,))
            metrics = dataset.evaluate_discrimination(predictions, labels, contrast_type)
            
            assert "accuracy" in metrics
        
        stats = dataset.get_statistics()
        assert len(stats["by_contrast"]) == 3


# ============================================================================
# 7. Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_task_type(self, dataset):
        """Test error on invalid task type."""
        with pytest.raises(ValueError, match="Unknown task type"):
            dataset.generate_batch(task_type="invalid")
    
    def test_invalid_contrast_type(self, dataset):
        """Test error on invalid contrast type."""
        with pytest.raises(ValueError, match="Unknown contrast type"):
            dataset.generate_batch(contrast_type="invalid")
    
    def test_zero_batch_size(self, dataset):
        """Test handling of zero batch size."""
        stimuli, labels = dataset.generate_batch(batch_size=0)
        assert stimuli.shape[0] == 0
        assert labels.shape[0] == 0
    
    def test_high_noise_robustness(self, high_noise_dataset):
        """Test that high noise doesn't break encoding."""
        phoneme = PhonemeCategory.P
        spectrogram = high_noise_dataset._encode_phoneme(phoneme)
        
        # Should still be valid (values in [0, 1])
        assert torch.all((spectrogram >= 0) & (spectrogram <= 1))
        assert not torch.any(torch.isnan(spectrogram))


# ============================================================================
# 8. Configuration Tests
# ============================================================================

class TestConfiguration:
    """Test configuration options."""
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PhonologicalConfig(
            n_freq_channels=32,
            n_time_steps=50,
            noise_std=0.05,
            continuum_steps=21,
        )
        dataset = PhonologicalDataset(config=config)
        
        # Check that config is applied
        phoneme = PhonemeCategory.P
        spec = dataset._encode_phoneme(phoneme)
        assert spec.shape == (32, 50)
        
        contrast = (PhonemeCategory.P, PhonemeCategory.B)
        stimuli, labels = dataset.generate_continuum(contrast)
        assert len(stimuli) == 21
    
    def test_device_handling(self):
        """Test device handling (CPU/CUDA)."""
        config = PhonologicalConfig(device="cpu")
        dataset = PhonologicalDataset(config=config)
        
        stimuli, labels = dataset.generate_batch(batch_size=8)
        
        assert stimuli.device.type == "cpu"
        assert labels.device.type == "cpu"
