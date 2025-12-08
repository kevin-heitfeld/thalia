"""
Unit tests for multi-language phonology dataset (English, German, Spanish).

Tests language-specific phonemes, contrasts, and acoustic encoding.
"""

import pytest
import torch

from thalia.datasets.phonology import (
    PhonologicalDataset,
    PhonologicalConfig,
    PhonemeCategory,
    Language,
    LANGUAGE_PHONEMES,
    LANGUAGE_CONTRASTS,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def dataset_english():
    """Create English phonology dataset."""
    config = PhonologicalConfig(language=Language.ENGLISH)
    return PhonologicalDataset(config=config)


@pytest.fixture
def dataset_german():
    """Create German phonology dataset."""
    config = PhonologicalConfig(language=Language.GERMAN)
    return PhonologicalDataset(config=config)


@pytest.fixture
def dataset_spanish():
    """Create Spanish phonology dataset."""
    config = PhonologicalConfig(language=Language.SPANISH)
    return PhonologicalDataset(config=config)


# ============================================================================
# 1. Language Selection Tests
# ============================================================================

class TestLanguageSelection:
    """Test language configuration."""
    
    def test_default_language_english(self):
        """Test that default language is English."""
        dataset = PhonologicalDataset()
        assert dataset.language == Language.ENGLISH
    
    def test_explicit_language_config(self):
        """Test explicit language in config."""
        for language in [Language.ENGLISH, Language.GERMAN, Language.SPANISH]:
            config = PhonologicalConfig(language=language)
            dataset = PhonologicalDataset(config=config)
            assert dataset.language == language
    
    def test_language_override_in_constructor(self):
        """Test language override in constructor."""
        config = PhonologicalConfig(language=Language.ENGLISH)
        dataset = PhonologicalDataset(config=config, language=Language.GERMAN)
        assert dataset.language == Language.GERMAN


# ============================================================================
# 2. English-Specific Tests
# ============================================================================

class TestEnglishPhonology:
    """Test English phoneme inventory."""
    
    def test_english_phoneme_inventory(self):
        """Test that English phonemes are loaded."""
        phonemes = LANGUAGE_PHONEMES[Language.ENGLISH]
        
        # Should have stops
        assert PhonemeCategory.P in phonemes
        assert PhonemeCategory.B in phonemes
        
        # Should have English vowels
        assert PhonemeCategory.IY in phonemes  # beat
        assert PhonemeCategory.IH in phonemes  # bit
        assert PhonemeCategory.AE in phonemes  # cat
    
    def test_english_contrasts(self, dataset_english):
        """Test English contrast types."""
        assert len(dataset_english.voicing_contrasts) > 0
        assert len(dataset_english.vowel_contrasts) > 0
        assert len(dataset_english.place_contrasts) > 0
        
        # Should NOT have German/Spanish-specific contrasts
        assert not hasattr(dataset_english, 'tap_trill_contrasts')
    
    def test_english_vowel_contrast(self, dataset_english):
        """Test English-specific vowel contrast (beat vs bit)."""
        contrast = (PhonemeCategory.IY, PhonemeCategory.IH)
        assert contrast in dataset_english.vowel_contrasts
        
        # Generate discrimination pair
        stim1, stim2, label = dataset_english.generate_discrimination_pair(contrast, same=False)
        assert stim1.shape == (64, 100)
        assert label == 0  # Different
    
    def test_english_batch_generation(self, dataset_english):
        """Test batch generation with English contrasts."""
        stimuli, labels = dataset_english.generate_batch(
            task_type="discrimination",
            batch_size=16,
            contrast_type="vowel"
        )
        assert stimuli.shape[0] == 16


# ============================================================================
# 3. German-Specific Tests
# ============================================================================

class TestGermanPhonology:
    """Test German phoneme inventory."""
    
    def test_german_phoneme_inventory(self):
        """Test that German phonemes are loaded."""
        phonemes = LANGUAGE_PHONEMES[Language.GERMAN]
        
        # Should have German-specific vowels
        assert PhonemeCategory.UE in phonemes  # ü
        assert PhonemeCategory.OE in phonemes  # ö
        
        # Should have German consonants
        assert PhonemeCategory.X in phonemes   # Bach-Laut
        assert PhonemeCategory.R_UVULAR in phonemes  # German r
    
    def test_german_umlaut_encoding(self, dataset_german):
        """Test encoding of German umlauts."""
        # /ü/ - high front rounded
        ue_spec = dataset_german._encode_phoneme(PhonemeCategory.UE, add_noise=False)
        assert ue_spec.shape == (64, 100)
        assert ue_spec.sum() > 0  # Has energy
        
        # /ö/ - mid front rounded
        oe_spec = dataset_german._encode_phoneme(PhonemeCategory.OE, add_noise=False)
        assert oe_spec.shape == (64, 100)
        assert oe_spec.sum() > 0
        
        # Should be different from each other
        correlation = torch.corrcoef(torch.stack([ue_spec.flatten(), oe_spec.flatten()]))[0, 1]
        assert correlation < 0.95  # Not too similar
    
    def test_german_contrasts(self, dataset_german):
        """Test German-specific contrasts."""
        assert len(dataset_german.voicing_contrasts) > 0
        assert len(dataset_german.vowel_contrasts) > 0
        assert len(dataset_german.fricative_contrasts) > 0
        
        # Should have German vowel contrasts
        contrasts_flat = [c for contrast_list in LANGUAGE_CONTRASTS[Language.GERMAN].values() 
                         for c in contrast_list]
        assert (PhonemeCategory.UE, PhonemeCategory.IY) in contrasts_flat  # ü vs i
    
    def test_german_fricative_contrast(self, dataset_german):
        """Test German fricative contrast (Bach-Laut)."""
        # /x/ vs /k/
        contrast = (PhonemeCategory.X, PhonemeCategory.K)
        assert contrast in dataset_german.fricative_contrasts
        
        stim1, stim2, label = dataset_german.generate_discrimination_pair(contrast, same=False)
        assert stim1.shape == (64, 100)
        assert label == 0
    
    def test_german_batch_generation(self, dataset_german):
        """Test batch generation with German fricative contrast."""
        stimuli, labels = dataset_german.generate_batch(
            task_type="discrimination",
            batch_size=16,
            contrast_type="fricative"
        )
        assert stimuli.shape[0] == 16


# ============================================================================
# 4. Spanish-Specific Tests
# ============================================================================

class TestSpanishPhonology:
    """Test Spanish phoneme inventory."""
    
    def test_spanish_phoneme_inventory(self):
        """Test that Spanish phonemes are loaded."""
        phonemes = LANGUAGE_PHONEMES[Language.SPANISH]
        
        # Should have Spanish 5-vowel system
        assert PhonemeCategory.A_ES in phonemes
        assert PhonemeCategory.E_ES in phonemes
        assert PhonemeCategory.I_ES in phonemes
        assert PhonemeCategory.O_ES in phonemes
        assert PhonemeCategory.U_ES in phonemes
        
        # Should have tap/trill distinction
        assert PhonemeCategory.R_TAP in phonemes   # pero
        assert PhonemeCategory.R_TRILL in phonemes  # perro
        
        # Should have voiced fricatives
        assert PhonemeCategory.B_FRIC in phonemes  # β
        assert PhonemeCategory.D_FRIC in phonemes  # ð
        assert PhonemeCategory.G_FRIC in phonemes  # ɣ
    
    def test_spanish_tap_trill_encoding(self, dataset_spanish):
        """Test encoding of Spanish tap vs trill (CRITICAL contrast)."""
        # Single tap /ɾ/ - very short
        tap_spec = dataset_spanish._encode_phoneme(PhonemeCategory.R_TAP, add_noise=False)
        assert tap_spec.shape == (64, 100)
        
        # Trill /r/ - longer, multiple taps
        trill_spec = dataset_spanish._encode_phoneme(PhonemeCategory.R_TRILL, add_noise=False)
        assert trill_spec.shape == (64, 100)
        
        # Trill should have more total energy (longer duration)
        assert trill_spec.sum() > tap_spec.sum() * 2.0
    
    def test_spanish_contrasts(self, dataset_spanish):
        """Test Spanish-specific contrasts."""
        assert len(dataset_spanish.voicing_contrasts) > 0
        assert len(dataset_spanish.vowel_contrasts) > 0
        assert len(dataset_spanish.tap_trill_contrasts) > 0
        assert len(dataset_spanish.fricative_contrasts) > 0
        
        # Should have the critical pero/perro contrast
        assert (PhonemeCategory.R_TAP, PhonemeCategory.R_TRILL) in dataset_spanish.tap_trill_contrasts
    
    def test_spanish_tap_trill_discrimination(self, dataset_spanish):
        """Test pero vs perro discrimination."""
        contrast = (PhonemeCategory.R_TAP, PhonemeCategory.R_TRILL)
        
        stim1, stim2, label = dataset_spanish.generate_discrimination_pair(contrast, same=False)
        assert stim1.shape == (64, 100)
        assert label == 0  # Different
        
        # Should be acoustically different
        similarity = torch.corrcoef(torch.stack([stim1.flatten(), stim2.flatten()]))[0, 1]
        assert similarity < 0.9
    
    def test_spanish_voiced_fricatives(self, dataset_spanish):
        """Test Spanish voiced fricatives (intervocalic)."""
        # /b/ vs /β/
        b_stop = dataset_spanish._encode_phoneme(PhonemeCategory.B, add_noise=False)
        b_fric = dataset_spanish._encode_phoneme(PhonemeCategory.B_FRIC, add_noise=False)
        
        assert b_stop.shape == (64, 100)
        assert b_fric.shape == (64, 100)
        
        # Should be different (stop has burst, fricative is continuous)
        correlation = torch.corrcoef(torch.stack([b_stop.flatten(), b_fric.flatten()]))[0, 1]
        assert correlation < 0.95
    
    def test_spanish_batch_generation(self, dataset_spanish):
        """Test batch generation with Spanish tap/trill."""
        stimuli, labels = dataset_spanish.generate_batch(
            task_type="discrimination",
            batch_size=16,
            contrast_type="tap_trill"
        )
        assert stimuli.shape[0] == 16


# ============================================================================
# 5. Cross-Language Comparison Tests
# ============================================================================

class TestCrossLanguage:
    """Test cross-language comparisons."""
    
    def test_all_languages_have_voicing_contrasts(self):
        """Test that all languages have voicing contrasts."""
        for language in [Language.ENGLISH, Language.GERMAN, Language.SPANISH]:
            contrasts = LANGUAGE_CONTRASTS[language]
            assert "voicing" in contrasts
            assert len(contrasts["voicing"]) > 0
    
    def test_german_has_unique_phonemes(self):
        """Test that German has phonemes not in English."""
        english_phonemes = set(LANGUAGE_PHONEMES[Language.ENGLISH])
        german_phonemes = set(LANGUAGE_PHONEMES[Language.GERMAN])
        
        german_unique = german_phonemes - english_phonemes
        assert PhonemeCategory.UE in german_unique  # ü
        assert PhonemeCategory.OE in german_unique  # ö
        assert PhonemeCategory.X in german_unique   # Bach-Laut
    
    def test_spanish_has_unique_phonemes(self):
        """Test that Spanish has phonemes not in English."""
        english_phonemes = set(LANGUAGE_PHONEMES[Language.ENGLISH])
        spanish_phonemes = set(LANGUAGE_PHONEMES[Language.SPANISH])
        
        spanish_unique = spanish_phonemes - english_phonemes
        assert PhonemeCategory.R_TAP in spanish_unique
        assert PhonemeCategory.R_TRILL in spanish_unique
        assert PhonemeCategory.B_FRIC in spanish_unique
    
    def test_language_specific_contrast_types(self):
        """Test that each language has appropriate contrast types."""
        # English: no tap/trill or fricative
        english_contrasts = LANGUAGE_CONTRASTS[Language.ENGLISH]
        assert "tap_trill" not in english_contrasts
        
        # German: has fricative, no tap/trill
        german_contrasts = LANGUAGE_CONTRASTS[Language.GERMAN]
        assert "fricative" in german_contrasts
        assert "tap_trill" not in german_contrasts
        
        # Spanish: has both tap/trill and fricative
        spanish_contrasts = LANGUAGE_CONTRASTS[Language.SPANISH]
        assert "tap_trill" in spanish_contrasts
        assert "fricative" in spanish_contrasts


# ============================================================================
# 6. Error Handling Tests
# ============================================================================

class TestMultiLanguageErrorHandling:
    """Test error handling for multi-language features."""
    
    def test_invalid_contrast_type_for_language(self, dataset_english):
        """Test error on invalid contrast type for language."""
        with pytest.raises(ValueError, match="Unknown contrast type.*tap_trill"):
            dataset_english.generate_batch(contrast_type="tap_trill")
    
    def test_error_message_shows_available_contrasts(self, dataset_german):
        """Test that error message shows available contrast types."""
        try:
            dataset_german.generate_batch(contrast_type="invalid")
        except ValueError as e:
            error_msg = str(e)
            assert "voicing" in error_msg
            assert "vowel" in error_msg
            assert "fricative" in error_msg
    
    def test_spanish_tap_trill_in_english_raises_error(self, dataset_english):
        """Test that Spanish-specific contrasts aren't available in English."""
        with pytest.raises(ValueError):
            dataset_english.generate_batch(contrast_type="tap_trill")


# ============================================================================
# 7. Integration Tests
# ============================================================================

class TestMultiLanguageIntegration:
    """Test full multi-language workflow."""
    
    def test_train_on_multiple_languages_sequentially(self):
        """Test training on multiple languages in sequence."""
        results = {}
        
        for language in [Language.ENGLISH, Language.GERMAN, Language.SPANISH]:
            config = PhonologicalConfig(language=language)
            dataset = PhonologicalDataset(config=config)
            
            # Generate batch
            stimuli, labels = dataset.generate_batch(
                task_type="discrimination",
                batch_size=32,
                contrast_type="voicing"
            )
            
            results[language.value] = {
                "stimuli_shape": stimuli.shape,
                "n_voicing_contrasts": len(dataset.voicing_contrasts),
            }
        
        # All languages should work
        assert len(results) == 3
        for lang_result in results.values():
            assert lang_result["stimuli_shape"][0] == 32
    
    def test_language_specific_continuum(self):
        """Test continuum generation for language-specific contrasts."""
        # Spanish tap/trill continuum
        dataset_spanish = PhonologicalDataset(language=Language.SPANISH)
        contrast = (PhonemeCategory.R_TAP, PhonemeCategory.R_TRILL)
        
        stimuli, labels = dataset_spanish.generate_continuum(contrast, n_steps=11)
        
        assert len(stimuli) == 11
        assert len(labels) == 11
        
        # Should transition from tap to trill
        assert labels[0] == 0  # Tap
        assert labels[-1] == 1  # Trill
