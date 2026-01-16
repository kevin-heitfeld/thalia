"""
Test Noise Scheduler - Verify curriculum-based noise scheduling.

This test validates:
1. NoiseScheduler initialization and configuration
2. Stage-specific noise profiles
3. Adaptive modulation based on performance/criticality
4. Region-specific noise scaling
5. Integration with curriculum trainer

Author: Thalia Project
Date: December 15, 2025
"""

from thalia.config.curriculum_growth import CurriculumStage
from thalia.constants.neuron import NOISE_STD_LOW
from thalia.training.curriculum import (
    NoiseScheduler,
    NoiseSchedulerConfig,
)


def test_noise_scheduler_init():
    """Test NoiseScheduler initialization."""
    config = NoiseSchedulerConfig(
        enabled=True,
        enable_criticality_adaptation=True,
        enable_performance_adaptation=True,
        verbose=False,
    )
    scheduler = NoiseScheduler(config)

    # Test public contract - config is set correctly
    assert scheduler.config.enabled is True
    assert scheduler.config.enable_criticality_adaptation is True
    assert scheduler.config.enable_performance_adaptation is True

    # Test behavior - should start with sensorimotor profile
    initial_profile = scheduler.get_current_profile()
    assert initial_profile.membrane_noise_std == NOISE_STD_LOW
    assert initial_profile.enable_weight_noise is False


def test_stage_specific_profiles():
    """Test that each stage has appropriate noise levels."""
    scheduler = NoiseScheduler(NoiseSchedulerConfig(verbose=False))

    # Stage 0: Sensorimotor - Low noise
    profile_s = scheduler.get_noise_profile(CurriculumStage.SENSORIMOTOR)
    assert profile_s.membrane_noise_std == NOISE_STD_LOW
    assert profile_s.enable_weight_noise is False
    assert profile_s.augmentation_strength == 0.05

    # Stage 1: Phonology - Low noise, weight noise enabled
    profile_p = scheduler.get_noise_profile(CurriculumStage.PHONOLOGY)
    assert profile_p.membrane_noise_std == NOISE_STD_LOW
    assert profile_p.enable_weight_noise is True
    assert profile_p.weight_noise_std == 0.02

    # Stage 4: Abstract - Higher noise
    profile_a = scheduler.get_noise_profile(CurriculumStage.ABSTRACT)
    assert profile_a.membrane_noise_std == 0.03
    assert profile_a.enable_weight_noise is True
    assert profile_a.augmentation_strength == 0.20


def test_criticality_adaptation():
    """Test noise adaptation based on criticality."""
    config = NoiseSchedulerConfig(
        enable_criticality_adaptation=True,
        criticality_noise_boost=1.5,
        criticality_noise_reduction=0.5,
        verbose=False,
    )
    scheduler = NoiseScheduler(config)
    scheduler.set_stage(CurriculumStage.PHONOLOGY)

    # Baseline
    baseline = scheduler.get_current_profile()
    baseline_noise = baseline.membrane_noise_std

    # Subcritical - should boost noise
    scheduler.update(
        CurriculumStage.PHONOLOGY,
        performance=0.75,
        criticality=0.90,  # Subcritical
    )
    boosted = scheduler.get_current_profile()
    assert boosted.membrane_noise_std > baseline_noise

    # Supercritical - should reduce noise
    scheduler.update(
        CurriculumStage.PHONOLOGY,
        performance=0.75,
        criticality=1.20,  # Supercritical
    )
    reduced = scheduler.get_current_profile()
    assert reduced.membrane_noise_std < baseline_noise


def test_performance_adaptation():
    """Test noise adaptation based on performance."""
    config = NoiseSchedulerConfig(
        enable_performance_adaptation=True,
        performance_threshold_low=0.6,
        performance_threshold_high=0.85,
        verbose=False,
    )
    scheduler = NoiseScheduler(config)
    scheduler.set_stage(CurriculumStage.TODDLER)

    # Baseline
    baseline = scheduler.get_current_profile()
    baseline_noise = baseline.membrane_noise_std

    # Low performance - should reduce noise
    scheduler.update(
        CurriculumStage.TODDLER,
        performance=0.50,  # Low
        criticality=1.0,
    )
    reduced = scheduler.get_current_profile()
    assert reduced.membrane_noise_std < baseline_noise

    # High performance - should increase noise
    scheduler.update(
        CurriculumStage.TODDLER,
        performance=0.90,  # High
        criticality=1.0,
    )
    boosted = scheduler.get_current_profile()
    assert boosted.membrane_noise_std > baseline_noise


def test_region_specific_noise():
    """Test region-specific noise scaling."""
    scheduler = NoiseScheduler(NoiseSchedulerConfig(verbose=False))
    scheduler.set_stage(CurriculumStage.GRAMMAR)

    # Get noise for different regions
    cortex_noise = scheduler.get_membrane_noise_for_region('cortex_l4')
    hippocampus_noise = scheduler.get_membrane_noise_for_region('hippocampus')
    cerebellum_noise = scheduler.get_membrane_noise_for_region('cerebellum')
    pfc_noise = scheduler.get_membrane_noise_for_region('prefrontal')

    # Hippocampus should be more variable
    assert hippocampus_noise > cortex_noise

    # Cerebellum should be less variable (precise timing)
    assert cerebellum_noise < cortex_noise

    # PFC should be slightly lower (stability for WM)
    assert pfc_noise < cortex_noise


def test_disabled_noise():
    """Test that noise can be disabled entirely."""
    config = NoiseSchedulerConfig(enabled=False, verbose=False)
    scheduler = NoiseScheduler(config)

    profile = scheduler.get_noise_profile(CurriculumStage.ABSTRACT)

    assert profile.membrane_noise_std == 0.0
    assert profile.enable_weight_noise is False
    assert profile.augmentation_strength == 0.0


def test_stage_progression():
    """Test noise increases across stages."""
    scheduler = NoiseScheduler(NoiseSchedulerConfig(verbose=False))

    stages = [
        CurriculumStage.SENSORIMOTOR,
        CurriculumStage.PHONOLOGY,
        CurriculumStage.TODDLER,
        CurriculumStage.GRAMMAR,
        CurriculumStage.READING,
        CurriculumStage.ABSTRACT,
    ]

    noise_levels = []
    for stage in stages:
        profile = scheduler.get_noise_profile(stage)
        noise_levels.append(profile.membrane_noise_std)

    # Generally increasing (with some stages having same level)
    # Check that abstract > sensorimotor
    assert noise_levels[-1] > noise_levels[0]


def test_weight_noise_enabled_stages():
    """Test weight noise is appropriately enabled across stages."""
    scheduler = NoiseScheduler(NoiseSchedulerConfig(verbose=False))

    # Sensorimotor: OFF (learning basics)
    profile_s = scheduler.get_noise_profile(CurriculumStage.SENSORIMOTOR)
    assert profile_s.enable_weight_noise is False

    # Phonology and beyond: ON (exploration)
    for stage in [CurriculumStage.PHONOLOGY, CurriculumStage.TODDLER,
                  CurriculumStage.GRAMMAR, CurriculumStage.ABSTRACT]:
        profile = scheduler.get_noise_profile(stage)
        assert profile.enable_weight_noise is True


def test_augmentation_progression():
    """Test data augmentation strength increases with stage."""
    scheduler = NoiseScheduler(NoiseSchedulerConfig(verbose=False))

    # Early stages: minimal augmentation
    profile_s = scheduler.get_noise_profile(CurriculumStage.SENSORIMOTOR)
    assert profile_s.augmentation_strength <= 0.05

    # Middle stages: moderate
    profile_g = scheduler.get_noise_profile(CurriculumStage.GRAMMAR)
    assert 0.10 <= profile_g.augmentation_strength <= 0.20

    # Late stages: higher (but conservative)
    profile_a = scheduler.get_noise_profile(CurriculumStage.ABSTRACT)
    assert profile_a.augmentation_strength == 0.20  # Max conservative


def test_wm_noise_consistent():
    """Test working memory noise is stable across stages."""
    scheduler = NoiseScheduler(NoiseSchedulerConfig(verbose=False))

    # WM noise should be consistent (not stage-dependent)
    for stage in [CurriculumStage.TODDLER, CurriculumStage.GRAMMAR,
                  CurriculumStage.ABSTRACT]:
        profile = scheduler.get_noise_profile(stage)
        # Should be in reasonable range for WM maintenance
        assert 0.015 <= profile.wm_noise_std <= 0.035


def test_rem_noise_scaling():
    """Test REM consolidation noise scales appropriately."""
    scheduler = NoiseScheduler(NoiseSchedulerConfig(verbose=False))

    # REM noise should increase for more abstract stages
    profile_p = scheduler.get_noise_profile(CurriculumStage.PHONOLOGY)
    profile_a = scheduler.get_noise_profile(CurriculumStage.ABSTRACT)

    # Abstract should have higher REM noise for complex schema extraction
    assert profile_a.rem_noise_std >= profile_p.rem_noise_std


def test_adaptation_clamping():
    """Test that adaptation is bounded to reasonable range."""
    scheduler = NoiseScheduler(NoiseSchedulerConfig(verbose=False))
    scheduler.set_stage(CurriculumStage.GRAMMAR)

    # Get baseline profile
    baseline_profile = scheduler.get_current_profile()
    baseline_noise = baseline_profile.membrane_noise_std

    # Extreme conditions
    scheduler.update(
        CurriculumStage.GRAMMAR,
        performance=0.01,  # Terrible
        criticality=0.5,   # Very subcritical
    )

    # Profile should still be reasonable (multiplier is clamped)
    adapted_profile = scheduler.get_current_profile()
    adapted_noise = adapted_profile.membrane_noise_std

    # Noise should be adapted but within reasonable bounds (0.5x to 2x)
    assert 0.5 * baseline_noise <= adapted_noise <= 2.0 * baseline_noise


def test_oscillator_phase_noise_progression():
    """Test that oscillator phase noise increases appropriately across stages."""
    scheduler = NoiseScheduler()

    # Sensorimotor (stage -0.5) should have zero oscillator phase noise
    scheduler.set_stage(CurriculumStage.SENSORIMOTOR)
    noise_sensorimotor = scheduler.get_oscillator_phase_noise_std()
    assert abs(noise_sensorimotor) < 1e-6, "Sensorimotor stage should have zero oscillator phase noise"

    # Oscillator phase noise should increase across later stages
    scheduler.set_stage(CurriculumStage.PHONOLOGY)
    noise_phonology = scheduler.get_oscillator_phase_noise_std()

    scheduler.set_stage(CurriculumStage.GRAMMAR)
    noise_grammar = scheduler.get_oscillator_phase_noise_std()

    scheduler.set_stage(CurriculumStage.ABSTRACT)
    noise_abstract = scheduler.get_oscillator_phase_noise_std()

    assert noise_phonology > noise_sensorimotor, "Phonology should have more oscillator noise than Sensorimotor"
    assert noise_grammar > noise_phonology, "Grammar should have more oscillator noise than Phonology"
    assert noise_abstract > noise_grammar, "Abstract should have more oscillator noise than Grammar"

    # Verify expected values (from stage profiles)
    assert abs(noise_phonology - 0.03) < 1e-6, f"Expected 0.03 rad for Phonology, got {noise_phonology}"
    assert abs(noise_grammar - 0.05) < 1e-6, f"Expected 0.05 rad for Grammar, got {noise_grammar}"
    assert abs(noise_abstract - 0.07) < 1e-6, f"Expected 0.07 rad for Abstract, got {noise_abstract}"


if __name__ == '__main__':
    # Run tests
    test_noise_scheduler_init()
    print("✓ Initialization test passed")

    test_stage_specific_profiles()
    print("✓ Stage-specific profiles test passed")

    test_criticality_adaptation()
    print("✓ Criticality adaptation test passed")

    test_performance_adaptation()
    print("✓ Performance adaptation test passed")

    test_region_specific_noise()
    print("✓ Region-specific noise test passed")

    test_disabled_noise()
    print("✓ Disabled noise test passed")

    test_stage_progression()
    print("✓ Stage progression test passed")

    test_weight_noise_enabled_stages()
    print("✓ Weight noise stages test passed")

    test_augmentation_progression()
    print("✓ Augmentation progression test passed")

    test_wm_noise_consistent()
    print("✓ WM noise consistency test passed")

    test_rem_noise_scaling()
    print("✓ REM noise scaling test passed")

    test_adaptation_clamping()
    print("✓ Adaptation clamping test passed")

    test_oscillator_phase_noise_progression()
    print("✓ Oscillator phase noise progression test passed")

    print("\n✅ All noise scheduler tests passed!")
