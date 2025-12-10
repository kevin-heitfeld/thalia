"""
Test Manim visualization module.

These tests verify that the visualization module loads correctly
and handles missing Manim gracefully.
"""

import pytest


def test_visualization_module_imports():
    """Test that visualization module can be imported."""
    try:
        from thalia.visualization import (
            BrainActivityVisualization,
            BrainArchitectureScene,
            SpikeActivityScene,
            LearningScene,
            GrowthScene,
            MANIM_AVAILABLE,
        )
        
        # Should always succeed (even if Manim not installed)
        assert BrainActivityVisualization is not None
        assert isinstance(MANIM_AVAILABLE, bool)
        
    except ImportError as e:
        pytest.fail(f"Failed to import visualization module: {e}")


def test_manim_availability_flag():
    """Test MANIM_AVAILABLE flag is set correctly."""
    from thalia.visualization import MANIM_AVAILABLE
    
    assert isinstance(MANIM_AVAILABLE, bool)
    
    # If available, should be able to import manim
    if MANIM_AVAILABLE:
        try:
            import manim
            assert manim is not None
        except ImportError:
            pytest.fail("MANIM_AVAILABLE=True but manim import failed")


def test_visualization_in_main_init():
    """Test that visualization exports are in main __init__."""
    from thalia import BrainActivityVisualization, MANIM_AVAILABLE
    
    # Should be importable from thalia
    assert BrainActivityVisualization is not None or not MANIM_AVAILABLE
    assert isinstance(MANIM_AVAILABLE, bool)


@pytest.mark.skipif(
    not __import__('importlib').util.find_spec('manim'),
    reason="Requires manim installation"
)
def test_brain_activity_visualization_creation():
    """Test BrainActivityVisualization can be instantiated (if Manim available)."""
    from thalia.visualization import BrainActivityVisualization, MANIM_AVAILABLE
    
    if not MANIM_AVAILABLE:
        pytest.skip("Manim not available")
    
    # Should be able to create visualizer without checkpoint
    viz = BrainActivityVisualization()
    assert viz is not None


@pytest.mark.skipif(
    not __import__('importlib').util.find_spec('manim'),
    reason="Requires manim installation"
)
def test_scene_classes_exist():
    """Test that scene classes can be imported (if Manim available)."""
    from thalia.visualization import (
        BrainArchitectureScene,
        SpikeActivityScene,
        LearningScene,
        GrowthScene,
        MANIM_AVAILABLE,
    )
    
    if not MANIM_AVAILABLE:
        pytest.skip("Manim not available")
    
    # Scene classes should be importable
    assert BrainArchitectureScene is not None
    assert SpikeActivityScene is not None
    assert LearningScene is not None
    assert GrowthScene is not None
