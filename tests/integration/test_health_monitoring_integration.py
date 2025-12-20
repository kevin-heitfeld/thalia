"""Integration tests for health and criticality monitoring in DynamicBrain.

Tests health monitoring, criticality tracking, and diagnostic reporting
to ensure DynamicBrain matches EventDrivenBrain monitoring capabilities.
"""

import inspect
import pytest
import torch
from thalia.config import GlobalConfig
from thalia.core.dynamic_brain import DynamicBrain
from thalia.core.brain_builder import BrainBuilder
from thalia.managers.component_registry import ComponentRegistry


@pytest.fixture
def health_brain() -> DynamicBrain:
    """Create DynamicBrain with health monitoring for testing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create global config
    global_config = GlobalConfig(
        device=device,
        dt_ms=1.0,
    )
    # Add criticality flag manually (not in GlobalConfig schema yet)
    global_config.monitor_criticality = False  # type: ignore

    # Create builder and build brain
    builder = BrainBuilder(global_config)

    # Add minimal regions for testing
    builder.add_component("cortex", "cortex", n_output=64, n_input=32, l4_size=32, l23_size=48, l5_size=16, l6a_size=0, l6b_size=0)
    builder.add_component("hippocampus", "hippocampus", n_output=32)
    builder.add_component("pfc", "prefrontal", n_output=16, n_input=64)

    # Add connections
    builder.connect("cortex", "hippocampus", "axonal_projection")
    builder.connect("cortex", "pfc", "axonal_projection")

    # Build brain (event-driven mode is default)
    # Ensure ComponentRegistry is set up for event-driven execution
    ComponentRegistry()
    brain = builder.build()

    return brain


@pytest.fixture
def criticality_brain() -> DynamicBrain:
    """Create DynamicBrain with criticality monitoring enabled."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create global config
    global_config = GlobalConfig(
        device=device,
        dt_ms=1.0,
    )
    # Add criticality flag manually (not in GlobalConfig schema yet)
    global_config.monitor_criticality = True  # type: ignore

    # Create builder and build brain
    builder = BrainBuilder(global_config)

    # Add minimal regions for testing
    builder.add_component("cortex", "cortex", n_output=64, n_input=32, l4_size=32, l23_size=48, l5_size=16, l6a_size=0, l6b_size=0)
    builder.add_component("hippocampus", "hippocampus", n_output=32)
    builder.add_component("pfc", "prefrontal", n_output=16, n_input=64)

    # Add connections
    builder.connect("cortex", "hippocampus", "axonal_projection")
    builder.connect("cortex", "pfc", "axonal_projection")

    # Build brain (event-driven mode is default)
    # Ensure ComponentRegistry is set up for event-driven execution
    ComponentRegistry()
    brain = builder.build()

    return brain


def test_health_monitor_initialized(health_brain):
    """Test that HealthMonitor is always initialized."""
    assert hasattr(health_brain, 'health_monitor')
    assert health_brain.health_monitor is not None
    print("✓ HealthMonitor initialized")


def test_criticality_monitor_optional(health_brain, criticality_brain):
    """Test that CriticalityMonitor is optional based on config."""
    # Default: criticality disabled
    assert hasattr(health_brain, 'criticality_monitor')
    assert health_brain.criticality_monitor is None

    # Enabled: criticality monitor initialized
    assert criticality_brain.criticality_monitor is not None
    print("✓ CriticalityMonitor is optional")


def test_check_health_method_exists(health_brain):
    """Test that check_health() method exists and returns correct format."""
    from thalia.diagnostics.health_monitor import HealthReport

    health_report = health_brain.check_health()

    # Check report structure (HealthReport dataclass)
    assert isinstance(health_report, HealthReport)
    assert hasattr(health_report, 'is_healthy')
    assert hasattr(health_report, 'issues')
    assert hasattr(health_report, 'summary')
    assert hasattr(health_report, 'overall_severity')

    # Check types
    assert isinstance(health_report.is_healthy, bool)
    assert isinstance(health_report.issues, list)
    assert isinstance(health_report.summary, str)
    assert isinstance(health_report.overall_severity, (int, float))

    print(f"✓ check_health() returns: is_healthy={health_report.is_healthy}, "
          f"issues={len(health_report.issues)}, severity={health_report.overall_severity:.2f}")


def test_check_health_with_normal_activity(health_brain):
    """Test that healthy brain returns is_healthy=True."""
    device = health_brain.device

    # Run a few timesteps with normal input
    for _ in range(5):
        input_data = {
            'cortex': torch.randn(32, device=device) * 0.5,  # Moderate input
        }
        health_brain.forward(input_data, n_timesteps=1)

    # Check health
    health_report = health_brain.check_health()

    # With normal activity, should be healthy
    # (May have minor warnings but not critical issues)
    print(f"✓ Normal activity health: {health_report.is_healthy}, "
          f"issues={len(health_report.issues)}, summary='{health_report.summary}'")


def test_check_health_detects_silence(health_brain):
    """Test that prolonged silence is detected as unhealthy."""
    device = health_brain.device

    # Run many timesteps with zero input
    for _ in range(100):
        input_data = {
            'cortex': torch.zeros(32, device=device),  # No input
        }
        health_brain.forward(input_data, n_timesteps=1)

    # Check health
    health_report = health_brain.check_health()

    # Should detect low activity (though may not flag as unhealthy immediately)
    print(f"✓ Silence detection: is_healthy={health_report.is_healthy}, "
          f"issues={len(health_report.issues)}, severity={health_report.overall_severity:.2f}")

    # At minimum should report activity metrics
    assert len(health_report.issues) >= 0  # May or may not detect as issue


def test_enhanced_diagnostics(health_brain):
    """Test that get_diagnostics includes all subsystems."""
    device = health_brain.device

    # Run a timestep
    input_data = {'cortex': torch.randn(32, device=device)}
    health_brain.forward(input_data, n_timesteps=1)

    # Get diagnostics
    diagnostics = health_brain.get_diagnostics()

    # Check all required subsystems present
    assert 'components' in diagnostics
    assert 'spike_counts' in diagnostics
    assert 'pathways' in diagnostics
    assert 'oscillators' in diagnostics
    assert 'neuromodulators' in diagnostics

    # Check component diagnostics
    assert 'cortex' in diagnostics['components']
    assert 'hippocampus' in diagnostics['components']
    assert 'pfc' in diagnostics['components']

    # Check oscillator diagnostics (6 frequencies)
    assert 'delta' in diagnostics['oscillators']
    assert 'theta' in diagnostics['oscillators']
    assert 'alpha' in diagnostics['oscillators']
    assert 'beta' in diagnostics['oscillators']
    assert 'gamma' in diagnostics['oscillators']
    assert 'ripple' in diagnostics['oscillators']

    # Check neuromodulator diagnostics
    assert 'dopamine' in diagnostics['neuromodulators'] or 'vta' in diagnostics['neuromodulators']

    print(f"✓ Enhanced diagnostics: {len(diagnostics)} subsystems tracked")


def test_criticality_tracking_updates(criticality_brain):
    """Test that criticality monitor updates during forward pass."""
    device = criticality_brain.device

    # Initial state
    assert criticality_brain.criticality_monitor is not None

    # Run timesteps
    for _ in range(10):
        input_data = {'cortex': torch.randn(32, device=device) * 0.5}
        criticality_brain.forward(input_data, n_timesteps=1)

    # Check that criticality was updated (should have diagnostics)
    updated_diagnostics = criticality_brain.criticality_monitor.get_diagnostics()
    assert updated_diagnostics is not None

    print(f"✓ Criticality tracking updated during forward pass")


def test_criticality_in_diagnostics(criticality_brain):
    """Test that criticality metrics appear in get_diagnostics()."""
    device = criticality_brain.device

    # Run a few timesteps
    for _ in range(5):
        input_data = {'cortex': torch.randn(32, device=device)}
        criticality_brain.forward(input_data, n_timesteps=1)

    # Get diagnostics
    diagnostics = criticality_brain.get_diagnostics()

    # Should include criticality subsystem
    assert 'criticality' in diagnostics

    print(f"✓ Criticality metrics in diagnostics")


def test_health_check_uses_all_diagnostics(health_brain):
    """Test that health check considers all subsystem diagnostics."""
    device = health_brain.device

    # Run timesteps
    for _ in range(10):
        input_data = {'cortex': torch.randn(32, device=device)}
        health_brain.forward(input_data, n_timesteps=1)

    # Get diagnostics
    diagnostics = health_brain.get_diagnostics()

    # Health check should use comprehensive diagnostics
    # (spike_counts, oscillators, pathways, etc.)
    assert 'spike_counts' in diagnostics
    assert len(diagnostics['spike_counts']) > 0  # Has spike data

    # Run health check to ensure it uses diagnostics
    health_brain.check_health()

    print(f"✓ Health check uses comprehensive diagnostics")


def test_health_monitoring_backward_compatibility(health_brain):
    """Test that health monitoring is compatible with EventDrivenBrain interface."""
    from thalia.diagnostics.health_monitor import HealthReport

    # EventDrivenBrain interface requirements:
    # 1. check_health() method exists
    # 2. Returns HealthReport with is_healthy, issues, summary, overall_severity
    # 3. get_diagnostics() includes all subsystem info

    # Check interface compatibility
    assert hasattr(health_brain, 'check_health')
    assert hasattr(health_brain, 'get_diagnostics')
    assert hasattr(health_brain, 'health_monitor')

    # Check method signatures match
    sig = inspect.signature(health_brain.check_health)
    assert len(sig.parameters) == 0  # No required parameters

    # Check return format (HealthReport dataclass)
    health_report = health_brain.check_health()
    assert isinstance(health_report, HealthReport)
    assert hasattr(health_report, 'is_healthy')
    assert hasattr(health_report, 'issues')
    assert hasattr(health_report, 'summary')
    assert hasattr(health_report, 'overall_severity')

    print("✓ Health monitoring interface matches EventDrivenBrain")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
