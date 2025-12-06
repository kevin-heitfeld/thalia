# Integration Tests

Integration tests verify that multiple components work together correctly.
Unlike unit tests which test components in isolation, integration tests
check interactions between components at different complexity levels.

## Complexity Level Hierarchy

```
Level 0: PRIMITIVES
├── LIFNeuron, ConductanceLIF, DendriticNeuron
├── SpikeTraces, Synapses
└── Test: Basic neuron dynamics with synthetic inputs

Level 1: LEARNING RULES
├── STDP, BCM, Hebbian, ThreeFactor
├── Test: Apply to random weight matrices
└── Depends on: Level 0 traces

Level 2: STABILITY MECHANISMS
├── UnifiedHomeostasis, EIBalance, DivisiveNormalization
├── IntrinsicPlasticity, MetabolicConstraints
├── Test: Can stabilize runaway/collapse scenarios
└── Depends on: Level 0-1

Level 3: REGIONS (Isolated)
├── LayeredCortex, TrisynapticHippocampus
├── Striatum, Prefrontal, Cerebellum
├── Test: Each region with dummy inputs
└── Depends on: Level 0-2

Level 4: INTEGRATION
├── EventDrivenBrain, Pathways
├── Inter-region communication
├── Test: Full system integration
└── Depends on: Level 0-3
```

## Test Organization

### Current Integration Tests

- `test_stdp_with_homeostasis.py` - Learning + Stability (Levels 1+2)
- More to be added...

### Planned Integration Tests

- `test_cortex_with_dopamine.py` - Region + Neuromodulation
- `test_hippocampus_with_theta.py` - Region + Oscillations
- `test_two_region_communication.py` - Inter-region pathways
- `test_full_brain_assembly.py` - Complete system

## Writing Integration Tests

Integration tests should:

1. **Test interactions, not features**: Don't repeat unit test functionality
2. **Use fixtures from conftest.py**: Leverage pre-configured components
3. **Check health**: Use HealthMonitor to detect pathological states
4. **Be realistic**: Use biologically plausible parameters
5. **Document dependencies**: Clearly state which levels are being tested

### Example Structure

```python
class TestCortexWithDopamine:
    """Test LayeredCortex with dopamine modulation.
    
    Complexity Levels: 3 (Region) + Neuromodulation
    """
    
    def test_dopamine_enhances_plasticity(
        self,
        layered_cortex,
        health_monitor
    ):
        # Setup test scenario
        cortex = layered_cortex
        
        # Run with low dopamine
        # ...
        
        # Run with high dopamine
        # ...
        
        # Verify dopamine effect
        assert high_da_plasticity > low_da_plasticity
        
        # Check health
        diagnostics = cortex.get_diagnostics()
        report = health_monitor.check_health(diagnostics)
        assert report.is_healthy
```

## Running Integration Tests

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run specific test file
pytest tests/integration/test_stdp_with_homeostasis.py -v

# Run with health monitoring output
pytest tests/integration/ -v -s
```

## Health Monitoring

All integration tests should use the `health_monitor` fixture to
detect pathological states:

- Activity collapse (too few spikes)
- Seizure risk (too many spikes)
- Weight explosion/collapse
- E/I imbalance
- Criticality drift

If an integration test fails health checks, it indicates that the
component interaction creates unstable dynamics.

## Adding New Integration Tests

1. Identify which complexity levels you're testing
2. Add fixtures to `conftest.py` if needed
3. Create test file with clear documentation
4. Use health monitoring
5. Run tests: `pytest tests/integration/your_test.py -v`
6. Update this README with new test descriptions

## Success Criteria

Integration tests pass if:

1. ✅ Components work together without errors
2. ✅ Health monitor reports healthy state
3. ✅ Emergent behavior matches expectations
4. ✅ No mysterious failures or instabilities
