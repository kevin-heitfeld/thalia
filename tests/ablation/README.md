# Ablation Tests

Ablation tests answer the question: **"What breaks when we remove this mechanism?"**

These tests systematically disable components to understand their value
and identify unnecessary complexity.

## Purpose

1. **Document mechanism value**: Quantify what each component contributes
2. **Identify unnecessary complexity**: Find components that can be removed
3. **Validate design choices**: Confirm that complex mechanisms are worth their cost
4. **Guide simplification**: Provide data for Phase 2 (Config Profiles)

## Test Structure

Each ablation test follows this pattern:

```python
class TestWithoutMechanism:
    """Test performance without [mechanism name].
    
    Baseline: FULL configuration with all mechanisms
    Ablated: Same config but with [mechanism] disabled
    """
    
    def test_task_performance_without_mechanism(self):
        # Create baseline (full config)
        baseline_brain = create_brain(RobustnessConfig.full())
        baseline_perf = run_task(baseline_brain)
        
        # Create ablated (mechanism disabled)
        ablated_config = RobustnessConfig.full()
        ablated_config.enable_mechanism = False
        ablated_brain = create_brain(ablated_config)
        ablated_perf = run_task(ablated_brain)
        
        # Compare performance
        performance_drop = (baseline_perf - ablated_perf) / baseline_perf
        
        # Document results
        print(f"Performance drop: {performance_drop:.1%}")
        
        # Assert reasonable bounds
        assert performance_drop < 0.50, \
            "Mechanism is critical (>50% performance drop)"
```

## Planned Ablation Tests

### Stability Mechanisms

- [ ] `test_without_bcm.py` - Remove BCM learning
- [ ] `test_without_homeostasis.py` - Remove homeostatic plasticity
- [ ] `test_without_ei_balance.py` - Remove E/I balance regulation
- [ ] `test_without_divisive_norm.py` - Remove divisive normalization
- [ ] `test_without_intrinsic_plasticity.py` - Remove IP
- [ ] `test_without_metabolic.py` - Remove metabolic constraints

### Oscillations

- [ ] `test_without_theta.py` - Remove theta oscillations
- [ ] `test_without_gamma.py` - Remove gamma oscillations
- [ ] `test_without_nested_oscillations.py` - Remove theta-gamma coupling

### Neuromodulation

- [ ] `test_without_dopamine.py` - Remove dopamine system
- [ ] `test_without_tonic_dopamine.py` - Remove tonic component only
- [ ] `test_without_phasic_dopamine.py` - Remove phasic component only

### Anatomical Features

- [ ] `test_without_dendrites.py` - Remove dendritic nonlinearity
- [ ] `test_without_axonal_delays.py` - Remove event-driven delays
- [ ] `test_without_predictive_coding.py` - Remove predictive coding

## Metrics to Track

For each ablation, measure:

1. **Task Performance**
   - Accuracy on standard tasks
   - Learning speed (timesteps to threshold)
   - Final asymptotic performance

2. **Network Health**
   - Spike rate stability
   - Weight magnitude stability
   - E/I ratio
   - Criticality (branching ratio)

3. **Computational Cost**
   - Training time
   - Memory usage
   - Number of parameters

4. **Failure Modes**
   - Does it crash?
   - Does it collapse/explode?
   - Does it produce pathological behavior?

## Ablation Report Format

```markdown
## Mechanism: [Name]

**Purpose**: [What it's supposed to do]

**Ablation Results**:
- Task Performance: -X% (compared to baseline)
- Health Status: Healthy/Unhealthy
- Failure Modes: [None/List]
- Computational Savings: +Y% faster, -Z MB memory

**Recommendation**:
- [ ] KEEP - Critical for performance
- [ ] KEEP - Critical for stability
- [ ] OPTIONAL - Small benefit, high cost
- [ ] REMOVE - No measurable benefit
```

## Running Ablation Tests

```bash
# Run all ablation tests
pytest tests/ablation/ -v

# Run specific ablation
pytest tests/ablation/test_without_bcm.py -v

# Generate ablation report
pytest tests/ablation/ -v --tb=short > ablation_report.txt
```

## Interpreting Results

### Critical Mechanisms (>20% performance drop)
- Must keep
- May need to simplify implementation
- Should be enabled by default

### Beneficial Mechanisms (5-20% improvement)
- Keep for FULL profile
- Optional for MINIMAL profile
- Document tradeoffs

### Marginal Mechanisms (<5% improvement)
- Consider removing
- Only include if computationally cheap
- Document why we kept them

### No Effect (0% difference)
- Strong candidate for removal
- May indicate redundancy with other mechanisms
- Check if implementation is correct

## Success Criteria

Ablation tests succeed if they:

1. ✅ Run to completion without crashes
2. ✅ Generate quantitative comparison
3. ✅ Include health monitoring
4. ✅ Document recommendations

The goal is **data-driven decision making** about system complexity.

## Adding New Ablation Tests

1. Choose a mechanism to ablate
2. Define baseline configuration
3. Create ablated configuration
4. Run same task on both
5. Compare performance and health
6. Document recommendation
7. Update this README

## Example Ablation Results (Hypothetical)

| Mechanism | Perf. Drop | Health | Recommendation |
|-----------|-----------|---------|----------------|
| BCM | -5% | Healthy | KEEP (prevents saturation) |
| Homeostasis | -15% | Healthy | KEEP (critical stability) |
| E/I Balance | -25% | Seizures | KEEP (critical) |
| Div. Norm | -8% | Healthy | KEEP (good stability) |
| Intrinsic Plast. | -3% | Healthy | OPTIONAL |
| Metabolic | -1% | Healthy | OPTIONAL |
| Theta Osc. | -12% | Healthy | KEEP (sequence learning) |
| Gamma Osc. | -6% | Healthy | OPTIONAL |
| Dendrites | -20% | Healthy | KEEP (credit assignment) |

This data would guide the creation of config profiles in Phase 2.
