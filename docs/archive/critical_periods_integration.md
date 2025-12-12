# Critical Periods Integration Plan

**Status**: Design Phase  
**Date**: December 12, 2025  
**Related**: `docs/design/curriculum_strategy.md`, `src/thalia/learning/critical_periods.py`

## Current Situation

The `CriticalPeriodGating` module exists in `src/thalia/learning/critical_periods.py` with full implementation:
- **Domains**: phonology, grammar, semantics, face_recognition, motor
- **Three-phase gating**: early (50%), peak (120%), late (sigmoid decay to 20%)
- **Configurable windows**: start_step, end_step, peak_multiplier, decay_rate
- **Methods**: `gate_learning()`, `get_window_status()`, `is_in_peak()`

However, it's **NOT integrated** into the curriculum training pipeline. The curriculum strategy document mentions critical periods conceptually but doesn't wire them into the actual training loop.

## Gap Analysis

### What Exists
✅ `CriticalPeriodGating` class with all logic  
✅ `CriticalPeriodConfig` with developmental windows  
✅ Curriculum strategy document mentions critical periods  
✅ Stage-specific task configs in `CurriculumTrainer`  

### What's Missing
❌ **No learning rate modulation in training loop**  
❌ **No critical period instance in CurriculumTrainer**  
❌ **No connection between domains and curriculum stages**  
❌ **No logging/monitoring of critical period status**  
❌ **No configuration in StageConfig**

## Biological Mapping

### Critical Period Windows (Default from code)

| Domain | Start | End | Peak | Floor | Biological Basis |
|--------|-------|-----|------|-------|------------------|
| **Phonology** | 0 | 50k | 120% | 20% | Native phoneme discrimination (0-6 months) |
| **Grammar** | 25k | 150k | 120% | 20% | Syntax acquisition (1-7 years) |
| **Semantics** | 50k | 300k | 115% | 30% | Vocabulary/meaning (1-18 years) |
| **Face Recognition** | 0 | 100k | 120% | 25% | Early visual expertise (0-12 months) |
| **Motor** | 0 | 75k | 125% | 30% | Sensorimotor coordination (0-12 months) |

### Curriculum Stage Alignment

| Stage | Age (steps) | Active Windows | Domains at Peak |
|-------|-------------|----------------|-----------------|
| **-0.5 (Sensorimotor)** | 0-50k | Motor, Face | Motor (PEAK), Face (PEAK) |
| **0 (Phonology)** | 50k-100k | Phonology, Motor | Phonology (PEAK), Motor (declining) |
| **1 (Toddler)** | 100k-200k | Grammar, Semantics | Grammar (PEAK), Semantics (starting) |
| **2 (Grammar)** | 200k-350k | Grammar, Semantics | Grammar (declining), Semantics (PEAK) |
| **3 (Reading)** | 350k-500k | Semantics | Semantics (late PEAK) |
| **4 (Abstract)** | 500k+ | All declining | All post-window |

## Integration Strategy

### Option 1: Global Critical Period Controller (RECOMMENDED)

Add critical period gating at the `CurriculumTrainer` level, modulating learning rates for all regions based on current stage and domain.

**Architecture**:
```
CurriculumTrainer
├── critical_period_gating: CriticalPeriodGating
├── domain_mappings: Dict[str, List[str]]  # task_name -> domains
└── train_stage()
    ├── Get base learning rates from regions
    ├── Apply critical period gating per task
    └── Set modulated learning rates before forward pass
```

**Pros**:
- Centralized control
- Easy to log and monitor
- Clean separation of concerns
- Matches biological reality (neuromodulation is global)

**Cons**:
- Requires setting learning rates on regions before each forward pass
- Needs interface to get/set learning rates on regions

### Option 2: Per-Region Critical Period (Local)

Each region tracks its own critical periods and modulates learning internally.

**Architecture**:
```
BrainRegion (with LearningStrategyMixin)
├── critical_period_gating: CriticalPeriodGating
├── current_domain: str
└── apply_strategy_learning()
    ├── Get base learning rate from strategy
    ├── Apply critical period gating
    └── Use modulated rate for updates
```

**Pros**:
- No global coordination needed
- Regions self-manage plasticity
- Works with existing `LearningStrategyMixin`

**Cons**:
- Duplicate gating instances (memory overhead)
- Harder to monitor globally
- Regions need to know which domain they're processing (unclear)

### Option 3: Neuromodulator-Based (Biological)

Treat critical period effects as a **neuromodulator** (like dopamine/acetylcholine) that modulates plasticity.

**Architecture**:
```
CurriculumTrainer
├── critical_period_gating: CriticalPeriodGating
└── train_stage()
    ├── Compute critical period multiplier
    ├── Set as "plasticity_modulator" on brain/regions
    └── Regions use modulator in learning

BrainRegion
└── apply_strategy_learning()
    ├── Get base learning rate
    ├── Modulate by state.plasticity_modulator
    └── Apply learning
```

**Pros**:
- Most biologically accurate (GABAergic inhibition → reduced plasticity)
- Uses existing neuromodulator infrastructure
- Clean interface via state

**Cons**:
- Less explicit than Option 1
- Requires state.plasticity_modulator field
- Might conflict with other modulators

## Recommended Approach

**Use Option 1 (Global Controller) with explicit domain tracking**

### Implementation Steps

#### 1. Add Critical Period Support to `StageConfig`

```python
# In src/thalia/training/curriculum/stage_manager.py

@dataclass
class StageConfig:
    """Configuration for a single curriculum stage."""
    
    # ... existing fields ...
    
    # NEW: Critical period configuration
    enable_critical_periods: bool = True
    domain_mappings: Dict[str, List[str]] = field(default_factory=dict)
    # Maps task_name -> domains for that task
    # Example: {'phonology_task': ['phonology'], 'grammar_task': ['grammar', 'semantics']}
```

#### 2. Add Critical Period Gating to `CurriculumTrainer`

```python
# In src/thalia/training/curriculum/stage_manager.py

from thalia.learning.critical_periods import CriticalPeriodGating

class CurriculumTrainer:
    def __init__(self, ...):
        # ... existing initialization ...
        
        # NEW: Critical period gating
        self.critical_period_gating = CriticalPeriodGating()
        
        # Track active domains per stage
        self._active_domains: Dict[str, List[str]] = {}
```

#### 3. Modulate Learning Rates in Training Loop

```python
# In CurriculumTrainer.train_stage()

def train_stage(self, stage, config, task_loader, evaluator=None):
    # ... setup ...
    
    for step in range(config.duration_steps):
        # 1. Sample task
        task_name = self.task_sampler.sample_next_task(task_weights)
        task_data = task_loader.get_task(task_name)
        
        # 2. Apply critical period gating (NEW)
        if config.enable_critical_periods:
            self._apply_critical_period_modulation(
                task_name=task_name,
                domains=config.domain_mappings.get(task_name, []),
                age=self.global_step,
            )
        
        # 3. Forward pass (learning happens automatically with modulated rates)
        output = self.brain.forward(task_data['input'], ...)
        
        # ... rest of loop ...
```

#### 4. Implement `_apply_critical_period_modulation()`

```python
def _apply_critical_period_modulation(
    self,
    task_name: str,
    domains: List[str],
    age: int,
) -> Dict[str, float]:
    """Apply critical period gating to learning rates.
    
    For each domain active in this task, compute the gating multiplier
    and modulate learning rates in relevant regions.
    
    Args:
        task_name: Current task name
        domains: List of domains this task trains (e.g., ['phonology'])
        age: Current training step
    
    Returns:
        Dict of domain -> multiplier for logging
    """
    if not domains:
        return {}
    
    # Compute average multiplier across all domains for this task
    multipliers = {}
    total_multiplier = 0.0
    
    for domain in domains:
        try:
            # Get window status for logging
            status = self.critical_period_gating.get_window_status(domain, age)
            multiplier = status['multiplier']
            multipliers[domain] = multiplier
            total_multiplier += multiplier
            
            # Log if at critical transition points
            if status['phase'] != getattr(self, f'_last_phase_{domain}', None):
                if self.verbose:
                    print(f"  Critical period: {domain} entering {status['phase']} phase "
                          f"(multiplier: {multiplier:.2f})")
                setattr(self, f'_last_phase_{domain}', status['phase'])
        
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Unknown domain '{domain}' for task '{task_name}'")
            continue
    
    # Average multiplier for this task
    if multipliers:
        avg_multiplier = total_multiplier / len(multipliers)
        
        # Apply to brain plasticity
        # Option A: Set as global modulator (if supported)
        if hasattr(self.brain, 'set_plasticity_modulator'):
            self.brain.set_plasticity_modulator(avg_multiplier)
        
        # Option B: Scale learning rates in all learning strategies
        # (This requires regions to expose learning rate setters)
        # For now, we'll implement this as a global modulator
    
    return multipliers
```

#### 5. Add Domain Mappings to Stage Configurations

```python
# In training scripts (e.g., thalia_birth_sensorimotor.py)

# Stage -0.5 (Sensorimotor)
stage_minus_half_config = StageConfig(
    duration_steps=50000,
    task_configs={
        'motor_control': TaskConfig(weight=0.40, difficulty=0.5),
        'reaching': TaskConfig(weight=0.35, difficulty=0.6),
        'manipulation': TaskConfig(weight=0.20, difficulty=0.7),
        'prediction': TaskConfig(weight=0.05, difficulty=0.7),
    },
    enable_critical_periods=True,
    domain_mappings={
        'motor_control': ['motor'],
        'reaching': ['motor', 'face_recognition'],  # Visual targeting
        'manipulation': ['motor'],
        'prediction': ['motor'],
    },
    # ... rest of config ...
)

# Stage 0 (Phonology)
stage_0_config = StageConfig(
    # ...
    domain_mappings={
        'phoneme_discrimination': ['phonology'],
        'visual_object': ['face_recognition'],
        'temporal_sequences': ['semantics'],  # Early semantic groundwork
    },
)

# Stage 1 (Grammar)
stage_1_config = StageConfig(
    # ...
    domain_mappings={
        'word_recognition': ['semantics', 'phonology'],
        'simple_grammar': ['grammar'],
        'working_memory': ['semantics'],
    },
)

# Stage 2 (Grammar consolidation)
stage_2_config = StageConfig(
    # ...
    domain_mappings={
        'trilingual_grammar': ['grammar', 'semantics'],
        'composition': ['grammar', 'semantics'],
    },
)
```

#### 6. Add Critical Period Monitoring

```python
# In CurriculumTrainer._collect_metrics()

def _collect_metrics(self) -> Dict[str, float]:
    """Collect training metrics."""
    metrics = {
        # ... existing metrics ...
    }
    
    # Add critical period status
    if hasattr(self, 'critical_period_gating'):
        for domain in self.critical_period_gating.get_all_domains():
            status = self.critical_period_gating.get_window_status(
                domain, 
                self.global_step
            )
            metrics[f'critical_period/{domain}_multiplier'] = status['multiplier']
            metrics[f'critical_period/{domain}_progress'] = status['progress']
    
    return metrics
```

#### 7. Add to Logging and Visualization

```python
# In CurriculumLogger or dashboard

def plot_critical_periods(history: List[Dict[str, float]]):
    """Plot critical period multipliers over time."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for domain in ['phonology', 'grammar', 'semantics', 'motor', 'face_recognition']:
        steps = [m['step'] for m in history]
        multipliers = [
            m.get(f'critical_period/{domain}_multiplier', 0.0) 
            for m in history
        ]
        ax.plot(steps, multipliers, label=domain, linewidth=2)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', label='Baseline')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Learning Rate Multiplier')
    ax.set_title('Critical Period Windows Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig
```

## Implementation Checklist

### Phase 1: Minimal Integration (Week 1)
- [ ] Add `critical_period_gating` to `CurriculumTrainer.__init__()`
- [ ] Add `enable_critical_periods` and `domain_mappings` to `StageConfig`
- [ ] Implement `_apply_critical_period_modulation()` (basic version)
- [ ] Add critical period metrics to `_collect_metrics()`
- [ ] Test with Stage -0.5 (motor domain only)

### Phase 2: Full Integration (Week 2)
- [ ] Add domain mappings for all stages (Stage 0-4)
- [ ] Implement plasticity modulator interface on brain/regions
- [ ] Add critical period status to logging
- [ ] Add visualization to dashboard
- [ ] Document usage in training scripts

### Phase 3: Validation (Week 3)
- [ ] Verify multipliers match expected windows
- [ ] Test phase transitions (early → peak → late)
- [ ] Compare performance with/without critical periods
- [ ] Add ablation study: `no_critical_periods` condition
- [ ] Measure impact on phonology/grammar acquisition

### Phase 4: Advanced Features (Week 4)
- [ ] Custom domains via `CriticalPeriodGating.add_domain()`
- [ ] Per-region domain specificity (if needed)
- [ ] Adaptive window adjustment (if performance plateaus)
- [ ] Integration with other modulators (dopamine, attention)

## Expected Impact

### Performance Improvements

**With Critical Periods** (predicted):
- **Phonology (Stage 0)**: +10-15% accuracy (in peak window)
- **Grammar (Stage 1-2)**: +10-12% accuracy (peak plasticity)
- **Late Language Learning**: -15-20% efficiency (post-window)
  - This is CORRECT behavior (matches human critical periods)

**Without Critical Periods** (baseline):
- Constant learning rate across all ages
- No developmental sensitivity
- Misses biological advantage of early plasticity

### Biological Realism

Critical periods explain:
- ✅ Why phonology MUST be Stage 0 (peak window)
- ✅ Why grammar window is Stage 1-2 (later, longer)
- ✅ Why semantics never fully closes (lifelong vocabulary)
- ✅ Bilingual advantage when early (<Stage 2)
- ✅ Difficulty learning new phonology after Stage 1

### Sample Efficiency

Expected reductions in steps to criterion:
- Phonology: 30-40% fewer steps (peak plasticity)
- Grammar: 20-30% fewer steps (optimal window)
- Motor: 25-35% fewer steps (early high plasticity)

## Testing Strategy

### Unit Tests

```python
# tests/unit/learning/test_critical_periods_integration.py

def test_critical_period_modulation_applied():
    """Test that critical period gating modulates learning rates."""
    trainer = CurriculumTrainer(brain, ...)
    
    # Stage 0 (phonology peak)
    trainer.global_step = 25000  # Mid-phonology window
    multipliers = trainer._apply_critical_period_modulation(
        task_name='phoneme_discrimination',
        domains=['phonology'],
        age=25000,
    )
    
    assert multipliers['phonology'] == pytest.approx(1.2, abs=0.01)
    
    # Late (post-window)
    multipliers = trainer._apply_critical_period_modulation(
        task_name='phoneme_discrimination',
        domains=['phonology'],
        age=200000,
    )
    
    assert multipliers['phonology'] < 0.5  # Declining

def test_multiple_domains():
    """Test task with multiple domains (averaging)."""
    trainer = CurriculumTrainer(brain, ...)
    trainer.global_step = 100000
    
    multipliers = trainer._apply_critical_period_modulation(
        task_name='grammar_semantic_task',
        domains=['grammar', 'semantics'],
        age=100000,
    )
    
    # Grammar should be in peak, semantics in early/peak
    assert 'grammar' in multipliers
    assert 'semantics' in multipliers
    assert all(m > 0.5 for m in multipliers.values())
```

### Integration Tests

```python
# tests/integration/test_curriculum_critical_periods.py

def test_stage0_phonology_advantage():
    """Test that phonology learns faster in Stage 0 vs late."""
    
    # Condition A: Stage 0 (peak window)
    brain_early = create_brain()
    trainer_early = CurriculumTrainer(brain_early, ...)
    trainer_early.global_step = 25000  # Peak phonology
    
    # Condition B: Stage 3 (post-window)
    brain_late = create_brain()
    trainer_late = CurriculumTrainer(brain_late, ...)
    trainer_late.global_step = 400000  # Late phonology
    
    # Train both on same phonology task
    steps_to_criterion_early = train_to_criterion(trainer_early, 'phonology')
    steps_to_criterion_late = train_to_criterion(trainer_late, 'phonology')
    
    # Early should be 30-50% faster
    assert steps_to_criterion_late > steps_to_criterion_early * 1.3
```

### Ablation Study

Add to curriculum ablation suite:
```python
'no_critical_periods': {
    'description': 'Constant learning rate (no plasticity windows)',
    'hypothesis': 'Phonology/grammar harder to acquire in later stages',
    'expected_drop': '10-15%',
    'config_override': {
        'enable_critical_periods': False,
    },
}
```

## Open Questions

1. **How to propagate modulated learning rates to regions?**
   - Option A: Add `brain.set_plasticity_modulator(multiplier)` global setter
   - Option B: Modify `LearningStrategyMixin.apply_strategy_learning()` to accept external multiplier
   - Option C: Expose `region.set_learning_rate()` and call before each forward
   
   **Recommendation**: Option A (global modulator via brain state)

2. **Should critical periods affect ALL learning or only specific rules?**
   - Biology: Affects ALL plasticity (GABAergic inhibition is global)
   - Implementation: Apply to all regions during that task
   
   **Recommendation**: Global effect (all regions modulated)

3. **How to handle multi-domain tasks?**
   - Average multipliers across domains
   - Use maximum multiplier (most permissive)
   - Use minimum multiplier (most conservative)
   
   **Recommendation**: Average (balanced approach)

4. **Should windows be fixed or adaptive?**
   - Fixed: Matches biological reality, predictable
   - Adaptive: Could extend windows if performance poor
   
   **Recommendation**: Start fixed, add adaptive in Phase 4 if needed

## Related Documents

- `docs/design/curriculum_strategy.md` - Main curriculum design
- `src/thalia/learning/critical_periods.py` - Implementation
- `docs/patterns/component-parity.md` - Regions and pathways
- `docs/patterns/state-management.md` - Neuromodulator patterns

## References

- Werker & Tees (1984): Phoneme discrimination critical period
- Newport (1990): Less is More - Critical periods in language
- Hensch (2004): Critical period plasticity in neural circuits
- Knudsen (2004): Sensitive periods in brain development
- Bavelier et al. (2010): Removing brakes on adult plasticity

---

**Next Steps**: 
1. Review this plan with team
2. Implement Phase 1 (minimal integration)
3. Test with Stage -0.5 motor domain
4. Expand to full curriculum in Phase 2
