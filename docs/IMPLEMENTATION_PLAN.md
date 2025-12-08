# Thalia Curriculum Implementation Plan

**Date**: December 8, 2025
**Status**: ✅ **Priority 1 Complete** (Week 1 Done!)
**Goal**: Implement missing curriculum-specific components before training begins

## Executive Summary

**Core Infrastructure Status**: ✅ **EXCELLENT** (A+ grade)
- All learning rules, regions, pathways, and training infrastructure exist
- Gaps are curriculum-specific mechanisms that layer on top of core

**Implementation Timeline**: 2-4 weeks before Stage -0.5 training
**Estimated Components**: 13 new/enhanced modules
**Priority**: ✅ 3 critical completed, 10 stage-specific remaining

**🎉 MILESTONE ACHIEVED**: All Priority 1 (Critical) components implemented and tested!

---

## Priority 1: 🔴 CRITICAL (Week 1-2, Must Have Before ANY Training)

### 1.1 Critical Period Gating ✅ COMPLETE
**Curriculum**: Stage 0 phonology (0-50k steps), grammar (25k-150k steps)
**Status**: ✅ **Implemented** (December 8, 2025)
**Actual Time**: ~6 hours (26 tests, 0.29s)
**Impact**: Time-windowed plasticity modulation for developmental learning

**Implementation**:
- **File**: `src/thalia/learning/critical_periods.py`
- **Class**: `CriticalPeriodGating`
- **Methods**: `gate_learning(learning_rate, domain, age)`
- **Integration**: Wrap `LocalTrainer` learning rate calls

**Code Structure**:
```python
class CriticalPeriodGating:
    """Modulate learning rates based on developmental windows."""

    def __init__(self):
        self.plasticity_window = {
            'phonology': (0, 50000),      # Peak: Stage 0
            'grammar': (25000, 150000),    # Peak: Stage 1-2
            'semantics': (50000, 300000),  # Extended window
        }

    def gate_learning(self, learning_rate, domain, age):
        """Apply critical period modulation."""
        window = self.plasticity_window[domain]
        if age < window[0]:
            return learning_rate * 0.5  # Too early
        elif age > window[1]:
            # Sigmoidal decay after window closes
            decay = 1.0 / (1.0 + np.exp((age - window[1]) / 20000))
            return learning_rate * (0.2 + 0.8 * decay)
        else:
            return learning_rate * 1.2  # Peak plasticity
```

**Tests**: `tests/unit/test_critical_periods.py`

---

### 1.2 Curriculum Training Infrastructure ✅ COMPLETE
**Curriculum**: All stages, continuous
**Status**: ✅ **Implemented** (December 8, 2025)
**Actual Time**: ~1 day (36 tests, 0.52s)
**Impact**: Core curriculum mechanics (interleaving, spacing, testing effect)

**Implementation**:
- **File**: `src/thalia/training/curriculum.py`
- **Classes**: 6 new classes

#### 1.2.1 InterleavedCurriculumSampler
**Purpose**: Multinomial task sampling (not blocked)
```python
class InterleavedCurriculumSampler:
    """Sample tasks from distribution each step."""

    def sample_next_task(self, stage_weights):
        """
        stage_weights: {0: 0.05, 1: 0.10, 2: 0.15, 4: 0.70}
        Returns: stage_id
        """
        return np.random.choice(stages, p=stage_weights)
```

#### 1.2.2 SpacedRepetitionScheduler
**Purpose**: Leitner expanding intervals
```python
class SpacedRepetitionScheduler:
    """Calculate review intervals based on performance."""

    def calculate_review_schedule(self, stage_history, performance):
        """
        Expanding intervals:
        - Performance > 0.92: interval × 1.5
        - Performance < 0.85: interval = 10k (reset)
        - Otherwise: interval = 25k
        """
```

#### 1.2.3 TestingPhaseProtocol
**Purpose**: Retrieval without feedback (testing effect)
```python
class TestingPhaseProtocol:
    """Low-stakes testing without immediate feedback."""

    def testing_phase(self, brain, test_set, frequency=0.15):
        """15% of steps are tests (no learning signal)."""
```

#### 1.2.4 ProductiveFailurePhase
**Purpose**: Intentional struggle before teaching
```python
class ProductiveFailurePhase:
    """Allow 20% success rate before scaffolding."""
```

#### 1.2.5 CurriculumDifficultyCalibrator
**Purpose**: Maintain 75% success rate (ZPD)
```python
class CurriculumDifficultyCalibrator:
    """Adjust difficulty to maintain optimal learning."""

    def calibrate(self, success_rate, current_difficulty):
        if success_rate > 0.90: difficulty += 0.05  # Too easy
        elif success_rate < 0.60: difficulty -= 0.05  # Too hard
```

#### 1.2.6 StageTransitionProtocol
**Purpose**: Gradual ramps during stage changes
```python
class StageTransitionProtocol:
    """Smooth transitions with 4-week difficulty ramp."""

    def transition(self, old_stage, new_stage):
        # Week 1: 0.3 difficulty, 70% old stage
        # Week 2: 0.5 difficulty, 50% old stage
        # Week 3: 0.7 difficulty, 30% old stage
        # Week 4+: 1.0 difficulty, 30% old stage
```

**Tests**: `tests/unit/test_curriculum_infrastructure.py`

---

### 1.3 Enhanced Consolidation ✅ COMPLETE
**Curriculum**: All stages, every 10-200k steps
**Status**: ✅ **Implemented** (December 8, 2025)
**Actual Time**: ~4 hours (29 tests, 0.30s)
**Impact**: Memory pressure detection, NREM/REM cycling, consolidation quality metrics

**Implementation**:
- **File**: `src/thalia/memory/consolidation.py` (NEW)
- **Replaced**: Old `src/thalia/core/sleep.py` (removed)
- **Components**: MemoryPressureDetector, SleepStageController, ConsolidationMetrics, ConsolidationTrigger

**Enhancements**:
```python
class SleepSystemMixin:
    def calculate_memory_pressure(self, brain):
        """
        Synaptic weight accumulation triggers consolidation.
        Returns: pressure [0, 1]
        """
        weight_changes = []
        for region in brain.regions:
            recent_dw = region.get_recent_weight_changes()
            weight_changes.append(recent_dw.abs().mean())
        return np.mean(weight_changes)

    def should_consolidate(self, step, last_consol, performance_delta, memory_pressure):
        """Adaptive consolidation triggers."""
        # Memory pressure override
        if memory_pressure > 0.8: return True, 3000

        # Forgetting detected
        if performance_delta < -0.05: return True, 5000

        # Standard adaptive schedule
        base_interval = {0: 15000, 1: 25000, 2: 40000, 3: 60000}[stage]
        if performance_delta > 0.10:
            interval = base_interval * 0.7  # More frequent
        elif performance_delta < 0.01:
            interval = base_interval * 1.5  # Less frequent
        else:
            interval = base_interval

        return (step - last_consol) >= interval, interval
```

**Tests**: `tests/unit/test_consolidation_enhanced.py`

---

## Priority 2: 🟡 STAGE-SPECIFIC (Week 3-4, Required Before Each Stage)

### 2.1 Stage -0.5: Sensorimotor Environment (Week 0-4)

#### 2.1.1 Sensorimotor Environment ✅ COMPLETE
**Curriculum**: Stage -0.5, 100% of training
**Status**: ✅ Implemented with Gymnasium + MuJoCo
**Time**: ~4 hours (library wrapper approach)
**Tests**: 35 tests passing (1.19s)
**Commit**: 16b6c7a

**Library Choice**: Gymnasium + MuJoCo
- Modern RL environment framework (v1.0+)
- MuJoCo physics for robotics simulation
- Target: Reacher-v4 (2-joint arm reaching)

**Implementation**:
- **File**: `src/thalia/environments/sensorimotor_wrapper.py` (600+ lines)
- **Config**: `SensorimotorConfig` dataclass
- **Features**:
  - 3 spike encoding methods (rate, population, temporal)
  - Population vector decoding (Georgopoulos algorithm)
  - Sensory noise (σ=0.02) for biological realism
  - Motor smoothing (exponential α=0.3)
  - 50 neurons per DOF (configurable)

**Components**:
1. `SensorimotorWrapper`: Main wrapper class
   - `reset()`: Initialize environment → spike observation
   - `step(motor_spikes)`: Execute action → next observation
   - `_encode_observation()`: Continuous → spikes (rate/population/temporal)
   - `_decode_motor_command()`: Spikes → continuous action
2. `motor_babbling()`: Random exploration task
3. `reaching_task()`: Goal-directed evaluation

**Observation Space**: [cos(θ), sin(θ), velocities, target] → 550 sensory neurons
**Action Space**: 2 torques → 100 motor neurons

**Tests**: `tests/unit/test_sensorimotor_wrapper.py`
- Initialization (4 tests)
- Observation encoding (7 tests) - rate, population, noise
- Motor decoding (5 tests) - population vector, smoothing
- Step mechanics (5 tests) - episode tracking, termination
- Motor babbling (3 tests) - statistics, resets
- Integration (4 tests) - full episodes, multiple encoding methods
- Edge cases (4 tests) - invalid params, extreme inputs
- Performance (3 tests) - encoding/decoding/step speed

---

### 2.2 Stage 0: Phonological Tasks (Week 4-12)

#### 2.2.1 Phonological Tasks ⚠️ NEW
**Curriculum**: Stage 0, Week 6-8 (45% of time)
**Status**: Missing
**Complexity**: Small (4-8 hours)
**Impact**: Critical period phonology learning

**Implementation**:
- **File**: `src/thalia/datasets/phonology.py`
- **Datasets**: /p/ vs /b/, /d/ vs /t/, vowel categories

**Structure**:
```python
class PhonologicalDataset:
    """Categorical perception tasks for phonemes."""

    def __init__(self):
        self.contrasts = [
            ('p', 'b'),  # Voicing distinction
            ('d', 't'),  # Voicing distinction
            ('a', 'i', 'u'),  # Vowel categories
        ]

    def generate_contrast_pair(self, contrast):
        """Generate phoneme pair for discrimination."""
```

**Tests**: `tests/unit/test_phonology_dataset.py`

---

### 2.3 Stage 1: Toddler Brain (Week 12-20)

#### 2.3.1 Executive Function Tasks (Stage 1) ⚠️ NEW
**Curriculum**: Stage 1, Week 18-20
**Status**: Missing
**Complexity**: Medium (1-2 days)
**Impact**: Inhibitory control foundation

**Implementation**:
- **File**: `src/thalia/tasks/executive_function.py`
- **Tasks**: Go/no-go, delayed gratification

```python
class ExecutiveFunctionTasks:
    def go_no_go(self, stimulus, rule):
        """
        Go: Respond to target
        No-go: Inhibit response to distractor
        """

    def delayed_gratification(self, reward_immediate, reward_delayed, delay_steps):
        """Choose smaller now vs larger later."""
```

**Tests**: `tests/unit/test_executive_function.py`

---

#### 2.3.2 Social Learning Module ⚠️ NEW
**Curriculum**: Stage 1, Week 10-11.5
**Status**: Missing
**Complexity**: Medium (2-3 days)
**Impact**: Fast learning from demonstration (2x LR)

**Implementation**:
- **File**: `src/thalia/learning/social_learning.py`
- **Class**: `SocialLearningModule`

```python
class SocialLearningModule:
    """Fast learning from social cues."""

    def imitation_learning(self, demonstration, base_lr):
        """2x learning rate from demonstration."""
        return base_lr * 2.0

    def pedagogy_boost(self, ostensive_cues, base_lr):
        """1.5x learning rate when teaching detected."""
        if self.detect_teaching_signal(ostensive_cues):
            return base_lr * 1.5
        return base_lr

    def joint_attention(self, gaze_direction, attention_weights):
        """Weight attention by gaze cues."""
        return attention_weights * self.gaze_modulation(gaze_direction)
```

**Tests**: `tests/unit/test_social_learning.py`

---

#### 2.3.3 Attention Mechanisms (Enhanced) ⚠️ ENHANCE EXISTING
**Curriculum**: Stage 1, continuous
**Status**: Partially implemented (top-down exists)
**Complexity**: Small (4-8 hours)
**Impact**: Bottom-up salience + developmental progression

**Implementation**:
- **File**: `src/thalia/integration/pathways/attention.py` (enhance)
- **Add**: Bottom-up salience, developmental weighting

```python
class AttentionMechanisms:
    """Two-pathway attention with developmental progression."""

    def bottom_up_salience(self, visual_input):
        """Stimulus-driven attention."""
        salience = (
            0.4 * brightness_contrast(visual_input) +
            0.4 * motion_saliency(visual_input) +
            0.2 * novelty_detector(visual_input)
        )
        return salience

    def top_down_modulation(self, visual_input, goal):
        """Goal-directed attention (existing SpikingAttentionPathway)."""
        return self.attention_pathway(visual_input, goal)

    def combined_attention(self, visual_input, goal, stage):
        """Developmental weighting."""
        bottom_up = self.bottom_up_salience(visual_input)
        top_down = self.top_down_modulation(visual_input, goal)

        # Stage-dependent weighting
        weights = {
            1: (0.7, 0.3),  # Stage 1: 70% bottom-up
            2: (0.5, 0.5),  # Stage 2: balanced
            3: (0.3, 0.7),  # Stage 3+: 70% top-down
        }
        w_bu, w_td = weights.get(stage, (0.3, 0.7))

        return w_bu * bottom_up + w_td * top_down
```

**Tests**: `tests/unit/test_attention_mechanisms.py`

---

#### 2.3.4 Metacognitive Monitor (Stage-Aware) ⚠️ NEW
**Curriculum**: Stage 1 (binary) → Stage 4 (calibrated)
**Status**: Partially implemented (ConfidenceEstimator exists)
**Complexity**: Medium (2-3 days)
**Impact**: Uncertainty estimation, abstention, active learning

**Implementation**:
- **File**: `src/thalia/diagnostics/metacognition.py`
- **Class**: `MetacognitiveMonitor`

```python
class MetacognitiveMonitor:
    """Stage-aware confidence estimation."""

    def __init__(self, confidence_estimator):
        self.estimator = confidence_estimator  # Existing component
        self.stage = 1

    def estimate_confidence(self, population_activity):
        """Stage-specific confidence levels."""
        raw_confidence = self.estimator.estimate(population_activity)

        if self.stage == 1:
            # Binary: know vs don't know
            return 1.0 if raw_confidence > 0.7 else 0.0

        elif self.stage == 2:
            # Coarse: high/medium/low
            if raw_confidence > 0.8: return 1.0
            elif raw_confidence > 0.5: return 0.5
            else: return 0.0

        elif self.stage == 3:
            # Continuous (poorly calibrated)
            return raw_confidence

        elif self.stage >= 4:
            # Calibrated (with training)
            return self.calibrated_confidence(raw_confidence)

    def should_abstain(self, confidence):
        """Decide whether to abstain from answering."""
        threshold = {1: 0.5, 2: 0.3, 3: 0.4, 4: 0.3}[self.stage]
        return confidence < threshold

    def calibrate(self, predicted_conf, actual_correct, dopamine):
        """Train calibration network (Stage 3-4)."""
        error = abs(predicted_conf - actual_correct)
        self.estimator.learn(error, dopamine)
```

**Tests**: `tests/unit/test_metacognitive_monitor.py`

---

#### 2.3.5 Theta-Gamma Working Memory ⚠️ NEW
**Curriculum**: Stage 1, Week 9-11
**Status**: Components exist (oscillators, prefrontal)
**Complexity**: Small (4-8 hours)
**Impact**: N-back tasks with phase coding

**Implementation**:
- **File**: `src/thalia/tasks/working_memory.py`
- **Function**: `theta_gamma_n_back()`

```python
def theta_gamma_n_back(prefrontal, stimulus_sequence, n=2):
    """
    N-back task using theta phase coding.
    Each item encoded at different theta phase within gamma cycle.
    """
    results = []

    for t, stimulus in enumerate(stimulus_sequence):
        # Theta phase: position within cycle (0-1)
        theta_phase = (t % 8) / 8.0  # 8 items per theta cycle
        gamma_phase = 0.5  # Peak excitability

        # Encode with phase information
        prefrontal.maintain(
            stimulus,
            theta_phase=theta_phase,
            gamma_phase=gamma_phase
        )

        # Retrieve item from n cycles ago
        target_phase = ((t - n) % 8) / 8.0
        retrieved = prefrontal.retrieve(theta_phase=target_phase)

        # Compare current to n-back
        is_match = (stimulus == retrieved)
        results.append(is_match)

    return results
```

**Tests**: `tests/unit/test_working_memory_tasks.py`

---

### 2.4 Stage 2: Grammar & Composition (Week 20-28)

#### 2.4.1 Executive Function Tasks (Stage 2) ⚠️ ADD TO EXISTING
**Curriculum**: Stage 2, Week 26-28
**Status**: Missing
**Complexity**: Medium (1-2 days)
**Impact**: Set shifting, cognitive flexibility

**Implementation**:
- **File**: `src/thalia/tasks/executive_function.py` (add to)
- **Tasks**: DCCS, task switching

```python
class ExecutiveFunctionTasks:
    # ... (Stage 1 tasks above)

    def dimensional_change_card_sort(self, card, current_rule):
        """
        DCCS: Sort by color, then switch to shape.
        Requires inhibiting old rule.
        """
        if current_rule == 'color':
            return self.sort_by_color(card)
        elif current_rule == 'shape':
            return self.sort_by_shape(card)  # Must inhibit color

    def task_switching(self, stimulus, task_cue):
        """Alternate between two tasks based on cue."""
        # Switch cost: slower/less accurate on switch trials
```

**Tests**: Update `tests/unit/test_executive_function.py`

---

#### 2.4.2 Cross-Modal Gamma Binding ⚠️ NEW
**Curriculum**: Stage 2, Week 24-26
**Status**: Missing
**Complexity**: Medium (1-2 days)
**Impact**: Synchronize visual + auditory via gamma

**Implementation**:
- **File**: `src/thalia/integration/pathways/crossmodal_binding.py`
- **Function**: `cross_modal_gamma_binding()`

```python
class CrossModalGammaBinding:
    """Force gamma synchrony across modalities."""

    def __init__(self, gamma_oscillator):
        self.gamma = gamma_oscillator

    def bind_modalities(self, visual_spikes, auditory_spikes):
        """
        Force both to spike at same gamma phase.
        Biology: Object + sound bound by gamma coherence.
        """
        # Current gamma phase
        gamma_phase = self.gamma.gamma_phase

        # Gate each modality by gamma phase
        gamma_window = self.compute_gamma_gate(gamma_phase)

        visual_gated = visual_spikes * gamma_window
        auditory_gated = auditory_spikes * gamma_window

        return visual_gated, auditory_gated

    def compute_gamma_gate(self, gamma_phase, width=0.3):
        """Gaussian gate around current phase."""
        return torch.exp(-4 * (gamma_phase - 0.5) ** 2)
```

**Tests**: `tests/unit/test_crossmodal_binding.py`

---

### 2.5 Stage 3-4: Reading & Abstract Reasoning (Week 28-70)

#### 2.5.1 Executive Function Tasks (Stage 3-4) ⚠️ ADD TO EXISTING
**Curriculum**: Stage 3 (Tower of Hanoi), Stage 4 (Raven's)
**Status**: Missing
**Complexity**: Large (1 week)
**Impact**: Planning, fluid reasoning

**Implementation**:
- **File**: `src/thalia/tasks/executive_function.py` (add to)
- **Tasks**: Tower of Hanoi, subgoaling, Raven's matrices, analogical reasoning

```python
class ExecutiveFunctionTasks:
    # ... (Stage 1-2 tasks above)

    def tower_of_hanoi(self, n_disks, start, target, auxiliary):
        """
        Planning task requiring subgoaling.
        Must decompose into subproblems.
        """
        # Track planning depth, subgoal creation

    def ravens_matrices(self, pattern_matrix):
        """
        Abstract rule induction from visual patterns.
        Stage 4 fluid reasoning.
        """
        # Extract abstract rules, test hypotheses

    def analogical_reasoning(self, source_domain, target_domain):
        """Map structure across domains."""
```

**Tests**: Update `tests/unit/test_executive_function.py`

---

## Priority 3: 🟢 ENHANCEMENTS (Nice to Have, Not Blocking)

### 3.1 Growth Integration with Curriculum
**Status**: GrowthManager exists, needs curriculum integration
**Complexity**: Small (configuration)
**Impact**: Auto-growth at 80% capacity

**Implementation**:
- **File**: `src/thalia/config/curriculum_config.py`
- **Add**: Stage-specific growth triggers

```python
class CurriculumConfig:
    growth_triggers = {
        0: {'threshold': 0.80, 'expansion': 0.15},  # 15% growth
        1: {'threshold': 0.80, 'expansion': 0.50},  # 50% growth
        2: {'threshold': 0.85, 'expansion': 0.35},  # 35% growth
        # ...
    }
```

---

### 3.2 Advanced Consolidation Features
**Status**: Basic replay exists
**Complexity**: Medium
**Impact**: Schema extraction, prototypical averaging

**Implementation**:
- **File**: `src/thalia/core/sleep.py` (enhance)
- **Add**: Cluster-based REM consolidation

```python
def rem_schema_extraction(self, replay_buffer, n_steps):
    """Extract schemas from similar episodes."""
    for step in range(n_steps // 2):
        # Sample similar episodes
        cluster = replay_buffer.sample_cluster(k=5, similarity=0.7)

        # Create prototypical average
        prototypical = torch.stack([ep['input'] for ep in cluster]).mean(dim=0)

        # Replay with noise (generalization)
        noisy = prototypical + torch.randn_like(prototypical) * 0.3
        brain.forward(noisy)
```

---

## Implementation Schedule

### Week 1: Critical Infrastructure ✅ COMPLETE
**Goal**: Core curriculum mechanics operational
**Actual Completion**: December 8, 2025 (Sunday evening)
**Total Tests**: 91 tests passing in ~1.1 seconds

**Completed Components**:
1. ✅ **Critical Period Gating** (26 tests)
   - CriticalPeriodWindow, CriticalPeriodConfig, CriticalPeriodGating
   - 5 default domains with 3-phase modulation
   - File: `src/thalia/learning/critical_periods.py`

2. ✅ **Curriculum Infrastructure** (36 tests)
   - InterleavedCurriculumSampler (interleaved practice)
   - SpacedRepetitionScheduler (Leitner algorithm)
   - TestingPhaseProtocol (retrieval practice)
   - ProductiveFailurePhase (intentional difficulty)
   - CurriculumDifficultyCalibrator (ZPD maintenance)
   - StageTransitionProtocol (gradual ramps)
   - File: `src/thalia/training/curriculum.py`

3. ✅ **Enhanced Consolidation** (29 tests)
   - MemoryPressureDetector (multi-factor triggers)
   - SleepStageController (NREM/REM alternation)
   - ConsolidationMetrics (quality tracking)
   - ConsolidationTrigger (high-level orchestration)
   - File: `src/thalia/memory/consolidation.py`
   - Replaced: Old `sleep.py` removed, `run_consolidation()` removed from Brain

**Git Commits**:
- a39ccea: Implement Critical Period Gating (Priority 1.1)
- 9746951: Implement Curriculum Training Infrastructure (Priority 1.2)
- aa87ddf: Implement Enhanced Consolidation (Priority 1.3)

---

### Week 2: Stage -0.5 Preparation
**Goal**: Sensorimotor foundation ready

**Monday-Wednesday**:
- ✅ Sensorimotor environment (library wrapper OR custom, 3 days)

**Thursday-Friday**:
- ✅ Phonological tasks dataset (8 hours)
- ✅ Tests for Stage -0.5 and Stage 0 components

---

### Week 3: Stage 1 Components
**Goal**: Toddler brain mechanisms operational

**Monday-Tuesday**:
- ✅ Social Learning Module (2 days)

**Wednesday**:
- ✅ Attention Mechanisms (bottom-up enhancement) (8 hours)

**Thursday-Friday**:
- ✅ Metacognitive Monitor (2 days)

---

### Week 4: Remaining Stage 1 + Stage 2
**Goal**: Complete all pre-training requirements

**Monday**:
- ✅ Executive Function Tasks (Stage 1) (8 hours)
- ✅ Theta-Gamma Working Memory (4 hours)

**Tuesday-Wednesday**:
- ✅ Executive Function Tasks (Stage 2) (1 day)
- ✅ Cross-Modal Gamma Binding (1 day)

**Thursday-Friday**:
- ✅ Executive Function Tasks (Stage 3-4) (2 days)
- ✅ Final integration tests
- ✅ Documentation updates

---

## Testing Strategy

### Unit Tests (Per Component)
Each component gets dedicated test file:
- `tests/unit/test_critical_periods.py`
- `tests/unit/test_curriculum_infrastructure.py`
- `tests/unit/test_consolidation_enhanced.py`
- `tests/unit/test_sensorimotor_env.py`
- `tests/unit/test_phonology_dataset.py`
- `tests/unit/test_executive_function.py`
- `tests/unit/test_social_learning.py`
- `tests/unit/test_attention_mechanisms.py`
- `tests/unit/test_metacognitive_monitor.py`
- `tests/unit/test_working_memory_tasks.py`
- `tests/unit/test_crossmodal_binding.py`

### Integration Tests
- `tests/integration/test_curriculum_pipeline.py`: Full curriculum flow
- `tests/integration/test_stage_transitions.py`: Stage-by-stage progression
- `tests/integration/test_sensorimotor_learning.py`: Stage -0.5 complete

### Validation Tests
Before training begins:
- ✅ All critical components operational
- ✅ Stage -0.5 environment functional
- ✅ Curriculum sampling/scheduling correct
- ✅ Consolidation triggers firing appropriately
- ✅ No regressions in existing functionality

---

## Success Criteria

### Before Stage -0.5 Training:
1. ✅ Critical Period Gating operational
2. ✅ Curriculum infrastructure complete (6 components)
3. ✅ Enhanced consolidation with memory pressure
4. ✅ Sensorimotor environment ready
5. ✅ All unit tests passing
6. ✅ Integration tests passing

### Before Each Stage:
- ✅ Stage-specific components implemented
- ✅ Tasks/datasets ready
- ✅ Tests passing
- ✅ Documentation updated

---

## Risk Mitigation

### Risk 1: Sensorimotor Environment Too Complex
**Mitigation**: Use existing library (Gym/PyBullet) instead of custom build
**Fallback**: Simplified 2D environment for initial testing

### Risk 2: Curriculum Infrastructure Bugs
**Mitigation**: Extensive unit tests, small-scale integration tests first
**Fallback**: Manual curriculum control for initial stages

### Risk 3: Timeline Slippage
**Mitigation**: Prioritize critical components, defer enhancements
**Fallback**: Start with Stage 0 (skip Stage -0.5) if sensorimotor delayed

---

## Documentation Updates

After implementation:
1. Update `docs/patterns/component-parity.md` (if pathways changed)
2. Update `docs/design/curriculum_strategy.md` (mark implemented components)
3. Create `docs/guides/curriculum_training.md` (usage guide)
4. Update `README.md` (new capabilities)

---

## Post-Implementation: Training Readiness Checklist

Before starting Stage -0.5 training:

**Infrastructure**:
- [ ] Critical Period Gating integrated with LocalTrainer
- [ ] Curriculum samplers/schedulers operational
- [ ] Enhanced consolidation with adaptive triggers
- [ ] Sensorimotor environment functional
- [ ] All tests passing (unit + integration)

**Stage -0.5 Requirements**:
- [ ] Motor control tasks ready
- [ ] Reaching/manipulation tasks ready
- [ ] Cerebellum forward/inverse models training correctly
- [ ] Proprioception integration working
- [ ] Success criteria defined and measurable

---

## Appendix: File Structure

```
src/thalia/
├── learning/
│   ├── critical_periods.py          # NEW
│   └── social_learning.py           # NEW
├── training/
│   └── curriculum.py                 # NEW (6 classes)
├── core/
│   └── sleep.py                      # ENHANCE
├── environments/
│   └── sensorimotor_wrapper.py      # NEW
├── datasets/
│   └── phonology.py                  # NEW
├── tasks/
│   ├── executive_function.py        # NEW
│   └── working_memory.py            # NEW
├── integration/pathways/
│   ├── attention.py                  # ENHANCE
│   └── crossmodal_binding.py        # NEW
└── diagnostics/
    └── metacognition.py              # NEW

tests/
├── unit/
│   ├── test_critical_periods.py     # NEW
│   ├── test_curriculum_infrastructure.py  # NEW
│   ├── test_consolidation_enhanced.py     # NEW
│   ├── test_sensorimotor_env.py     # NEW
│   ├── test_phonology_dataset.py    # NEW
│   ├── test_executive_function.py   # NEW
│   ├── test_social_learning.py      # NEW
│   ├── test_attention_mechanisms.py # NEW
│   ├── test_metacognitive_monitor.py # NEW
│   ├── test_working_memory_tasks.py # NEW
│   └── test_crossmodal_binding.py   # NEW
└── integration/
    ├── test_curriculum_pipeline.py   # NEW
    ├── test_stage_transitions.py     # NEW
    └── test_sensorimotor_learning.py # NEW
```

---

## Next Actions

1. **Commit this plan** to repository
2. **Start Week 1 implementation**: Critical Period Gating
3. **Daily progress updates** to this document
4. **Weekly reviews** with stakeholders

**Status**: Ready to begin implementation
**Expected Completion**: December 22-29, 2025 (2-3 weeks)
**Training Start**: January 2026
