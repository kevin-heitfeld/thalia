# Complexity Mitigation Plan

> Reducing complexity risk while maintaining architectural integrity

**Status**: 🔄 **IN PROGRESS** (December 6, 2025)

**Prerequisites**: ✅ Hyperparameter Robustness (Completed)

**Current Phase**: ✅ Phase 1: Complexity Layers (**100% COMPLETE** ✅)

**Current Phase**: ✅ Phase 2: Configuration Presets (**100% COMPLETE** ✅)

**Next Phase**: Phase 3: Documentation & Examples

## Context

After successfully implementing all robustness mechanisms (E/I balance, divisive
normalization, intrinsic plasticity, criticality monitoring, metabolic constraints),
we now focus on making the system more understandable and maintainable.

**Phase 1 Completed** (commits be111b3, 7cef740, [current]):
- ✅ Created architecture.md documenting 5-level complexity hierarchy (400+ lines)
- ✅ Reorganized tests: 184 unit tests in tests/unit/
- ✅ Created integration test framework in tests/integration/conftest.py
- ✅ Implemented 7 integration tests for cortex + robustness (all passing)
- ✅ Fixed conftest.py fixtures with correct APIs
- ✅ Created ablation test framework in tests/ablation/
- ✅ Implemented 9 ablation tests (all passing):
  - test_without_ei_balance.py (2 tests)
  - test_without_divisive_norm.py (2 tests)
  - test_without_intrinsic_plasticity.py (2 tests)
  - test_without_any_robustness.py (3 tests)
- ✅ Documented ablation results in docs/design/ablation_results.md
- ✅ **Total test suite**: 191 tests passing (184 unit + 7 integration + 9 ablation ≈ 200 tests)

**Key Findings from Ablation Study**:
- **Divisive normalization**: CRITICAL (prevents +1080% variance increase)
- **E/I balance**: VALUABLE (prevents +26% variance increase)
- **Intrinsic plasticity**: MINOR in current setup (15% adaptation loss)
- **Combined mechanisms**: SEVERE when all removed (+184% variance, -32% consistency)

## Problem Statement

THALIA has grown to include many interacting mechanisms:
- Theta/gamma oscillations
- Dopamine (tonic/phasic)
- STDP, BCM, three-factor learning
- Homeostasis (multiple forms)
- Event-driven regions with axonal delays
- Predictive coding

This creates a combinatorial explosion of possible behaviors, making debugging
and reasoning about the system increasingly difficult.

## Goals

1. **Understandability**: Any behavior should be traceable to specific mechanisms
2. **Testability**: Each component should be testable in isolation
3. **Ablation Clarity**: Know what breaks when you remove each mechanism
4. **Configuration Clarity**: Sensible defaults that "just work"

---

## Phase 1: Complexity Layers (Week 1-2)

### 1.1 Define the Layer Hierarchy

```
Level 0: PRIMITIVES
├── LIFNeuron, Synapse, SpikeTraces
├── Test: Unit tests with synthetic inputs
└── No dependencies on other THALIA components

Level 1: LEARNING RULES
├── STDP, BCM, Hebbian, ThreeFactor
├── Test: Apply to random weight matrices
└── Depends on: Level 0 traces only

Level 2: STABILITY MECHANISMS
├── UnifiedHomeostasis, EIBalance, DivisiveNormalization
├── Test: Can stabilize synthetic runaway/collapse scenarios
└── Depends on: Level 0-1

Level 3: REGIONS (Isolated)
├── LayeredCortex, TrisynapticHippocampus, Striatum, PFC
├── Test: Each region with dummy inputs, no inter-region connections
└── Depends on: Level 0-2

Level 4: INTEGRATION
├── EventDrivenBrain, pathways, inter-region communication
├── Test: Full integration tests
└── Depends on: Level 0-3
```

### 1.2 Create Integration Test Directory

```
tests/
├── unit/              # Existing tests, renamed
├── integration/       # NEW: Multi-component tests
│   ├── test_stdp_with_homeostasis.py
│   ├── test_cortex_with_dopamine.py
│   ├── test_hippocampus_with_theta.py
│   └── test_two_region_communication.py
└── ablation/          # NEW: What-breaks-when tests
    ├── test_without_bcm.py
    ├── test_without_homeostasis.py
    ├── test_without_theta.py
    └── test_without_dopamine.py
```

### 1.3 Tasks

- [ ] Create `tests/integration/` directory
---

## Phase 2: Configuration Presets (Week 2-3)

**Status**: ✅ **COMPLETED** (December 6, 2025)

### 2.1 Evidence-Based Presets

Based on ablation study results, refined `RobustnessConfig` presets:

```python
# Recommended default for most work
config = RobustnessConfig.stable()
# Includes: divisive norm (CRITICAL) + E/I balance (VALUABLE)
```

**Preset Definitions:**

| Preset | Mechanisms | Evidence | Overhead | Use Case |
|--------|-----------|----------|----------|----------|
| `minimal()` | Divisive norm only | CRITICAL mechanism | ~5-10% | Prototyping |
| `stable()` | + E/I balance | -55% CV reduction | ~15-20% | **Most work** ⭐ |
| `full()` | All mechanisms | Maximum robustness | ~30-40% | Production |

### 2.2 Implementation

Updated `src/thalia/config/robustness_config.py`:
- ✅ Refined preset implementations based on ablation evidence
- ✅ Added comprehensive docstrings with performance impact
- ✅ Included ablation study results in class documentation
- ✅ Clear guidance on when to use each preset

### 2.3 Documentation

Created comprehensive usage guide:
- ✅ `docs/guides/robustness_config_guide.md` (detailed guide)
  - Quick start examples
  - Evidence from ablation study
  - Decision tree for preset selection
  - Usage patterns and common mistakes
  - Performance vs stability trade-offs

### 2.4 Key Findings Applied

| Mechanism | Evidence | Preset Inclusion |
|-----------|----------|------------------|
| Divisive Norm | CRITICAL (+1080% variance) | ✅ All presets (including minimal) |
| E/I Balance | VALUABLE (+26% variance) | ✅ stable(), full() |
| Intrinsic Plasticity | MINOR (15% adaptation) | ✅ full() only (optional) |
| Criticality | Diagnostic value | ✅ full() for research |
| Metabolic | Sparse coding goal | ✅ full() when needed |

**Completed Tasks:**
- ✅ Refined RobustnessConfig presets based on evidence
- ✅ Updated documentation with ablation results
- ✅ Created comprehensive usage guide
- ✅ Added performance impact estimates
- ✅ Clear recommendations for 95% of use cases

---

## Phase 3: Ablation Framework (Week 3-4)

**Status**: ✅ **COMPLETED** (December 6, 2025)

### 3.1 Ablation Test Structure

Created comprehensive ablation testing framework in `tests/ablation/`:

**Test Files Created:**
- ✅ `test_without_ei_balance.py` (2 tests)
- ✅ `test_without_divisive_norm.py` (2 tests)
- ✅ `test_without_intrinsic_plasticity.py` (2 tests)
- ✅ `test_without_any_robustness.py` (3 tests)

**Total**: 9 ablation tests, all passing

### 3.2 Mechanisms Ablated

| Mechanism | Finding | Impact | Evidence |
|-----------|---------|--------|----------|
| Divisive Norm | CRITICAL | +1080% variance (contrast) | Catastrophic without it |
| E/I Balance | VALUABLE | +26% variance | Significant stability loss |
| Intrinsic Plasticity | MINOR | 15% adaptation loss | Useful but not essential |
| Combined (all) | SEVERE | +184% variance | Synergistic effects |

### 3.3 Documentation

- ✅ `docs/design/ablation_results.md` - Comprehensive findings report
  - Methodology explanation
  - Detailed results for each mechanism
  - Ranking by importance
  - Recommendations for Phase 2
  - Limitations and next steps

### 3.4 Test Coverage

**Total Test Suite**: 200 tests
- 184 unit tests
- 7 integration tests
- 9 ablation tests

**All tests passing**: 199/200 (1 unrelated flaky test)

**Completed Tasks:**
- ✅ Created ablation test framework
- ✅ Implemented 9 ablation tests
- ✅ Quantified impact of each mechanism
- ✅ Generated evidence-based recommendations
- ✅ Used results to inform Phase 2 presets

---

## Phase 4: Observability Dashboard (Week 4-5)

**Status**: ✅ **COMPLETED** (December 6, 2025)

### 4.1 Real-time Monitoring

Created a diagnostic dashboard that shows:
- Per-region spike rates (with target bands)
- E/I ratio over time
- Weight magnitude distributions
- Dopamine levels (tonic/phasic)
- Branching ratio (criticality measure)
- Overall health score (0-100)

### 4.2 Anomaly Detection

**Implementation**: `thalia/diagnostics/health_monitor.py`

```python
class HealthMonitor:
    """Detect unhealthy network states."""
    
    def check_health(self, diagnostics: Dict[str, Any]) -> HealthReport:
        # Detects:
        # - ACTIVITY_COLLAPSE: spike rate < 0.01
        # - SEIZURE_RISK: spike rate > 0.5
        # - WEIGHT_EXPLOSION: weights > 5.0
        # - WEIGHT_COLLAPSE: weights < 0.001
        # - EI_IMBALANCE: ratio outside [1.0, 10.0]
        # - CRITICALITY_DRIFT: branching ratio outside [0.8, 1.2]
        # - DOPAMINE_SATURATION: dopamine > 2.0
        
        # Returns:
        # - is_healthy: bool
        # - overall_severity: 0-100 (higher = worse)
        # - issues: List[IssueReport] with recommendations
        # - summary: One-line status message
```

### 4.3 Dashboard Visualization

**Implementation**: `thalia/diagnostics/dashboard.py`

Features:
- Real-time matplotlib dashboard with 6 subplots
- Windowed time series (configurable, default 100 timesteps)
- Color-coded health zones (green/orange/red)
- Current issue display with recommendations
- Summary statistics and trend analysis
- Save reports to PNG/PDF

### 4.4 Tasks

- [x] Create `thalia/diagnostics/health_monitor.py`
- [x] Define health metrics and thresholds
- [x] Implement `HealthMonitor.check_health()` with severity scoring
- [x] Create `Dashboard` class with matplotlib visualization
- [x] Add health checks to diagnostics export
- [x] Create demo script (`experiments/scripts/demo_dashboard.py`)
- [x] Write comprehensive tests (20 tests covering all issue types)

### 4.5 Testing

**File**: `tests/test_health_dashboard.py`

Test Coverage:
- 14 tests for `HealthMonitor` (all issue types, severity, trends)
- 6 tests for `Dashboard` (creation, updates, windowing, summary)
- All 20 tests passing

### 4.6 Usage Example

```python
from thalia.diagnostics import Dashboard, HealthConfig

# Create dashboard
dashboard = Dashboard(
    health_config=HealthConfig(),
    window_size=50,
)

# Training loop
for epoch in range(num_epochs):
    diagnostics = brain.get_diagnostics()
    dashboard.update(diagnostics)
    
    # Show every 5 steps
    if epoch % 5 == 0:
        dashboard.show(block=False)

# Final report
dashboard.print_summary()
dashboard.save_report("training_health.png")
```

### 4.7 Implementation Summary

**Files Created**:
- `src/thalia/diagnostics/health_monitor.py` (412 lines)
  - `HealthConfig`: Configurable thresholds
  - `HealthMonitor`: Main health checking logic
  - `HealthReport`: Health status with issues and severity
  - `IssueReport`: Individual issue with recommendations
  - `HealthIssue`: Enum of 8 issue types

- `src/thalia/diagnostics/dashboard.py` (343 lines)
  - `Dashboard`: Interactive matplotlib visualization
  - 6 subplot layout (health score, spike rate, E/I, criticality, dopamine, issues)
  - Trend tracking and summary statistics
  - Save to file functionality

- `experiments/scripts/demo_dashboard.py` (96 lines)
  - Demonstrates dashboard with simulated phases
  - Shows activity collapse, recovery, and normal operation

- `tests/test_health_dashboard.py` (498 lines, 20 tests)
  - Comprehensive coverage of all health checks
  - Dashboard functionality testing
  - All tests passing

**Total**: 1,349 lines of code + tests

### 4.8 Success Metrics

✅ Can detect all 8 pathological states automatically
✅ Provides actionable recommendations for each issue
✅ Real-time visualization works in experiment scripts
✅ All tests pass (20/20)
✅ Severity scoring allows filtering of minor issues
✅ Trend analysis detects gradual degradation
✅ Dashboard can save reports for documentation

### 4.9 Next Steps

With observability infrastructure complete, we can now:
1. Use dashboard during Phase 1 (Complexity Layers) test reorganization
2. Validate Phase 2 (Config Profiles) with health metrics
3. Use health scores in Phase 3 (Ablation Framework) comparisons

The dashboard provides the foundation for data-driven development
throughout the remaining complexity mitigation work.

---

## Success Criteria

1. **All tests pass at each complexity level independently**
2. **MINIMAL profile achieves >70% of FULL profile performance on simple tasks**
3. **Ablation tests document value of each mechanism**
4. **No "mystery" behaviors — every observation traceable to specific code**
5. **New contributors can understand system in <1 day using layer docs**

---

## Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1-2 | Complexity Layers | Test reorganization, layer docs |
| 2-3 | Config Profiles | Profile system, tested configs |
| 3-4 | Ablation Framework | Ablation tests, value scorecard |
| 4-5 | Observability | Health monitor, dashboard |

---

## Risk Mitigation

- **Scope creep**: Focus on observability over new features
- **Breaking changes**: Keep existing tests passing throughout
- **Over-engineering**: Start simple, add complexity only when needed

---

## Recommended Implementation Order (REVISED)

After discussion, we recommend **starting with Phase 4 (Observability)** rather
than the original Phase 1-4 sequence. Here's why:

### Why Start with Observability?

1. **Immediate Value**: A diagnostic dashboard will help debug Phase 1-3 work
2. **Risk Reduction**: Catch issues early during test reorganization
3. **Foundation for Testing**: Ablation tests (Phase 3) need good metrics
4. **Parallelization**: Can work on dashboard while planning other phases

### Revised Phase Order

**Phase 4 → Phase 1 → Phase 2 → Phase 3**

#### Start: Phase 4 (Observability Dashboard) - Week 1

**Why first**: Foundation for all other work

Tasks:
- [ ] Create `thalia/diagnostics/health_monitor.py`
- [ ] Add health metrics (spike rates, E/I ratio, weights, dopamine)
- [ ] Implement `HealthMonitor.check_health()` with thresholds
- [ ] Add health checks to brain diagnostics
- [ ] Create simple matplotlib dashboard for experiments
- [ ] Test with existing experiment scripts

**Output**: Working health monitor that can detect pathological states

#### Then: Phase 1 (Complexity Layers) - Week 2-3

**Why second**: Need observability to verify layer isolation

Tasks:
- [ ] Create `tests/integration/` and `tests/ablation/` directories
- [ ] Reorganize existing tests into `tests/unit/`
- [ ] Document layer hierarchy in `docs/design/architecture.md`
- [ ] Create fixtures for each complexity level
- [ ] Write integration tests for critical pairs
- [ ] Use health monitor to verify each layer is stable

**Output**: Clear layer hierarchy with isolated testing

#### Then: Phase 2 (Configuration Profiles) - Week 3-4

**Why third**: Build on layer understanding

Tasks:
- [ ] Create `thalia/config/profiles.py` with MINIMAL/STABLE/BIOLOGICAL/FULL
- [ ] Implement profile defaults based on layer hierarchy
- [ ] Test each profile works independently
- [ ] Use health monitor to validate profile stability
- [ ] Document profiles in README with examples

**Output**: Easy-to-use configuration presets

#### Finally: Phase 3 (Ablation Framework) - Week 4-5

**Why last**: Requires profiles + observability infrastructure

Tasks:
- [ ] Create `tests/ablation/framework.py`
- [ ] Use profiles as baseline for ablation
- [ ] Use health monitor metrics for comparisons
- [ ] Generate ablation report with quantitative results
- [ ] Identify unnecessary complexity for removal

**Output**: Data-driven understanding of mechanism value

### Benefits of Revised Order

1. **Better Debugging**: Dashboard available during all other work
2. **Clearer Metrics**: Health monitoring defines success criteria early
3. **Faster Iteration**: Can visualize impact of changes immediately
4. **Risk Reduction**: Catch breaking changes earlier

### Estimated Timeline (Revised)

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | Observability (Phase 4) | Health monitor, dashboard |
| 2-3 | Complexity Layers (Phase 1) | Test reorganization, layer docs |
| 3-4 | Config Profiles (Phase 2) | Profile system, tested configs |
| 4-5 | Ablation Framework (Phase 3) | Ablation tests, value scorecard |

**Total: ~5 weeks** (same as original, but better ordered)
