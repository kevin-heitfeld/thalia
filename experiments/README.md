# Experiments Directory

This directory contains experiments for validating the Thalia brain regions library.

## Structure

- `scripts/regions/` - Brain region experiments organized by phase
  - `phase1/` - Single region validation (✅ complete)
  - `phase2/` - Two-region compositions
  - `phase3/` - Multi-region systems
  - `phase4/` - Emergent cognition
- `results/regions/` - Generated figures and JSON results
- `notebooks/` - Jupyter notebooks for exploration

## Current Status

**Phase 1: Single Region Validation** ✅ COMPLETE

All five brain regions validated independently:

| Experiment | Region | Task | Result |
|------------|--------|------|--------|
| exp1_cortex_predictive | Cortex | Predictive coding on MNIST | ✅ PASSED |
| exp2_cerebellum_timing | Cerebellum | Interval timing | ✅ PASSED |
| exp3_striatum_bandit | Striatum | Multi-armed bandit RL | ✅ PASSED |
| exp4_hippocampus_memory | Hippocampus | Hetero-associative memory | ✅ PASSED |
| exp5_prefrontal_delay | Prefrontal | Delayed match-to-sample | ✅ PASSED |

## Running Experiments

```bash
# Set PYTHONPATH and run individual experiments
set PYTHONPATH=.
python experiments/scripts/regions/phase1/exp1_cortex_predictive.py

# Or use cmd /c for single command
cmd /c "set PYTHONPATH=. && python experiments/scripts/regions/phase1/exp1_cortex_predictive.py"
```

## Brain Regions

| Region | Learning Rule | Biological Function |
|--------|---------------|---------------------|
| **Cortex** | Predictive Coding | Hierarchical feature learning |
| **Cerebellum** | Delta rule (error-corrective) | Precise timing and motor learning |
| **Striatum** | Three-factor RL | Reward-based action selection |
| **Hippocampus** | One-shot Hebbian | Episodic memory, pattern completion |
| **Prefrontal** | Gated working memory | Cognitive control, context maintenance |

See `EXPERIMENT_PLAN.md` for detailed specifications and Phase 2-4 plans.
