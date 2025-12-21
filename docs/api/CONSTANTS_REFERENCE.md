# Constants Reference

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2025-12-21 19:26:47
> Generated from: `scripts/generate_api_docs.py`

This document catalogs all module-level constants in Thalia. These include biological time constants, default values, and thresholds.

Total: 59 constants

## Constants by Category

### Dopamine (DA) - Reward Prediction Error and Reinforcement

| Constant | Value | Description |
|----------|-------|-------------|
| `DA_ACH_SUPPRESSION` | `0.3` | Dopamine (DA) - Reward Prediction Error and Reinforcement |
| `NE_ACH_ENHANCEMENT` | `0.2` | ============================================================================= |
| `NE_GAIN_MAX` | `1.5` | Phasic dopamine decay (reuptake by DAT transporters) |
| `NE_GAIN_MIN` | `1.0` | No description |
| `TARGET_NEUROMODULATOR_LEVEL` | `0.5` | ============================================================================= |

**Source**: `thalia\neuromodulation\constants.py`

---

### General

| Constant | Value | Description |
|----------|-------|-------------|
| `ACH_BASELINE` | `0.3` | No description |
| `ACH_DECAY_PER_MS` | `0.98` | No description |
| `ACH_ENCODING_LEVEL` | `0.8` | No description |
| `ACH_RETRIEVAL_LEVEL` | `0.2` | No description |
| `DATASET_WEIGHT_GAZE` | `0.1` | No description |
| `DATASET_WEIGHT_MNIST` | `0.4` | No description |
| `DATASET_WEIGHT_PHONOLOGY` | `0.3` | No description |
| `DATASET_WEIGHT_TEMPORAL` | `0.2` | No description |
| `DA_BASELINE` | `0.3` | No description |
| `DA_BURST_MAGNITUDE` | `1.0` | No description |
| `DA_DIP_MAGNITUDE` | `-0.5` | No description |
| `DA_PHASIC_DECAY_PER_MS` | `0.995` | No description |
| `DA_TONIC_ALPHA` | `0.05` | No description |
| `HOMEOSTATIC_TAU` | `0.999` | No description |
| `MAX_RECEPTOR_SENSITIVITY` | `1.5` | No description |
| `MIN_RECEPTOR_SENSITIVITY` | `0.5` | No description |
| `NE_AROUSAL_ALPHA` | `0.1` | No description |
| `NE_BASELINE` | `0.3` | No description |
| `NE_BURST_MAGNITUDE` | `1.0` | No description |
| `NE_DECAY_PER_MS` | `0.99` | No description |
| `PROGRESS_BAR_HEIGHT` | `0.5` | No description |
| `PROPRIOCEPTION_NOISE_SCALE` | `0.1` | No description |
| `SENSORIMOTOR_WEIGHT_MANIPULATION` | `0.25` | No description |
| `SENSORIMOTOR_WEIGHT_MOTOR_CONTROL` | `0.25` | No description |
| `SENSORIMOTOR_WEIGHT_PREDICTION` | `0.25` | No description |
| `SENSORIMOTOR_WEIGHT_REACHING` | `0.25` | No description |
| `SPIKE_PROBABILITY_HIGH` | `0.5` | No description |
| `SPIKE_PROBABILITY_LOW` | `0.15` | No description |
| `SPIKE_PROBABILITY_MEDIUM` | `0.3` | No description |
| `TEXT_POSITION_BOTTOM_RIGHT_X` | `0.98` | No description |
| `TEXT_POSITION_BOTTOM_RIGHT_Y` | `0.98` | No description |
| `TEXT_POSITION_CENTER` | `0.5` | No description |
| `TEXT_POSITION_TOP_LEFT` | `0.1` | No description |

**Source**: `thalia\neuromodulation\constants.py`

---

### MOTOR SPIKE PROBABILITIES

| Constant | Value | Description |
|----------|-------|-------------|
| `MATCH_PROBABILITY_DEFAULT` | `0.3` | No description |
| `REWARD_SCALE_PREDICTION` | `1.0` | No description |
| `STIMULUS_STRENGTH_HIGH` | `1.0` | No description |
| `WEIGHT_INIT_SCALE_SMALL` | `0.05` | MOTOR SPIKE PROBABILITIES |
| `WORKSPACE_SIZE_DEFAULT` | `1.0` | No description |

**Source**: `thalia\training\datasets\constants.py`

---

### Text Positioning

| Constant | Value | Description |
|----------|-------|-------------|
| `ALPHA_HIGHLIGHT` | `0.2` | No description |
| `ALPHA_SEMI_TRANSPARENT` | `0.5` | No description |
| `AXIS_MARGIN_NEGATIVE` | `-0.5` | No description |
| `AXIS_MARGIN_POSITIVE` | `0.5` | Text Positioning |
| `FIRING_RATE_RUNAWAY_THRESHOLD` | `0.9` | No description |
| `FIRING_RATE_SILENCE_THRESHOLD` | `0.01` | No description |
| `TARGET_SPIKE_RATE_LOWER` | `0.05` | No description |
| `TARGET_SPIKE_RATE_UPPER` | `0.15` | No description |

**Source**: `thalia\training\visualization\constants.py`

---

### UI Element Dimensions

| Constant | Value | Description |
|----------|-------|-------------|
| `CALIBRATION_EXCELLENT_ECE` | `0.1` | No description |
| `CALIBRATION_GOOD_ECE` | `0.15` | No description |
| `DIFFICULTY_RANGE_MAX` | `0.9` | No description |
| `DIFFICULTY_RANGE_MIN` | `0.3` | No description |
| `PERFORMANCE_ACCEPTABLE` | `0.85` | No description |
| `PERFORMANCE_EXCELLENT` | `0.95` | UI Element Dimensions |
| `PERFORMANCE_GOOD` | `0.9` | No description |
| `PERFORMANCE_POOR` | `0.7` | No description |

**Source**: `thalia\training\visualization\constants.py`

---

