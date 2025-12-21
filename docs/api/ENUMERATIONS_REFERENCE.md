# Enumerations Reference

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2025-12-21 20:10:00
> Generated from: `scripts/generate_api_docs.py`

This document catalogs all enumeration types used in Thalia.

Total: 47 enumerations

## Enumerations by Category

### Components

#### `CodingStrategy` (Enum)

Spike coding strategies (shared across encoders/decoders).

**Source**: `components\coding\spike_coding.py`

**Members**:

- `RATE` — Spike count encodes value
- `TEMPORAL` — Spike timing encodes value
- `POPULATION` — Population codes
- `PHASE` — Phase relative to oscillation
- `BURST` — Burst patterns
- `SDR` — Sparse distributed representation
- `WTA` — Winner-take-all

---

#### `InitStrategy` (Enum)

Weight initialization strategies.

**Source**: `components\synapses\weight_init.py`

**Members**:

- `GAUSSIAN` — Gaussian (normal) distribution
- `UNIFORM` — Uniform distribution
- `XAVIER` — Xavier/Glorot initialization
- `KAIMING` — Kaiming/He initialization
- `SPARSE_RANDOM` — Sparse random connectivity
- `TOPOGRAPHIC` — Topographic (spatial) connectivity
- `ORTHOGONAL` — Orthogonal initialization
- `ZEROS` — All zeros
- `ONES` — All ones
- `IDENTITY` — Identity matrix
- `CONSTANT` — Constant value

---

#### `STPType` (Enum)

Predefined synapse types based on Markram et al. (1998) classification.

**Source**: `components\synapses\stp.py`

**Members**:

- `DEPRESSING` — Strong initial, rapid fatigue
- `DEPRESSING_FAST` — Very fast depression, quick recovery
- `FACILITATING` — Weak initial, builds up with activity
- `FACILITATING_STRONG` — Very strong facilitation
- `PSEUDOLINEAR` — Balanced, roughly linear response
- `NONE` — 'none'

---

### Config

#### `CortexType` (Enum)

Types of cortex implementation.

LAYERED: Standard feedforward layered cortex (L4 → L2/3 → L5)
PREDICTIVE: Layered cortex with predictive coding (local error signals)

**Source**: `config\brain_config.py`

**Members**:

- `LAYERED` — 'layered'
- `PREDICTIVE` — 'predictive'

---

#### `CurriculumStage` (IntEnum)

Curriculum stages matching main training plan.

**Source**: `config\curriculum_growth.py`

**Members**:

- `SENSORIMOTOR` — Stage -0.5 (motor control)
- `PHONOLOGY` — Stage 0 (phonological learning)
- `TODDLER` — Stage 1 (first words, joint attention)
- `GRAMMAR` — Stage 2 (grammar, composition)
- `READING` — Stage 3 (reading, planning)
- `ABSTRACT` — Stage 4 (abstract reasoning)

---

#### `DecodingType` (Enum)

Types of spike decoding strategies.

**Source**: `config\language_config.py`

**Members**:

- `RATE` — 'rate'
- `TEMPORAL` — 'temporal'
- `POPULATION` — 'population'
- `ATTENTION` — 'attention'

---

#### `EncodingType` (Enum)

Types of spike encoding strategies.

**Source**: `config\language_config.py`

**Members**:

- `RATE` — 'rate'
- `TEMPORAL` — 'temporal'
- `PHASE` — 'phase'
- `BURST` — 'burst'
- `SDR` — 'sdr'

---

#### `RegionType` (Enum)

Types of brain regions.

**Source**: `config\brain_config.py`

**Members**:

- `CORTEX` — 'cortex'
- `HIPPOCAMPUS` — 'hippocampus'
- `PFC` — 'pfc'
- `STRIATUM` — 'striatum'
- `CEREBELLUM` — 'cerebellum'

---

### Core

#### `DiagnosticLevel` (Enum)

Verbosity levels for diagnostics.

**Source**: `core\diagnostics.py`

**Members**:

- `OFF` — No diagnostics
- `SUMMARY` — Epoch-level summaries only
- `DETAILED` — Per-trial key metrics
- `TRACE` — Full per-timestep traces (expensive!)

---

### Datasets

#### `AgreementType` (Enum)

Subject-verb agreement types.

**Source**: `datasets\grammar.py`

**Members**:

- `SINGULAR` — 'singular'
- `PLURAL` — 'plural'

---

#### `GrammarRule` (Enum)

Types of grammar rules to test.

**Source**: `datasets\grammar.py`

**Members**:

- `SUBJECT_VERB_AGREEMENT` — 'sv_agreement'
- `NOUN_ADJECTIVE` — 'noun_adj'
- `WORD_ORDER_SVO` — 'word_order_svo'
- `WORD_ORDER_SOV` — 'word_order_sov'
- `PLURAL_MORPHOLOGY` — 'plural_morph'
- `TENSE_MORPHOLOGY` — 'tense_morph'

---

#### `Language` (Enum)

Supported languages for grammar tasks.

**Source**: `datasets\grammar.py`

**Members**:

- `ENGLISH` — Language to use
- `GERMAN` — 'de'
- `SPANISH` — 'es'

---

#### `Language` (Enum)

Supported languages for phonological training.

**Source**: `datasets\phonology.py`

**Members**:

- `ENGLISH` — ===== ENGLISH: Vowel categories =====
- `GERMAN` — ===== GERMAN: Unique vowels =====
- `SPANISH` — ===== SPANISH: Vowels (5-vowel system) =====

---

#### `Language` (Enum)

Supported languages for reading tasks.

**Source**: `datasets\reading.py`

**Members**:

- `ENGLISH` — Language to use
- `GERMAN` — 'de'
- `SPANISH` — 'es'

---

#### `PatternType` (Enum)

Types of sequential patterns.

**Source**: `datasets\temporal_sequences.py`

**Members**:

- `ABC` — Linear sequence A→B→C
- `ABA` — Repetition with gap A→B→A
- `AAB` — Immediate repetition A→A→B
- `ABAC` — Hierarchical A→B→A→C
- `RANDOM` — No structure (control)

---

#### `PhonemeCategory` (Enum)

Phoneme categories for discrimination tasks (multi-language).

**Source**: `datasets\phonology.py`

**Members**:

- `P` — Voiceless bilabial stop (VOT ~60ms)
- `B` — Voiced bilabial stop (VOT ~0ms)
- `T` — ===== UNIVERSAL: Voicing contrasts (VOT continuum) =====
- `D` — Voiced alveolar stop (VOT ~0ms)
- `K` — Voiceless velar stop (VOT ~80ms)
- `G` — Voiced velar stop (VOT ~0ms)
- `AA` — /ɑ/ as in "father" (F1=730, F2=1090)
- `AE` — /æ/ as in "cat" (F1=660, F2=1720)
- `AH` — /ʌ/ as in "but" (F1=640, F2=1190)
- `EH` — /ɛ/ as in "bed" (F1=530, F2=1840)
- `IH` — /ɪ/ as in "bit" (F1=390, F2=1990)
- `IY` — /i/ as in "beat" (F1=270, F2=2290)
- `UH` — /ʊ/ as in "book" (F1=440, F2=1020)
- `UW` — /u/ as in "boot" (F1=300, F2=870)
- `UE` — /y/ as in "über" (F1=270, F2=2100) - high front rounded
- `OE` — /ø/ as in "schön" (F1=390, F2=1680) - mid front rounded
- `AE_DE` — /ɛː/ as in "Käse" (F1=530, F2=1840) - long mid front
- `X` — /x/ as in "Bach" (F1=1500, F2=2500) - voiceless velar fricative
- `R_UVULAR` — /ʁ/ German uvular r (F1=500, F2=1400)
- `A_ES` — /a/ as in "casa" (F1=700, F2=1200)
- `E_ES` — /e/ as in "peso" (F1=400, F2=2000)
- `I_ES` — /i/ as in "piso" (F1=280, F2=2250)
- `O_ES` — /o/ as in "poco" (F1=400, F2=800)
- `U_ES` — /u/ as in "puro" (F1=300, F2=700)
- `R_TAP` — /ɾ/ single tap as in "pero" (duration ~30ms)
- `R_TRILL` — /r/ trill as in "perro" (duration ~100ms, multiple taps)
- `B_FRIC` — /β/ voiced bilabial fricative (intervocalic b)
- `D_FRIC` — /ð/ voiced dental fricative (intervocalic d)
- `G_FRIC` — /ɣ/ voiced velar fricative (intervocalic g)
- `M` — ===== GERMAN: Unique vowels =====
- `N` — ===== UNIVERSAL: Voicing contrasts (VOT continuum) =====
- `NG` — ===== ENGLISH: Vowel categories =====

---

#### `ReadingTask` (Enum)

Types of reading tasks.

**Source**: `datasets\reading.py`

**Members**:

- `PHONEME_TO_WORD` — Decode phonemes → word
- `WORD_TO_MEANING` — Map word → semantic features
- `SENTENCE_COMPLETION` — Fill in missing word
- `SIMPLE_QA` — Who/what/where questions
- `SEMANTIC_ROLE` — Agent/action/patient labeling

---

### Decision Making

#### `SelectionMode` (Enum)

Action selection strategies.

**Source**: `decision_making\action_selection.py`

**Members**:

- `SOFTMAX` — Temperature-based probabilistic selection
- `GREEDY` — Always choose highest-value action
- `EPSILON_GREEDY` — ε chance of random, 1-ε chance of greedy
- `UCB` — Upper Confidence Bound (pure exploration)

---

### Diagnostics

#### `CriticalityState` (Enum)

Network criticality state.

**Source**: `diagnostics\criticality.py`

**Members**:

- `SUBCRITICAL` — 'subcritical'
- `CRITICAL` — 'critical'
- `SUPERCRITICAL` — 'supercritical'

---

#### `HealthIssue` (Enum)

Types of network health issues.

**Source**: `diagnostics\health_monitor.py`

**Members**:

- `ACTIVITY_COLLAPSE` — 'activity_collapse'
- `SEIZURE_RISK` — 'seizure_risk'
- `WEIGHT_EXPLOSION` — 'weight_explosion'
- `WEIGHT_COLLAPSE` — 'weight_collapse'
- `EI_IMBALANCE` — 'ei_imbalance'
- `CRITICALITY_DRIFT` — 'criticality_drift'
- `DOPAMINE_SATURATION` — 'dopamine_saturation'
- `LEARNING_STALL` — 'learning_stall'
- `OSCILLATOR_PATHOLOGY` — 'oscillator_pathology'

---

#### `IssueSeverity` (Enum)

Severity levels for health issues.

Values represent severity scores (0-100, higher = worse).

**Source**: `diagnostics\health_monitor.py`

**Members**:

- `LOW` — Minor issues, informational
- `MEDIUM` — Moderate issues, should be addressed
- `HIGH` — Critical issues, need immediate attention
- `CRITICAL` — Catastrophic issues, system failure imminent

---

#### `MetacognitiveStage` (Enum)

Developmental stages of metacognitive ability.

**Source**: `diagnostics\metacognition.py`

**Members**:

- `TODDLER` — Binary: know vs don't know
- `PRESCHOOL` — Coarse: high/medium/low
- `SCHOOL_AGE` — Continuous but poorly calibrated
- `ADOLESCENT` — Well-calibrated with training

---

#### `OscillatorIssue` (Enum)

Types of oscillator health issues.

**Source**: `diagnostics\oscillator_health.py`

**Members**:

- `FREQUENCY_DRIFT` — 'frequency_drift'
- `PHASE_LOCKING` — 'phase_locking'
- `ABNORMAL_AMPLITUDE` — 'abnormal_amplitude'
- `COUPLING_FAILURE` — 'coupling_failure'
- `SYNCHRONY_LOSS` — 'synchrony_loss'
- `PATHOLOGICAL_COUPLING` — 'pathological_coupling'
- `OSCILLATOR_DEAD` — 'oscillator_dead'
- `CROSS_REGION_DESYNCHRONY` — 'cross_region_desynchrony'

---

### Environments

#### `SpikeEncoding` (Enum)

Spike encoding strategies for proprioception.

**Source**: `environments\sensorimotor_wrapper.py`

**Members**:

- `RATE` — Rate coding (firing rate ∝ value)
- `POPULATION` — Population coding (Gaussian tuning curves)
- `TEMPORAL` — Temporal coding (spike timing)

---

### Io

#### `DType` (IntEnum)

Supported data types.

**Source**: `io\tensor_encoding.py`

**Members**:

- `FLOAT32` — 0
- `FLOAT64` — 1
- `INT32` — 2
- `INT64` — 3
- `BOOL` — 4
- `FLOAT16` — Half precision

---

#### `EncodingType` (IntEnum)

Tensor encoding types.

**Source**: `io\tensor_encoding.py`

**Members**:

- `DENSE` — 0
- `SPARSE_COO` — 1

---

### Language

#### `PositionEncodingType` (Enum)

Types of position encoding.

**Source**: `language\position.py`

**Members**:

- `SINUSOIDAL` — Classic transformer-style
- `OSCILLATORY` — Neural oscillation-based
- `PHASE_PRECESSION` — Hippocampal-style
- `NESTED_GAMMA` — Theta-nested gamma

---

### Learning

#### `SocialCueType` (Enum)

Types of social cues.

**Source**: `learning\social_learning.py`

**Members**:

- `DEMONSTRATION` — Observed action
- `OSTENSIVE` — Teaching signal (eye contact, motherese)
- `GAZE` — Gaze direction
- `JOINT_ATTENTION` — Shared focus
- `NONE` — 'none'

---

### Memory

#### `SleepStage` (Enum)

Sleep stages during consolidation.

**Source**: `memory\consolidation\consolidation.py`

**Members**:

- `NREM` — Non-REM: Hippocampus → Cortex transfer
- `REM` — Non-REM: Hippocampus → Cortex transfer

---

### Pathways

#### `AttentionStage` (Enum)

Developmental stages of attention.

**Source**: `pathways\attention\attention.py`

**Members**:

- `INFANT` — Stage 0: Pure bottom-up
- `TODDLER` — Stage 1: Mostly bottom-up (70%)
- `PRESCHOOL` — Stage 2: Balanced (50/50)
- `SCHOOL_AGE` — Stage 3+: Mostly top-down (70%)

---

#### `Modality` (Enum)

Sensory modalities.

**Source**: `pathways\sensory_pathways.py`

**Members**:

- `VISION` — 'vision'
- `AUDITION` — 'audition'
- `LANGUAGE` — 'language'
- `TOUCH` — 'touch'
- `PROPRIOCEPTION` — 'proprioception'

---

### Regions

#### `ErrorType` (Enum)

Types of prediction errors.

**Source**: `regions\cortex\predictive_coding.py`

**Members**:

- `POSITIVE` — Actual > Predicted (under-prediction)
- `NEGATIVE` — Actual < Predicted (over-prediction)
- `SIGNED` — Single population with +/- values

---

#### `GoalStatus` (Enum)

Status of a goal in the hierarchy.

**Source**: `regions\prefrontal_hierarchy.py`

**Members**:

- `PENDING` — Not started
- `ACTIVE` — Currently pursuing
- `COMPLETED` — Successfully achieved
- `FAILED` — Could not achieve
- `PAUSED` — Temporarily suspended

---

#### `HERStrategy` (Enum)

Strategy for selecting hindsight goals.

**Source**: `regions\hippocampus\hindsight_relabeling.py`

**Members**:

- `FINAL` — Use final achieved state as goal
- `FUTURE` — Sample from future achieved states
- `EPISODE` — Sample from any state in episode
- `RANDOM` — Sample random goal (baseline)

---

#### `LearningRule` (Enum)

Types of learning rules used in different brain regions.

**Source**: `regions\base.py`

**Members**:

- `HEBBIAN` — Basic Hebbian: Δw ∝ pre × post
- `STDP` — Spike-Timing Dependent Plasticity
- `BCM` — Bienenstock-Cooper-Munro with sliding threshold
- `ERROR_CORRECTIVE` — Delta rule: Δw ∝ pre × (target - actual)
- `PERCEPTRON` — Binary error correction
- `THREE_FACTOR` — Δw ∝ eligibility × dopamine
- `ACTOR_CRITIC` — Policy gradient with value function
- `REWARD_MODULATED_STDP` — Δw ∝ STDP_eligibility × dopamine (striatum uses D1/D2 variant)
- `ONE_SHOT` — Single-exposure learning
- `THETA_PHASE` — Phase-dependent encoding/retrieval
- `PREDICTIVE_STDP` — Δw ∝ STDP × prediction_error (three-factor)

---

#### `ReplayMode` (Enum)

Replay execution mode.

**Source**: `regions\hippocampus\replay_engine.py`

**Members**:

- `SEQUENCE` — Gamma-driven sequence replay
- `SINGLE` — Single-state replay (fallback)
- `RIPPLE` — Sharp-wave ripple replay

---

### Tasks

#### `MovementDirection` (Enum)

Basic movement directions.

**Source**: `tasks\sensorimotor.py`

**Members**:

- `LEFT` — 0
- `RIGHT` — 1
- `UP` — 2
- `DOWN` — 3
- `FORWARD` — 4
- `BACK` — 5
- `STOP` — 6

---

#### `SensorimotorTaskType` (Enum)

Types of sensorimotor tasks.

**Source**: `tasks\sensorimotor.py`

**Members**:

- `MOTOR_CONTROL` — 'motor_control'
- `REACHING` — 'reaching'
- `MANIPULATION` — 'manipulation'
- `PREDICTION` — 'prediction'

---

#### `StimulusType` (Enum)

Stimulus categories for Go/No-Go.

**Source**: `tasks\executive_function.py`

**Members**:

- `TARGET` — Go signal
- `DISTRACTOR` — No-go signal
- `NEUTRAL` — 'neutral'

---

#### `TaskType` (Enum)

Types of executive function tasks.

**Source**: `tasks\executive_function.py`

**Members**:

- `GO_NO_GO` — 'go_no_go'
- `DELAYED_GRATIFICATION` — 'delayed_gratification'
- `DCCS` — Dimensional Change Card Sort
- `TASK_SWITCHING` — 'task_switching'
- `TOWER_OF_HANOI` — 'tower_of_hanoi'
- `RAVENS_MATRICES` — 'ravens_matrices'
- `ANALOGICAL_REASONING` — 'analogical_reasoning'

---

### Training

#### `GateDecision` (Enum)

Gate decision outcomes.

**Source**: `training\curriculum\stage_gates.py`

**Members**:

- `PROCEED` — 'proceed'
- `EXTEND` — 'extend_stage'
- `ROLLBACK` — 'rollback_checkpoint'
- `EMERGENCY_STOP` — 'emergency_stop'

---

#### `InterventionType` (Enum)

Types of interventions that can be triggered.

**Source**: `training\curriculum\stage_monitoring.py`

**Members**:

- `NONE` — 'none'
- `REDUCE_LOAD` — 'reduce_load'
- `CONSOLIDATE` — 'consolidate'
- `TEMPORAL_SEPARATION` — 'temporal_separation'
- `EMERGENCY_STOP` — 'emergency_stop'
- `ROLLBACK` — 'rollback'

---

#### `LogLevel` (Enum)

Logging levels for curriculum training.

**Source**: `training\curriculum\logger.py`

**Members**:

- `DEBUG` — 'DEBUG'
- `INFO` — 'INFO'
- `WARNING` — 'WARNING'
- `ERROR` — 'ERROR'

---

#### `MechanismPriority` (IntEnum)

Priority levels for cognitive mechanisms.

**Source**: `training\curriculum\stage_manager.py`

**Members**:

- `CRITICAL` — Cannot be disabled (e.g., basic perception)
- `HIGH` — Core mechanisms for current stage
- `MEDIUM` — Supporting mechanisms
- `LOW` — Optional enhancements

---

#### `NoiseType` (Enum)

Types of noise that can be scheduled.

**Source**: `training\curriculum\noise_scheduler.py`

**Members**:

- `MEMBRANE` — Neuron membrane potential noise
- `WEIGHT` — Synaptic weight perturbation
- `SPIKE` — Temporal jitter
- `INPUT` — Data augmentation

---

#### `PhonologyTaskType` (Enum)

Task types for phonology stage.

**Source**: `training\datasets\loaders.py`

**Members**:

- `MNIST` — MNIST
- `TEMPORAL` — 'temporal'
- `PHONOLOGY` — 'phonology'
- `GAZE_FOLLOWING` — 'gaze_following'

---

#### `TaskType` (Enum)

Task types for sensorimotor stage.

**Source**: `training\datasets\loaders.py`

**Members**:

- `MOTOR_CONTROL` — 'motor_control'
- `REACHING` — 'reaching'
- `MANIPULATION` — 'manipulation'
- `PREDICTION` — 'prediction'

---

