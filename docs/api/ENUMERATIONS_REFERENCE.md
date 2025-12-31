# Enumerations Reference

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2025-12-31 19:33:39
> Generated from: `scripts/generate_api_docs.py`

This document catalogs all enumeration types used in Thalia.

Total: 47 enumerations

## 📑 Table of Contents

Jump to category:

- [Thalia/Components](#thaliacomponents) (3 enums)
- [Thalia/Config](#thaliaconfig) (5 enums)
- [Thalia/Core](#thaliacore) (1 enums)
- [Thalia/Datasets](#thaliadatasets) (8 enums)
- [Thalia/Decision Making](#thaliadecision-making) (1 enums)
- [Thalia/Diagnostics](#thaliadiagnostics) (5 enums)
- [Thalia/Environments](#thaliaenvironments) (1 enums)
- [Thalia/Io](#thaliaio) (2 enums)
- [Thalia/Language](#thalialanguage) (1 enums)
- [Thalia/Learning](#thalialearning) (1 enums)
- [Thalia/Memory](#thaliamemory) (1 enums)
- [Thalia/Pathways](#thaliapathways) (1 enums)
- [Thalia/Regions](#thaliaregions) (5 enums)
- [Thalia/Tasks](#thaliatasks) (4 enums)
- [Thalia/Training](#thaliatraining) (8 enums)

## Enumerations by Category

### Thalia/Components

#### [``CodingStrategy``](../../src/thalia/components/coding/spike_coding.py#L37) (Enum)

Spike coding strategies (shared across encoders/decoders).

**Source**: [`thalia/components/coding/spike_coding.py`](../../src/thalia/components/coding/spike_coding.py)

**Members**:

- `RATE` — Spike count encodes value
- `TEMPORAL` — Spike timing encodes value
- `POPULATION` — Population codes
- `PHASE` — Phase relative to oscillation
- `BURST` — Burst patterns
- `SDR` — Sparse distributed representation
- `WTA` — Winner-take-all

---

#### [``InitStrategy``](../../src/thalia/components/synapses/weight_init.py#L81) (Enum)

Weight initialization strategies.

**Source**: [`thalia/components/synapses/weight_init.py`](../../src/thalia/components/synapses/weight_init.py)

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

#### [``STPType``](../../src/thalia/components/synapses/stp.py#L53) (Enum)

Predefined synapse types based on Markram et al. (1998) classification.

**Source**: [`thalia/components/synapses/stp.py`](../../src/thalia/components/synapses/stp.py)

**Members**:

- `DEPRESSING` — Strong initial, rapid fatigue
- `DEPRESSING_FAST` — Very fast depression, quick recovery
- `FACILITATING` — Weak initial, builds up with activity
- `FACILITATING_STRONG` — Very strong facilitation
- `PSEUDOLINEAR` — Balanced, roughly linear response
- `NONE` — 'none'

---

### Thalia/Config

#### [``CortexType``](../../src/thalia/config/brain_config.py#L38) (Enum)

Types of cortex implementation.

LAYERED: Standard feedforward layered cortex (L4 → L2/3 → L5)
PREDICTIVE: Layered cortex with predictive coding (local error signals)

**Source**: [`thalia/config/brain_config.py`](../../src/thalia/config/brain_config.py)

**Members**:

- `LAYERED` — 'layered'
- `PREDICTIVE` — 'predictive'

---

#### [``CurriculumStage``](../../src/thalia/config/curriculum_growth.py#L50) (IntEnum)

Curriculum stages matching main training plan.

**Source**: [`thalia/config/curriculum_growth.py`](../../src/thalia/config/curriculum_growth.py)

**Members**:

- `SENSORIMOTOR` — Stage -0.5 (motor control)
- `PHONOLOGY` — Stage 0 (phonological learning)
- `TODDLER` — Stage 1 (first words, joint attention)
- `GRAMMAR` — Stage 2 (grammar, composition)
- `READING` — Stage 3 (reading, planning)
- `ABSTRACT` — Stage 4 (abstract reasoning)

---

#### [``DecodingType``](../../src/thalia/config/language_config.py#L30) (Enum)

Types of spike decoding strategies.

**Source**: [`thalia/config/language_config.py`](../../src/thalia/config/language_config.py)

**Members**:

- `RATE` — 'rate'
- `TEMPORAL` — 'temporal'
- `POPULATION` — 'population'
- `ATTENTION` — 'attention'

---

#### [``EncodingType``](../../src/thalia/config/language_config.py#L21) (Enum)

Types of spike encoding strategies.

**Source**: [`thalia/config/language_config.py`](../../src/thalia/config/language_config.py)

**Members**:

- `RATE` — 'rate'
- `TEMPORAL` — 'temporal'
- `PHASE` — 'phase'
- `BURST` — 'burst'
- `SDR` — 'sdr'

---

#### [``RegionType``](../../src/thalia/config/brain_config.py#L29) (Enum)

Types of brain regions.

**Source**: [`thalia/config/brain_config.py`](../../src/thalia/config/brain_config.py)

**Members**:

- `CORTEX` — 'cortex'
- `HIPPOCAMPUS` — 'hippocampus'
- `PFC` — 'pfc'
- `STRIATUM` — 'striatum'
- `CEREBELLUM` — 'cerebellum'

---

### Thalia/Core

#### [``DiagnosticLevel``](../../src/thalia/core/diagnostics.py#L46) (Enum)

Verbosity levels for diagnostics.

**Source**: [`thalia/core/diagnostics.py`](../../src/thalia/core/diagnostics.py)

**Members**:

- `OFF` — No diagnostics
- `SUMMARY` — Epoch-level summaries only
- `DETAILED` — Per-trial key metrics
- `TRACE` — Full per-timestep traces (expensive!)

---

### Thalia/Datasets

#### [``AgreementType``](../../src/thalia/datasets/grammar.py#L42) (Enum)

Subject-verb agreement types.

**Source**: [`thalia/datasets/grammar.py`](../../src/thalia/datasets/grammar.py)

**Members**:

- `SINGULAR` — 'singular'
- `PLURAL` — 'plural'

---

#### [``GrammarRule``](../../src/thalia/datasets/grammar.py#L32) (Enum)

Types of grammar rules to test.

**Source**: [`thalia/datasets/grammar.py`](../../src/thalia/datasets/grammar.py)

**Members**:

- `SUBJECT_VERB_AGREEMENT` — 'sv_agreement'
- `NOUN_ADJECTIVE` — 'noun_adj'
- `WORD_ORDER_SVO` — 'word_order_svo'
- `WORD_ORDER_SOV` — 'word_order_sov'
- `PLURAL_MORPHOLOGY` — 'plural_morph'
- `TENSE_MORPHOLOGY` — 'tense_morph'

---

#### [``Language``](../../src/thalia/datasets/grammar.py#L25) (Enum)

Supported languages for grammar tasks.

**Source**: [`thalia/datasets/grammar.py`](../../src/thalia/datasets/grammar.py)

**Members**:

- `ENGLISH` — Language to use
- `GERMAN` — 'de'
- `SPANISH` — 'es'

---

#### [``Language``](../../src/thalia/datasets/phonology.py#L26) (Enum)

Supported languages for phonological training.

**Source**: [`thalia/datasets/phonology.py`](../../src/thalia/datasets/phonology.py)

**Members**:

- `ENGLISH` — ===== ENGLISH: Vowel categories =====
- `GERMAN` — ===== GERMAN: Unique vowels =====
- `SPANISH` — ===== SPANISH: Vowels (5-vowel system) =====

---

#### [``Language``](../../src/thalia/datasets/reading.py#L24) (Enum)

Supported languages for reading tasks.

**Source**: [`thalia/datasets/reading.py`](../../src/thalia/datasets/reading.py)

**Members**:

- `ENGLISH` — Language to use
- `GERMAN` — 'de'
- `SPANISH` — 'es'

---

#### [``PatternType``](../../src/thalia/datasets/temporal_sequences.py#L23) (Enum)

Types of sequential patterns.

**Source**: [`thalia/datasets/temporal_sequences.py`](../../src/thalia/datasets/temporal_sequences.py)

**Members**:

- `ABC` — Linear sequence A→B→C
- `ABA` — Repetition with gap A→B→A
- `AAB` — Immediate repetition A→A→B
- `ABAC` — Hierarchical A→B→A→C
- `RANDOM` — No structure (control)

---

#### [``PhonemeCategory``](../../src/thalia/datasets/phonology.py#L33) (Enum)

Phoneme categories for discrimination tasks (multi-language).

**Source**: [`thalia/datasets/phonology.py`](../../src/thalia/datasets/phonology.py)

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

#### [``ReadingTask``](../../src/thalia/datasets/reading.py#L31) (Enum)

Types of reading tasks.

**Source**: [`thalia/datasets/reading.py`](../../src/thalia/datasets/reading.py)

**Members**:

- `PHONEME_TO_WORD` — Decode phonemes → word
- `WORD_TO_MEANING` — Map word → semantic features
- `SENTENCE_COMPLETION` — Fill in missing word
- `SIMPLE_QA` — Who/what/where questions
- `SEMANTIC_ROLE` — Agent/action/patient labeling

---

### Thalia/Decision Making

#### [``SelectionMode``](../../src/thalia/decision_making/action_selection.py#L26) (Enum)

Action selection strategies.

**Source**: [`thalia/decision_making/action_selection.py`](../../src/thalia/decision_making/action_selection.py)

**Members**:

- `SOFTMAX` — Temperature-based probabilistic selection
- `GREEDY` — Always choose highest-value action
- `EPSILON_GREEDY` — ε chance of random, 1-ε chance of greedy
- `UCB` — Upper Confidence Bound (pure exploration)

---

### Thalia/Diagnostics

#### [``CriticalityState``](../../src/thalia/diagnostics/criticality.py#L65) (Enum)

Network criticality state.

**Source**: [`thalia/diagnostics/criticality.py`](../../src/thalia/diagnostics/criticality.py)

**Members**:

- `SUBCRITICAL` — 'subcritical'
- `CRITICAL` — 'critical'
- `SUPERCRITICAL` — 'supercritical'

---

#### [``HealthIssue``](../../src/thalia/diagnostics/health_monitor.py#L49) (Enum)

Types of network health issues.

**Source**: [`thalia/diagnostics/health_monitor.py`](../../src/thalia/diagnostics/health_monitor.py)

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

#### [``IssueSeverity``](../../src/thalia/diagnostics/health_monitor.py#L62) (Enum)

Severity levels for health issues.

Values represent severity scores (0-100, higher = worse).

**Source**: [`thalia/diagnostics/health_monitor.py`](../../src/thalia/diagnostics/health_monitor.py)

**Members**:

- `LOW` — Minor issues, informational
- `MEDIUM` — Moderate issues, should be addressed
- `HIGH` — Critical issues, need immediate attention
- `CRITICAL` — Catastrophic issues, system failure imminent

---

#### [``MetacognitiveStage``](../../src/thalia/diagnostics/metacognition.py#L30) (Enum)

Developmental stages of metacognitive ability.

**Source**: [`thalia/diagnostics/metacognition.py`](../../src/thalia/diagnostics/metacognition.py)

**Members**:

- `TODDLER` — Binary: know vs don't know
- `PRESCHOOL` — Coarse: high/medium/low
- `SCHOOL_AGE` — Continuous but poorly calibrated
- `ADOLESCENT` — Well-calibrated with training

---

#### [``OscillatorIssue``](../../src/thalia/diagnostics/oscillator_health.py#L56) (Enum)

Types of oscillator health issues.

**Source**: [`thalia/diagnostics/oscillator_health.py`](../../src/thalia/diagnostics/oscillator_health.py)

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

### Thalia/Environments

#### [``SpikeEncoding``](../../src/thalia/environments/sensorimotor_wrapper.py#L133) (Enum)

Spike encoding strategies for proprioception.

**Source**: [`thalia/environments/sensorimotor_wrapper.py`](../../src/thalia/environments/sensorimotor_wrapper.py)

**Members**:

- `RATE` — Rate coding (firing rate ∝ value)
- `POPULATION` — Population coding (Gaussian tuning curves)
- `TEMPORAL` — Temporal coding (spike timing)

---

### Thalia/Io

#### [``DType``](../../src/thalia/io/tensor_encoding.py#L29) (IntEnum)

Supported data types.

**Source**: [`thalia/io/tensor_encoding.py`](../../src/thalia/io/tensor_encoding.py)

**Members**:

- `FLOAT32` — 0
- `FLOAT64` — 1
- `INT32` — 2
- `INT64` — 3
- `BOOL` — 4
- `FLOAT16` — Half precision

---

#### [``EncodingType``](../../src/thalia/io/tensor_encoding.py#L23) (IntEnum)

Tensor encoding types.

**Source**: [`thalia/io/tensor_encoding.py`](../../src/thalia/io/tensor_encoding.py)

**Members**:

- `DENSE` — 0
- `SPARSE_COO` — 1

---

### Thalia/Language

#### [``PositionEncodingType``](../../src/thalia/language/position.py#L56) (Enum)

Types of position encoding.

**Source**: [`thalia/language/position.py`](../../src/thalia/language/position.py)

**Members**:

- `SINUSOIDAL` — Classic transformer-style
- `OSCILLATORY` — Neural oscillation-based
- `PHASE_PRECESSION` — Hippocampal-style
- `NESTED_GAMMA` — Theta-nested gamma

---

### Thalia/Learning

#### [``SocialCueType``](../../src/thalia/learning/social_learning.py#L21) (Enum)

Types of social cues.

**Source**: [`thalia/learning/social_learning.py`](../../src/thalia/learning/social_learning.py)

**Members**:

- `DEMONSTRATION` — Observed action
- `OSTENSIVE` — Teaching signal (eye contact, motherese)
- `GAZE` — Gaze direction
- `JOINT_ATTENTION` — Shared focus
- `NONE` — 'none'

---

### Thalia/Memory

#### [``SleepStage``](../../src/thalia/memory/consolidation/consolidation.py#L108) (Enum)

Sleep stages during consolidation.

**Source**: [`thalia/memory/consolidation/consolidation.py`](../../src/thalia/memory/consolidation/consolidation.py)

**Members**:

- `NREM` — Non-REM: Hippocampus → Cortex transfer
- `REM` — Non-REM: Hippocampus → Cortex transfer

---

### Thalia/Pathways

#### [``Modality``](../../src/thalia/pathways/sensory_pathways.py#L119) (Enum)

Sensory modalities.

**Source**: [`thalia/pathways/sensory_pathways.py`](../../src/thalia/pathways/sensory_pathways.py)

**Members**:

- `VISION` — 'vision'
- `AUDITION` — 'audition'
- `LANGUAGE` — 'language'
- `TOUCH` — 'touch'
- `PROPRIOCEPTION` — 'proprioception'

---

### Thalia/Regions

#### [``ErrorType``](../../src/thalia/regions/cortex/predictive_coding.py#L92) (Enum)

Types of prediction errors.

**Source**: [`thalia/regions/cortex/predictive_coding.py`](../../src/thalia/regions/cortex/predictive_coding.py)

**Members**:

- `POSITIVE` — Actual > Predicted (under-prediction)
- `NEGATIVE` — Actual < Predicted (over-prediction)
- `SIGNED` — Single population with +/- values

---

#### [``GoalStatus``](../../src/thalia/regions/prefrontal_hierarchy.py#L36) (Enum)

Status of a goal in the hierarchy.

**Source**: [`thalia/regions/prefrontal_hierarchy.py`](../../src/thalia/regions/prefrontal_hierarchy.py)

**Members**:

- `PENDING` — Not started
- `ACTIVE` — Currently pursuing
- `COMPLETED` — Successfully achieved
- `FAILED` — Could not achieve
- `PAUSED` — Temporarily suspended

---

#### [``HERStrategy``](../../src/thalia/regions/hippocampus/hindsight_relabeling.py#L33) (Enum)

Strategy for selecting hindsight goals.

**Source**: [`thalia/regions/hippocampus/hindsight_relabeling.py`](../../src/thalia/regions/hippocampus/hindsight_relabeling.py)

**Members**:

- `FINAL` — Use final achieved state as goal
- `FUTURE` — Sample from future achieved states
- `EPISODE` — Sample from any state in episode
- `RANDOM` — Sample random goal (baseline)

---

#### [``LearningRule``](../../src/thalia/regions/base.py#L24) (Enum)

Types of learning rules used in different brain regions.

**Source**: [`thalia/regions/base.py`](../../src/thalia/regions/base.py)

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

#### [``ReplayMode``](../../src/thalia/regions/hippocampus/replay_engine.py#L42) (Enum)

Replay execution mode.

**Source**: [`thalia/regions/hippocampus/replay_engine.py`](../../src/thalia/regions/hippocampus/replay_engine.py)

**Members**:

- `SEQUENCE` — Gamma-driven sequence replay
- `SINGLE` — Single-state replay (fallback)
- `RIPPLE` — Sharp-wave ripple replay

---

### Thalia/Tasks

#### [``MovementDirection``](../../src/thalia/tasks/sensorimotor.py#L52) (Enum)

Basic movement directions.

**Source**: [`thalia/tasks/sensorimotor.py`](../../src/thalia/tasks/sensorimotor.py)

**Members**:

- `LEFT` — 0
- `RIGHT` — 1
- `UP` — 2
- `DOWN` — 3
- `FORWARD` — 4
- `BACK` — 5
- `STOP` — 6

---

#### [``SensorimotorTaskType``](../../src/thalia/tasks/sensorimotor.py#L44) (Enum)

Types of sensorimotor tasks.

**Source**: [`thalia/tasks/sensorimotor.py`](../../src/thalia/tasks/sensorimotor.py)

**Members**:

- `MOTOR_CONTROL` — 'motor_control'
- `REACHING` — 'reaching'
- `MANIPULATION` — 'manipulation'
- `PREDICTION` — 'prediction'

---

#### [``StimulusType``](../../src/thalia/tasks/executive_function.py#L53) (Enum)

Stimulus categories for Go/No-Go.

**Source**: [`thalia/tasks/executive_function.py`](../../src/thalia/tasks/executive_function.py)

**Members**:

- `TARGET` — Go signal
- `DISTRACTOR` — No-go signal
- `NEUTRAL` — 'neutral'

---

#### [``TaskType``](../../src/thalia/tasks/executive_function.py#L37) (Enum)

Types of executive function tasks.

**Source**: [`thalia/tasks/executive_function.py`](../../src/thalia/tasks/executive_function.py)

**Members**:

- `GO_NO_GO` — 'go_no_go'
- `DELAYED_GRATIFICATION` — 'delayed_gratification'
- `DCCS` — Dimensional Change Card Sort
- `TASK_SWITCHING` — 'task_switching'
- `TOWER_OF_HANOI` — 'tower_of_hanoi'
- `RAVENS_MATRICES` — 'ravens_matrices'
- `ANALOGICAL_REASONING` — 'analogical_reasoning'

---

### Thalia/Training

#### [``AttentionStage``](../../src/thalia/training/curriculum/constants.py#L35) (Enum)

Developmental stages of attention control.

Represents the shift from reactive (bottom-up) to proactive (top-down)
attention control across development, matching curriculum stages.

Biological basis:
- Infant: Pure bottom-up (novelty, salience, motion)
- Toddler: Mostly bottom-up with emerging goal-directed control
- Preschool: Balanced control (conflict monitoring emerges)
- School-age: Top-down dominant (strategic attention allocation)

Implementation:
- Controls thalamic gating strength (alpha suppression)
- Modulates PFC→thalamus feedback gain
- Adjusts NE gain modulation sensitivity

References:
- Posner & Petersen (1990): Attention networks
- Colombo (2001): Infant attention development
- Diamond (2013): Executive function emergence

**Source**: [`thalia/training/curriculum/constants.py`](../../src/thalia/training/curriculum/constants.py)

**Members**:

- `INFANT` — Stage 0: Pure bottom-up (100% reactive)
- `TODDLER` — Stage 1: Mostly bottom-up (70% reactive, 30% goal-directed)
- `PRESCHOOL` — Stage 2: Balanced (50% reactive, 50% goal-directed)
- `SCHOOL_AGE` — Stage 3+: Top-down dominant (30% reactive, 70% goal-directed)

---

#### [``GateDecision``](../../src/thalia/training/curriculum/stage_gates.py#L29) (Enum)

Gate decision outcomes.

**Source**: [`thalia/training/curriculum/stage_gates.py`](../../src/thalia/training/curriculum/stage_gates.py)

**Members**:

- `PROCEED` — 'proceed'
- `EXTEND` — 'extend_stage'
- `ROLLBACK` — 'rollback_checkpoint'
- `EMERGENCY_STOP` — 'emergency_stop'

---

#### [``InterventionType``](../../src/thalia/training/curriculum/stage_monitoring.py#L28) (Enum)

Types of interventions that can be triggered.

**Source**: [`thalia/training/curriculum/stage_monitoring.py`](../../src/thalia/training/curriculum/stage_monitoring.py)

**Members**:

- `NONE` — 'none'
- `REDUCE_LOAD` — 'reduce_load'
- `CONSOLIDATE` — 'consolidate'
- `TEMPORAL_SEPARATION` — 'temporal_separation'
- `EMERGENCY_STOP` — 'emergency_stop'
- `ROLLBACK` — 'rollback'

---

#### [``LogLevel``](../../src/thalia/training/curriculum/logger.py#L56) (Enum)

Logging levels for curriculum training.

**Source**: [`thalia/training/curriculum/logger.py`](../../src/thalia/training/curriculum/logger.py)

**Members**:

- `DEBUG` — 'DEBUG'
- `INFO` — 'INFO'
- `WARNING` — 'WARNING'
- `ERROR` — 'ERROR'

---

#### [``MechanismPriority``](../../src/thalia/training/curriculum/stage_manager.py#L191) (IntEnum)

Priority levels for cognitive mechanisms.

**Source**: [`thalia/training/curriculum/stage_manager.py`](../../src/thalia/training/curriculum/stage_manager.py)

**Members**:

- `CRITICAL` — Cannot be disabled (e.g., basic perception)
- `HIGH` — Core mechanisms for current stage
- `MEDIUM` — Supporting mechanisms
- `LOW` — Optional enhancements

---

#### [``NoiseType``](../../src/thalia/training/curriculum/noise_scheduler.py#L57) (Enum)

Types of noise that can be scheduled.

**Source**: [`thalia/training/curriculum/noise_scheduler.py`](../../src/thalia/training/curriculum/noise_scheduler.py)

**Members**:

- `MEMBRANE` — Neuron membrane potential noise
- `WEIGHT` — Synaptic weight perturbation
- `SPIKE` — Temporal jitter
- `INPUT` — Data augmentation

---

#### [``PhonologyTaskType``](../../src/thalia/training/datasets/loaders.py#L434) (Enum)

Task types for phonology stage.

**Source**: [`thalia/training/datasets/loaders.py`](../../src/thalia/training/datasets/loaders.py)

**Members**:

- `MNIST` — MNIST
- `TEMPORAL` — 'temporal'
- `PHONOLOGY` — 'phonology'
- `GAZE_FOLLOWING` — 'gaze_following'

---

#### [``TaskType``](../../src/thalia/training/datasets/loaders.py#L108) (Enum)

Task types for sensorimotor stage.

**Source**: [`thalia/training/datasets/loaders.py`](../../src/thalia/training/datasets/loaders.py)

**Members**:

- `MOTOR_CONTROL` — 'motor_control'
- `REACHING` — 'reaching'
- `MANIPULATION` — 'manipulation'
- `PREDICTION` — 'prediction'

---

