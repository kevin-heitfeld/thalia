# Module Exports Reference

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2025-12-22 13:58:06
> Generated from: `scripts/generate_api_docs.py`

This document catalogs all public exports (`__all__`) from Thalia modules. These are the recommended imports for external code.

Total: 46 modules, 556 exports

## Module Exports

### `thalia`

**Source**: `thalia\__init__.py`

**Exports** (20):

- `__version__`
- `ThaliaConfig`
- `GlobalConfig`
- `BrainConfig`
- `RegionSizes`
- `ComponentGraph`
- `ConnectionGraph`
- `TopologyGraph`
- `SourceSpec`
- `SourcePort`
- `TargetPort`
- `SourceOutputs`
- `InputSizes`
- `SynapticWeights`
- `LearningStrategies`
- `StateDict`
- `CheckpointMetadata`
- `DiagnosticsDict`
- `NeuromodulatorLevels`
- `BatchData`

**Usage**:

```python
from thalia import __version__
```

---

### `thalia.components`

**Source**: `thalia\components\__init__.py`

**Exports** (20):

- `ConductanceLIF`
- `ConductanceLIFConfig`
- `TAU_MEM_STANDARD`
- `TAU_MEM_FAST`
- `TAU_MEM_SLOW`
- `TAU_SYN_EXCITATORY`
- `TAU_SYN_INHIBITORY`
- `TAU_SYN_NMDA`
- `V_THRESHOLD_STANDARD`
- `V_RESET_STANDARD`
- `V_REST_STANDARD`
- `E_LEAK`
- `E_EXCITATORY`
- `E_INHIBITORY`
- `G_LEAK_STANDARD`
- `G_LEAK_FAST`
- `G_LEAK_SLOW`
- `ADAPT_INCREMENT_NONE`
- `ADAPT_INCREMENT_MODERATE`
- `ADAPT_INCREMENT_STRONG`

**Usage**:

```python
from thalia.components import ConductanceLIF
```

---

### `thalia.components.coding`

**Source**: `thalia\components\coding\__init__.py`

**Exports** (12):

- `CodingStrategy`
- `SpikeCodingConfig`
- `SpikeEncoder`
- `SpikeDecoder`
- `RateEncoder`
- `RateDecoder`
- `compute_spike_similarity`
- `compute_firing_rate`
- `compute_spike_count`
- `compute_spike_density`
- `is_silent`
- `is_saturated`

**Usage**:

```python
from thalia.components.coding import CodingStrategy
```

---

### `thalia.components.neurons`

**Source**: `thalia\components\neurons\__init__.py`

**Exports** (20):

- `ConductanceLIF`
- `ConductanceLIFConfig`
- `TAU_MEM_STANDARD`
- `TAU_MEM_FAST`
- `TAU_MEM_SLOW`
- `TAU_SYN_EXCITATORY`
- `TAU_SYN_INHIBITORY`
- `TAU_SYN_NMDA`
- `V_THRESHOLD_STANDARD`
- `V_RESET_STANDARD`
- `V_REST_STANDARD`
- `E_LEAK`
- `E_EXCITATORY`
- `E_INHIBITORY`
- `G_LEAK_STANDARD`
- `G_LEAK_FAST`
- `G_LEAK_SLOW`
- `ADAPT_INCREMENT_NONE`
- `ADAPT_INCREMENT_MODERATE`
- `ADAPT_INCREMENT_STRONG`

**Usage**:

```python
from thalia.components.neurons import ConductanceLIF
```

---

### `thalia.components.synapses`

**Source**: `thalia\components\synapses\__init__.py`

**Exports** (17):

- `InitStrategy`
- `WeightInitializer`
- `ShortTermPlasticity`
- `STPConfig`
- `STPType`
- `STPSynapse`
- `STP_PRESETS`
- `STPPreset`
- `get_stp_config`
- `list_presets`
- `SpikeTrace`
- `PairedTraces`
- `TraceConfig`
- `compute_stdp_update`
- `create_trace`
- `update_trace`
- `compute_decay`

**Usage**:

```python
from thalia.components.synapses import InitStrategy
```

---

### `thalia.config`

**Source**: `thalia\config\__init__.py`

**Exports** (20):

- `ThaliaConfig`
- `print_config`
- `validate_thalia_config`
- `validate_brain_config`
- `validate_global_consistency`
- `validate_region_sizes`
- `check_config_and_warn`
- `ConfigValidationError`
- `ValidatedConfig`
- `ValidatorRegistry`
- `BaseConfig`
- `NeuralComponentConfig`
- `LearningComponentConfig`
- `PathwayConfig`
- `GlobalConfig`
- `BrainConfig`
- `RegionSizes`
- `CortexType`
- `NeuromodulationConfig`
- `TrainingConfig`

**Usage**:

```python
from thalia.config import ThaliaConfig
```

---

### `thalia.coordination`

**Source**: `thalia\coordination\__init__.py`

**Exports** (9):

- `OscillatorManager`
- `BrainOscillator`
- `OscillatorConfig`
- `SinusoidalOscillator`
- `OscillatorCoupling`
- `GrowthManager`
- `GrowthEvent`
- `CapacityMetrics`
- `GrowthCoordinator`

**Usage**:

```python
from thalia.coordination import OscillatorManager
```

---

### `thalia.core`

**Source**: `thalia\core\__init__.py`

**Exports** (20):

- `ThaliaError`
- `ComponentError`
- `ConfigurationError`
- `BiologicalPlausibilityError`
- `CheckpointError`
- `IntegrationError`
- `validate_spike_tensor`
- `validate_device_consistency`
- `validate_weight_matrix`
- `validate_positive`
- `validate_probability`
- `validate_temporal_causality`
- `DiagnosticLevel`
- `DiagnosticsConfig`
- `DiagnosticsManager`
- `StriatumDiagnostics`
- `HippocampusDiagnostics`
- `BrainSystemDiagnostics`
- `Resettable`
- `Learnable`

**Usage**:

```python
from thalia.core import ThaliaError
```

---

### `thalia.core.base`

**Source**: `thalia\core\base\__init__.py`

**Exports** (3):

- `NeuralComponentConfig`
- `LearningComponentConfig`
- `PathwayConfig`

**Usage**:

```python
from thalia.core.base import NeuralComponentConfig
```

---

### `thalia.core.protocols`

**Source**: `thalia\core\protocols\__init__.py`

**Exports** (14):

- `BrainComponent`
- `BrainComponentBase`
- `BrainComponentMixin`
- `LearnableComponent`
- `RoutingComponent`
- `Resettable`
- `Learnable`
- `Forwardable`
- `Diagnosable`
- `WeightContainer`
- `Configurable`
- `NeuralComponentProtocol`
- `Checkpointable`
- `CheckpointableWithNeuromorphic`

**Usage**:

```python
from thalia.core.protocols import BrainComponent
```

---

### `thalia.datasets`

**Source**: `thalia\datasets\__init__.py`

**Exports** (20):

- `PhonologicalDataset`
- `PhonologicalConfig`
- `PhonemeCategory`
- `PhonemeFeatures`
- `PHONEME_FEATURES`
- `Language`
- `LANGUAGE_PHONEMES`
- `LANGUAGE_CONTRASTS`
- `TemporalSequenceDataset`
- `SequenceConfig`
- `PatternType`
- `create_stage0_temporal_dataset`
- `CIFARForThalia`
- `CIFARConfig`
- `create_stage1_cifar_datasets`
- `GrammarDataset`
- `GrammarConfig`
- `GrammarVocabulary`
- `GrammarRule`
- `AgreementType`

**Usage**:

```python
from thalia.datasets import PhonologicalDataset
```

---

### `thalia.decision_making`

**Source**: `thalia\decision_making\__init__.py`

**Exports** (3):

- `ActionSelector`
- `ActionSelectionConfig`
- `SelectionMode`

**Usage**:

```python
from thalia.decision_making import ActionSelector
```

---

### `thalia.diagnostics`

**Source**: `thalia\diagnostics\__init__.py`

**Exports** (20):

- `CriticalityConfig`
- `CriticalityMonitor`
- `CriticalityState`
- `AvalancheAnalyzer`
- `HealthConfig`
- `HealthMonitor`
- `HealthReport`
- `IssueReport`
- `HealthIssue`
- `OscillatorHealthConfig`
- `OscillatorHealthMonitor`
- `OscillatorHealthReport`
- `OscillatorIssueReport`
- `OscillatorIssue`
- `PerformanceProfiler`
- `PerformanceStats`
- `quick_profile`
- `Dashboard`
- `auto_diagnostics`
- `MetacognitiveMonitor`

**Usage**:

```python
from thalia.diagnostics import CriticalityConfig
```

---

### `thalia.environments`

**Source**: `thalia\environments\__init__.py`

**Exports** (3):

- `SensorimotorWrapper`
- `SensorimotorConfig`
- `SpikeEncoding`

**Usage**:

```python
from thalia.environments import SensorimotorWrapper
```

---

### `thalia.io`

**Source**: `thalia\io\__init__.py`

**Exports** (12):

- `BrainCheckpoint`
- `CheckpointManager`
- `save_checkpoint`
- `load_checkpoint`
- `compress_file`
- `decompress_file`
- `CompressionError`
- `save_delta_checkpoint`
- `load_delta_checkpoint`
- `PrecisionPolicy`
- `PRECISION_POLICIES`
- `get_precision_statistics`

**Usage**:

```python
from thalia.io import BrainCheckpoint
```

---

### `thalia.language`

**Source**: `thalia\language\__init__.py`

**Exports** (9):

- `SpikeEncoder`
- `SpikeEncoderConfig`
- `SparseDistributedRepresentation`
- `SpikeDecoder`
- `SpikeDecoderConfig`
- `OscillatoryPositionEncoder`
- `PositionEncoderConfig`
- `LanguageBrainInterface`
- `LanguageInterfaceConfig`

**Usage**:

```python
from thalia.language import SpikeEncoder
```

---

### `thalia.learning`

**Source**: `thalia\learning\__init__.py`

**Exports** (20):

- `BCMRule`
- `BCMConfig`
- `CriticalPeriodGating`
- `CriticalPeriodConfig`
- `CriticalPeriodWindow`
- `SocialLearningModule`
- `SocialLearningConfig`
- `SocialContext`
- `SocialCueType`
- `compute_shared_attention`
- `UnifiedHomeostasis`
- `UnifiedHomeostasisConfig`
- `StriatumHomeostasis`
- `EIBalanceConfig`
- `EIBalanceRegulator`
- `LayerEIBalance`
- `IntrinsicPlasticityConfig`
- `IntrinsicPlasticity`
- `PopulationIntrinsicPlasticity`
- `MetabolicConfig`

**Usage**:

```python
from thalia.learning import BCMRule
```

---

### `thalia.learning.eligibility`

**Source**: `thalia\learning\eligibility\__init__.py`

**Exports** (2):

- `EligibilityTraceManager`
- `STDPConfig`

**Usage**:

```python
from thalia.learning.eligibility import EligibilityTraceManager
```

---

### `thalia.learning.homeostasis`

**Source**: `thalia\learning\homeostasis\__init__.py`

**Exports** (12):

- `UnifiedHomeostasis`
- `UnifiedHomeostasisConfig`
- `StriatumHomeostasis`
- `IntrinsicPlasticityConfig`
- `IntrinsicPlasticity`
- `PopulationIntrinsicPlasticity`
- `MetabolicConfig`
- `MetabolicConstraint`
- `RegionalMetabolicBudget`
- `HomeostaticConfig`
- `HomeostaticRegulator`
- `NeuromodulatorCoordination`

**Usage**:

```python
from thalia.learning.homeostasis import UnifiedHomeostasis
```

---

### `thalia.learning.rules`

**Source**: `thalia\learning\rules\__init__.py`

**Exports** (16):

- `BCMRule`
- `BCMConfig`
- `LearningConfig`
- `BaseStrategy`
- `HebbianConfig`
- `STDPConfig`
- `BCMStrategyConfig`
- `ThreeFactorConfig`
- `ErrorCorrectiveConfig`
- `HebbianStrategy`
- `STDPStrategy`
- `BCMStrategy`
- `ThreeFactorStrategy`
- `ErrorCorrectiveStrategy`
- `CompositeStrategy`
- `create_strategy`

**Usage**:

```python
from thalia.learning.rules import BCMRule
```

---

### `thalia.managers`

**Source**: `thalia\managers\__init__.py`

**Exports** (7):

- `ComponentRegistry`
- `register_region`
- `register_pathway`
- `register_module`
- `BaseManager`
- `ManagerContext`
- `BaseCheckpointManager`

**Usage**:

```python
from thalia.managers import ComponentRegistry
```

---

### `thalia.memory`

**Source**: `thalia\memory\__init__.py`

**Exports** (20):

- `SequenceMemory`
- `SequenceContext`
- `ContextBuffer`
- `ContextBufferConfig`
- `MemoryPressureDetector`
- `MemoryPressureConfig`
- `SleepStageController`
- `SleepStageConfig`
- `SleepStage`
- `ConsolidationMetrics`
- `ConsolidationSnapshot`
- `ConsolidationTrigger`
- `ConsolidationTriggerConfig`
- `SchemaExtractionConsolidation`
- `SchemaExtractionConfig`
- `Schema`
- `SemanticReorganization`
- `SemanticReorganizationConfig`
- `InterferenceResolution`
- `InterferenceResolutionConfig`

**Usage**:

```python
from thalia.memory import SequenceMemory
```

---

### `thalia.memory.consolidation`

**Source**: `thalia\memory\consolidation\__init__.py`

**Exports** (18):

- `MemoryPressureDetector`
- `MemoryPressureConfig`
- `SleepStageController`
- `SleepStageConfig`
- `SleepStage`
- `ConsolidationMetrics`
- `ConsolidationSnapshot`
- `ConsolidationTrigger`
- `ConsolidationTriggerConfig`
- `SchemaExtractionConsolidation`
- `SchemaExtractionConfig`
- `Schema`
- `SemanticReorganization`
- `SemanticReorganizationConfig`
- `InterferenceResolution`
- `InterferenceResolutionConfig`
- `run_advanced_consolidation`
- `ConsolidationManager`

**Usage**:

```python
from thalia.memory.consolidation import MemoryPressureDetector
```

---

### `thalia.mixins`

**Source**: `thalia\mixins\__init__.py`

**Exports** (6):

- `DeviceMixin`
- `ResettableMixin`
- `ConfigurableMixin`
- `DiagnosticCollectorMixin`
- `DiagnosticsMixin`
- `GrowthMixin`

**Usage**:

```python
from thalia.mixins import DeviceMixin
```

---

### `thalia.neuromodulation`

**Source**: `thalia\neuromodulation\__init__.py`

**Exports** (13):

- `VTADopamineSystem`
- `VTA`
- `VTAConfig`
- `LocusCoeruleusSystem`
- `LocusCoeruleus`
- `LocusCoeruleusConfig`
- `NucleusBasalisSystem`
- `NucleusBasalis`
- `NucleusBasalisConfig`
- `NeuromodulatorManager`
- `NeuromodulatorHomeostasis`
- `NeuromodulatorHomeostasisConfig`
- `NeuromodulatorMixin`

**Usage**:

```python
from thalia.neuromodulation import VTADopamineSystem
```

---

### `thalia.neuromodulation.systems`

**Source**: `thalia\neuromodulation\systems\__init__.py`

**Exports** (9):

- `VTADopamineSystem`
- `VTA`
- `VTAConfig`
- `LocusCoeruleusSystem`
- `LocusCoeruleus`
- `LocusCoeruleusConfig`
- `NucleusBasalisSystem`
- `NucleusBasalis`
- `NucleusBasalisConfig`

**Usage**:

```python
from thalia.neuromodulation.systems import VTADopamineSystem
```

---

### `thalia.pathways`

**Source**: `thalia\pathways\__init__.py`

**Exports** (11):

- `NeuralPathway`
- `AxonalProjection`
- `SensoryPathway`
- `VisualPathway`
- `AuditoryPathway`
- `LanguagePathway`
- `AttentionMechanisms`
- `AttentionMechanismsConfig`
- `AttentionStage`
- `CrossModalGammaBinding`
- `CrossModalBindingConfig`

**Usage**:

```python
from thalia.pathways import NeuralPathway
```

---

### `thalia.pathways.attention`

**Source**: `thalia\pathways\attention\__init__.py`

**Exports** (5):

- `AttentionMechanisms`
- `AttentionMechanismsConfig`
- `AttentionStage`
- `CrossModalGammaBinding`
- `CrossModalBindingConfig`

**Usage**:

```python
from thalia.pathways.attention import AttentionMechanisms
```

---

### `thalia.planning`

**Source**: `thalia\planning\__init__.py`

**Exports** (5):

- `MentalSimulationCoordinator`
- `SimulationConfig`
- `Rollout`
- `DynaPlanner`
- `DynaConfig`

**Usage**:

```python
from thalia.planning import MentalSimulationCoordinator
```

---

### `thalia.regions`

**Source**: `thalia\regions\__init__.py`

**Exports** (20):

- `NeuralComponent`
- `LearningRule`
- `NeuralComponentConfig`
- `NeuralComponentState`
- `RegionFactory`
- `RegionRegistry`
- `register_region`
- `LayeredCortex`
- `LayeredCortexConfig`
- `PredictiveCortex`
- `PredictiveCortexConfig`
- `Cerebellum`
- `CerebellumConfig`
- `Striatum`
- `StriatumConfig`
- `Prefrontal`
- `PrefrontalConfig`
- `Hippocampus`
- `HippocampusConfig`
- `HippocampusState`

**Usage**:

```python
from thalia.regions import NeuralComponent
```

---

### `thalia.regions.cerebellum`

**Source**: `thalia\regions\cerebellum\__init__.py`

**Exports** (5):

- `GranuleCellLayer`
- `EnhancedPurkinjeCell`
- `DeepCerebellarNuclei`
- `Cerebellum`
- `CerebellumConfig`

**Usage**:

```python
from thalia.regions.cerebellum import GranuleCellLayer
```

---

### `thalia.regions.cortex`

**Source**: `thalia\regions\cortex\__init__.py`

**Exports** (7):

- `LayeredCortex`
- `LayeredCortexConfig`
- `LayeredCortexState`
- `calculate_layer_sizes`
- `PredictiveCortex`
- `PredictiveCortexConfig`
- `PredictiveCortexState`

**Usage**:

```python
from thalia.regions.cortex import LayeredCortex
```

---

### `thalia.regions.hippocampus`

**Source**: `thalia\regions\hippocampus\__init__.py`

**Exports** (7):

- `Hippocampus`
- `HippocampusConfig`
- `HippocampusState`
- `Episode`
- `ReplayEngine`
- `ReplayConfig`
- `ReplayMode`

**Usage**:

```python
from thalia.regions.hippocampus import Hippocampus
```

---

### `thalia.regions.striatum`

**Source**: `thalia\regions\striatum\__init__.py`

**Exports** (8):

- `Striatum`
- `StriatumConfig`
- `ActionSelectionMixin`
- `TDLambdaConfig`
- `TDLambdaTraces`
- `TDLambdaLearner`
- `compute_n_step_return`
- `compute_lambda_return`

**Usage**:

```python
from thalia.regions.striatum import Striatum
```

---

### `thalia.regulation`

**Source**: `thalia\regulation\__init__.py`

**Exports** (20):

- `TARGET_FIRING_RATE_STANDARD`
- `TARGET_FIRING_RATE_LOW`
- `TARGET_FIRING_RATE_MEDIUM`
- `TARGET_FIRING_RATE_HIGH`
- `HOMEOSTATIC_TAU_STANDARD`
- `HOMEOSTATIC_TAU_FAST`
- `HOMEOSTATIC_TAU_SLOW`
- `SYNAPTIC_SCALING_RATE`
- `SYNAPTIC_SCALING_MIN`
- `SYNAPTIC_SCALING_MAX`
- `INTRINSIC_PLASTICITY_RATE`
- `FIRING_RATE_WINDOW_MS`
- `MIN_FIRING_RATE_HZ`
- `MAX_FIRING_RATE_HZ`
- `LEARNING_RATE_DEFAULT`
- `LEARNING_RATE_STDP`
- `LEARNING_RATE_BCM`
- `LEARNING_RATE_HEBBIAN`
- `TAU_ELIGIBILITY_STANDARD`
- `TAU_BCM_THRESHOLD`

**Usage**:

```python
from thalia.regulation import TARGET_FIRING_RATE_STANDARD
```

---

### `thalia.stimuli`

**Source**: `thalia\stimuli\__init__.py`

**Exports** (5):

- `StimulusPattern`
- `Sustained`
- `Transient`
- `Sequential`
- `Programmatic`

**Usage**:

```python
from thalia.stimuli import StimulusPattern
```

---

### `thalia.surgery`

**Source**: `thalia\surgery\__init__.py`

**Exports** (11):

- `lesion_region`
- `partial_lesion`
- `temporary_lesion`
- `restore_region`
- `ablate_pathway`
- `restore_pathway`
- `freeze_region`
- `unfreeze_region`
- `freeze_pathway`
- `unfreeze_pathway`
- `add_region_to_trained_brain`

**Usage**:

```python
from thalia.surgery import lesion_region
```

---

### `thalia.synapses`

**Source**: `thalia\synapses\__init__.py`

**Exports** (2):

- `AfferentSynapses`
- `AfferentSynapsesConfig`

**Usage**:

```python
from thalia.synapses import AfferentSynapses
```

---

### `thalia.tasks`

**Source**: `thalia\tasks\__init__.py`

**Exports** (20):

- `ExecutiveFunctionTasks`
- `TaskType`
- `StimulusType`
- `GoNoGoConfig`
- `DelayedGratificationConfig`
- `DCCSConfig`
- `TaskResult`
- `NBackTask`
- `ThetaGammaEncoder`
- `WorkingMemoryTaskConfig`
- `theta_gamma_n_back`
- `create_n_back_sequence`
- `SensorimotorTaskType`
- `MovementDirection`
- `MotorControlConfig`
- `ReachingConfig`
- `ManipulationConfig`
- `MotorControlTask`
- `ReachingTask`
- `ManipulationTask`

**Usage**:

```python
from thalia.tasks import ExecutiveFunctionTasks
```

---

### `thalia.training`

**Source**: `thalia\training\__init__.py`

**Exports** (20):

- `TextDataPipeline`
- `DataConfig`
- `InterleavedCurriculumSampler`
- `InterleavedCurriculumSamplerConfig`
- `SpacedRepetitionScheduler`
- `SpacedRepetitionSchedulerConfig`
- `TestingPhaseProtocol`
- `TestingPhaseConfig`
- `ProductiveFailurePhase`
- `ProductiveFailureConfig`
- `CurriculumDifficultyCalibrator`
- `DifficultyCalibratorConfig`
- `StageTransitionProtocol`
- `StageTransitionConfig`
- `TransitionWeekConfig`
- `CurriculumTrainer`
- `StageConfig`
- `TaskConfig`
- `TrainingResult`
- `MechanismPriority`

**Usage**:

```python
from thalia.training import TextDataPipeline
```

---

### `thalia.training.curriculum`

**Source**: `thalia\training\curriculum\__init__.py`

**Exports** (20):

- `InterleavedCurriculumSampler`
- `InterleavedCurriculumSamplerConfig`
- `SpacedRepetitionScheduler`
- `SpacedRepetitionSchedulerConfig`
- `TestingPhaseProtocol`
- `TestingPhaseConfig`
- `ProductiveFailurePhase`
- `ProductiveFailureConfig`
- `CurriculumDifficultyCalibrator`
- `DifficultyCalibratorConfig`
- `StageTransitionProtocol`
- `StageTransitionConfig`
- `TransitionWeekConfig`
- `CurriculumTrainer`
- `StageConfig`
- `TaskConfig`
- `TrainingResult`
- `MechanismPriority`
- `ActiveMechanism`
- `CognitiveLoadMonitor`

**Usage**:

```python
from thalia.training.curriculum import InterleavedCurriculumSampler
```

---

### `thalia.training.datasets`

**Source**: `thalia\training\datasets\__init__.py`

**Exports** (20):

- `BaseTaskLoader`
- `SensorimotorTaskLoader`
- `SensorimotorConfig`
- `PhonologyTaskLoader`
- `PhonologyConfig`
- `TaskLoaderRegistry`
- `create_sensorimotor_loader`
- `create_phonology_loader`
- `SPIKE_PROBABILITY_LOW`
- `SPIKE_PROBABILITY_MEDIUM`
- `SPIKE_PROBABILITY_HIGH`
- `SENSORIMOTOR_WEIGHT_MOTOR_CONTROL`
- `SENSORIMOTOR_WEIGHT_REACHING`
- `SENSORIMOTOR_WEIGHT_MANIPULATION`
- `SENSORIMOTOR_WEIGHT_PREDICTION`
- `DATASET_WEIGHT_MNIST`
- `DATASET_WEIGHT_TEMPORAL`
- `DATASET_WEIGHT_PHONOLOGY`
- `DATASET_WEIGHT_GAZE`
- `REWARD_SCALE_PREDICTION`

**Usage**:

```python
from thalia.training.datasets import BaseTaskLoader
```

---

### `thalia.training.evaluation`

**Source**: `thalia\training\evaluation\__init__.py`

**Exports** (5):

- `MetacognitiveCalibrator`
- `CalibrationSample`
- `CalibrationPrediction`
- `CalibrationMetrics`
- `create_simple_task_generator`

**Usage**:

```python
from thalia.training.evaluation import MetacognitiveCalibrator
```

---

### `thalia.training.visualization`

**Source**: `thalia\training\visualization\__init__.py`

**Exports** (4):

- `TrainingMonitor`
- `quick_monitor`
- `LiveDiagnostics`
- `quick_diagnostics`

**Usage**:

```python
from thalia.training.visualization import TrainingMonitor
```

---

### `thalia.utils`

**Source**: `thalia\utils\__init__.py`

**Exports** (13):

- `clamp_weights`
- `cosine_similarity_safe`
- `ensure_1d`
- `zeros_like_config`
- `ones_like_config`
- `assert_single_instance`
- `CircularDelayBuffer`
- `compute_theta_encoding_retrieval`
- `compute_ach_recurrent_suppression`
- `compute_gamma_phase_gate`
- `compute_theta_gamma_coupling_gate`
- `compute_oscillator_modulated_gain`
- `compute_learning_rate_modulation`

**Usage**:

```python
from thalia.utils import clamp_weights
```

---

### `thalia.visualization`

**Source**: `thalia\visualization\__init__.py`

**Exports** (3):

- `visualize_brain_topology`
- `export_topology_to_graphviz`
- `plot_connectivity_matrix`

**Usage**:

```python
from thalia.visualization import visualize_brain_topology
```

---

