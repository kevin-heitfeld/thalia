# Module Exports Reference

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2026-01-19 05:37:19
> Generated from: `scripts/generate_api_docs.py`

This document catalogs all public exports (`__all__`) from Thalia modules. These are the recommended imports for external code.

Total: 47 modules, 488 exports

## 📑 Table of Contents

Quick jump to module:

- [thalia](#thalia) | - [thalia.components](#thaliacomponents) | - [thalia.components.coding](#thaliacomponentscoding)
- [thalia.components.neurons](#thaliacomponentsneurons) | - [thalia.components.synapses](#thaliacomponentssynapses) | - [thalia.config](#thaliaconfig)
- [thalia.constants](#thaliaconstants) | - [thalia.coordination](#thaliacoordination) | - [thalia.core](#thaliacore)
- [thalia.core.base](#thaliacorebase) | - [thalia.core.protocols](#thaliacoreprotocols) | - [thalia.datasets](#thaliadatasets)
- [thalia.decision_making](#thaliadecision_making) | - [thalia.diagnostics](#thaliadiagnostics) | - [thalia.environments](#thaliaenvironments)
- [thalia.io](#thaliaio) | - [thalia.language](#thalialanguage) | - [thalia.learning](#thalialearning)
- [thalia.learning.eligibility](#thalialearningeligibility) | - [thalia.learning.homeostasis](#thalialearninghomeostasis) | - [thalia.learning.rules](#thalialearningrules)
- [thalia.managers](#thaliamanagers) | - [thalia.memory](#thaliamemory) | - [thalia.mixins](#thaliamixins)
- [thalia.neuromodulation](#thalianeuromodulation) | - [thalia.neuromodulation.systems](#thalianeuromodulationsystems) | - [thalia.pathways](#thaliapathways)
- [thalia.regions](#thaliaregions) | - [thalia.regions.cerebellum](#thaliaregionscerebellum) | - [thalia.regions.cortex](#thaliaregionscortex)
- [thalia.regions.hippocampus](#thaliaregionshippocampus) | - [thalia.regions.prefrontal](#thaliaregionsprefrontal) | - [thalia.regions.striatum](#thaliaregionsstriatum)
- [thalia.regions.thalamus](#thaliaregionsthalamus) | - [thalia.regulation](#thaliaregulation) | - [thalia.replay](#thaliareplay)
- [thalia.stimuli](#thaliastimuli) | - [thalia.surgery](#thaliasurgery) | - [thalia.synapses](#thaliasynapses)
- [thalia.tasks](#thaliatasks) | - [thalia.training](#thaliatraining) | - [thalia.training.curriculum](#thaliatrainingcurriculum)
- [thalia.training.datasets](#thaliatrainingdatasets) | - [thalia.training.evaluation](#thaliatrainingevaluation) | - [thalia.training.visualization](#thaliatrainingvisualization)
- [thalia.utils](#thaliautils) | - [thalia.visualization](#thaliavisualization) | 

## Module Exports

### `thalia`

**Source**: [`thalia/__init__.py`](../../src/thalia/__init__.py)

**Exports** (20):

- `__version__`
- `ThaliaConfig`
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
- `NeuromodulatorLevels`
- `BatchData`
- `DynamicBrain`
- `NeuralComponentConfig`

**Usage**:

```python
from thalia import __version__
```

---

### `thalia.components`

**Source**: [`thalia/components/__init__.py`](../../src/thalia/components/__init__.py)

**Exports** (20):

- `ConductanceLIF`
- `ConductanceLIFConfig`
- `create_pyramidal_neurons`
- `create_relay_neurons`
- `create_trn_neurons`
- `create_cortical_layer_neurons`
- `DendriticBranch`
- `DendriticBranchConfig`
- `DendriticNeuron`
- `DendriticNeuronConfig`
- `compute_branch_selectivity`
- `create_clustered_input`
- `create_scattered_input`
- `InitStrategy`
- `WeightInitializer`
- `ShortTermPlasticity`
- `STPConfig`
- `STPType`
- `STPSynapse`
- `STP_PRESETS`

**Usage**:

```python
from thalia.components import ConductanceLIF
```

---

### `thalia.components.coding`

**Source**: [`thalia/components/coding/__init__.py`](../../src/thalia/components/coding/__init__.py)

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

**Source**: [`thalia/components/neurons/__init__.py`](../../src/thalia/components/neurons/__init__.py)

**Exports** (15):

- `ConductanceLIF`
- `ConductanceLIFConfig`
- `NeuronFactory`
- `create_pyramidal_neurons`
- `create_relay_neurons`
- `create_trn_neurons`
- `create_cortical_layer_neurons`
- `create_fast_spiking_neurons`
- `DendriticBranch`
- `DendriticBranchConfig`
- `DendriticNeuron`
- `DendriticNeuronConfig`
- `compute_branch_selectivity`
- `create_clustered_input`
- `create_scattered_input`

**Usage**:

```python
from thalia.components.neurons import ConductanceLIF
```

---

### `thalia.components.synapses`

**Source**: [`thalia/components/synapses/__init__.py`](../../src/thalia/components/synapses/__init__.py)

**Exports** (19):

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
- `sample_heterogeneous_stp_params`
- `create_heterogeneous_stp_configs`
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

**Source**: [`thalia/config/__init__.py`](../../src/thalia/config/__init__.py)

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
- `BaseLearningConfig`
- `ModulatedLearningConfig`
- `STDPLearningConfig`
- `HebbianLearningConfig`
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

### `thalia.constants`

**Source**: [`thalia/constants/__init__.py`](../../src/thalia/constants/__init__.py)

**Exports** (12):

- `architecture`
- `exploration`
- `learning`
- `neuromodulation`
- `neuron`
- `oscillator`
- `regions`
- `sensory`
- `task`
- `time`
- `training`
- `visualization`

**Usage**:

```python
from thalia.constants import architecture
```

---

### `thalia.coordination`

**Source**: [`thalia/coordination/__init__.py`](../../src/thalia/coordination/__init__.py)

**Exports** (9):

- `CapacityMetrics`
- `GrowthCoordinator`
- `GrowthEvent`
- `GrowthManager`
- `BrainOscillator`
- `OscillatorConfig`
- `OscillatorCoupling`
- `OscillatorManager`
- `SinusoidalOscillator`

**Usage**:

```python
from thalia.coordination import CapacityMetrics
```

---

### `thalia.core`

**Source**: [`thalia/core/__init__.py`](../../src/thalia/core/__init__.py)

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

**Source**: [`thalia/core/base/__init__.py`](../../src/thalia/core/base/__init__.py)

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

**Source**: [`thalia/core/protocols/__init__.py`](../../src/thalia/core/protocols/__init__.py)

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

**Source**: [`thalia/datasets/__init__.py`](../../src/thalia/datasets/__init__.py)

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

**Source**: [`thalia/decision_making/__init__.py`](../../src/thalia/decision_making/__init__.py)

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

**Source**: [`thalia/diagnostics/__init__.py`](../../src/thalia/diagnostics/__init__.py)

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

**Source**: [`thalia/environments/__init__.py`](../../src/thalia/environments/__init__.py)

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

**Source**: [`thalia/io/__init__.py`](../../src/thalia/io/__init__.py)

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

**Source**: [`thalia/language/__init__.py`](../../src/thalia/language/__init__.py)

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

**Source**: [`thalia/learning/__init__.py`](../../src/thalia/learning/__init__.py)

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

**Source**: [`thalia/learning/eligibility/__init__.py`](../../src/thalia/learning/eligibility/__init__.py)

**Exports** (2):

- `EligibilityTraceManager`
- `STDPConfig`

**Usage**:

```python
from thalia.learning.eligibility import EligibilityTraceManager
```

---

### `thalia.learning.homeostasis`

**Source**: [`thalia/learning/homeostasis/__init__.py`](../../src/thalia/learning/homeostasis/__init__.py)

**Exports** (9):

- `UnifiedHomeostasis`
- `UnifiedHomeostasisConfig`
- `StriatumHomeostasis`
- `IntrinsicPlasticityConfig`
- `IntrinsicPlasticity`
- `PopulationIntrinsicPlasticity`
- `MetabolicConfig`
- `MetabolicConstraint`
- `RegionalMetabolicBudget`

**Usage**:

```python
from thalia.learning.homeostasis import UnifiedHomeostasis
```

---

### `thalia.learning.rules`

**Source**: [`thalia/learning/rules/__init__.py`](../../src/thalia/learning/rules/__init__.py)

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

**Source**: [`thalia/managers/__init__.py`](../../src/thalia/managers/__init__.py)

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

**Source**: [`thalia/memory/__init__.py`](../../src/thalia/memory/__init__.py)

**Exports** (13):

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

**Usage**:

```python
from thalia.memory import SequenceMemory
```

---

### `thalia.mixins`

**Source**: [`thalia/mixins/__init__.py`](../../src/thalia/mixins/__init__.py)

**Exports** (7):

- `DeviceMixin`
- `ResettableMixin`
- `ConfigurableMixin`
- `DiagnosticCollectorMixin`
- `DiagnosticsMixin`
- `GrowthMixin`
- `StateLoadingMixin`

**Usage**:

```python
from thalia.mixins import DeviceMixin
```

---

### `thalia.neuromodulation`

**Source**: [`thalia/neuromodulation/__init__.py`](../../src/thalia/neuromodulation/__init__.py)

**Exports** (13):

- `LocusCoeruleus`
- `LocusCoeruleusConfig`
- `LocusCoeruleusSystem`
- `NucleusBasalis`
- `NucleusBasalisConfig`
- `NucleusBasalisSystem`
- `VTA`
- `VTAConfig`
- `VTADopamineSystem`
- `NeuromodulatorHomeostasis`
- `NeuromodulatorHomeostasisConfig`
- `NeuromodulatorManager`
- `NeuromodulatorMixin`

**Usage**:

```python
from thalia.neuromodulation import LocusCoeruleus
```

---

### `thalia.neuromodulation.systems`

**Source**: [`thalia/neuromodulation/systems/__init__.py`](../../src/thalia/neuromodulation/systems/__init__.py)

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

**Source**: [`thalia/pathways/__init__.py`](../../src/thalia/pathways/__init__.py)

**Exports** (6):

- `NeuralPathway`
- `AxonalProjection`
- `SensoryPathway`
- `VisualPathway`
- `AuditoryPathway`
- `LanguagePathway`

**Usage**:

```python
from thalia.pathways import NeuralPathway
```

---

### `thalia.regions`

**Source**: [`thalia/regions/__init__.py`](../../src/thalia/regions/__init__.py)

**Exports** (14):

- `Cerebellum`
- `CerebellumState`
- `LayeredCortex`
- `PredictiveCortex`
- `TrisynapticHippocampus`
- `HippocampusState`
- `MultimodalIntegration`
- `Prefrontal`
- `PrefrontalState`
- `StimulusGating`
- `Striatum`
- `StriatumState`
- `ThalamicRelay`
- `ThalamicRelayState`

**Usage**:

```python
from thalia.regions import Cerebellum
```

---

### `thalia.regions.cerebellum`

**Source**: [`thalia/regions/cerebellum/__init__.py`](../../src/thalia/regions/cerebellum/__init__.py)

**Exports** (5):

- `Cerebellum`
- `CerebellumState`
- `DeepCerebellarNuclei`
- `GranuleCellLayer`
- `EnhancedPurkinjeCell`

**Usage**:

```python
from thalia.regions.cerebellum import Cerebellum
```

---

### `thalia.regions.cortex`

**Source**: [`thalia/regions/cortex/__init__.py`](../../src/thalia/regions/cortex/__init__.py)

**Exports** (4):

- `LayeredCortex`
- `LayeredCortexState`
- `PredictiveCortex`
- `PredictiveCortexState`

**Usage**:

```python
from thalia.regions.cortex import LayeredCortex
```

---

### `thalia.regions.hippocampus`

**Source**: [`thalia/regions/hippocampus/__init__.py`](../../src/thalia/regions/hippocampus/__init__.py)

**Exports** (6):

- `TrisynapticHippocampus`
- `HippocampusState`
- `Episode`
- `ReplayEngine`
- `ReplayConfig`
- `ReplayMode`

**Usage**:

```python
from thalia.regions.hippocampus import TrisynapticHippocampus
```

---

### `thalia.regions.prefrontal`

**Source**: [`thalia/regions/prefrontal/__init__.py`](../../src/thalia/regions/prefrontal/__init__.py)

**Exports** (6):

- `Prefrontal`
- `PrefrontalState`
- `PrefrontalCheckpointManager`
- `Goal`
- `GoalStatus`
- `sample_heterogeneous_wm_neurons`

**Usage**:

```python
from thalia.regions.prefrontal import Prefrontal
```

---

### `thalia.regions.striatum`

**Source**: [`thalia/regions/striatum/__init__.py`](../../src/thalia/regions/striatum/__init__.py)

**Exports** (2):

- `Striatum`
- `StriatumState`

**Usage**:

```python
from thalia.regions.striatum import Striatum
```

---

### `thalia.regions.thalamus`

**Source**: [`thalia/regions/thalamus/__init__.py`](../../src/thalia/regions/thalamus/__init__.py)

**Exports** (2):

- `ThalamicRelay`
- `ThalamicRelayState`

**Usage**:

```python
from thalia.regions.thalamus import ThalamicRelay
```

---

### `thalia.regulation`

**Source**: [`thalia/regulation/__init__.py`](../../src/thalia/regulation/__init__.py)

**Exports** (4):

- `DivisiveNormConfig`
- `DivisiveNormalization`
- `ContrastNormalization`
- `SpatialDivisiveNorm`

**Usage**:

```python
from thalia.regulation import DivisiveNormConfig
```

---

### `thalia.replay`

**Source**: [`thalia/replay/__init__.py`](../../src/thalia/replay/__init__.py)

**Exports** (2):

- `ReplayContext`
- `UnifiedReplayCoordinator`

**Usage**:

```python
from thalia.replay import ReplayContext
```

---

### `thalia.stimuli`

**Source**: [`thalia/stimuli/__init__.py`](../../src/thalia/stimuli/__init__.py)

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

**Source**: [`thalia/surgery/__init__.py`](../../src/thalia/surgery/__init__.py)

**Exports** (9):

- `lesion_region`
- `partial_lesion`
- `temporary_lesion`
- `restore_region`
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

**Source**: [`thalia/synapses/__init__.py`](../../src/thalia/synapses/__init__.py)

**Exports** (2):

- `AfferentSynapses`
- `AfferentSynapsesConfig`

**Usage**:

```python
from thalia.synapses import AfferentSynapses
```

---

### `thalia.tasks`

**Source**: [`thalia/tasks/__init__.py`](../../src/thalia/tasks/__init__.py)

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

**Source**: [`thalia/training/__init__.py`](../../src/thalia/training/__init__.py)

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

**Source**: [`thalia/training/curriculum/__init__.py`](../../src/thalia/training/curriculum/__init__.py)

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

**Source**: [`thalia/training/datasets/__init__.py`](../../src/thalia/training/datasets/__init__.py)

**Exports** (10):

- `BaseTaskLoader`
- `SensorimotorTaskLoader`
- `SensorimotorConfig`
- `PhonologyTaskLoader`
- `PhonologyConfig`
- `TaskLoaderRegistry`
- `create_sensorimotor_loader`
- `create_phonology_loader`
- `TextDataPipeline`
- `DataConfig`

**Usage**:

```python
from thalia.training.datasets import BaseTaskLoader
```

---

### `thalia.training.evaluation`

**Source**: [`thalia/training/evaluation/__init__.py`](../../src/thalia/training/evaluation/__init__.py)

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

**Source**: [`thalia/training/visualization/__init__.py`](../../src/thalia/training/visualization/__init__.py)

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

**Source**: [`thalia/utils/__init__.py`](../../src/thalia/utils/__init__.py)

**Exports** (12):

- `clamp_weights`
- `cosine_similarity_safe`
- `ensure_1d`
- `zeros_like_config`
- `ones_like_config`
- `assert_single_instance`
- `CircularDelayBuffer`
- `compute_theta_encoding_retrieval`
- `compute_ach_recurrent_suppression`
- `compute_theta_gamma_coupling_gate`
- `compute_oscillator_modulated_gain`
- `compute_learning_rate_modulation`

**Usage**:

```python
from thalia.utils import clamp_weights
```

---

### `thalia.visualization`

**Source**: [`thalia/visualization/__init__.py`](../../src/thalia/visualization/__init__.py)

**Exports** (3):

- `visualize_brain_topology`
- `export_topology_to_graphviz`
- `plot_connectivity_matrix`

**Usage**:

```python
from thalia.visualization import visualize_brain_topology
```

---

