# Training Module Reorganization - Migration Guide

**Date**: December 12, 2025  
**Change Type**: Breaking (import paths changed)  
**Status**: Completed

## Overview

The training module has been reorganized into four focused subdirectories for better code organization and discoverability:

```
training/
  curriculum/    - Stage management, curriculum strategies, evaluation
  datasets/      - Task loaders, data pipeline, constants  
  evaluation/    - Metacognition, metrics
  visualization/ - Monitoring, live diagnostics
```

## What Changed

### File Moves and Renames

| Old Location | New Location | Notes |
|-------------|-------------|-------|
| `training/curriculum_trainer.py` | `training/curriculum/stage_manager.py` | Renamed for clarity |
| `training/curriculum_logger.py` | `training/curriculum/logger.py` | Shorter name |
| `training/stage_evaluation.py` | `training/curriculum/stage_evaluation.py` | Moved to curriculum |
| `training/curriculum.py` | `training/curriculum/curriculum.py` | Moved to curriculum |
| `training/task_loaders.py` | `training/datasets/loaders.py` | Clearer context |
| `training/task_constants.py` | `training/datasets/constants.py` | Shorter name |
| `training/data_pipeline.py` | `training/datasets/pipeline.py` | Grouped with datasets |
| `training/metacognition.py` | `training/evaluation/metacognition.py` | Moved to evaluation |
| `training/monitor.py` | `training/visualization/monitor.py` | Moved to visualization |
| `training/live_diagnostics.py` | `training/visualization/live_diagnostics.py` | Moved to visualization |

## Migration Path

### Option 1: Use Main Module Re-exports (Recommended, No Code Changes)

All public APIs are still available from `thalia.training`:

```python
# This still works - NO CHANGES NEEDED
from thalia.training import (
    CurriculumTrainer,
    SensorimotorTaskLoader,
    MetacognitiveCalibrator,
    TrainingMonitor,
    LiveDiagnostics,
    TextDataPipeline,
)
```

### Option 2: Import from Subdirectories (More Explicit)

You can now import from organized subdirectories:

```python
# Curriculum components
from thalia.training.curriculum import (
    CurriculumTrainer,
    StageConfig,
    evaluate_stage_sensorimotor,
    CurriculumLogger,
)

# Dataset components
from thalia.training.datasets import (
    BaseTaskLoader,
    SensorimotorTaskLoader,
    SPIKE_PROBABILITY_LOW,
    TextDataPipeline,
)

# Evaluation components
from thalia.training.evaluation import (
    MetacognitiveCalibrator,
    CalibrationMetrics,
)

# Visualization components
from thalia.training.visualization import (
    TrainingMonitor,
    quick_monitor,
    LiveDiagnostics,
)
```

### Breaking Changes

Direct imports of old module names will fail:

```python
# ❌ BROKEN - Old import style
from thalia.training.curriculum_trainer import CurriculumTrainer
from thalia.training.task_loaders import SensorimotorTaskLoader
from thalia.training.task_constants import SPIKE_PROBABILITY_LOW

# ✅ FIXED - Option 1: Use main module
from thalia.training import CurriculumTrainer, SensorimotorTaskLoader
from thalia.training.datasets import SPIKE_PROBABILITY_LOW

# ✅ FIXED - Option 2: Use subdirectory
from thalia.training.curriculum.stage_manager import CurriculumTrainer
from thalia.training.datasets.loaders import SensorimotorTaskLoader
from thalia.training.datasets.constants import SPIKE_PROBABILITY_LOW
```

## Quick Reference

### Import Mapping

| Old Import | New Import (Subdirectory) | Import (Main Module) |
|-----------|--------------------------|---------------------|
| `from thalia.training.curriculum_trainer import CurriculumTrainer` | `from thalia.training.curriculum.stage_manager import CurriculumTrainer` | `from thalia.training import CurriculumTrainer` |
| `from thalia.training.task_loaders import BaseTaskLoader` | `from thalia.training.datasets.loaders import BaseTaskLoader` | `from thalia.training import BaseTaskLoader` |
| `from thalia.training.task_constants import SPIKE_PROBABILITY_LOW` | `from thalia.training.datasets.constants import SPIKE_PROBABILITY_LOW` | Use subdirectory (not re-exported) |
| `from thalia.training.metacognition import MetacognitiveCalibrator` | `from thalia.training.evaluation.metacognition import MetacognitiveCalibrator` | `from thalia.training import MetacognitiveCalibrator` |
| `from thalia.training.monitor import TrainingMonitor` | `from thalia.training.visualization.monitor import TrainingMonitor` | `from thalia.training import TrainingMonitor` |

## Benefits

### 1. Clear Separation of Concerns
- **curriculum/**: Everything related to training stages and curriculum strategies
- **datasets/**: All data loading, task generation, and constants
- **evaluation/**: Metrics, metacognition, assessment tools
- **visualization/**: Monitoring and diagnostic displays

### 2. Improved Discoverability
- Looking for dataset constants? Check `datasets/constants.py`
- Looking for stage evaluation? Check `curriculum/stage_evaluation.py`
- Looking for monitoring tools? Check `visualization/monitor.py`

### 3. Independent Development
- Curriculum team can work in `curriculum/` without affecting visualization
- Dataset additions don't touch evaluation code
- Cleaner git history and easier code reviews

### 4. Better Documentation Structure
Each subdirectory has its own `__init__.py` with focused documentation:
- `curriculum/__init__.py`: Explains curriculum mechanics
- `datasets/__init__.py`: Explains data loading approach
- `evaluation/__init__.py`: Explains evaluation tools
- `visualization/__init__.py`: Explains monitoring tools

## Testing

All imports verified with test script:

```python
# Main module imports
from thalia.training import (
    CurriculumTrainer, SensorimotorTaskLoader,
    MetacognitiveCalibrator, TrainingMonitor
)

# Subdirectory imports
from thalia.training.curriculum import CurriculumTrainer
from thalia.training.datasets import BaseTaskLoader
from thalia.training.evaluation import MetacognitiveCalibrator
from thalia.training.visualization import TrainingMonitor
```

All tests pass! ✓

## Files Affected

### Created
- `training/curriculum/__init__.py`
- `training/datasets/__init__.py`
- `training/evaluation/__init__.py`
- `training/visualization/__init__.py`

### Moved/Renamed
- 10 Python files reorganized into subdirectories

### Updated
- `training/__init__.py` (updated to import from subdirectories)
- `training/thalia_birth_sensorimotor.py` (import path updated)
- `tasks/sensorimotor.py` (import path updated)
- `tasks/working_memory.py` (import path updated)

## Rollback

If needed, rollback is possible by:
1. Moving files back to original locations
2. Reverting import path changes
3. Deleting subdirectory `__init__.py` files

However, **rollback is not recommended** as the new structure provides significant organizational benefits.

## Questions?

See the architecture review document for full rationale:
- `docs/reviews/architecture-review-2025-12-12.md` (Section T3.4)

---

**Summary**: Most code requires **no changes** - use imports from `thalia.training` as before. Only direct module imports (e.g., `curriculum_trainer`) need updating.
