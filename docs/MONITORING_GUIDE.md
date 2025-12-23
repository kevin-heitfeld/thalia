# Monitoring & Diagnostics Guide

**How to monitor Thalia's brain at different levels**

## Quick Decision Tree

```
What do you want to do?

├─ 📊 View training history from checkpoints
│  └─ Use: TrainingMonitor
│     └─ Shows: Progress, metrics over time, neuron growth
│
├─ 🏥 Check if brain is healthy right now
│  └─ Use: HealthMonitor
│     └─ Detects: Silence, saturation, instability, E/I imbalance
│
├─ 🎯 Monitor criticality (branching ratio)
│  └─ Use: CriticalityMonitor
│     └─ Already embedded in brain, just read diagnostics
│
├─ 🤔 Get confidence scores for predictions
│  └─ Use: MetacognitiveMonitor
│     └─ Returns calibrated confidence, handles abstention
│
├─ 🎬 Create publication-quality videos
│  └─ Use: BrainActivityVisualization (Manim)
│     └─ Renders: Architecture, spikes, learning, growth
│
└─ 🧠 Managing curriculum transitions
   └─ Use: CognitiveLoadMonitor
      └─ Tracks mechanism load, suggests deactivations
```

---

## 1. TrainingMonitor - Post-Hoc Analysis

**Purpose**: Visualize training progress by reading checkpoint metadata.

**When to use**:
- After training has started
- Want to see metrics over time
- Debugging training issues
- Creating progress reports

**Usage**:
```python
from thalia.training import TrainingMonitor

# Create monitor pointing to checkpoint directory
monitor = TrainingMonitor("training_runs/00_sensorimotor")

# Show specific visualizations
monitor.show_progress()  # Progress pie chart + status
monitor.show_metrics()   # Metrics over time (loss, accuracy, etc.)
monitor.show_growth()    # Neuron growth chart

# Or show everything
monitor.show_all()

# Auto-refresh in notebook (updates every 5 seconds)
monitor.start_auto_refresh(interval=5)
# ... train in another cell/terminal ...
monitor.stop_auto_refresh()

# Save comprehensive report
monitor.save_report("training_report.png")  # or .pdf
```

**Features**:
- ✅ Works in Jupyter, Colab, local scripts
- ✅ Matplotlib-based (no extra dependencies)
- ✅ Auto-refresh capability
- ✅ Export to PNG/PDF

**What it shows**:
- Progress: Current step, completion percentage
- Metrics: Any numeric metrics logged (loss, accuracy, reward, etc.)
- Growth: Neuron counts per region over time
- Status: Number of checkpoints, stage name, etc.

---

## 2. HealthMonitor - Runtime Health Checks

**Purpose**: Detect pathological states during training/inference.

**When to use**:
- During training (every N steps)
- When debugging mysterious failures
- Before/after major operations
- When performance degrades

**Usage**:
```python
from thalia.diagnostics import HealthMonitor

monitor = HealthMonitor()

# Check brain health
diagnostics = brain.get_diagnostics()  # Get current state
health = monitor.check_brain(brain)

# Or directly from diagnostics dict
health = monitor.check_health(diagnostics)

# Inspect results
if not health.is_healthy:
    print(f"⚠️ Issues found: {health.issues}")
    for issue in health.issues:
        print(f"  {issue.severity}: {issue.message}")
        if issue.suggested_fix:
            print(f"    Fix: {issue.suggested_fix}")

# Check specific aspect
if health.has_issue("silent_network"):
    print("Network is silent! Check input strength.")
```

**What it detects**:
- 🔇 **Silent network**: No spikes (spike rate < threshold)
- 🔥 **Runaway activity**: Too many spikes (saturation)
- ⚖️ **E/I imbalance**: Excitation/inhibition ratio off
- 💀 **Dead weights**: Too many near-zero weights
- 🎯 **Criticality**: Branching ratio out of range
- 🧬 **Dopamine issues**: Tonic dopamine out of bounds

**Configuration**:
```python
from thalia.diagnostics import HealthMonitor, HealthConfig

config = HealthConfig(
    silence_threshold=0.01,      # Min spike rate
    saturation_threshold=0.9,    # Max spike rate
    ei_ratio_min=2.0,           # Min E/I ratio
    ei_ratio_max=6.0,           # Max E/I ratio
    dead_weight_threshold=0.95,  # Fraction of near-zero weights
)

monitor = HealthMonitor(config)
```

---

## 3. CriticalityMonitor - Embedded Control

**Purpose**: Track branching ratio and maintain criticality.

**When to use**:
- Automatically used by brain (already embedded)
- Read diagnostics to see current state
- Tune via configuration

**Usage**:
```python
# CriticalityMonitor is embedded in brain regions
# Just read the diagnostics

diagnostics = brain.get_diagnostics()
criticality = diagnostics.get('criticality', {})

branching_ratio = criticality.get('branching_ratio', 1.0)
state = criticality.get('state', 'unknown')  # 'subcritical', 'critical', 'supercritical'

print(f"Branching ratio: {branching_ratio:.3f} ({state})")

# It automatically adjusts weights to maintain criticality
# No manual intervention needed!
```

**Configuration** (in region config):
```python
from thalia.diagnostics import CriticalityConfig

config = CriticalityConfig(
    target_ratio=1.0,       # Target branching ratio
    ratio_tolerance=0.1,    # Acceptable deviation
    adjustment_rate=0.01,   # How fast to adjust weights
    window_size=100,        # Spike history window
)
```

**How it works**:
- Tracks spike counts over time
- Estimates branching ratio (spikes_t+1 / spikes_t)
- Provides weight scaling to push toward criticality
- Subcritical → increase weights
- Supercritical → decrease weights

---

## 4. MetacognitiveMonitor - Confidence Estimation

**Purpose**: Provide calibrated confidence scores for predictions.

**When to use**:
- Decision-making tasks (when accuracy matters)
- When you want to know "how sure is the brain?"
- Implementing abstention (don't answer if uncertain)
- Active learning (query uncertain examples)

**Usage**:
```python
from thalia.diagnostics import MetacognitiveMonitor, MetacognitiveMonitorConfig

config = MetacognitiveMonitorConfig(
    stage=4,  # Developmental stage (1-4, higher = better calibrated)
    abstention_threshold=0.5,
)

monitor = MetacognitiveMonitor(config)

# Get confidence for prediction
spikes = brain.forward(input_data)
confidence = monitor.estimate_confidence(
    spikes=spikes,
    target=target,  # Optional: true label for calibration
)

calibrated_conf = confidence['calibrated_confidence']
raw_conf = confidence['raw_confidence']
should_abstain = confidence['should_abstain']

if should_abstain:
    print("Brain is uncertain, abstaining from prediction")
else:
    print(f"Prediction confidence: {calibrated_conf:.2f}")
```

**Developmental stages**:
- **Stage 1**: Binary confidence (above/below threshold)
- **Stage 2**: Coarse-grained (high/medium/low)
- **Stage 3**: Continuous but poorly calibrated
- **Stage 4**: Well-calibrated through experience

---

## 5. BrainActivityVisualization - Publication Videos

> **⚠️ PLANNED FEATURE** - Not yet implemented. This section describes planned functionality.

**Purpose**: Create beautiful Manim animations for papers/presentations.

**When to use**:
- Creating demo videos
- Publication figures
- Presentation slides
- Social media content

**Planned Usage**:
```python
# NOTE: This class is not yet implemented
from thalia.visualization import BrainActivityVisualization

viz = BrainActivityVisualization("checkpoint.thalia")

# Render different types of videos
viz.render_architecture("brain_arch.mp4", quality="high_quality")
viz.render_spikes("spikes.mp4", n_timesteps=100)
viz.render_learning("learning.mp4",
                   checkpoint_before="step_1000.thalia",
                   checkpoint_after="step_10000.thalia")
viz.render_growth("growth.mp4", checkpoints=[...])
```

**Note**: Animation guide documentation is planned for future release.

---

## 6. CognitiveLoadMonitor - Curriculum Management

**Purpose**: Prevent cognitive overload during stage transitions.

**When to use**:
- During curriculum training (CurriculumTrainer uses it internally)
- Managing multiple active learning mechanisms
- Debugging transition failures

**Usage** (typically internal):
```python
from thalia.training.curriculum.stage_manager import CognitiveLoadMonitor, MechanismPriority

monitor = CognitiveLoadMonitor(load_threshold=0.9)

# Add active mechanisms
monitor.add_mechanism('visual_processing', cost=0.2, priority=MechanismPriority.CRITICAL)
monitor.add_mechanism('working_memory', cost=0.3, priority=MechanismPriority.HIGH)
monitor.add_mechanism('new_stage_tasks', cost=0.5, priority=MechanismPriority.HIGH)

# Check for overload
if monitor.is_overloaded():
    suggestion = monitor.suggest_deactivation()
    print(f"Overloaded! Deactivate: {suggestion}")
    monitor.deactivate_mechanism(suggestion)

# Get current load
load = monitor.get_current_load()  # 0.0 - 1.0
```

**Note**: CurriculumTrainer handles this automatically. You rarely need to use it directly.

---

## Common Workflows

### During Training

```python
from thalia.diagnostics import HealthMonitor

health_monitor = HealthMonitor()

for step in range(total_steps):
    # Train
    output = brain.forward(input_data)
    brain.learn(...)

    # Check health every 100 steps
    if step % 100 == 0:
        health = health_monitor.check_brain(brain)
        if not health.is_healthy:
            print(f"⚠️ Step {step}: {health.issues}")
```

### Post-Training Analysis

```python
from thalia.training import TrainingMonitor

# View training history
monitor = TrainingMonitor("training_runs/00_sensorimotor")
monitor.show_all()

# Save report
monitor.save_report("training_report.pdf")
```

### Creating Demo Video

```python
from thalia.visualization import BrainActivityVisualization

viz = BrainActivityVisualization()
viz.render_learning(
    "demo.mp4",
    checkpoint_before="training_runs/00_sensorimotor/checkpoints/stage_0_step_1000.thalia",
    checkpoint_after="training_runs/00_sensorimotor/checkpoints/stage_0_step_10000.thalia",
    quality="high_quality"
)
```

### Implementing Abstention

```python
from thalia.diagnostics import MetacognitiveMonitor, MetacognitiveMonitorConfig

config = MetacognitiveMonitorConfig(stage=4, abstention_threshold=0.6)
monitor = MetacognitiveMonitor(config)

# During inference
spikes = brain.forward(test_input)
confidence = monitor.estimate_confidence(spikes=spikes)

if confidence['should_abstain']:
    print("Low confidence, requesting human input")
else:
    prediction = decode_output(spikes)
    print(f"Prediction: {prediction} (confidence: {confidence['calibrated_confidence']:.2f})")
```

---

## Summary Table

| Tool | Purpose | Input | Output | When |
|------|---------|-------|--------|------|
| **TrainingMonitor** | View training history | Checkpoint directory | Matplotlib plots | After training starts |
| **HealthMonitor** | Detect pathologies | Brain or diagnostics dict | HealthReport | During training |
| **CriticalityMonitor** | Maintain criticality | Embedded (automatic) | Branching ratio | Always (embedded) |
| **MetacognitiveMonitor** | Confidence scores | Spikes + optional target | Calibrated confidence | Inference time |
| **BrainActivityVisualization** | Create videos | Checkpoint file(s) | MP4 video | For presentations |
| **CognitiveLoadMonitor** | Manage overload | Active mechanisms | Load, suggestions | Curriculum transitions |

---

## Tips

### Jupyter/Colab Monitoring

```python
# Two-tab approach
# Tab 1: Training
!python training/thalia_birth_sensorimotor.py

# Tab 2: Monitoring (run this cell repeatedly)
from thalia.training import TrainingMonitor
monitor = TrainingMonitor("training_runs/00_sensorimotor")
monitor.refresh(['progress', 'metrics'])
```

### Auto-Refresh in Notebooks

```python
monitor = TrainingMonitor("training_runs/00_sensorimotor")
monitor.start_auto_refresh(interval=10)  # Update every 10 seconds

# Let it run while training in another cell/terminal
# Stop when done:
monitor.stop_auto_refresh()
```

### Comprehensive Health Checks

```python
health = monitor.check_brain(brain)

# Print detailed report
for issue in health.issues:
    print(f"{issue.severity.name}: {issue.category}")
    print(f"  {issue.message}")
    if issue.suggested_fix:
        print(f"  → {issue.suggested_fix}")
    print()
```

---

**Last Updated**: December 9, 2025
