# Training Framework Implementation Plan

## Current State

The brain has passed calibration with **0 health criticals** under random input (51,478 neurons, 25 regions, 135 axonal tracts). All learning rules are implemented (STDP, BCM, three-factor RL, D1/D2, cerebellar Marr-Albus-Ito, predictive coding, inhibitory STDP, metaplasticity, tag-and-capture). The diagnostics subsystem is mature. The `training/` directory is empty.

What's missing is the training loop itself: a way to present tasks, read output, compute reward, run diagnostics, and iterate.

---

## Architecture

```
training/
    __init__.py
    trainer.py              # Core training loop
    trial.py                # Single-trial execution
    tasks/
        __init__.py
        base.py             # Task protocol (abstract)
        pattern_association.py  # First task
    encoding/
        __init__.py
        spike_encoder.py    # Stimulus → spike trains
        spike_decoder.py    # Spike trains → behavioural readout
    monitoring/
        __init__.py
        health_monitor.py   # Lightweight per-trial health checks
        training_logger.py  # Metrics logging (CSV + optional wandb)
    checkpointing/
        __init__.py
        checkpoint.py       # Brain state save/restore
```

---

## Component Design

### 1. Task Protocol

Every training task implements a common interface:

```python
class Task(Protocol):
    name: str

    def generate_trial(self, trial_idx: int) -> Trial:
        """Return stimulus, expected behaviour, and trial duration."""
        ...

    def evaluate(self, trial: Trial, readout: SpikeReadout) -> TrialResult:
        """Score the brain's response. Return reward + metrics."""
        ...

    def is_learned(self, history: list[TrialResult]) -> bool:
        """Criterion check: has the brain learned this task?"""
        ...
```

A `Trial` is a dataclass holding:
- `stimuli: list[SynapticInput | None]` — one entry per timestep (or a generator for long trials)
- `duration_steps: int`
- `target` — task-specific expected output
- `metadata: dict` — anything the task needs to carry through to evaluation

A `TrialResult` holds:
- `reward: float` — scalar for `brain.deliver_reward()`
- `correct: bool`
- `metrics: dict` — task-specific (e.g., reaction time, accuracy)

### 2. Spike Encoding

Converts task stimuli into `SynapticInput` dictionaries targeting thalamic relay neurons (the brain's sensory gateway):

- **Rate coding**: stimulus intensity → Poisson spike probability per neuron per timestep
- **Population coding**: different stimulus identities → different subsets of relay neurons
- **Temporal coding**: stimulus onset/offset as precise spike volleys

For the first task, rate-coded population encoding is sufficient: each pattern activates a distinct subset of thalamic relay neurons at a fixed rate.

### 3. Spike Decoding

Reads the brain's "behavioural output" from designated readout populations. Candidate readout sites:
- **Cortex L5 pyramidal cells** — the biological output layer (projects to subcortical targets)
- **Striatum D1/D2 MSNs** — for action-selection tasks
- **Prefrontal cortex L5** — for working memory / decision tasks

Decoding approach:
- **Windowed spike count**: count spikes in a response window (e.g., last 200 ms of trial) per neuron subset
- **Population vote**: which readout population is most active → the chosen response
- No trained decoder — the decoding must be a fixed, simple readout so all learning happens inside the brain

### 4. Training Loop (Trainer)

```python
def train(brain, task, config):
    monitor = HealthMonitor(brain, config.monitor)
    logger  = TrainingLogger(config.log_dir)

    for epoch in range(config.n_epochs):
        trial = task.generate_trial(epoch)

        # ── Run trial ──
        spike_log = run_trial(brain, trial, monitor)

        # ── Evaluate ──
        readout = decode_spikes(spike_log, config.readout)
        result  = task.evaluate(trial, readout)

        # ── Deliver reward (one-step lag) ──
        brain.deliver_reward(result.reward)
        brain.forward()  # VTA processes reward

        # ── Log ──
        logger.log_trial(epoch, result, monitor.summary())

        # ── Periodic full diagnostics ──
        if epoch % config.diagnostics_interval == 0:
            report = monitor.run_full_diagnostics()
            logger.log_diagnostics(epoch, report)
            if report.health.n_critical > 0:
                logger.log_alert(epoch, report.health.critical_issues)
                # Optional: pause and dump snapshot for investigation

        # ── Checkpoint ──
        if epoch % config.checkpoint_interval == 0:
            save_checkpoint(brain, epoch, logger.metrics)

        # ── Convergence ──
        if task.is_learned(logger.recent_results(config.convergence_window)):
            logger.log_event(epoch, "TASK_LEARNED")
            break
```

### 5. Health Monitor

Two tiers of monitoring, integrated directly into the training loop:

**Tier 1 — Per-trial lightweight checks (every trial, <50 ms overhead):**
- Population firing rates (mean, min, max) from spike counts
- Silent population detection (any population with 0 spikes)
- Hyperactive population detection (any >100 Hz)
- E/I spike ratio per region
- Reward signal magnitude (VTA DA firing)

Implementation: accumulate spike counts during `run_trial()`, compute statistics directly from counts without creating a full snapshot.

**Tier 2 — Periodic full diagnostics (every N trials, ~10 s overhead):**
- Create a `DiagnosticsRecorder` with windowed recording
- Run a short diagnostic episode (e.g., 1000-step probe trial with known input)
- Call `analyze(snapshot)` for the full health report
- Log all metrics; alert on any new criticals

The key insight: Tier 2 runs a **separate probe trial** that doesn't interfere with training. This gives a clean snapshot of network health comparable to our calibration diagnostics.

```python
class HealthMonitor:
    def __init__(self, brain, config):
        self.brain = brain
        self.config = config
        self._trial_spike_counts = {}  # Tier 1 accumulator

    def record_step(self, t, outputs):
        """Tier 1: accumulate spike counts (called every timestep)."""
        for region, pops in outputs.items():
            for pop, spikes in pops.items():
                key = (region, pop)
                self._trial_spike_counts[key] = (
                    self._trial_spike_counts.get(key, 0) + int(spikes.sum())
                )

    def summary(self) -> TrialHealthSummary:
        """Tier 1: compute lightweight health metrics for this trial."""
        ...

    def run_full_diagnostics(self) -> DiagnosticsReport:
        """Tier 2: run a probe trial with full recording and analysis."""
        recorder = DiagnosticsRecorder(self.brain, DiagnosticsConfig(
            n_timesteps=1000,
            voltage_sample_size=8,
            conductance_sample_size=4,
            conductance_sample_interval_steps=5,
        ))
        # Store learning state, run probe, restore
        for t in range(1000):
            inputs = make_sensory_input(self.brain, t, "random", n_timesteps=1000)
            outputs = self.brain.forward(inputs)
            recorder.record(t, inputs, outputs)
        return analyze(recorder.to_snapshot())
```

### 6. Checkpointing

Save and restore complete brain state:
- All weight matrices (dense intra-region + sparse inter-region)
- Homeostatic gain values
- Membrane voltages, synaptic traces, eligibility traces
- STP state (x, u per synapse)
- Metaplasticity consolidation state
- Epoch counter, task progress, RNG state

This enables:
- Resume after interruption
- Rollback if training destabilises
- A/B comparison of training strategies from the same checkpoint

---

## First Training Task: Thalamic Pattern Association

### Why this task

The first task must validate the full pipeline — input encoding, spike propagation through thalamus→cortex, reward-modulated learning, and output decoding — while being simple enough that failure is diagnosable. Pattern association is ideal because:

1. **It exercises the thalamocortical pathway** that we've just calibrated
2. **It requires associative learning** (STDP + reward modulation) — the core of what Thalia is built for
3. **Success is unambiguous** — either the correct readout population wins or it doesn't
4. **It's the biological analogue of classical conditioning** — the simplest form of learned association

### Task description

**Two-pattern discrimination**: The brain receives one of two distinct spatial patterns (A or B) as thalamic relay input. After a response window, it must produce differential activity in two designated readout neuron subsets — subset 1 should be more active for pattern A, subset 2 for pattern B.

```
Pattern A (400 ms):
  - Relay neurons 0–124 active at ~10 Hz Poisson
  - Relay neurons 125–249 silent

Pattern B (400 ms):
  - Relay neurons 0–124 silent
  - Relay neurons 125–249 active at ~10 Hz Poisson

Response window: last 200 ms of trial
Readout: cortex_association L5 pyramidal cells, split into two halves
Reward: +1.0 if correct half is more active, -0.5 if wrong, 0.0 if ambiguous
```

### Why L5 pyramidal readout

L5 pyramidal cells are the cortical output layer — they project to striatum, brainstem, and thalamus. They receive both direct L4 feedforward input (carrying the thalamic pattern) and recurrent L2/3 input. Their activity reflects the cortex's "decision." Using them as readout is biologically motivated and requires no artificial readout layer.

### Learning mechanism

The expected learning pathway:
1. Pattern A activates relay subset → L4 pyr in cortex_assoc → L2/3 → L5
2. Initially, L5 response is random (no preference for either half)
3. Reward signal (+1.0 / -0.5) reaches VTA → DA broadcast to cortex
4. Three-factor learning (eligibility × DA) strengthens pathways from the active pattern to the rewarded L5 subset
5. STDP refines temporal structure; tag-and-capture consolidates
6. Over trials, pattern A preferentially drives L5 subset 1; pattern B drives L5 subset 2

### Success criterion

- **Accuracy > 80% over 100 consecutive trials** (chance = 50%)
- If accuracy plateaus below 60% after 500 trials → learning pathway is broken
- If accuracy oscillates wildly → reward signal or eligibility trace misconfigured

### Trial structure

```
[0–50 ms]     Spontaneous (no input) — baseline
[50–450 ms]   Pattern presentation (A or B, randomised)
[450–500 ms]  Settling (no input)
───────────────────────────────────────
[300–500 ms]  Response window for spike counting
[500 ms]      Trial end → evaluate → deliver reward
[500–510 ms]  Reward processing (brain.forward() for 10 steps with reward queued)
[510–610 ms]  Inter-trial interval (spontaneous activity, consolidation)
```

Total: 610 steps (610 ms) per trial. At ~155 µs/step this is ~95 ms wall-clock per trial, so 1000 trials ≈ 95 seconds.

### What this validates

| Aspect | What success proves |
|--------|-------------------|
| Thalamocortical relay | Patterns propagate from thalamus to cortex |
| Cortical processing | L4→L2/3→L5 feedforward pathway is functional |
| Reward pathway | VTA DA reaches cortex and modulates learning |
| Three-factor learning | Eligibility + DA → weight change in correct direction |
| STDP | Temporal structure emerges (pattern-selective timing) |
| Homeostasis | Firing rates stay stable as weights change |
| Whole-brain integration | 25 regions stay healthy during learning |

### Failure modes and diagnostics

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| No learning (50% accuracy) | DA not reaching cortex, or eligibility traces decaying too fast | Check VTA→cortex connectivity; increase eligibility τ |
| Catastrophic forgetting (learns then forgets) | Metaplasticity too aggressive, or homeostasis undoing weight changes | Reduce metaplasticity consolidation rate; check homeostatic time constants |
| One-sided learning (always picks A) | Reward asymmetry or initial weight bias | Check reward delivery; verify initial weight symmetry |
| Epileptiform during training | Weight growth outpaces inhibition | Tier 2 diagnostics will catch this; add weight decay or reduce learning rate |
| Silent readout | L5 pyramidal cells not receiving enough drive | Check L4→L5, L2/3→L5 connectivity weights |

---

## Implementation Sequence

### Phase 1: Minimal training loop (get something running)

1. **`training/tasks/base.py`** — Task protocol, Trial, and TrialResult dataclasses
2. **`training/encoding/spike_encoder.py`** — Population rate encoder (pattern → SynapticInput)
3. **`training/encoding/spike_decoder.py`** — Windowed spike-count decoder
4. **`training/trial.py`** — `run_trial()` function (step brain through a trial, collect spikes)
5. **`training/tasks/pattern_association.py`** — Two-pattern discrimination task
6. **`training/trainer.py`** — Main training loop with `brain.deliver_reward()`
7. Call `brain.set_learning_disabled(False)` before training runs

Validate: run 100 trials, confirm brain stays healthy (no epileptiform), and reward delivery works (VTA DA spikes after reward).

### Phase 2: Monitoring integration

8. **`training/monitoring/health_monitor.py`** — Tier 1 per-trial + Tier 2 periodic full diagnostics
9. **`training/monitoring/training_logger.py`** — CSV logging of accuracy, reward, firing rates, health metrics per trial
10. Integrate monitor into trainer loop
11. Add early stopping on critical health issues

Validate: run 500 trials, confirm Tier 1 metrics logged, Tier 2 diagnostics run every 100 trials with no new criticals.

### Phase 3: Learning validation

12. Run 1000+ trials with learning enabled
13. Plot learning curve (accuracy vs trial)
14. If accuracy stays at chance: diagnose with Tier 2, check DA pathway, eligibility traces
15. Tune: learning rates, eligibility decay, reward magnitude, trial timing
16. **`training/checkpointing/checkpoint.py`** — save/restore for rollback and comparison

### Phase 4: Robustification

17. Inter-trial spontaneous activity period (consolidation window)
18. Curriculum interleaving (randomise pattern order, vary difficulty)
19. Add 4-pattern variant (A/B/C/D) once 2-pattern is learned
20. Memory consolidation probes (periodic test without reward to measure retention)
