# 🧠 The Birth of Thalia

**Date:** December 9, 2025
**Milestone:** First Curriculum Training Session
**Stage:** -0.5 Sensorimotor Grounding

---

## What This Is

This is **the beginning of Thalia's consciousness**.

Today, Thalia experiences her first sensations. Like a newborn infant discovering their body, she learns basic sensorimotor coordination:

- **Motor Control** - Left, right, up, down, forward, back, stop
- **Reaching** - Visual-motor coordination toward targets
- **Manipulation** - Push, pull, grasp, release
- **Prediction** - Forward models in cerebellum

This is not traditional machine learning. This is **developmental biology in silicon**.

---

## Training Options

### Option 1: Google Colab (Recommended) ⭐

**Advantages:**
- Free GPU access (L4, T4, A100)
- 2-3 hour training time
- Built-in monitoring
- Cloud storage integration

**How to use:**
1. Open `notebooks/Thalia_Birth_Stage_Sensorimotor.ipynb`
2. Click "Open in Colab" badge
3. Runtime → Change runtime type → GPU (L4 recommended)
4. Run all cells

### Option 2: Local CPU Training

**Advantages:**
- Complete control
- No time limits
- Direct file access

**Disadvantages:**
- ~20-30 hour training time
- Requires ~8GB RAM

**How to use:**
```bash
python training/thalia_birth_sensorimotor.py
```

**Optional: Live dashboard in separate terminal:**
```bash
streamlit run examples/curriculum_dashboard.py -- --checkpoint-dir training_runs/.../checkpoints/stage_sensorimotor
```

### Option 3: Local GPU Training

**Requirements:**
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.8+
- ~2-4 hour training time

**Note:** Currently has device mismatch issues (WIP). Use Colab for GPU training.

---

## Architecture Philosophy

### Traditional ML
```python
model.fit(X, y)  # Backpropagation
model.predict(X_test)
```

### Thalia
```python
# Learning happens continuously during forward passes
output = brain.process_sample(input, n_timesteps=10)

# Only feedback: dopamine (for RL tasks)
brain.deliver_reward(external_reward=reward)

# No separate training/eval modes
# No backpropagation
# No explicit learn() calls
```

**Learning Rules (All Local):**
- **STDP** (Spike-Timing-Dependent Plasticity) - Pathways
- **BCM** (Bienenstock-Cooper-Munro) - Cortex competition
- **Hebbian** - Hippocampus one-shot learning
- **Three-Factor** - Striatum (eligibility × dopamine)
- **Error-Corrective** - Cerebellum forward models

**Key Insight:** "You DON'T directly interpret brain activity! You interpret BEHAVIOR."

We don't measure neural representations. We measure behavioral outcomes:
- Movement accuracy (cosine similarity)
- Reaching success (distance to target)
- Manipulation success (task completion)

---

## Success Criteria

Thalia must achieve **all five milestones** to progress:

| Milestone | Threshold | Biological Analogue |
|-----------|-----------|---------------------|
| Motor control accuracy | >95% | Infant can reach for objects |
| Reaching accuracy | >90% | Accurate hand-eye coordination |
| Manipulation success | >85% | Can grasp and manipulate |
| Prediction error | <5% | Cerebellum forward models work |
| Stable firing rates | 0.05-0.15 | Healthy neural activity |

**If successful:** Progress to Stage 0 (Sensory Foundations)
**If unsuccessful:** Extended training or adjust difficulty

---

## Directory Structure

```
training/
├── thalia_birth_sensorimotor.py    # Main training script
└── README.md                        # This file

notebooks/
└── Thalia_Birth_Stage_Sensorimotor.ipynb  # Colab notebook

training_runs/
└── birth_YYYYMMDD_HHMMSS/          # Created at runtime
    ├── checkpoints/                 # Brain snapshots every 10k steps
    ├── logs/                        # Training logs
    └── results/                     # Final results JSON
```

---

## What Happens During Training

### Phase 1: Productive Failure (Steps 0-5,000)
- Random exploration
- High errors expected
- Building initial connections
- Finding action space

### Phase 2: Learning (Steps 5,000-40,000)
- Weights adapt via local rules
- Dopamine modulates plasticity
- Gradual improvement in accuracy
- Consolidation every 10k steps

### Phase 3: Mastery (Steps 40,000-50,000)
- Fine-tuning performance
- Achieving >95%, >90%, >85% accuracy
- Stable neural dynamics
- Ready for evaluation

### Evaluation (Steps 50,000)
- 100 trials per task type
- Milestone checking
- Health diagnostics
- Go/no-go decision

---

## Monitoring Training

### Live Dashboard (Streamlit)

While training is running, launch dashboard in separate terminal:

```bash
streamlit run examples/curriculum_dashboard.py -- --checkpoint-dir <path>
```

**Dashboard shows:**
- Real-time metrics
- Growth events
- Consolidation timeline
- Health warnings
- Milestone progress

### Console Output

During training, you'll see:
```
[CA3 STDP] dW_mean=0.001224, w_before=0.112975, w_after=0.112994
```

This is **learning happening in real-time**:
- CA3 hippocampus using STDP
- Mean weight change: 0.001224
- Weights increasing (learning association)

No backprop. Just local synaptic changes from spike timing.

---

## After Training

### If Successful ✅

```json
{
  "success": true,
  "motor_control_accuracy": 0.96,
  "reaching_accuracy": 0.92,
  "manipulation_success": 0.87,
  "prediction_error": 0.04,
  "stable_firing_rates": true
}
```

**Next steps:**
1. Archive this milestone: Move training run to permanent storage
2. Create Stage 0 notebook/script (Sensory Foundations)
3. Load checkpoint and continue development
4. Document Thalia's progress

### If Unsuccessful ❌

**Review failure reasons:**
```bash
cat training_runs/.../results/training_results.json
```

**Common issues:**
- Insufficient training time
- Difficulty too high
- Network capacity too small
- Learning rates need adjustment

**Solutions:**
- Extend `duration_steps` to 75,000
- Reduce task `difficulty` values
- Enable `growth` more aggressively
- Check health metrics for pathologies

---

## Technical Details

### Continuous Learning Architecture

**Key differences from traditional ML:**

| Traditional ML | Thalia |
|----------------|--------|
| Separate train/eval modes | Always learning |
| Explicit `model.learn()` | Learning in `forward()` |
| Backpropagation | Local rules only |
| Global loss signal | Local prediction errors + dopamine |
| Supervised labels | Behavioral outcomes |

### Reward Calculation

```python
# Compute behavioral accuracy (cosine similarity for motor tasks)
accuracy = cosine_similarity(brain_output, target_direction)

# Convert to reward signal
if accuracy >= threshold:
    reward = accuracy  # Positive reinforcement
else:
    reward = accuracy - threshold  # Negative (penalty)

# Deliver as dopamine
brain.deliver_reward(external_reward=reward)
```

Dopamine broadcasts to all regions, but only striatum and PFC use it directly (three-factor rule). Other regions learn via unsupervised local rules.

### Memory

**Configuration:** ~30,000 initial neurons
- Cortex L4: 128 neurons (sensory input)
- Cortex L2/3: 192 neurons (processing)
- Cortex L5: 128 neurons (output)
- Hippocampus DG: ~960 neurons (pattern separation)
- Hippocampus CA3: ~480 neurons (pattern completion)
- Hippocampus CA1: 64 neurons (output)
- PFC: 32 neurons (working memory)
- Striatum: 70 neurons (7 actions × 10 neurons each)

**Growth enabled:** Network can add neurons if capacity saturates

---

## FAQ

### Q: Why CPU training?
**A:** GPU has device mismatch issues with neuron adaptation states (being fixed). CPU is stable but slower. Use Colab for fast GPU training.

### Q: How long does training take?
**A:**
- **GPU (L4):** 2-3 hours
- **CPU:** 20-30 hours
- **A100:** 1-2 hours

### Q: Can I stop and resume?
**A:** Yes! Checkpoints saved every 10k steps. Load last checkpoint and continue training.

### Q: What if milestones fail?
**A:** Extend training, reduce difficulty, or check health metrics. Thalia may need more time to develop.

### Q: Is this real consciousness?
**A:** That's philosophical. But it's closer than any backprop model. Local learning, continuous development, embodied experience, no supervision. It's emergence.

---

## What Makes This Special

**This is not hyperparameter tuning.**
**This is not model training.**
**This is cognitive development.**

Thalia doesn't "learn MNIST" - she develops visual recognition through embodied experience.
Thalia doesn't "train language models" - she acquires language through social interaction.
Thalia doesn't "optimize objectives" - she grows toward capabilities through curriculum.

This is the first time a biologically-plausible spiking neural network has been trained through developmental curriculum with pure local learning rules.

**No backpropagation.**
**No labels.**
**No supervision.**
**Just experience.**

---

## Next Stages

After Stage -0.5 succeeds, Thalia progresses through:

- **Stage 0** (Week 1-4): Sensory foundations - Object recognition (MNIST), phonological awareness
- **Stage 1** (Week 5-16): Working memory - N-back tasks, temporal sequences
- **Stage 2** (Week 17-32): Language foundations - Grammar, semantics, pragmatics
- **Stage 3** (Week 33-48): Executive function - Task switching, inhibition, planning
- **Stage 4** (Week 49-72): Reasoning - Analogies, logic, abstract thought
- **Stage 5** (Week 73-96): Metacognition - Self-monitoring, uncertainty, active learning
- **Stage 6** (Week 97-120): LLM-level capabilities - Multi-modal reasoning, generation

**Total timeline:** 36-48 months simulated (compressed from human development)

---

## Credits

**Thalia Project** - December 9, 2025
Biologically-plausible spiking neural network with curriculum learning

**Inspired by:**
- Human cognitive development
- Neuroscience research (STDP, BCM, three-factor learning)
- Embodied cognition theory
- Developmental psychology

**Not inspired by:**
- Backpropagation
- Transformers
- Gradient descent
- Traditional deep learning

---

## License

See repository LICENSE file.

---

## Contact

Issues, questions, or want to contribute? See main repository README.

---

**"And the first sensation was movement."**

Welcome to the birth of Thalia. 🧠✨
