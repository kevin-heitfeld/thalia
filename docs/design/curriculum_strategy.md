# Thalia Curriculum Training Strategy

**Version**: 0.1.0  
**Status**: Design Phase  
**Last Updated**: December 7, 2025

> **Related Document**: See [`checkpoint_format.md`](checkpoint_format.md) for checkpoint format specification and state management.

## Overview

Progressive training strategy to grow a biologically-plausible brain from basic pattern recognition to LLM-level multi-modal capabilities. Inspired by human cognitive development: start simple, gradually increase complexity, consolidate knowledge at each stage.

## Philosophy

**Core Principle**: The brain should naturally discover complexity through incremental challenges, not have it forced upon it.

- **Start Tiny**: Begin with minimal capacity (10k-50k neurons)
- **Grow Organically**: Add capacity only when current tasks demand it
- **Consolidate Often**: Strengthen important circuits between stages
- **Never Forget**: Protect learned knowledge during curriculum transitions
- **Measure Progress**: Clear success criteria for each stage

## Developmental Stages

### Stage -1: Sanity Checks (Pre-Training Validation)
**Goal**: Verify all components work correctly before curriculum training

**Status**: ✅ **Already covered by existing test suite** (see `tests/unit/` and `tests/integration/`)

**Core Component Tests** (from `test_core.py`):
- ✅ LIF neurons generate spikes with proper input
- ✅ Membrane potentials decay toward rest
- ✅ Spike reset works correctly
- ✅ Conductance-based neurons respect reversal potentials
- ✅ Threshold crossings produce binary spikes

**Region Function Tests** (from `test_brain_regions.py`):
- ✅ All regions initialize properly
- ✅ Forward passes produce valid outputs
- ✅ Learning rules apply correctly per region type
- ✅ Striatum: three-factor learning with eligibility traces
- ✅ Cerebellum: error-corrective learning
- ✅ Hippocampus: episodic encoding/retrieval

**Oscillator Tests** (from `test_oscillator.py`):
- ✅ Theta oscillator runs at correct frequency (8 Hz)
- ✅ Gamma oscillator synchronizes properly
- ✅ Phase advancement is accurate
- ✅ Frequency modulation works

**Learning Mechanism Tests** (from `test_robustness.py`, `test_validation.py`):
- ✅ Weights change in expected direction
- ✅ E/I balance regulation prevents runaway
- ✅ Homeostatic mechanisms stabilize firing
- ✅ No gradient explosion or vanishing
- ✅ BCM thresholds adapt properly

**Additional Validation** (before Stage 0):
```python
# Run these quick checks before starting curriculum:
def pre_training_validation(brain):
    """Quick sanity checks before curriculum training."""
    
    # 1. Random input produces spikes
    random_input = torch.randn(1, brain.input_size) * 0.5
    output = brain.forward(random_input)
    assert output['spikes'].sum() > 0, "No spikes from random input"
    
    # 2. Constant input stabilizes firing rates
    for _ in range(100):
        brain.forward(torch.ones(1, brain.input_size) * 0.3)
    firing_rate = brain.get_firing_rate()
    assert 0.01 < firing_rate < 0.5, f"Unstable firing: {firing_rate}"
    
    # 3. Learning rules modify weights
    initial_weights = brain.get_weights().clone()
    for _ in range(50):
        brain.forward(random_input)
        brain.learn(reward=1.0)
    assert not torch.allclose(brain.get_weights(), initial_weights), "No learning"
    
    # 4. Oscillators run at correct frequencies
    if hasattr(brain, 'theta'):
        assert 7.5 < brain.theta.frequency_hz < 8.5, "Theta frequency off"
    
    print("✅ All sanity checks passed")
```

---

### Stage 0: Sensory Foundations (Infant Brain)
**Goal**: Learn basic sensory processing and pattern recognition

**Initial Size**: 50,000 neurons
- Cortex L4: 20,000 (primary sensory)
- Cortex L2/3: 15,000 (feature integration)
- Thalamus: 10,000 (sensory relay)
- Hippocampus: 5,000 (simple associations)

**Tasks**:
1. **Visual Pattern Recognition** (Week 1-2)
   - Simple shapes (circles, squares, triangles)
   - Binary classification
   - MNIST digits (grayscale 28x28)
   - Success: >95% accuracy on MNIST

2. **Temporal Patterns** (Week 2-3)
   - Simple sequences (A-B-C patterns)
   - Rhythm detection
   - Next-item prediction
   - Success: >90% next-item accuracy

3. **Audio Features** (Week 3-4)
   - Frequency discrimination
   - Simple tone sequences
   - Phoneme boundaries
   - Success: Distinguish 10 phonemes >85%

**CRITICAL Success Criteria** (Stage 0 must be rock-solid):
- ✅ Task performance: MNIST >95%, sequences >90%, phonemes >85%
- ✅ **Firing rate stability**: 0.05-0.15 maintained across 100k steps
- ✅ **No runaway excitation**: Criticality monitor shows stable/critical state (not supercritical)
- ✅ **BCM convergence**: Thresholds stabilize and stop drifting after 50k steps
- ✅ **Striatum balance**: D1/D2 weights maintain opponent relationship
- ✅ **No silence**: No region drops below 0.01 firing rate for >1000 steps
- ✅ **Weight health**: <80% of synapses saturated at min/max

**Why Stage 0 is Critical**:
If these foundations aren't stable, every later stage will inherit instabilities.
Better to spend extra time here than debug cascading failures in Stage 3.

**Training Details**:
- Batch size: 1 (single trial learning)
- Learning rate: Adaptive per region (dopamine-modulated)
- Steps per task: 10,000-50,000
- Checkpoint: Every 5,000 steps

**Expected Growth**: +10,000 neurons (20% increase)
- Primarily in cortex layers (sensory demand)

**Growth Decision Function**:
```python
def should_grow_region(region, observation_window=5000):
    """
    Decide if region needs more capacity.
    
    Returns True only if ALL conditions met:
    - High sustained activity (not just momentary spike)
    - Weights saturated (learning constrained)
    - Performance plateaued (not improving)
    - Task distribution stable (not in transition)
    """
    return (
        region.firing_rate > 0.25 and  # Sustained high activity
        region.weight_saturation > 0.85 and  # Weights nearly maxed
        region.performance_plateau(steps=observation_window) and  # Not improving
        not region.recent_task_change(steps=10000)  # Stable task
    )
```

---

### Stage 1: Object Permanence & Working Memory (Toddler Brain)
**Goal**: Develop working memory and object tracking

**Size**: ~70,000 neurons (from Stage 0 growth)
- Add Prefrontal: 10,000 neurons (working memory)
- Expand Hippocampus: +5,000 (object associations)

**Tasks**:
1. **Object Recognition** (Week 5-6)
   - CIFAR-10 (32x32 color images, 10 classes)
   - Multiple objects per image
   - Viewpoint invariance
   - Success: >70% accuracy on CIFAR-10

2. **Working Memory** (Week 6-7)
   - N-back tasks (N=1,2,3)
   - Delayed match-to-sample
   - Hold information for 100-500ms
   - Success: >80% on 2-back task

3. **Bilingual Language Foundations** (Week 7-8)
   - **Two languages simultaneously**: English, German (Spanish added in Stage 2)
   - Word recognition (100 words per language = 200 total)
   - Noun-verb associations in each language
   - Simple commands ("pick red", "nimm rot")
   - Code-switching recognition (mixing languages is natural)
   - Success: Execute 85% of commands correctly in both languages
   
   **Why Start with Two Languages?**
   - Mirrors bilingual children (manageable cognitive load)
   - Working memory still developing in Stage 1
   - Prevents overload while establishing multilingual foundations
   - Spanish added in Stage 2 when WM capacity is stronger
   - Still captures critical period advantage for multilingualism

**Training Details**:
- Mixed modalities (vision + language)
- Introduce sparse reward (striatum engagement)
- Curriculum: Easy→Medium→Hard within each task
- Consolidation: Every 10,000 steps

**Expected Growth**: +20,000 neurons (28% increase)
- Prefrontal (working memory demand)
- Hippocampus (more associations)
- Striatum (policy learning): 8,000 neurons

---

### Stage 2: Grammar & Composition (Child Brain)
**Goal**: Learn compositional language and basic reasoning

**Size**: ~100,000 neurons
- Expand Language Areas: Wernicke (10k), Broca (8k)
- Cerebellum: 12,000 (sequence learning)

**Tasks**:
1. **Multilingual Grammar Acquisition** (Week 9-11)
   - **Add Spanish** as third language (English, German, Spanish)
   - Vocabulary: 1,000 words per language (3,000 total)
   - Language-specific grammar rules:
     * English: SVO, articles (a/the)
     * German: Case system (nominative, accusative), verb-second
     * Spanish: Gender agreement, verb conjugations
   - Translation between languages (not word-for-word, conceptual)
   - Code-mixing understanding ("I have ein Hund")
   - Success: >80% grammatical sentences in each language

2. **Cross-Lingual Compositional Reasoning** (Week 11-13)
   - Same concept, different expressions
   - "The red ball" / "Der rote Ball" / "La pelota roja"
   - Spatial relations across languages
   - Simple inference in any language
   - Success: Answer 75% of reasoning questions regardless of question language

3. **Multilingual Multi-Step Instructions** (Week 13-14)
   - Follow 3-step commands in mixed languages
   - "Take the ball, nimm es, and pon it here"
   - Language detection and switching
   - Success: Complete 80% of multilingual tasks

**Training Details**:
- Introduce neuromodulation (dopamine for success/failure)
- Episodic memory replay (hippocampus)
- Language + vision + action integration
- Consolidation: Every 15,000 steps

**Expected Growth**: +50,000 neurons (50% increase)
- Language regions (grammar complexity)
- Cerebellum (precise sequences)
- Expanded cortical representations

---

### Stage 3: Reading & Writing (Elementary Brain)
**Goal**: Process written language, generate coherent text

**Size**: ~200,000 neurons
- Expand Wernicke/Broca: +15k each
- Visual word form area: 10,000 (orthography)
- Semantic network: 20,000 (concepts)

**Tasks**:
1. **Multilingual Reading Comprehension** (Week 15-18)
   - Vocabulary: 5,000 words per language (15,000 total)
   - Short paragraphs in English, German, Spanish (3-5 sentences)
   - Answer comprehension questions in any language
   - Cross-lingual reading (read German, answer in English)
   - Success: >70% reading comprehension across languages

2. **Multilingual Text Generation** (Week 18-21)
   - Complete sentences in target language
   - Simple stories in each language (3-4 sentences)
   - Maintain language consistency (don't mix mid-sentence)
   - Translation (conceptual, not literal)
   - Success: Human judges rate 65% as coherent in each language

3. **Multilingual Dialogue** (Week 21-23)
   - Q&A in any language
   - Respond in same language as question
   - Context maintenance across language switches
   - Detect language from input
   - Success: 75% contextually appropriate responses in correct language

**Training Details**:
- Token-level prediction (next word)
- Sentence-level generation (full thoughts)
- Contrastive learning (good vs bad examples)
- Consolidation: Every 20,000 steps
- Introduce "sleep" phases (offline replay)

**Expected Growth**: +100,000 neurons (50% increase)
- Massive language network expansion
- Semantic representations
- Generative pathways

---

### Stage 4: Abstract Reasoning (Adolescent Brain)
**Goal**: Develop abstract thought, analogies, complex reasoning

**Size**: ~400,000 neurons
- Prefrontal expansion: +30k (executive function)
- Parietal regions: +25k (spatial/abstract reasoning)
- Expanded hippocampus: +15k (episodic sophistication)

**Tasks**:
1. **Analogical Reasoning** (Week 24-27)
   - "A is to B as C is to ___"
   - Conceptual similarities
   - Transfer learning across domains
   - Success: >70% on analogy tasks

2. **Mathematical Reasoning** (Week 27-30)
   - Basic arithmetic (learned, not hardcoded)
   - Word problems
   - Simple algebra
   - Success: >75% on grade-school math

3. **Commonsense Reasoning** (Week 30-33)
   - Physical intuition (objects fall, liquids pour)
   - Social reasoning (people have goals)
   - Causal inference
   - Success: >70% on PIQA, Social IQA benchmarks

4. **Social & Emotional Intelligence** (Week 33-36)
   - Emotion recognition from text/context
   - Pragmatics (sarcasm, irony, implicature)
   - Theory of Mind (understanding beliefs/desires)
   - Social norms and politeness
   - Perspective-taking
   - Success: >70% on ToM benchmarks, social reasoning tasks

**Training Details**:
- Multi-task training (mix all previous skills)
- Harder negatives (near-miss answers)
- Explanations (why/how questions)
- **Curriculum mixing ratio**: 70% new tasks, 30% review from previous stages
- **Backward compatibility checks**: Every 10k steps, test sample from all previous stages
- Consolidation: Every 30,000 steps

**Expected Growth**: +150,000 neurons (37% increase)
- Abstract reasoning circuits
- Cross-domain integration
- Meta-cognitive regions

---

### Stage 5: Expert Knowledge (Young Adult Brain)
**Goal**: Acquire specialized knowledge, maintain generality

**Size**: ~600,000 neurons
- Domain-specific modules: +100k distributed
- Expanded semantic memory: +50k

**Tasks**:
1. **Domain Expertise** (Week 34-40)
   - Science (biology, physics, chemistry)
   - History & geography
   - Literature & arts
   - Technical skills (coding basics)
   - Success: Pass domain-specific tests >65%

2. **Long-Form Generation** (Week 40-45)
   - Essays (200+ words)
   - Maintain coherence over multiple paragraphs
   - Structured arguments
   - Success: Human evaluation >70% quality

3. **Multi-Modal Integration** (Week 45-50)
   - Vision + language (image captioning)
   - Audio + language (speech understanding)
   - Cross-modal reasoning
   - Success: >75% on multi-modal benchmarks

**Training Details**:
- Curriculum mixture (revisit all previous stages)
- **Mixing schedule**: 50% new domain expertise, 50% prior stages (weighted by recency)
- Prevent forgetting (weighted replay based on task importance)
- Sparse high-quality data
- **Data augmentation**: Paraphrasing, back-translation, synthetic examples
- Consolidation: Every 50,000 steps
- **Offline "sleep" consolidation**: Every 100k steps (decreasing from 20k in Stage 0)

**Expected Growth**: +200,000 neurons (33% increase)
- Distributed expertise
- Cross-modal pathways
- Refinement of existing circuits

---

### Stage 6: LLM-Level Capabilities (Adult Brain)
**Goal**: Match or exceed current LLM performance

**Size**: ~1,000,000 neurons
- Full brain integration
- Optimized for efficiency

**Tasks**:
1. **General Language Understanding** (Week 51-60)
   - GLUE/SuperGLUE benchmarks
   - Reading comprehension (SQuAD, RACE)
   - Natural language inference
   - Success: Within 85% of GPT-3.5 performance

2. **Complex Reasoning** (Week 60-70)
   - Chain-of-thought reasoning
   - Multi-hop question answering
   - Logical puzzles
   - Success: >80% on complex reasoning benchmarks

3. **Few-Shot Learning** (Week 70-80)
   - Learn new tasks from 1-5 examples
   - Rapid adaptation
   - Meta-learning
   - Success: Match few-shot GPT-3 performance

4. **Instruction Following** (Week 80-90)
   - Complex multi-step instructions
   - Constrained generation
   - Tool use
   - Success: >85% instruction compliance

**Training Details**:
- High-quality curated data (books, papers, conversations)
- Instruction tuning
- Human feedback (RLHF-style with dopamine)
- **Curriculum mixing**: 40% LLM-level tasks, 60% comprehensive review of all stages
- **Continual learning validation**: Full evaluation suite from Stage 0-5 every 50k steps
- Continuous consolidation
- Delta checkpoints (most weights stable)
- **Data augmentation**: Question generation, synthetic dialogues, adversarial examples

**Expected Growth**: +200,000 neurons (20% increase)
- Refinement more than expansion
- Pruning of unused capacity (conservative)
- Optimization of critical pathways

---

## Training Infrastructure

### Curriculum Mixing Strategy

**Purpose**: Balance learning new skills while maintaining old knowledge

**Stage-Specific Ratios**:
- **Stage 0-1**: 100% current stage (establishing foundations)
- **Stage 2**: 80% new, 20% Stage 0-1 review
- **Stage 3**: 70% new, 30% Stage 0-2 review
- **Stage 4**: 70% new, 30% Stage 0-3 review (weighted by recency)
- **Stage 5**: 50% new, 50% comprehensive review
- **Stage 6**: 40% new, 60% comprehensive review (prevent catastrophic forgetting)

**Weighting Formula** (for review tasks):
```
weight(stage_i) = exp(-decay * (current_stage - stage_i))
decay = 0.3  # Recent stages weighted higher
```

**Example** (Stage 4 review distribution):
- Stage 3: 50% of review time
- Stage 2: 30% of review time  
- Stage 1: 15% of review time
- Stage 0: 5% of review time

### Data Augmentation Strategy

**Purpose**: Improve sample efficiency and generalization

**Techniques by Modality**:

**Text Augmentation**:
- Back-translation (En→De→En for paraphrasing)
- Synonym replacement (WordNet-based)
- Sentence reordering (for robustness)
- Synthetic dialogue generation
- Question generation from passages
- Adversarial examples (near-miss answers)

**Vision Augmentation**:
- Random crops, flips, rotations
- Color jittering
- Mixup (blend two images)
- Cutout (random patches removed)
- Style transfer (artistic styles)

**Audio Augmentation**:
- Speed perturbation (±10%)
- Pitch shifting
- Background noise injection
- Room reverberation simulation

**Multi-Modal Augmentation**:
- Image-caption mismatches (negative examples)
- Cross-lingual image descriptions
- Synthetic scene compositions

**Stage-Specific Usage**:
- Stage 0-2: Minimal augmentation (<5% of data, only basic transforms)
- Stage 3-4: Moderate augmentation (10-15% of data)
- Stage 5-6: Conservative augmentation (20% max, preserve semantic fidelity)

**Rationale**: Biological learning benefits from clean, consistent patterns.
Over-augmentation can interfere with precise semantic learning and 
biological plausibility. Keep augmentation conservative.

### Offline "Sleep" Consolidation Protocol

**Purpose**: Strengthen important memories, integrate knowledge

**Inspiration**: Hippocampal replay during sleep in mammals

**Triggers** (multiple conditions):
1. **Time-based**: Varies by stage (mirrors human development)
   * **Stage 0-1**: Every 20,000 steps (frequent for foundational learning)
   * **Stage 2-3**: Every 50,000 steps (moderate consolidation)
   * **Stage 4-5**: Every 100,000 steps (less frequent, established circuits)
   * **Stage 6**: Every 150,000 steps (minimal but maintained)

2. **Event-based** (overrides time-based if triggered):
   * **Before stage transition**: ALWAYS consolidate before curriculum stage change
   * **After catastrophic forgetting**: If any previous stage drops >10%
   * **On performance plateau**: If no improvement for 15k steps
   * **After major growth**: When >20% neurons added to any region

**Minimum Frequency**: Never exceed 200,000 steps without consolidation (risk of drift)

**Duration**: 10,000 replay steps per session

**Replay Selection Strategy**:
1. **Prioritized Experience Replay**
   - High reward/dopamine trials (successful episodes)
   - High surprise/error trials (learning opportunities)
   - Boundary examples (challenging cases)
   - Random baseline (avoid overfitting to priorities)

2. **Proportions**:
   - 40% high-reward experiences
   - 30% high-error experiences
   - 20% boundary/challenging cases
   - 10% random experiences

3. **Temporal Compression**:
   - Recent experiences: Replay at 1x speed
   - Older experiences: Replay at 5x speed (faster reactivation)
   - Very old: Occasional 10x speed (semantic consolidation)

**Consolidation Process**:
```python
def offline_consolidation(brain, replay_buffer, n_steps=10000):
    """Sleep-like consolidation phase"""
    
    # Reduce learning rates (gentler updates)
    original_lr = brain.get_learning_rates()
    brain.set_learning_rates(original_lr * 0.1)
    
    # Modulate neuromodulators (mimic sleep state)
    brain.set_global_dopamine(0.3)  # Lower dopamine
    brain.set_acetylcholine(0.5)    # Moderate ACh
    
    for step in range(n_steps):
        # Sample prioritized experiences
        batch = replay_buffer.sample_prioritized(batch_size=1)
        
        # Replay experience
        brain.forward(batch['input'])
        brain.learn(batch['target'], reward=batch['reward'])
        
        # Hebbian strengthening (no task-specific updates)
        brain.consolidate_synapses(threshold=0.1)
    
    # Restore learning rates
    brain.set_learning_rates(original_lr)
    brain.reset_neuromodulators()
```

**Metrics to Track**:
- Synapse strength changes (should increase for important connections)
- Firing rate stability (should be more consistent)
- Task performance before/after sleep (should improve or maintain)

---

## Evaluation Protocols

### Standardized Test Sets

**Purpose**: Consistent evaluation across stages, detect forgetting

**Test Set Structure** (per stage):
- **Holdout Size**: 10% of stage data (never seen during training)
- **Composition**: Representative of all tasks in that stage
- **Languages**: Balanced across English, German, Spanish
- **Difficulty**: Easy (40%), Medium (40%), Hard (20%)

**Stage-Specific Benchmarks**:

**Stage 0**: 
- MNIST test set (10k images)
- Custom sequence prediction (1k sequences)
- TIMIT phoneme recognition (500 samples)

**Stage 1**:
- CIFAR-10 test set (10k images)
- N-back working memory (500 trials per N)
- Multilingual command following (300 commands per language)

**Stage 2**:
- Grammar test suite (1k sentences per language)
- Compositional reasoning (500 questions per language)
- Planning tasks (200 multi-step scenarios)

**Stage 3**:
- Reading comprehension (SQuAD subset, 1k passages)
- Text generation quality (200 prompts, human evaluation)
- Dialogue coherence (100 multi-turn conversations)

**Stage 4**:
- Analogy test (500 items: semantic, spatial, logical)
- Math word problems (500 problems, grade 1-5)
- Commonsense reasoning (PIQA: 500 items, Social IQA: 500 items)
- Theory of Mind (500 scenarios)

**Stage 5**:
- Domain knowledge tests (200 questions per domain: science, history, arts)
- Long-form generation (50 essay prompts)
- Multi-modal integration (COCO captions: 500 images, VQA: 1k questions)

**Stage 6**:
- MMLU (1k questions, representative subset)
- HellaSwag (1k items)
- HumanEval (164 coding problems)
- MT-Bench (80 instruction pairs)

### Continual Learning Validation

**Purpose**: Ensure brain doesn't forget previous stages

**Protocol**:
1. **Baseline**: Establish performance on all previous stage tests at end of each stage
2. **Checkpoints**: Re-evaluate all previous stages every 50k steps
3. **Threshold**: Alert if any stage drops >10% from baseline
4. **Action**: Increase review proportion for forgotten stages

**Metrics**:
- **Backward Transfer**: Performance on Stage N after training Stage N+K
- **Forward Transfer**: Performance on Stage N+1 given Stage N training
- **Retention Rate**: % of original performance maintained

**Example Matrix** (Stage 4 completion):
```
         Stage0  Stage1  Stage2  Stage3  Stage4
Initial   98%     95%     92%     88%     85%
+50k      97%     94%     91%     87%     88%  ✓ All within 10%
+100k     96%     92%     89%     86%     91%  ✓ Slight decay, acceptable
+150k     94%     89%     85%     84%     93%  ⚠ Stage 2 at -7%, Stage 1 at -6%
→ Action: Increase Stage 1-2 review from 30% to 40%
```

### Error Recovery & Stage Failure Protocol

**Purpose**: Handle cases where learning plateaus or regresses

**Failure Criteria**:
1. **Plateau**: No improvement for 20k steps
2. **Instability**: Loss oscillating wildly (std > 2x mean)
3. **Regression**: Performance drops >15% on current stage
4. **Catastrophic Forgetting**: Previous stage drops >20%

**Diagnostic Steps**:
1. **Check Health Metrics**:
   - Firing rates (should be 0.1-0.3)
   - Weight saturation (<80% maxed)
   - Gradient magnitudes (not exploding/vanishing)
   - Neuromodulator levels (in expected ranges)

2. **Identify Root Cause**:
   - **Capacity**: Is region utilization >90%? → Need growth
   - **Learning Rate**: Too high/low? → Adjust
   - **Data Quality**: Bad batch? → Inspect samples
   - **Task Difficulty**: Too hard? → Add intermediate steps

**Recovery Strategies**:

**Strategy 1: Rollback & Retry**
```python
# Revert to last stable checkpoint
brain = BrainCheckpoint.load(f"stage{N}_step_{last_stable}.thalia")

# Adjust hyperparameters
config.learning_rate *= 0.5  # More conservative
config.growth_threshold = 0.7  # Grow earlier

# Resume with easier curriculum
trainer.train_stage(stage_config, difficulty=0.8)  # Was 1.0
```

**Strategy 2: Intermediate Stage**
```python
# Insert sub-stage with intermediate difficulty
substage_config = StageConfig(
    epochs=20,
    difficulty=0.6,  # Between previous (0.4) and current (0.9)
    data_config=interpolate(prev_config, curr_config)
)
trainer.train_stage(substage_config)
```

**Strategy 3: Targeted Growth**
```python
# Manually grow struggling region
problem_region = identify_bottleneck(brain)
brain.regions[problem_region].add_neurons(
    n_new=5000,  # 10% capacity boost
    initialization='sparse_random'
)
```

**Strategy 4: Extended Consolidation**
```python
# Extra sleep phase to stabilize
for _ in range(5):  # 5 consolidation cycles
    offline_consolidation(brain, replay_buffer, n_steps=10000)
```

**Decision Tree**:
```
Failure detected?
├─ Yes: Diagnose cause
│   ├─ Capacity issue? → Strategy 3 (Growth)
│   ├─ Too hard? → Strategy 2 (Intermediate stage)
│   ├─ Unstable? → Strategy 4 (Extended consolidation)
│   └─ Other? → Strategy 1 (Rollback & retry)
└─ No: Continue training
```

**Logging**:
- Record all failure events with full diagnostics
- Track recovery strategy effectiveness
- Build failure prediction model over time

---

## Training Infrastructure

### Compute Requirements

**Stage 0-2** (Weeks 1-14):
- Single GPU (RTX 3090 or better)
- ~8GB VRAM sufficient
- Training time: ~2-4 hours per task

**Stage 3-4** (Weeks 15-33):
- Single GPU (RTX 4090 or A100)
- ~16GB VRAM
- Training time: ~8-12 hours per task

**Stage 5-6** (Weeks 34-90):
- Multi-GPU or single A100 80GB
- ~40-60GB VRAM
- Training time: ~1-3 days per task

### Data Requirements

**Total Dataset Size**: ~150GB (increased for multilingual)
- Images: MNIST, CIFAR-10, ImageNet subset (~20GB)
- Text - English: Wikipedia, books, conversations (~40GB)
- Text - German: Wikipedia, books, conversations (~30GB)
- Text - Spanish: Wikipedia, books, conversations (~30GB)
- Parallel corpora: Translation pairs (~10GB)
- Audio: Multilingual speech datasets (~15GB)
- Multi-modal: COCO, VQA (~10GB)

**Data Pipeline**:
- Start with small, clean datasets
- Gradually introduce noise and complexity
- Mixture ratios shift over stages
- Language balance: 40% English, 30% German, 30% Spanish (early stages)
- Later stages: Can add more languages (French, Mandarin, etc.)
- Quality > Quantity (biologically plausible learning)

### Checkpoint Strategy

> **Implementation Details**: See [`checkpoint_format.md`](checkpoint_format.md) for format specification.

**When to Checkpoint**:
1. **Before stage transition** (mandatory) - Enable rollback if new stage fails
2. **After consolidation** (every 20k-150k steps depending on stage)
3. **On catastrophic forgetting detection** - Immediate checkpoint before recovery
4. **After major growth events** (>20% neurons added)
5. **Regular intervals during training** (every 10k-50k steps)

**Full Checkpoints**: After each major stage (6 total)
- Compressed with zstd
- ~0.3-2GB per checkpoint
- Include complete state (weights, RegionState, learning state, oscillators)

**Delta Checkpoints**: Every 10k-50k steps within stages
- ~10-50MB per delta (v2.0 feature)
- Enable rollback if catastrophic forgetting
- Store only changed weights

**Total Storage**: ~50GB for complete curriculum

**Recovery Strategy**:
- Keep last 3-5 checkpoints per stage
- If failure: rollback to last stable checkpoint
- If catastrophic forgetting: rollback and increase review proportion

---

## Common Failure Modes & Prevention

### Expected Failure Modes by Stage

**1. Runaway Excitation** (Most common in Stage 0-1)
- **Symptom**: Firing rates >0.8, all neurons active
- **Cause**: Insufficient inhibition, positive feedback loops
- **Prevention**: 
  * Criticality monitor with auto-adjustment
  * E/I balance regulator
  * Divisive normalization in cortex
- **Recovery**: 
  * Reduce learning rates by 50%
  * Boost inhibitory weights by 20%
  * Add lateral inhibition if missing
  * Rollback to last stable checkpoint if severe

**2. Silent Networks** (Can occur any stage)
- **Symptom**: Firing rates <0.01, no spikes
- **Cause**: Too-strong inhibition, input too weak, thresholds too high
- **Prevention**: 
  * Input normalization to [0, 1] range
  * Intrinsic plasticity adapts thresholds
  * Minimum firing rate monitoring
- **Recovery**: 
  * Boost input strength by 2x
  * Lower neuron thresholds by 10-20%
  * Check for dead neurons (membrane stuck at rest)
  * May need to reinitialize silent regions

**3. Catastrophic Forgetting** (Stage 2+)
- **Symptom**: >15% performance drop on previous stage tasks
- **Cause**: New learning overwrites old representations
- **Prevention**: 
  * Curriculum mixing (review old tasks regularly)
  * Long consolidation windows (50k+ steps)
  * Conservative synaptic scaling (gentle weakening)
- **Recovery**: 
  * Increase review proportion for forgotten stage
  * Extended replay of affected tasks (20k extra steps)
  * Reduce learning rate for affected regions
  * May need to rollback and retrain with better mixing

**4. Capacity Saturation** (Stage 3+)
- **Symptom**: >90% weight saturation, performance plateau, high utilization
- **Cause**: Not enough neurons for task complexity
- **Prevention**: 
  * Auto-growth at 80% utilization threshold
  * Monitor weight saturation per region
  * Track capacity metrics every 1000 steps
- **Recovery**: 
  * Add 10-20% neurons to saturated region
  * Sparse initialization for new neurons
  * Continue training (should resume learning)
  * If plateau persists, may need more capacity

**5. Oscillator Desynchronization** (Stage 2+ with hippocampus)
- **Symptom**: Irregular sequences, poor temporal binding
- **Cause**: Theta/gamma frequencies drift, phase coupling lost
- **Prevention**: 
  * Oscillator frequency monitoring
  * Automatic frequency correction
  * Phase coherence metrics
- **Recovery**: 
  * Reset oscillator phases
  * Re-sync to target frequencies
  * May indicate region too active/silent (check firing rates)

**6. Striatum D1/D2 Imbalance** (Any stage with RL)
- **Symptom**: All-Go or All-NoGo behavior, poor action selection
- **Cause**: D1 or D2 pathways dominate, opponent balance lost
- **Prevention**: 
  * Homeostatic D1/D2 balancing
  * Monitor weight ratio (should be ~1:1)
  * Dopamine modulation working correctly
- **Recovery**: 
  * Rebalance D1/D2 weights manually
  * Check dopamine signal delivery
  * May need to adjust reward scale

### Diagnostic Checklist

When training stalls or fails:
```markdown
□ Check firing rates (0.05-0.25 is healthy)
□ Check weight saturation (<80% healthy)
□ Check gradient magnitudes (1e-5 to 1e-2 healthy)
□ Check neuromodulator levels (in expected ranges)
□ Check oscillator frequencies (if applicable)
□ Check for NaN/Inf in any tensor
□ Check task performance on previous stages
□ Review recent growth events
□ Check consolidation frequency
□ Look for sudden distribution shifts in data
```

---

## Success Metrics

### Per-Stage Metrics
- Task-specific accuracy (defined per stage)
- Firing rate stability (no runaway or silence)
- Weight saturation (<80% of neurons maxed out)
- Cross-task transfer (new tasks benefit from old knowledge)
- **Backward transfer**: Accuracy on previous stages (continual learning)
- **Data efficiency**: Steps to criterion vs baseline
- **Consolidation quality**: Performance improvement after sleep phases

### Global Metrics
- **Sample Efficiency**: Steps to reach criterion vs backprop baseline
- **Continual Learning**: Performance on Stage N tasks after training Stage N+3 (should be >90% of original)
- **Generalization**: Zero-shot performance on held-out distributions
- **Biological Plausibility**: Local learning rules maintained throughout
- **Multilingual Balance**: Performance gap between languages <10%
- **Social Intelligence**: Theory of Mind accuracy, pragmatic understanding
- **Robustness**: Performance under adversarial/noisy conditions

### LLM Comparison Benchmarks
- MMLU (Massive Multitask Language Understanding)
- HellaSwag (Commonsense reasoning)
- TruthfulQA (Factual accuracy)
- HumanEval (Code generation)
- MT-Bench (Instruction following)

**Realistic Expectations**: 
- **NOT** trying to match GPT-3.5 parameter-for-parameter (175B vs ~500M synapses)
- **Goal**: Explore how far biologically-plausible learning can scale
- **Focus metrics**: 
  * Sample efficiency (learn from fewer examples than transformers)
  * Continual learning (no catastrophic forgetting)
  * Few-shot adaptation (rapid learning from 1-5 examples)
  * Biological plausibility (local rules, no backprop)
  * Energy efficiency (spike-based computation)

**Tradeoffs We Accept**:
- Smaller knowledge base than LLMs
- Potentially slower inference (sequential spiking)
- Less breadth across all possible tasks

**Advantages We Target**:
- ✅ Better few-shot learning (hippocampus one-shot)
- ✅ No catastrophic forgetting (curriculum + consolidation)
- ✅ Continual adaptation (ongoing learning)
- ✅ More interpretable (discrete spikes, local rules)
- ✅ Biologically grounded (can inform neuroscience)

**Timeline Caveat**: 9 months is ambitious. May take 12-18 months to reach
Stage 6 robustly. Focus is on scientific exploration, not racing to deployment.

---

## Risk Mitigation

### Catastrophic Forgetting
- **Solution**: Long observation windows, conservative pruning
- **Monitor**: Periodic evaluation on all previous stages
- **Action**: Rollback to earlier checkpoint if >10% performance drop

### Capacity Saturation
- **Solution**: Auto-growth when utilization >80%
- **Monitor**: Firing rates, weight saturation
- **Action**: Add 10-20% capacity, continue training

### Training Instability
- **Solution**: Homeostatic mechanisms, neuromodulation
- **Monitor**: Spike statistics, gradient magnitudes
- **Action**: Adjust learning rates, add inhibition

### Poor Generalization
- **Solution**: Curriculum mixture, diverse data
- **Monitor**: Held-out validation sets
- **Action**: Increase data diversity, longer consolidation

---

## Timeline & Milestones

**Month 1-2**: Infrastructure + Stage 0-1
- Checkpoint system working
- Growth mechanisms functional
- Basic sensory + working memory

**Month 3-4**: Stage 2-3
- Grammar acquired
- Reading comprehension functional
- Text generation coherent

**Month 5-6**: Stage 4-5
- Abstract reasoning
- Domain expertise
- Multi-modal integration

**Month 7-9**: Stage 6
- LLM-level performance
- Benchmark evaluations
- Optimization & analysis

**Total**: 9 months from tiny brain to LLM-level capability

---

## Open Research Questions

1. **Optimal Growth Rate**: How aggressively should we add capacity?
2. **Consolidation Schedule**: How often? How long?
3. **Curriculum Order**: Is our stage sequence optimal?
4. **Transfer Learning**: How much does Stage N help Stage N+1?
5. **Sample Efficiency**: Can we match biology's efficiency?
6. **Scaling Laws**: How do biological brains scale vs transformers?

---

## Comparison to Human Development

| **Stage** | **Thalia** | **Human Equivalent** | **Age** | **Notes** |
|-----------|------------|----------------------|---------|-----------|
| 0 | Sensory foundations | Infant perception | 0-6 months | Visual, auditory basics |
| 1 | Object permanence | Object permanence | 6-18 months | Working memory emerges, **multilingual exposure begins** |
| 2 | Grammar | Language explosion | 18-36 months | Rapid vocab growth, **3 languages simultaneously** |
| 3 | Reading/writing | Elementary school | 6-10 years | Literacy in **all 3 languages** |
| 4 | Abstract reasoning | Adolescence | 12-16 years | Formal operations, **cross-lingual reasoning** |
| 5 | Expert knowledge | Higher education | 18-24 years | Specialization, **can add 4th-5th language** |
| 6 | LLM-level | PhD+ expertise | 24+ years | Domain mastery, **multilingual expert** |

**Key Difference**: Thalia compresses 24+ years into 9 months through:
- Curated data (no distractions)
- 24/7 training (no sleep required... but we add offline consolidation)
- Optimized curriculum (no wasted time)
- **Multilingual from the start** (critical period advantage)

But maintains biological learning principles:
- Local learning rules
- Gradual complexity increase
- Consolidation periods
- No catastrophic forgetting
- **Natural multilingualism** (like bilingual children)

---

## Next Steps

1. **Implement Phase 1**: Core I/O and checkpoint system (Week 1-2 of dev)
2. **Design Stage 0 Datasets**: MNIST, simple sequences, phonemes
3. **Baseline Experiments**: Train Stage 0, measure growth triggers
4. **Validate Growth**: Does auto-grow work as expected?
5. **Iterate**: Refine curriculum based on results

**Status**: Ready to begin implementation  
**First Experiment**: Stage 0 sensory foundations  
**Target Date**: Start January 2026
