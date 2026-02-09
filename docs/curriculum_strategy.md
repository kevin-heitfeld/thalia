# Thalia Curriculum Training Strategy

**Version**: 0.1.0
**Status**: Design Phase
**Date**: 2026-02-03

---

## Overview

Progressive training strategy to grow a biologically-plausible brain from basic pattern recognition to LLM-level multi-modal capabilities. Inspired by human cognitive development: start with **bootstrap/developmental initialization**, then gradually increase complexity through curriculum stages, consolidating knowledge at each step.

---

## Philosophy

**Core Principle**: The brain should naturally discover complexity through incremental challenges, not have it forced upon it.

**Bootstrap Principle**: Before curriculum learning can begin, the brain must be "brought to life" through developmental initialization—stronger initial connections, spontaneous activity, and elevated plasticity. Skip this and learning never starts (the vicious cycle: no activity → no learning → no activity).

- **Bootstrap First**: Establish functional connectivity BEFORE task learning (Stage 0)
- **Start Strong**: Initialize with biologically-inspired weights (0.3-0.5, not random)
- **Start Tiny**: Begin with minimal capacity (10k-50k neurons)
- **Grow Organically**: Add capacity only when current tasks demand it
- **Consolidate Often**: Strengthen important circuits between stages
- **Never Forget**: Protect learned knowledge during curriculum transitions
- **Measure Progress**: Clear success criteria for each stage

---

## Developmental Stages

### Stage 0: Bootstrap & Developmental Initialization (Pre-Natal Development)
**Goal**: Establish functional connectivity before task learning begins

**Duration**: Week -2 to 0 (2 weeks BEFORE curriculum starts)

**Critical Insight**: Real brains don't "learn from scratch"—they start with genetic prewiring, spontaneous activity, and elevated developmental plasticity. This stage simulates prenatal/early postnatal development to solve the **bootstrap problem**: getting learning started when initial connections are weak.

**The Bootstrap Problem**:
- Weak random weights (0.16) → neurons don't fire
- Neurons don't fire → Hebbian learning can't strengthen weights
- **Vicious cycle**: silence → no learning → continued silence → weight decay → catastrophic collapse
- Observed in Run 9: 96.5% weight collapse despite biological homeostasis mechanisms

**How Real Brains Solve It**
1. **Genetic prewiring**: Thalamus→L4 starts at 0.3-0.5 conductance (not random)
2. **Spontaneous activity**: Retinal waves, cortical spindle bursts drive learning before sensory input
3. **Critical period plasticity**: 10-100x higher learning rates during development
4. **Elevated homeostasis**: Synaptic scaling rates 10-20x faster initially
5. **Feedforward bias**: Thalamic input 3-5x stronger than recurrent connections
6. **Low thresholds**: Higher intrinsic excitability in young neurons

**Initial Size**: 30,000 neurons (minimal functional brain)
- Thalamus: 8,000 (sensory relay)
- Cortex L4: 5,000 (primary sensory)
- Cortex L2/3: 3,000 (feature integration)
- Cortex L5: 3,000 (output)
- Hippocampus: 2,000 (associations)
- Striatum: 5,000 (action selection)
- Motor cortex: 4,000 (if sensorimotor included)

**Bootstrap Configuration** (Elevated Parameters):

```python
bootstrap_config = {
    # INITIAL WEIGHTS (Much stronger than random)
    'thalamus_to_l4': {
        'mean': 0.40,  # vs 0.164 standard (2.4x stronger)
        'std': 0.15,
        'distribution': 'log_normal',
        'connectivity': 0.30,  # 30% sparse
    },
    'l4_to_l23': {
        'mean': 0.20,  # vs 0.10 standard (2x stronger)
        'std': 0.10,
        'distribution': 'log_normal',
        'connectivity': 0.25,
    },
    'l23_recurrent': {
        'mean': 0.08,  # vs 0.05 standard (1.6x stronger)
        'std': 0.05,
        'distribution': 'log_normal',
        'connectivity': 0.15,  # Very sparse initially
    },

    # LEARNING RATES (Elevated for critical period)
    'bcm_learning_rate': 0.05,  # vs 0.01 standard (5x higher)
    'stdp_learning_rate': 0.03,  # vs 0.01 standard (3x higher)
    'dopamine_modulation': 0.60,  # Elevated tonic dopamine

    # HOMEOSTATIC MECHANISMS (Much faster)
    'synaptic_scaling_lr': 0.02,  # vs 0.001 standard (20x faster!)
    'synaptic_scaling_min_activity': 0.005,
    'synaptic_scaling_max_factor': 2.5,

    # WEIGHT DECAY (Disabled during bootstrap)
    'weight_decay': 0.0001,  # Standard rate
    'silent_decay_factor': 0.0,  # NO decay when silent (vs 0.1)
    'min_activity_for_decay': 0.010,  # Higher threshold

    # EXCITABILITY (Lower thresholds, higher gain)
    'initial_threshold_factor': 0.70,  # Lower thresholds (easier to fire)
    'background_excitation': 0.10,  # Tonic depolarization (vs 0.0)
    'homeostatic_gain_lr': 0.005,  # Faster gain adaptation

    # NEUROMODULATORS (Elevated during development)
    'dopamine_baseline': 0.60,  # vs 0.20 adult (3x higher)
    'acetylcholine_baseline': 0.60,  # vs 0.30 adult (2x higher)
    'norepinephrine_baseline': 0.50,  # vs 0.20 adult (2.5x higher)
}
```

**Training Protocol**:

#### Phase 0A: Spontaneous Activity (Week -2 to -1.5)
**Goal**: Refine genetically-specified connections through spontaneous activity

**No external input** (simulates prenatal/eyes-closed period):
- Inject weak Ornstein-Uhlenbeck noise to all regions (σ=0.05)
- Allow spontaneous population bursts (wave-like activity)
- Hebbian/STDP refinement of initial connections
- Establish stable firing rates (target: 0.08-0.12)
- BCM thresholds settle to activity levels

**Success Criteria**:
- [ ] All regions firing 0.05-0.15 (stable spontaneous activity)
- [ ] No silent regions (>0.01 minimum for 5000 consecutive steps)
- [ ] No runaway excitation (max firing <0.5)
- [ ] Weight distributions remain log-normal (not collapsing)
- [ ] Thalamus→L4 weights stable or increasing (>0.35 mean)

#### Phase 0B: Simple Pattern Exposure (Week -1.5 to -0.5)
**Goal**: Verify learning can strengthen connections with minimal input

**Ultra-simple stimuli** (simulates early sensory input):
- Single pixels on/off (2 patterns total)
- Pure tones (2 frequencies)
- 100x repetition per pattern
- No reward signal (pure Hebbian/STDP)
- Very short sequences (5 timesteps)

**Tasks**:
1. **Thalamic Relay**: Thalamus receives input, relays to L4
2. **Cortical Response**: L4 fires reliably (>80% of trials)
3. **Pattern Discrimination**: L4 activity differs for pattern A vs B
4. **Weight Strengthening**: Thalamus→L4 weights increase from activity

**Success Criteria**:
- [ ] Cortex fires on >90% of stimulus presentations
- [ ] Can discriminate 2 patterns (>85% accuracy)
- [ ] Thalamus→L4 weights increased: mean >0.42 (strengthened from 0.40)
- [ ] No catastrophic forgetting (spontaneous activity still present)
- [ ] Firing rates stable (0.08-0.15)

#### Phase 0C: Parameter Transition (Week -0.5 to 0)
**Goal**: Gradually transition from developmental to adult parameters

**Critical Period Closure Simulation**:
- Synaptic scaling rate: 0.02 → 0.010 → 0.005 → 0.002 (linear decay)
- Learning rates: 0.05 → 0.03 → 0.02 → 0.01 (linear decay)
- Silent decay factor: 0.0 → 0.03 → 0.07 → 0.10 (linear increase)
- Background excitation: 0.10 → 0.05 → 0.02 → 0.0 (linear decay)
- Threshold factor: 0.70 → 0.80 → 0.90 → 1.0 (return to standard)
- Neuromodulator baselines: Decay to adult levels over 5000 steps

**Continue simple pattern training** during transition:
- Increase to 4 patterns (2x complexity)
- Verify learning still works as plasticity reduces
- Monitor for any instabilities

**Success Criteria**:
- [ ] All regions still firing 0.05-0.15 after parameter transition
- [ ] Thalamus→L4 weights stable: mean >0.35 (no collapse)
- [ ] Can discriminate 4 patterns (>80% accuracy)
- [ ] Homeostatic mechanisms functional (gains adjusting appropriately)
- [ ] No runaway excitation during transition
- [ ] System ready for curriculum learning (Stage 1)

**Final Bootstrap Validation** (End of Week 0):
- [ ] **Functional connectivity established**: All pathways responsive
- [ ] **Weight health**: Thalamus→L4 mean >0.35 (>2x initial random)
- [ ] **Activity stability**: All regions 0.05-0.15 firing rate
- [ ] **Learning functional**: Weights strengthen with correlated activity
- [ ] **Homeostasis working**: Synaptic scaling prevents silence/runaway
- [ ] **BCM converged**: Thresholds stable (not drifting)
- [ ] **Discrimination capability**: >80% on 4-pattern task
- [ ] **No pathologies**: No silence, runaway, or weight saturation

**If any validation fails**: DO NOT proceed to Stage 1. Debug bootstrap configuration.

**Why This Stage is CRITICAL**:
- **Solves the bootstrap problem**: Gets learning started before curriculum
- **Prevents Run 9 failure**: 96.5% weight collapse won't happen with 0.40 init + 0.02 scaling
- **Matches biology**: Real brains have prenatal development (not learning from scratch)
- **One-time cost**: 2 weeks upfront investment prevents months of failed training
- **Enables curriculum**: Stage 1+ assume functional brain (this stage delivers it)

**Comparison to Current Approach**:
| Parameter | Current (Failed Run 9) | Bootstrap Stage 0 | Improvement |
|-----------|------------------------|-------------------|-------------|
| Thalamus→L4 init | 0.164 | 0.40 | 2.4x stronger |
| Synaptic scaling LR | 0.001 | 0.02 | 20x faster |
| Silent decay factor | 0.1 | 0.0 | No decay |
| Learning rate | 0.01 | 0.05 | 5x higher |
| Result after 200 seq | 0.0058 (collapse) | >0.35 (stable) | 60x better |

**After Stage 0**: Brain is "alive"—firing, learning, stable. Now ready for curriculum learning.

---

### Stage 1: Sensorimotor Grounding (Embodied Foundation)
**Goal**: Establish sensorimotor coordination and embodied representations

**Duration**: Week 0-4 (1 month - extended for robust grounding)

**Rationale for Extension**: Human infants spend ~6 months on sensorimotor coordination. While we compress timelines, rushing this foundation risks weak embodied representations that cascade through later stages. Better to over-invest here (1 month) than debug abstract reasoning failures in Stage 6.

**Initial Size**: 30,000 neurons
- Motor cortex: 10,000 (action generation)
- Somatosensory cortex: 8,000 (proprioception)
- Cerebellum: 7,000 (forward models)
- Cortex L4: 5,000 (visual input)

**Rationale**:
- Human infants spend 0-6 months learning basic motor control before object recognition
- Active exploration (not passive viewing) drives early learning
- Sensorimotor coordination is foundation for all later cognition
- Cerebellum forward models need motor experience early
- Grounded representations (not arbitrary features)

**Tasks** (Interleaved Practice):
1. **Basic Motor Control** (Week 0-2, 40% of time)
   - Simple movements: left/right, up/down, forward/back
   - Proprioceptive feedback: "where is my effector?"
   - Velocity and acceleration control
   - Stop/start commands
   - Success: >90% accurate movement execution

2. **Visual-Motor Coordination** (Week 0-4, 35% of time)
   - Reach toward visual target
   - Track moving objects with "gaze"
   - Predict: "If I move left, visual field shifts right"
   - Hand-eye coordination tasks
   - Success: >85% accurate reaching, <15% prediction error

3. **Object Manipulation** (Week 2-4, 20% of time)
   - Push/pull objects
   - Grasp and release
   - Understand object affordances (pushable, graspable)
   - Cause-effect relationships (push → object moves)
   - Success: >80% successful manipulation

4. **Sensorimotor Prediction** (Week 2-4, 5% of time)
   - Learn forward models: action → sensory outcome
   - Inverse models: desired outcome → action
   - Cerebellum trains on prediction errors
   - Foundation for all later learning
   - Success: <5% prediction error on familiar actions (more stringent with extended time)

**Training Details**:
- Environment: Simple 2D/3D grid world with physics
- Continuous sensorimotor loop (action → perception → action)
- Cerebellum learns forward/inverse models via error correction
- Motor babbling phase (explore action space)
- Proprioceptive feedback at every timestep

**Success Criteria**:
- >95% accurate basic movements (raised with extended time)
- >90% reaching accuracy toward targets (raised from 85%)
- >85% successful object manipulation (raised from 80%)
- <5% sensorimotor prediction error (more stringent)
- Stable firing rates (0.05-0.15)
- Cerebellum forward models functional
- Robust proprioceptive representations established
- Strong sensorimotor integration (foundation for all later stages)

**Expected Growth**: +5,000 neurons (17% increase)
- Primarily in motor cortex and cerebellum (sensorimotor demand)

**Why This Stage is Critical**:
- Provides grounded representations (not arbitrary features)
- Cerebellum training early enables better learning later
- Active exploration more biologically realistic than passive observation
- Motor-to-sensory feedback stabilizes representations
- Foundation for Stage 2 object recognition (now embodied)

---

### Stage 2: Sensory Foundations (Infant Brain)
**Goal**: Learn basic sensory processing and pattern recognition

**Initial Size**: 35,000 neurons (continuing from Stage 1 growth)
- Cortex L4: 15,000 (primary sensory, expanded from Stage 1)
- Cortex L2/3: 10,000 (feature integration)
- Thalamus: 8,000 (sensory relay)
- Hippocampus: 2,000 (simple associations)
- Motor/Somatosensory/Cerebellum: Retained from Stage 1

**Tasks** (Interleaved Practice - NOT Sequential):
1. **Multi-Modal Sensory Integration** (Week 4-8, 70% of time)
   - **Week 4-5**: 40% visual, 20% temporal, 40% audio + phonological
     * Visual: MNIST digits, simple shapes (with active "looking")
     * Temporal: A-B-C sequences, rhythm detection
     * Audio: **Phoneme categorical perception**
   - **Week 5-6**: 30% visual, 25% temporal, 45% audio + phonological
     * Continue MNIST with temporal patterns
     * Expand phonological: /p/ vs /b/, /d/ vs /t/ distinctions
   - **Week 6-8**: 30% visual, 25% temporal, 45% phonological foundations
     * Consolidate visual + temporal
     * Master phoneme boundaries (categorical perception)
     * Vowel categories (/a/ vs /i/ vs /u/)

   **Success Criteria**:
   - Visual: >95% accuracy on MNIST
   - Temporal: >90% next-item prediction
   - **Phonological: >90% phoneme discrimination** (NEW)
   - **Categorical perception curves match human infants** (NEW)

   **Rationale**:
   - Phoneme discrimination emerges 6-8 months in humans (Stage 2 timing!)
   - Earlier phonological foundation → better literacy (Stage 5)
   - Matches critical period for phonetic tuning
   - Interleaved practice prevents context-specific learning
   - Forces multi-modal integration from start

2. **Social Referencing Foundations** (Week 6-8, 30% of time) - NEW
   - **Gaze following**: Track where "caregiver" is looking
   - **Attention weighting**: Attended regions get learning boost
   - **Simple joint attention**: Look at what's being pointed at
   - Success: >80% gaze following accuracy

**CRITICAL Success Criteria** (Stage 2 must be rock-solid):
- Task performance: MNIST >95%, sequences >90%, **phonemes >90%** (NEW)
- **Categorical perception established** (sharp boundaries between phoneme categories)
- **Gaze following functional** (>80% accuracy) - NEW
- **Firing rate stability**: 0.05-0.15 maintained across 100k steps
- **No runaway excitation**: Criticality monitor shows stable/critical state (not supercritical)
- **BCM convergence**: Thresholds stabilize and stop drifting after 50k steps
- **Striatum balance**: D1/D2 weights maintain opponent relationship (if RL active)
- **No silence**: No region drops below 0.01 firing rate for >1000 steps
- **Weight health**: <80% of synapses saturated at min/max
- **Sensorimotor integration**: Visual-motor coordination from Stage 1 maintained

**Why Stage 2 is Critical**:
If these foundations aren't stable, every later stage will inherit instabilities.
Better to spend extra time here than debug cascading failures in Stage 5.
**NEW: Phonological foundation here enables natural literacy acquisition in Stage 5.**

**Training Details**:
- Batch size: 1 (single trial learning)
- Learning rate: Adaptive per region (dopamine-modulated)
- **Critical Period Gating**: Active for phonology (peak plasticity window)
- Steps per task: 15,000-60,000 (increased for phonological learning)
- Checkpoint: Every 5,000 steps
- **Temporal abstraction**: Single timescale (50ms bins, no chunking yet)

**Why Critical Periods Matter**:
- Explains why phonology MUST be Stage 2 (optimal window)
- Predicts learning rate differences across developmental stages
- Matches human bilingual advantage when early (<7 years ~ Stage 4)
- Grammar window opens later, stays open longer
- Semantic learning never fully closes (lifelong vocabulary learning)

**Expected Growth**: +15,000 neurons (43% increase to ~50,000 total)
- Primarily in cortex layers (sensory + phonological demand)
- Auditory cortex expansion for phonological processing
- Primarily in cortex layers (sensory demand)

---

### Stage 3: Object Permanence & Working Memory (Toddler Brain)
**Goal**: Develop working memory and object tracking

**Duration**: Week 8-16 (extended from Week 6-11 for language foundations)

**Size**: ~50,000 neurons (from Stage 2 growth)
- Add Prefrontal: 10,000 neurons (working memory with **theta-gamma coupling**)
- Expand Hippocampus: +5,000 (object associations)
- Expand Striatum: +3,000 (early policy learning)

**Tasks**:
1. **Object Recognition with Active Exploration** (Week 8-10)
   - CIFAR-10 (32x32 color images, 10 classes)
   - **Active viewing**: Use motor control from Stage 1 to "look around"
   - Multiple objects per image
   - Viewpoint invariance through active exploration
   - **Generation task**: Describe object from memory (not just recognize)
   - Success: >70% accuracy on CIFAR-10, >60% on object description

2. **Working Memory with Theta-Gamma Oscillations** (Week 9-11) - ENHANCED
   - **N-back tasks (N=1,2) using theta phase codes**
   - Delayed match-to-sample
   - Hold information for 100-500ms across theta cycles
   - **Productive failure**: Try 2-back before explicit teaching of strategies
   - Success: >80% on 2-back task

3. **Social Learning Foundations** (Week 10-11.5) - ENHANCED & EXPLICIT
   - **Imitation learning**: Copy demonstrated actions (fast learning!)
   - **Joint attention**: Gaze following, shared reference (building on Stage 2)
   - **Pedagogy detection**: Recognize teaching vs incidental observation
   - **Social referencing**: Use others' reactions to ambiguous stimuli
   - Success: >85% imitation accuracy, >80% joint attention

4. **Bilingual Language Foundations** (Week 11.5-13) - REVISED
   - **Two languages simultaneously**: English 60%, German 40%
   - Word recognition (100 words per language = 200 total)
   - Noun-verb associations in each language
   - Simple commands ("pick red", "nimm rot")
   - Code-switching recognition (mixing languages is natural)
   - **Phonological foundation from Stage 2**: Now map sounds → words
   - **Rhyme detection** (building on Stage 2 phoneme awareness)
   - **Syllable segmentation** (chunking phonemes → syllables)
   - **Generation over recognition**: Produce words, not just parse
   - Success: Execute 85% of commands in both languages, >80% phonological mapping

   **Why Phonology Already Established (Stage 2)?**
   - Stage 2: Phoneme categorical perception (sounds)
   - Stage 3: Map phonological representations → word meanings
   - Natural progression: sounds → words → grammar (next stage)

   **Why Start with Two Languages?**
   - Mirrors bilingual children (manageable cognitive load)
   - Working memory developing in Stage 3 (can handle two)
   - Prevents overload while establishing multilingual foundations
   - Spanish added gradually in Stage 4 when WM capacity is stronger
   - Still captures critical period advantage for multilingualism

5. **Binary Metacognitive Monitoring** (Week 12-13) - NEW, EARLIER
   - Learn to abstain: "I don't know" responses
   - Binary uncertainty only (no continuous confidence yet)
   - Provides signal for consolidation prioritization
   - Success: >70% correct abstention (abstain when wrong, respond when right)

   **Rationale**: Human children begin metacognitive awareness at 18-24 months.
   Starting here (not Stage 6) enables:
   - Earlier abstention (reduce "hallucinations")
   - Consolidation prioritization (replay high-uncertainty items)
   - Natural exploration (seek uncertain states)

6. **Executive Function: Inhibitory Control** (Week 12-13) - NEW
   - **Go/No-Go Tasks**: Respond to target, inhibit to non-target
   - **Simple delayed gratification**: Wait for larger reward
   - **Impulse control**: Suppress prepotent responses
   - Success: >75% correct inhibition

   **Why Critical**: Foundation for all later executive function and self-control

7. **Attention Mechanisms** (Week 11-13) - NEW
   - **Bottom-up salience**: Attend to bright, moving, novel stimuli
   - **Top-down task modulation**: Goal-directed attention (find red objects)
   - **Attentional control**: Resist distraction
   - Success: >70% target detection with distractors

   **Developmental Progression**:
   - Stage 3: Bottom-up dominant (70%), goal modulation weak (30%)
   - Stage 4: Balanced (50/50)
   - Stage 5+: Top-down dominant (30/70) - strategic attention control

---

#### Developmental Milestones

**Must achieve ALL before progressing to Stage 4:**

**Motor & Sensorimotor** (maintained from Stage 1):
- [ ] >95% accurate reaching toward targets
- [ ] <5% sensorimotor prediction error
- [ ] Stable proprioceptive representations
- [ ] Cerebellum forward models functional

**Perception**:
- [ ] >70% accuracy on CIFAR-10
- [ ] >95% MNIST accuracy (maintained from Stage 2)
- [ ] >90% phoneme discrimination (maintained from Stage 2)
- [ ] >80% gaze following accuracy

**Working Memory**:
- [ ] >80% accuracy on 2-back task
- [ ] Stable theta oscillations (7.5-8.5 Hz)
- [ ] Gamma-theta coupling functional
- [ ] Can maintain information across 100-500ms

**Language**:
- [ ] 100 words per language (200 total: English + German)
- [ ] >85% command following (both languages)
- [ ] >80% phonology→word mapping
- [ ] Simple noun-verb associations functional

**Executive Function**:
- [ ] >75% go/no-go accuracy
- [ ] Delayed gratification functional (wait for 1.5x reward)
- [ ] Impulse control demonstrated

**Attention**:
- [ ] >70% target detection with distractors
- [ ] Bottom-up salience functional (70% of attention)
- [ ] Top-down modulation emerging (30% of attention)

**Social**:
- [ ] >85% imitation accuracy
- [ ] >80% joint attention
- [ ] Pedagogy detection functional (1.5x learning boost)
- [ ] Social referencing from Stage 2 maintained

**Metacognition**:
- [ ] >70% correct abstention (binary "know" vs "don't know")
- [ ] Can signal uncertainty appropriately

**System Health**:
- [ ] Firing rates: 0.05-0.15 (stable)
- [ ] No runaway excitation episodes (>0.8 firing)
- [ ] Weight saturation <80%
- [ ] No region silence >1000 steps (>0.01 firing minimum)
- [ ] BCM thresholds stabilized
- [ ] Oscillator frequencies accurate (theta: 7.5-8.5 Hz)

**Backward Compatibility**:
- [ ] Stage 1 sensorimotor skills intact
- [ ] Stage 2 performance maintained (>90% of original)

**Growth & Capacity**:
- [ ] Grown to ~75,000 neurons (50% increase from Stage 2)
- [ ] Prefrontal cortex established (10k neurons)
- [ ] Language areas emerging (Wernicke/Broca precursors)

**Failure Modes Checked**:
- [ ] No runaway excitation in past 20,000 steps
- [ ] No silent regions in past 20,000 steps
- [ ] No catastrophic forgetting of Stage 2
- [ ] Striatum D1/D2 balance maintained (if RL active)

**Ready to Proceed to Stage 4 when ALL boxes checked**

---

**Training Details**:
- Mixed modalities (vision + language + motor)
- Introduce sparse reward (striatum engagement)
- Curriculum: Easy→Medium→Hard within each task
- Consolidation: Every 10,000 steps
- **Temporal abstraction**: Two-level hierarchy (50ms → 500ms)
  * Syllables as chunks (not phoneme sequences)
  * Object tracking across frames (not frame-by-frame)
  * Introduce "event" boundaries

**Expected Growth**: +25,000 neurons (50% increase to ~75,000 total)
- Prefrontal (working memory demand with theta-gamma)
- Hippocampus (more associations)
- Striatum (policy learning): +5,000 neurons
- Language areas: Early Wernicke/Broca precursors (+7,000)
- Prefrontal (working memory demand)
- Hippocampus (more associations)
- Striatum (policy learning): 8,000 neurons

---

### Stage 4: Grammar & Composition (Child Brain)
**Goal**: Learn compositional language and basic reasoning

**Duration**: Week 16-30 (extended from Week 11-20 for trilingual foundations)

**Size**: ~75,000 neurons (from Stage 3)
- Expand Language Areas: Wernicke (10k), Broca (8k)
- Cerebellum: +7,000 (sequence learning with temporal chunking)

**Tasks**:
1. **Multilingual Grammar Acquisition** (Week 16-24) - EXTENDED
   - **Gradual Spanish Introduction**:
     * Week 16-18: English 45%, German 35%, Spanish 20% (introduction)
     * Week 19-20: English 40%, German 35%, Spanish 25% (expansion) - 🔥 **Challenge Week**
     * Week 21-24: English 40%, German 30%, Spanish 30% (balanced)
   - Vocabulary: 1,000 words per language (3,000 total)
   - Language-specific grammar rules:
     * English: SVO, articles (a/the)
     * German: Case system (nominative, accusative), verb-second
     * Spanish: Gender agreement, verb conjugations
   - **GENERATION PRIORITIZED**: Produce sentences before parsing
   - Translation between languages (not word-for-word, conceptual)
   - Code-mixing understanding ("I have ein Hund")
   - **Productive failure**: Attempt Spanish grammar before explicit teaching
   - **Social learning**: Learn grammar from demonstrated examples (not just rules)
   - Success: >80% grammatical generation (not just recognition) in each language

   **Desirable Difficulties**:
   - Week 13-14 Challenge: 90% difficulty, interleaved syntax (English SVO + German V2)
   - Temporal spacing: Review Stage 2-3 tasks with 2-3 day gaps
   - Generation over recognition: Produce sentences before comprehension tests
   - **Testing effect**: Low-stakes grammar tests without feedback

2. **Cross-Lingual Compositional Reasoning** (Week 24-26)
   - Same concept, different expressions
   - "The red ball" / "Der rote Ball" / "La pelota roja"
   - Spatial relations across languages
   - Simple inference in any language
   - **Enhanced social cognition**: Intention recognition, simple false belief, perspective-taking
   - **Cultural learning**: Group conventions differ by language community
   - **Generation task**: Create novel concept descriptions in each language
   - **Cross-modal binding with gamma synchrony**: Visual object + auditory label
   - Success: Answer 75% of reasoning questions, generate 70% correct descriptions

3. **Multilingual Multi-Step Instructions** (Week 26-28)
   - Follow 3-step commands in mixed languages
   - "Take the ball, nimm es, and pon it here"
   - Language detection and switching
   - **Social learning**: Learn from demonstration (not just instruction)
   - **Imitation + pedagogy detection**: Fast learning from teachers
   - Success: Complete 80% of multilingual tasks

4. **Coarse Metacognitive Confidence** (Week 27-28) - PROGRESSIVE
   - Expand from binary (Stage 3) to coarse confidence (high/medium/low)
   - Still poorly calibrated (like 3-year-olds)
   - Provides richer signal for consolidation
   - Success: 3-level confidence somewhat correlated with accuracy (not well-calibrated yet)

5. **Executive Function: Set Shifting** (Week 26-28) - NEW
   - **Dimensional Change Card Sort (DCCS)**: Sort by color, then by shape
   - **Task switching**: Alternate between two rule sets
   - **Cognitive flexibility**: Inhibit old rule, activate new rule
   - Success: >70% on switch trials (vs >90% on repeat trials)

   **Why Critical**: Enables language switching, multi-task learning, flexible behavior

**Training Details**:
- Introduce neuromodulation (dopamine for success/failure)
- Episodic memory replay (hippocampus)
- Language + vision + action + social integration
- Consolidation: Every 15,000 steps
- **Temporal abstraction**: Three-level hierarchy (50ms → 500ms → 5s)
  * Words as chunks (not syllable sequences)
  * Action plans (not individual movements)
  * Sentence-level processing with theta-gamma nesting

**Expected Growth**: +50,000 neurons (67% increase to ~125,000 total)
- Language regions (grammar complexity)
- Cerebellum (precise sequences with chunking)
- Expanded cortical representations
- Social cognition circuits
- Language regions (grammar complexity)
- Cerebellum (precise sequences)
- Expanded cortical representations

---

### Stage 5: Reading & Writing (Elementary Brain)
**Goal**: Process written language, generate coherent text

**Duration**: Week 30-46 (extended from Week 20-32 for trilingual literacy)

**Size**: ~175,000 neurons (from Stage 4 growth)
- Expand Wernicke/Broca: +15k each
- Visual word form area: 12,000 (orthography)
- Semantic network: 25,000 (concepts)

**Tasks**:
1. **Multilingual Reading Comprehension** (Week 30-38) - EXTENDED
   - Vocabulary: 5,000 words per language (15,000 total)
   - Short paragraphs in English, German, Spanish (3-5 sentences)
   - **GENERATION FIRST**: Summarize passage without looking (retrieval practice)
   - Answer comprehension questions in any language
   - Cross-lingual reading (read German, answer in English)
   - **Testing effect**: Frequent low-stakes quizzes without feedback
   - **Leverage phonological foundation from Stage 2**: Use decoding strategies
   - **Letter-sound correspondences**: Now make sense (sounds learned in Stage 2)
   - Success: >70% reading comprehension, >65% summary quality across languages

2. **Multilingual Text Generation** (Week 38-42) - EXTENDED
   - **Generation prioritized**: Produce before recognizing
   - Complete sentences in target language
   - Simple stories in each language (3-4 sentences)
   - Maintain language consistency (don't mix mid-sentence)
   - Translation (conceptual, not literal)
   - **Create novel analogies** (not just recognize them)
   - **Productive failure**: Try complex narratives before scaffolding
   - Success: Human judges rate 65% as coherent in each language

3. **Multilingual Dialogue & Pragmatics** (Week 42-46) - EXTENDED
   - Q&A in any language
   - Respond in same language as question
   - Context maintenance across language switches
   - Detect language from input
   - **Advanced pragmatics**: Sarcasm detection, implicature, politeness
   - **Enhanced social cognition**: Understand speaker intentions, cultural norms
   - **Theory of Mind**: Predict what interlocutor believes/knows
   - **Generation focus**: Produce contextually appropriate responses
   - Success: 75% contextually appropriate responses in correct language

4. **Continuous Metacognitive Confidence** (Week 44-46) - PROGRESSIVE
   - Expand from 3-level (Stage 4) to continuous confidence (0-100%)
   - Begin calibration training
   - Use confidence to prioritize consolidation (high-error items)
   - Success: Continuous confidence estimates, calibration improving (ECE < 0.25)

**Metacognitive Calibration Training Protocol** (Week 44-46):
```python
class ConfidenceCalibrationTraining:
    """
    Explicit training to calibrate confidence estimates.

    Psychology: Metacognitive monitoring improves with feedback
    (Schraw & Dennison, 1994).
    """

    def calibration_feedback_loop(self, brain, test_set, training_fraction=0.20):
        """
        Train brain to match confidence to accuracy.

        Args:
            training_fraction: 20% of Stage 5 training time on calibration
        """

        for batch in test_set:
            # Get prediction + confidence
            output = brain.forward(batch['input'])
            confidence = brain.estimate_confidence(output)

            # Reveal ground truth
            actual_correct = (output.prediction == batch['label'])

            # Compute calibration error
            calibration_error = abs(confidence - float(actual_correct))

            # Feedback signal (dopamine modulated)
            if calibration_error < 0.10:
                dopamine = 1.0  # Well-calibrated!
            elif calibration_error > 0.30:
                dopamine = 0.3  # Poorly calibrated
            else:
                dopamine = 0.7  # Moderate

            # Update confidence estimation network
            brain.metacognitive_module.learn(
                error=calibration_error,
                dopamine=dopamine
            )

            # Log for analysis
            log_calibration_metrics(confidence, actual_correct, calibration_error)
```

**Calibration Schedule**:
- Week 44-46 (Stage 5): 20% of training time on calibration tasks
- Week 60-64 (Stage 6): 30% of training time on calibration refinement
- Goal: ECE < 0.25 by end of Stage 5, ECE < 0.15 by end of Stage 6

5. **Executive Function: Planning** (Week 42-46) - NEW
   - **Tower of Hanoi**: Multi-step planning with subgoals
   - **Maze solving**: Plan path before execution
   - **Goal decomposition**: Break complex goal into subgoals
   - Success: >60% on 3-step planning tasks

   **Why Critical**: Foundation for complex reasoning, problem-solving, goal-directed behavior

6. **Scaffolding and Fading** (Week 30-46) - NEW
   - **High scaffolding** (Week 30-38): Examples + hints provided
   - **Medium scaffolding** (Week 38-42): Partial hints only
   - **Low scaffolding** (Week 42-46): Minimal support
   - **Adaptive fading**: Based on performance

   **Why Critical**: Matches Zone of Proximal Development, optimal challenge level

**Training Details**:
- Token-level prediction (next word)
- Sentence-level generation (full thoughts)
- Contrastive learning (good vs bad examples)
- Consolidation: Every 20,000 steps
- Introduce "sleep" phases (offline replay)
- **Temporal abstraction**: Four-level hierarchy (50ms → 500ms → 5s → 30s)
  * Paragraph comprehension (multi-sentence integration)
  * Multi-step reasoning (plan across sentences)
  * Narrative structure (story arcs)
  * Theta-gamma-slow (1 Hz) nested oscillations

**Expected Growth**: +100,000 neurons (57% increase to ~275,000 total)
- Massive language network expansion
- Semantic representations
- Generative pathways
- Theory of Mind circuits

**Conservative Pruning Begins** (Stage 5):
- **Biological Rationale**: Human synaptic pruning begins ~age 3 but is gradual
  * Peak density at age 2 (Stage 3 equivalent)
  * ~2-3% reduction per year through adolescence
  * 50% total reduction from peak to age 18 (over 16 years)
- **Implementation**: 1% per consolidation cycle (conservative start)
  * Much more gradual than initial 5-10% proposal
  * Mirrors biological pruning rates (~3% annually)
  * Risk mitigation: Conservative approach prevents removing useful connections
- Remove inefficient synapses (low activity + low importance)
- Preserve high-importance pathways (importance > 0.3)
- Improves efficiency without forgetting

---

### Stage 6: Abstract Reasoning (Adolescent Brain)
**Goal**: Develop abstract thought, analogies, complex reasoning

**Duration**: Week 46-70 (extended from Week 32-56 for calibration maturation)

**Size**: ~375,000 neurons (from Stage 5 growth)
- Prefrontal expansion: +35k (executive function with dendritic computation)
- Parietal regions: +30k (spatial/abstract reasoning)
- Expanded hippocampus: +20k (episodic sophistication)

**Tasks**:
1. **Analogical Reasoning** (Week 46-54) - EXTENDED
   - "A is to B as C is to ___"
   - **GENERATION FOCUS**: Create novel analogies (not just solve them)
   - Conceptual similarities
   - Transfer learning across domains
   - **Testing effect**: Frequent retrieval practice
   - **Productive failure**: Attempt cross-domain analogies before teaching
   - Success: >70% solving analogies, >60% creating valid analogies

2. **Mathematical Reasoning** (Week 54-60) - EXTENDED
   - Basic arithmetic (learned, not hardcoded)
   - Word problems
   - Simple algebra
   - **Generation**: Explain solution steps (not just answer)
   - **Testing effect**: Problem-solving without immediate feedback
   - Success: >75% on grade-school math, >65% explanation quality

3. **Commonsense Reasoning** (Week 60-64) - EXTENDED
   - Physical intuition (objects fall, liquids pour)
   - Social reasoning (people have goals)
   - Causal inference
   - **Social learning**: Learn physics from observation, not just rules
   - **Generation**: Predict outcomes before seeing them
   - Success: >70% on PIQA, Social IQA benchmarks

4. **Advanced Social & Emotional Intelligence** (Week 64-68) - EXTENDED
   - Emotion recognition from text/context
   - **Complex Theory of Mind**: Second-order beliefs ("Alice thinks Bob believes...")
   - Moral reasoning and ethical dilemmas
   - Social norms and politeness (culture-specific)
   - Perspective-taking across cultures/languages
   - **Cultural learning**: Acquire group-specific conventions
   - Success: >70% on complex ToM benchmarks, social reasoning tasks

5. **Metacognitive Mastery & Active Learning** (Week 68-70) - REFINED (Not New!)
   - **Calibrated confidence**: Refine continuous estimates from Stage 5 (goal: ECE < 0.15)
   - **30% of training time on calibration refinement** (increased from Stage 5's 20%)
   - **Abstention mastery**: Know when to say "I don't know" (practiced since Stage 3)
   - **Calibration**: Match confidence to actual accuracy (goal: ECE < 0.15)
   - Monitor population variance → confidence signal
   - **🔥 Metacognitive curriculum control** (NEW capability):
     * Brain selects next task based on uncertainty
     * Active learning: Study what you don't know
     * Self-directed difficulty adjustment
   - Success: Calibration error <0.15, appropriate abstention rate, >70% self-selection accuracy

   **Developmental Progression**:
   - Stage 3: Binary uncertainty ("know" vs "don't know")
   - Stage 4: Coarse confidence (high/medium/low)
   - Stage 5: Continuous confidence (0-100%), poorly calibrated
   - Stage 6: **Well-calibrated confidence** + active learning control

6. **Dendritic Computation for Credit Assignment** (Week 68-70) - NEW
   - Use dendritic branches for compositional reasoning
   - Multi-step logical inference without backprop
   - "If A and B, then C" reasoning locally
   - Success: >65% on multi-premise reasoning tasks

7. **Executive Function: Fluid Reasoning** (Week 64-70) - NEW
   - **Raven's Progressive Matrices**: Abstract pattern induction
   - **Analogical reasoning**: Structure mapping across domains
   - **Hypothesis testing**: Generate and evaluate hypotheses
   - Success: >65% on matrix reasoning tasks

   **Developmental Summary (EF)**:
   - Stage 3 (12-24 mo): Inhibitory control (go/no-go)
   - Stage 4 (2-5 yr): Set shifting (DCCS, task switching)
   - Stage 5 (6-10 yr): Planning (Tower of Hanoi, subgoaling)
   - Stage 6 (12-18 yr): Fluid reasoning (Raven's, analogies)

   **Why This Sequence**: Matches prefrontal cortex maturation trajectory

**Training Details**:
- Multi-task training (mix all previous skills)
- Harder negatives (near-miss answers)
- Explanations (why/how questions)
- **Curriculum mixing ratio**: 70% new tasks, 30% review from previous stages
- **Metacognitive curriculum control** (Stage 6 only):
- **Backward compatibility checks**: Every 10k steps, test sample from all previous stages
- Consolidation: Every 30,000 steps (or when memory pressure high)

**Expected Growth**: +150,000 neurons (40% increase to ~525,000 total)
- Abstract reasoning circuits with dendritic computation
- Cross-domain integration
- Metacognitive regions (anterior cingulate analog)

**Conservative Pruning Continues** (Stage 6):
- **Rate**: 2% per consolidation cycle (modest increase from Stage 5's 1%)
- **Peak adolescent pruning**: Mirrors human brain refinement (12-18 years)
- **Biology**: Humans prune ~3% annually during adolescence
- **Target**: Remove redundant connections while preserving learned knowledge
- **Monitoring**: Track performance on all previous stages during pruning
- **Safety**: If any stage drops >adaptive threshold, pause pruning

---

### Stage 7: Expert Knowledge (Young Adult Brain)
**Goal**: Acquire specialized knowledge, maintain generality

**Duration**: Week 70-106 (extended from Week 56-88)

**Size**: ~675,000 neurons (from Stage 6 growth)
- Domain-specific modules: +100k distributed
- Expanded semantic memory: +50k

**Tasks**:
1. **Domain Expertise** (Week 70-88) - EXTENDED
   - Science (biology, physics, chemistry)
   - History & geography
   - Literature & arts
   - Technical skills (coding basics)
   - Success: Pass domain-specific tests >65%

2. **Long-Form Generation** (Week 88-96) - EXTENDED
   - Essays (200+ words)
   - Maintain coherence over multiple paragraphs
   - Structured arguments
   - Success: Human evaluation >70% quality

3. **Multi-Modal Integration** (Week 96-106) - EXTENDED
   - Vision + language (image captioning)
   - Audio + language (speech understanding)
   - Cross-modal reasoning with gamma synchrony
   - Success: >75% on multi-modal benchmarks

**Training Details**:
- Curriculum mixture (revisit all previous stages)
- **Mixing schedule**: 50% new domain expertise, 50% prior stages (weighted by recency)
- Prevent forgetting (weighted replay based on task importance)
- Sparse high-quality data
- **Data augmentation**: Paraphrasing, back-translation, synthetic examples
- Consolidation: Every 50,000 steps
- **Offline "sleep" consolidation**: Every 100k steps (decreasing from 20k in Stage 2)

**Expected Growth**: +200,000 neurons (30% increase to ~875,000 total)
- Distributed expertise
- Cross-modal pathways
- Refinement of existing circuits

**Pruning: Moderate and Declining** (Stage 7):
- **Rate**: 2% per consolidation cycle (maintained from Stage 6)
- **Focus**: Remove redundant domain-specific connections
- **Preserve**: Core competencies from all previous stages
- **Goal**: Optimize for efficiency while maintaining breadth
- **Biology**: Pruning continues but slows in young adulthood

---

### Stage 8: LLM-Level Capabilities (Adult Brain)
**Goal**: Match or exceed current LLM performance

**Duration**: Week 106-192 (extended from Week 88-168)

**Size**: ~1,000,000 neurons
- Full brain integration
- Optimized for efficiency

**Tasks**:
1. **General Language Understanding** (Week 106-130) - EXTENDED
   - GLUE/SuperGLUE benchmarks
   - Reading comprehension (SQuAD, RACE)
   - Natural language inference
   - Success: Within 85% of GPT-3.5 performance

2. **Complex Reasoning** (Week 130-154) - EXTENDED
   - Chain-of-thought reasoning
   - Multi-hop question answering
   - Logical puzzles
   - Success: >80% on complex reasoning benchmarks

3. **Few-Shot Learning** (Week 154-174) - EXTENDED
   - Learn new tasks from 1-5 examples
   - Rapid adaptation
   - Meta-learning
   - Success: Match few-shot GPT-3 performance

4. **Instruction Following** (Week 174-192) - EXTENDED
   - Complex multi-step instructions
   - Constrained generation
   - Tool use
   - **Calibrated confidence**: Report uncertainty appropriately
   - Success: >85% instruction compliance

5. **Metacognitive Mastery** (integrated throughout)
   - Use confidence to trigger knowledge search
   - Abstain when uncertain (avoid hallucinations)
   - Match human judge calibration
   - Explain reasoning and uncertainty

**Training Details**:
- High-quality curated data (books, papers, conversations)
- Instruction tuning
- Human feedback (RLHF-style with dopamine)
- **Curriculum mixing**: 40% LLM-level tasks, 60% comprehensive review of all stages
- **Continual learning validation**: Full evaluation suite from Stage 1 to 5 every 50k steps
- Continuous consolidation
- Delta checkpoints (most weights stable)
- **Data augmentation**: Question generation, synthetic dialogues, adversarial examples

**Expected Growth**: +200,000 neurons (23% increase to ~1,200,000 total)
- Refinement more than expansion
- Optimization of critical pathways

**Pruning: Minimal Refinement** (Stage 8):
- **Rate**: 1% per consolidation cycle (reduced for final optimization)
- **Focus**: Fine-tuning and refinement only
- **Conservative**: Avoid removing established expertise
- **Final network**: ~50% smaller than peak (mirrors human adolescence→adulthood)
- **Biology**: Adult pruning minimal, primarily maintenance

**Conservative Pruning Summary Across Stages**:
- Stage 4: 1% (gentle introduction)
- Stage 5: 1% (early pruning phase)
- Stage 6: 2% (peak adolescent pruning)
- Stage 7: 2% (continued refinement)
- Stage 8: 1% (minimal maintenance)
- **Total reduction**: ~30-40% from peak (vs 50% in humans, more conservative)
- **Rationale**: Better to under-prune than risk losing learned knowledge

---

## Training Infrastructure

### Curriculum Mixing Strategy

**Purpose**: Balance learning new skills while maintaining old knowledge

**Stage-Specific Ratios** (Initial):
- **Stage 2-3**: 100% current stage (establishing foundations)
- **Stage 4**: 80% new, 20% Stage 2-3 review
- **Stage 5**: 70% new, 30% Stage 2-4 review
- **Stage 6**: 70% new, 30% Stage 2-5 review (weighted by recency)
- **Stage 7**: 50% new, 50% comprehensive review
- **Stage 8**: 40% new, 60% comprehensive review (prevent catastrophic forgetting)

**Interleaved Practice Within Sessions**
**Spaced Repetition for Stage Review**
**Adaptive Loss-Weighted Replay** (Stage 4+)

**Example** (Stage 6 adaptive distribution with spaced repetition):
- If Stage 3 degraded to 85%: Gets 40% of review time + shortened interval (10k steps)
- If Stage 4 stable at 91%: Gets 15% of review time + extended interval (75k steps)
- If Stage 5 stable at 89%: Gets 20% of review time + moderate interval (25k steps)
- Stage 2 always gets baseline 5% (foundation preservation) + very long intervals (100k+ steps)

### Dynamic Difficulty Adjustment

**Purpose**: Maintain optimal learning zone (Vygotsky's ZPD)

**Usage**: Apply every 1000 steps, log difficulty trajectory for analysis

---

### Cognitive Load Monitoring

**Purpose**: Prevent cognitive overload from simultaneous demands

**Rationale**: Multiple mechanisms (working memory, language switching, executive function, attention control) impose cognitive load. Exceeding capacity causes performance degradation and learned helplessness.

**Usage Guidelines**:
- **Stage 2-3**: Capacity = 2-3 chunks (introduce 1 new mechanism at a time)
- **Stage 4**: Capacity = 3-4 chunks (can handle 2 mechanisms)
- **Stage 5+**: Capacity = 4-7 chunks (can handle multiple demands)

**Benefits**:
- Prevents overwhelming the brain with too many simultaneous demands
- Explains why certain stage transitions are extended
- Guides scheduling of new mechanism introductions
- Maintains optimal challenge (Vygotsky's ZPD)

---

### Testing Effect / Retrieval Practice

**Purpose**: Testing beats re-studying for long-term retention

**Evidence**: One of most robust findings in learning science (Roediger & Karpicke, 2006)

**Testing Schedule**:
- **Stage 2-3**: 10% of steps are tests (building foundations)
- **Stage 4-5**: 15% of steps are tests (expanding)
- **Stage 6-8**: 20% of steps are tests (mastery)

**Key Principles**:
1. **No immediate feedback** (delayed feedback better for retention)
2. **Low stakes** (no penalties, just practice)
3. **Frequent** (regular retrieval beats cramming)
4. **Varied** (mix question formats)

---

### Productive Failure Phases

**Purpose**: Let brain struggle BEFORE instruction → better learning

**Evidence**: Kapur (2008) - failure before teaching beats immediate success

**Usage**:
- Before each new stage (100-200 steps productive failure)
- Before introducing new task types (e.g., 3-back, Spanish grammar)
- NOT for foundational skills (Stage 2) - only after Stage 3

**Why It Works**:
- Activates relevant prior knowledge
- Reveals knowledge gaps
- Increases attention during subsequent instruction
- Creates "need to know" motivation

---

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
- Stage 2-4: Minimal augmentation (<5% of data, only basic transforms)
- Stage 5-6: Moderate augmentation (10-15% of data)
- Stage 7-8: Conservative augmentation (20% max, preserve semantic fidelity)

**Rationale**: Biological learning benefits from clean, consistent patterns.
Over-augmentation can interfere with precise semantic learning and
biological plausibility. Keep augmentation conservative.

### Offline "Sleep" Consolidation Protocol

**Purpose**: Strengthen important memories, integrate knowledge

**Inspiration**: Hippocampal replay during sleep in mammals

**Memory Pressure-Triggered Consolidation**

**Adaptive Spaced Consolidation** (enhanced with memory pressure):

**Event-Based Triggers** (override adaptive schedule):
1. **Before stage transition**: ALWAYS consolidate
2. **After catastrophic forgetting**: If any previous stage drops >10%
3. **On performance plateau**: If no improvement for 15k steps
4. **After major growth**: When >20% neurons added to any region

**Minimum Frequency**: Never exceed 200,000 steps without consolidation (risk of drift)

**Rationale**: Matches optimal spacing from memory research (expanding intervals),
consolidates during active learning not plateau, prevents overconsolidation.

**Duration**: 10,000 replay steps per session (can be distributed across ultradian cycles)

---

### Stage Transition Protocols

**Purpose**: Smooth transitions between curriculum stages to prevent learned helplessness from difficulty jumps

**Rationale**: Sudden difficulty increases cause frustration and performance drops. Gradual transitions preserve motivation and build confidence.

**Transition Checklist** (Applied at every stage boundary):
1. **Extended consolidation** (double normal duration)
2. **Milestone evaluation** (all criteria must pass)
3. **Gradual difficulty ramp** (4-week intro: 0.3 → 0.5 → 0.7 → 1.0)
4. **High initial review** (70% → 50% → 30% over 3 weeks)
5. **Cognitive load monitoring** (prevent overload during transition)
6. **Backward compatibility check** (previous stage maintained >90%)

**Example Transition** (Stage 4 → Stage 5):
```
Week 28-29: Extended consolidation (10 cycles instead of 5)
Week 29: Evaluate Stage 4 milestones
  Grammar: 82% (>80% threshold)
  Set shifting: 73% (>70% threshold)
  Coarse confidence: Working
  Stage 2-3 maintained: 91% and 89%
  → ALL PASSED, proceed to Stage 5

Week 30: Stage 5 begins
  - Difficulty: 0.3 (very easy reading tasks)
  - Mixing: 70% Stage 4 review, 30% Stage 5 intro
  - Load: MANAGEABLE (0.6 of capacity)

Week 31:
  - Difficulty: 0.5 (easy reading)
  - Mixing: 50% Stage 4, 50% Stage 5
  - Load: HIGH_LOAD (0.9 of capacity) ← Optimal!

Week 32:
  - Difficulty: 0.7 (moderate)
  - Mixing: 30% Stage 4, 70% Stage 5

Week 33+:
  - Difficulty: 1.0 (full)
  - Mixing: 30% Stage 4, 70% Stage 5 (maintained)
```

**Benefits**:
- Prevents "cliff" transitions that cause failures
- Builds confidence through early success
- Maintains old knowledge during new learning
- Provides clear go/no-go criteria
- Reduces rollback frequency

---

**Ultradian Sleep Cycles** ( Mimics 90-min REM/NNEM architecture)

**Why Ultradian Cycles Matter**:
- **Early SWS**: Stabilizes new memories (hippocampus→cortex transfer)
- **Late REM**: Extracts abstract schemas, integrates knowledge
- **Alternation**: Prevents interference between consolidation modes
- **Biologically accurate**: Matches human sleep architecture

**Temporal Mapping** (Steps → Biological Time):
```python
# Biological sleep: 7.5 hours = 5 cycles × 90 minutes
# Thalia equivalent mapping:
BIOLOGICAL_SLEEP_HOURS = 7.5
TRAINING_STEPS_PER_HOUR = 1333  # 10,000 steps = 7.5 hours
STEPS_PER_ULTRADIAN_CYCLE = 2000  # "90 minutes" equivalent
N_CYCLES_PER_CONSOLIDATION = 5

# Total: 10,000 steps = 5 cycles × 2000 steps/cycle
# Note: "Steps" are abstract time units, not wall-clock time
# Compression factor: ~1000x faster than biological real-time
```

**Implementation Note**: Steps represent computational time, not wall-clock time. One "training step" ≈ one forward pass + learning update. The biological equivalence is conceptual (for curriculum design) rather than literal timing.

**Replay Selection Strategy**:
1. **Prediction Error-Driven Prioritization**

2. **Enhanced Proportions**:
   - 35% high prediction-error experiences (learn from mistakes!)
   - 25% high-reward experiences (preserve successes)
   - 20% novel/boundary cases (challenging)
   - 10% low-error experiences (maintain stable knowledge)
   - 10% random baseline (avoid overfitting)

3. **Temporal Compression**:
   - Recent experiences: Replay at 1x speed
   - Older experiences: Replay at 5x speed (faster reactivation)
   - Very old: Occasional 10x speed (semantic consolidation)

**Metrics to Track**:
- Synapse strength changes (should increase for important connections)
- Firing rate stability (should be more consistent)
- Task performance before/after sleep (should improve or maintain)

---

### Compute Requirements

**Stage 2-4** (Weeks 1-14):
- Single GPU (RTX 3090 or better)
- ~8GB VRAM sufficient
- Training time: ~2-4 hours per task

**Stage 5-6** (Weeks 15-33):
- Single GPU (RTX 4090 or A100)
- ~16GB VRAM
- Training time: ~8-12 hours per task

**Stage 7-8** (Weeks 34-90):
- Multi-GPU or single A100 80GB
- ~40-60GB VRAM
- Training time: ~1-3 days per task

---

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

---

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

## Evaluation Protocols

### Standardized Test Sets

**Purpose**: Consistent evaluation across stages, detect forgetting

**Test Set Structure** (per stage):
- **Holdout Size**: 10% of stage data (never seen during training)
- **Composition**: Representative of all tasks in that stage
- **Languages**: Balanced across English, German, Spanish
- **Difficulty**: Easy (40%), Medium (40%), Hard (20%)

**Stage-Specific Benchmarks**:

**Stage 2**:
- MNIST test set (10k images)
- Custom sequence prediction (1k sequences)
- TIMIT phoneme recognition (500 samples)

**Stage 3**:
- CIFAR-10 test set (10k images)
- N-back working memory (500 trials per N)
- Multilingual command following (300 commands per language)

**Stage 4**:
- Grammar test suite (1k sentences per language)
- Compositional reasoning (500 questions per language)
- Planning tasks (200 multi-step scenarios)

**Stage 5**:
- Reading comprehension (SQuAD subset, 1k passages)
- Text generation quality (200 prompts, human evaluation)
- Dialogue coherence (100 multi-turn conversations)

**Stage 6**:
- Analogy test (500 items: semantic, spatial, logical)
- Math word problems (500 problems, grade 1-5)
- Commonsense reasoning (PIQA: 500 items, Social IQA: 500 items)
- Theory of Mind (500 scenarios)

**Stage 7**:
- Domain knowledge tests (200 questions per domain: science, history, arts)
- Long-form generation (50 essay prompts)
- Multi-modal integration (COCO captions: 500 images, VQA: 1k questions)

**Stage 8**:
- MMLU (1k questions, representative subset)
- HellaSwag (1k items)
- HumanEval (164 coding problems)
- MT-Bench (80 instruction pairs)

### Continual Learning Validation

**Purpose**: Ensure brain doesn't forget previous stages

**Protocol**:
1. **Baseline**: Establish performance on all previous stage tests at end of each stage
2. **Checkpoints**: Re-evaluate all previous stages every 50k steps
3. **Adaptive Threshold**: Alert threshold scales with brain size and stage
4. **Action**: Increase review proportion for forgotten stages

**Adaptive Forgetting Thresholds**:
```python
def adaptive_forgetting_threshold(stage, n_neurons):
    """
    Stricter thresholds as brain grows.

    Rationale: 10% of 1M neurons >> 10% of 50k neurons
    Larger brains have more to lose, need tighter monitoring.
    """
    base_threshold = 0.10  # 10% baseline

    # Scale inversely with size (larger = stricter)
    size_penalty = 1.0 - 0.3 * np.log10(n_neurons / 50000)

    # Scale with stage (more to lose in later stages)
    stage_penalty = 1.0 - 0.05 * stage

    threshold = base_threshold * size_penalty * stage_penalty
    return max(0.05, threshold)  # Floor at 5%
```

**Worked Examples** (Adaptive Threshold Calculation):

| Stage | N Neurons | Size Penalty | Stage Penalty | Threshold | Notes |
|-------|-----------|--------------|---------------|-----------|-------|
| 0 | 50,000 | 1.00 | 1.00 | **10.0%** | Baseline, small network |
| 1 | 75,000 | 0.93 | 0.95 | **8.8%** | Growing, slight tightening |
| 2 | 125,000 | 0.88 | 0.90 | **7.9%** | Moderate size, more to lose |
| 3 | 275,000 | 0.76 | 0.85 | **6.5%** | Large network, stricter |
| 4 | 525,000 | 0.70 | 0.80 | **5.6%** | Very large, tight monitoring |
| 5 | 875,000 | 0.62 | 0.75 | **4.7%** ⚠ Below floor → **5.0%** | Capped at minimum |
| 6 | 1,200,000 | 0.59 | 0.70 | **4.1%** ⚠ Below floor → **5.0%** | Capped at minimum |

**Interpretation**:
- **Stage 2-4**: More tolerant of forgetting (7-10% acceptable)
- **Stage 5-6**: Stricter monitoring as knowledge base grows (5-7%)
- **Stage 7-8**: Floor at 5% prevents over-sensitivity (still very strict)
- **Rationale**: Larger networks have more synapses at risk; can't tolerate same percentage drop

**Example** (Stage 6, ~525k neurons):
- **Backward Transfer**: Performance on Stage N after training Stage N+K
- **Forward Transfer**: Performance on Stage N+1 given Stage N training
- **Retention Rate**: % of original performance maintained

**Example Matrix** (Stage 6 completion, ~525k neurons):
```
         Stage0  Stage1  Stage2  Stage3  Stage4  Threshold
Initial   98%     95%     92%     88%     85%     -
+50k      97%     94%     91%     87%     88%     5.6%  ✓ All within threshold
+100k     96%     92%     89%     86%     91%     5.6%  ✓ Slight decay, acceptable
+150k     94%     89%     85%     84%     93%     5.6%  ⚠ Stage 4 at -7.6% (EXCEEDS), Stage 3 at -6.3% (EXCEEDS)
→ Action: Increase Stage 3-4 review from 30% to 40%

Note: Threshold = 5.6% for Stage 6 with 525k neurons (adaptive formula)
Earlier stages used: 10.0% (Stage 2), 8.8% (Stage 3), 7.9% (Stage 4), 6.5% (Stage 5)
Adaptive thresholds get stricter as brain grows and accumulates more knowledge.
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
brain.regions[problem_region].grow_output(
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

## Common Failure Modes & Prevention

### Expected Failure Modes by Stage

**1. Runaway Excitation** (Most common in Stage 2-3)
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

**3. Catastrophic Forgetting** (Stage 4+)
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

**4. Capacity Saturation** (Stage 5+)
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

**5. Oscillator Desynchronization** (Stage 4+ with hippocampus)
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
- [ ] Check firing rates (0.05-0.25 is healthy)
- [ ] Check weight saturation (<80% healthy)
- [ ] Check gradient magnitudes (1e-5 to 1e-2 healthy)
- [ ] Check neuromodulator levels (in expected ranges)
- [ ] Check oscillator frequencies (if applicable)
- [ ] Check for NaN/Inf in any tensor
- [ ] Check task performance on previous stages
- [ ] Review recent growth events
- [ ] Check consolidation frequency
- [ ] Look for sudden distribution shifts in data

---

## Success Metrics

### Metacognitive Monitoring (Stage 6+)

**Purpose**: Enable brain to gauge its own uncertainty and abstain when appropriate

**Training Schedule**:
- **Stage 6**: Train to report uncertainty ("I'm not sure...")
- **Stage 7**: Use confidence to trigger knowledge search
- **Stage 8**: Enable calibrated predictions (match human judges)

---

### Per-Stage Metrics
- Task-specific accuracy (defined per stage)
- Firing rate stability (no runaway or silence)
- Weight saturation (<80% of neurons maxed out)
- Cross-task transfer (new tasks benefit from old knowledge)
- **Backward transfer**: Accuracy on previous stages (continual learning)
- **Data efficiency**: Steps to criterion vs baseline
- **Consolidation quality**: Performance improvement after sleep phases
- **Confidence calibration** (Stage 6+): Match predicted probability to actual accuracy

### Global Metrics
- **Sample Efficiency**: Steps to reach criterion vs backprop baseline
- **Continual Learning**: Performance on Stage N tasks after training Stage N+3 (should be >90% of original)
- **Generalization**: Zero-shot performance on held-out distributions
- **Biological Plausibility**: Local learning rules maintained throughout
- **Multilingual Balance**: Performance gap between languages <10%
- **Social Intelligence**: Theory of Mind accuracy, pragmatic understanding
- **Robustness**: Performance under adversarial/noisy conditions

### Additional Learning Dynamics Metrics

**Sample Efficiency**:
- `steps_to_criterion`: How fast did it learn?
- `sample_efficiency_ratio`: vs. backprop baseline

**Generalization**:
- `ood_performance`: Out-of-distribution test set
- `compositional_generalization`: Novel combinations

**Stability**:
- `variance_over_time`: Is learning stable?
- `catastrophic_forgetting_index`: Sum of performance drops

**Biological Plausibility**:
- `local_learning_ratio`: % of updates truly local (should be 100%)
- `spike_efficiency`: Information per spike (bits/spike)
- `metabolic_cost`: Total spikes generated (lower is better)

**Network Efficiency** (Stage 5+):
- `pruning_ratio`: % synapses removed without performance loss
- `parameter_efficiency`: Performance per synapse
- `inference_speed`: Spikes per forward pass

**Confidence Calibration** (Stage 6+):
- `calibration_error`: Predicted probability vs actual accuracy
- `abstention_accuracy`: Correct when brain says "I don't know"
- `expected_calibration_error`: ECE metric

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
- Better few-shot learning (hippocampus one-shot)
- No catastrophic forgetting (curriculum + consolidation)
- Continual adaptation (ongoing learning)
- More interpretable (discrete spikes, local rules)
- Biologically grounded (can inform neuroscience)

**Timeline Caveat**: 50+ months (REVISED from 36-48 months with Stage 0 added). Local learning rules
require 10-100x more samples than backprop. **Bootstrap phase (Stage 0) is critical**: 2 weeks upfront
prevents months of failed training from weight collapse. Focus is on scientific exploration,
not racing to deployment. This realistic timeline accounts for biological constraints and the
bootstrap problem that real brains solve through prenatal development.

---

## Risk Mitigation

### Bootstrap Failure (Stage 0)
- **Problem**: Weight collapse, silent cortex, no learning
- **Solution**: Stage 0 with elevated parameters (0.40 init, 0.02 scaling, 0.0 silent decay)
- **Monitor**: Weight means, firing rates, discrimination accuracy
- **Action**: If validation fails, adjust init weights higher or extend spontaneous activity phase

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

**Month -0.5** (Week -2 to 0): Stage 0 (Bootstrap & Developmental Initialization)
- Genetic prewiring: Initialize weights (thal→L4: 0.40, L4→L2/3: 0.20)
- Phase 0A: Spontaneous activity with OU noise (1 week)
- Phase 0B: Simple 2-pattern discrimination (0.5 weeks)
- Phase 0C: Parameter transition to adult levels (0.5 weeks)
- Validation: All regions firing 0.05-0.15, weights >0.35, discrimination >80%
- **CRITICAL**: If validation fails, DO NOT proceed to Stage 1

**Month 1** (Week 0-4): Stage 1 (Sensorimotor Grounding - Extended)
- Embodied foundations established
- Motor control and proprioception functional
- Visual-motor coordination >90% accuracy
- Cerebellum forward models operational
- Success criteria more stringent with extended time (<5% prediction error)

**Month 2-3** (Week 4-8): Stage 2 (Extended Foundation Phase)
- Infrastructure + checkpoint system working
- Growth mechanisms functional
- Interleaved sensory integration
- Phonological categorical perception
- Memory pressure monitoring
- Prediction error-driven replay
- Social attention (gaze following)
- Testing effect implementation
- Extra time: Stage 2 critical, budget more for phonological foundation

**Month 3-4** (Week 8-16): Stage 3 (Extended Language Foundations)
- Bilingual foundations (English + German)
- Theta-gamma working memory (oscillatory implementation)
- Explicit social learning (imitation, joint attention, pedagogy detection)
- Binary metacognitive monitoring (early abstention)
- Executive function: Inhibitory control (go/no-go)
- Attention mechanisms (bottom-up + top-down)
- Phonological awareness → word mapping
- Milestone checklist must be completed

**Month 4-7.5** (Week 16-30): Stage 4 (Extended Trilingual Grammar)
- Gradual trilingual grammar acquisition (Spanish introduced Week 19)
- Desirable difficulties integration
- Productive failure phases
- Generation-focused tasks
- Cross-modal gamma binding (visual + auditory)
- Executive function: Set shifting (DCCS, task switching)
- Coarse metacognitive confidence (3 levels)
- Intention recognition and false belief
- Spaced repetition for Stage 1 to 1
- Extra time: Trilingual grammar more complex than initially estimated

**Month 7.5-11.5** (Week 30-46): Stage 5 (Extended Trilingual Literacy)
- Reading comprehension (leveraging Stage 2 phonological foundation)
- Text generation with pragmatics
- Advanced social cognition and Theory of Mind
- Continuous metacognitive confidence (0-100%, poorly calibrated)
- Metacognitive calibration training (20% of time Week 44-46)
- Executive function: Planning (Tower of Hanoi, subgoaling)
- Scaffolding fading protocol (high → medium → low support)
- Testing effect + retrieval practice
- REM generalization consolidation
- Conservative pruning begins (1% per cycle)
- Hierarchical temporal abstraction (four-level)
- Extra time: Literacy acquisition takes longer with three languages

**Month 11.5-17.5** (Week 46-70): Stage 6 (Extended Abstract Reasoning)
- Abstract reasoning and analogies
- Complex Theory of Mind
- Metacognitive calibration refinement (30% of time Week 68-70, goal: ECE < 0.15)
- Dendritic computation (compositional reasoning without backprop)
- Executive function: Fluid reasoning (Raven's matrices, hypothesis testing)
- Pruning increases (2% per cycle, peak adolescent phase)
- Adaptive pruning with performance monitoring
- Generation over recognition throughout
- Extra time: Abstract reasoning and metacognitive calibration need maturation

**Month 17.5-26.5** (Week 70-106): Stage 7 (Extended Domain Expertise)
- Domain expertise across multiple fields
- Multi-modal integration with gamma synchrony
- Continued conservative pruning (2% per cycle)
- Schema extraction during REM
- Long-form generation (essays, reports)
- Extra time: Domain breadth requires more learning

**Month 26.5-48** (Week 106-192): Stage 8 (Extended LLM-Level)
- LLM-level performance across benchmarks
- Metacognitive mastery refinement
- Minimal pruning (1% per cycle, maintenance only)
- Benchmark evaluations
- Final optimization and analysis
- Extra time: LLM-level capabilities require extensive training

**Month 48+**: Buffer for debugging, hyperparameter tuning, and validation
- Allow time for failure recovery
- Hyperparameter search per stage
- Multilingual data curation
- Final validation and analysis
- Extended benchmarking
- Ablation studies

**Total**: 50+ months (up from 48+ months with Stage 0 bootstrap addition)
- Week -2 to 0: Bootstrap/developmental initialization (Stage 0)
- Week 0 to 192: Curriculum learning (Stages 1-8)
- Realistic estimate with proper safeguards and biological constraints
- Budget 30-40% overhead for failure recovery and tuning
- Focus on foundation quality over speed
- Science > deployment speed
- **Bootstrap first**: 2 weeks upfront prevents months of failed training later
- Local learning rules require more samples than backprop (10-100x typical)
- Multilingual curriculum adds complexity but captures critical period advantages

---

## Open Research Questions

**Bootstrap & Initialization (Stage 0)**:
1. **Optimal Initial Weights**: Is 0.40 the right thalamus→L4 initialization? Sensitivity analysis needed.
2. **Spontaneous Activity Duration**: Is 1 week enough? Or do we need 2-3 weeks?
3. **Critical Period Closure Rate**: How fast should we transition from 0.02 to 0.002 scaling rate?
4. **Feedforward Bias**: Should thalamus→L4 be 3x, 4x, or 5x stronger than recurrent?
5. **Bootstrap Failure Detection**: What early warning signs predict bootstrap failure?
6. **Species Differences**: Different initialization for vision vs language vs motor systems?

**Curriculum & Learning**:
7. **Optimal Growth Rate**: How aggressively should we add capacity?
8. **Memory Pressure Thresholds**: What pressure level triggers consolidation optimally?
9. **Interleaving Ratios**: What task mixing proportions maximize retention?
10. **Spaced Repetition Parameters**: Optimal expansion rate for review intervals?
11. **Generation vs Recognition Balance**: How much generation is enough?
12. **Prediction Error Weighting**: How to weight TD-error vs reward vs novelty?
13. **Metacognitive Control Timing**: When should brain take over curriculum selection?
14. **Testing Frequency**: Optimal % of steps that should be tests?
15. **Productive Failure Duration**: How long to struggle before teaching?
16. **REM Schema Extraction**: How much noise for optimal generalization?
17. **Curriculum Order**: Is our stage sequence optimal?
18. **Transfer Learning**: How much does Stage N help Stage N+1?
19. **Sample Efficiency**: Can we match biology's efficiency?
20. **Scaling Laws**: How do biological brains scale vs transformers?
21. **Social Learning Impact**: How much faster is social vs individual learning?

---

## Advanced Mechanisms

### Oscillatory Coupling for Cross-Modal Binding

**Purpose**: Use theta-gamma synchrony for temporal sequences and cross-modal binding

**Why Critical**:
- Explains how brain binds "red" + "ball" + "rolling" into unified percept
- **Now actively used** (was unused despite having oscillator infrastructure)
- Essential for working memory (Stage 3) and multi-modal integration (Stage 4+)
- Biologically grounded mechanism with experimental support

**Curriculum Integration**:
- Stage 3: Theta-gamma working memory (n-back tasks)
- Stage 4: Cross-modal binding (visual + auditory words)
- Stage 5+: Hierarchical temporal abstraction (nested oscillations)
- Stage 7: Multi-modal integration (vision + language + audio)

---

### Dendritic Computation for Credit Assignment

**Status**: Now implemented in Stage 6 (Abstract Reasoning)

**Purpose**: Enable multi-step reasoning without global backprop

**Why Critical**:
- Dendritic spikes solve credit assignment locally
- Essential for abstract reasoning (Stage 6)
- Multi-premise logical inference without backprop
- Competitive advantage over pure rate-based SNNs

**Curriculum Integration**:
- Stage 6: Analogical reasoning (A:B::C:D requires premise integration)
- Stage 6: Mathematical reasoning (multi-step proofs)
- Stage 6: Commonsense reasoning (if-then chains)
- Stage 7+: Complex domain expertise (multi-constraint reasoning)

**Performance Target**: >65% on multi-premise reasoning tasks (Stage 6)

---

## Comparison to Human Development

| **Stage** | **Thalia** | **Human Equivalent** | **Age** | **Duration** | **Key Mechanisms** |
|-----------|------------|----------------------|---------|--------------|-----------|
| 0 | Bootstrap & development | Prenatal + neonatal | -2 months to birth | Week -2 to 0 (2 weeks) | Genetic prewiring (0.4 init weights), spontaneous activity (OU noise), elevated plasticity (5x LR, 20x scaling), critical period closure simulation |
| 1 | Sensorimotor grounding | Motor development | 0-6 months | Week 0-4 (1 month) | Active exploration, proprioception, cerebellum forward models, stringent thresholds (<5% error) |
| 2 | Sensory foundations + phonology | Infant perception | 6-12 months | Week 4-8 (1 month) | Critical period (phonology), interleaved multi-modal, phoneme discrimination, memory pressure, gaze following |
| 3 | Object permanence + WM + EF: Inhibition | Object permanence | 12-24 months | Week 8-16 (2 months) | Theta-gamma WM, bilingual + phonology→words, social learning (imitation), binary metacognition, go/no-go, attention (bottom-up/top-down), milestone checklist |
| 4 | Grammar + EF: Shifting | Language explosion | 2-5 years | Week 16-30 (3.5 months) | Trilingual generation, productive failure, gamma binding, DCCS/task switching, cultural learning, coarse confidence (3-level), cognitive load monitoring |
| 5 | Reading/writing + EF: Planning | Elementary school | 6-10 years | Week 30-46 (4 months) | Testing effect, generation-first, Theory of Mind, Tower of Hanoi, scaffolding fading, REM schemas, continuous confidence (0-100%), calibration training (20%), 4-level temporal abstraction, conservative pruning (1%) |
| 6 | Abstract reasoning + EF: Fluid | Adolescence | 12-18 years | Week 46-70 (6 months) | Metacognitive calibration refinement (30%, ECE < 0.15), active learning control, dendritic computation, Raven's matrices, peak pruning (2%), stage transition protocols |
| 7 | Expert knowledge | Higher education | 18-24 years | Week 70-106 (9 months) | Specialization, spaced repetition, continued pruning (2%), domain expertise, multi-modal integration |
| 8 | LLM-level | PhD+ expertise | 24-30+ years | Week 106-192 (21.5 months) | Domain mastery, calibrated confidence, schema mastery, minimal pruning (1%), LLM benchmarks |

**Key Difference**: Thalia compresses 24-30+ years into 50+ months through:
- **Stage 0 Bootstrap**: Solves cold-start with genetic prewiring + spontaneous activity + elevated plasticity
- Curated data (no distractions)
- 24/7 training (but with biologically-timed consolidation)
- Optimized curriculum with cutting-edge learning science:
  - Embodied sensorimotor foundation (Stage 1)
  - Earlier phonological development (Stage 2, not Stage 3)
  - Explicit social learning mechanisms (imitation, pedagogy, joint attention)
  - Theta-gamma oscillatory coupling (working memory, cross-modal binding)
  - Progressive metacognitive development (Stage 3→6, not just Stage 6)
  - Hierarchical temporal abstraction (explicit chunking at multiple timescales)
  - Dendritic computation (compositional reasoning without backprop)
  - Memory pressure-triggered consolidation (synaptic homeostasis)
  - Interleaved practice (better than blocked)
  - Spaced repetition (expanding intervals)
  - Generation over recognition (testing effect)
  - Prediction error-driven replay (learn from mistakes)
  - Productive failure (struggle before instruction)
  - REM generalization (schema extraction)
- Multilingual from the start (critical period advantage)
- Social learning throughout (human advantage)

But maintains biological learning principles:
- **Bootstrap before learning** (Stage 0 genetic scaffold + spontaneous activity)
- Local learning rules (no backprop)
- Gradual complexity increase (ZPD)
- Memory pressure-based consolidation (adenosine analog)
- No catastrophic forgetting (spaced repetition + consolidation)
- Natural multilingualism (like bilingual children)
- Generation-first learning (like human acquisition)
- Embodied grounding (sensorimotor foundation)
- Oscillatory mechanisms (theta-gamma coupling)
- Social scaffolding (learn from demonstrations)

---

## Planned Ablation Studies

**Purpose**: Validate contribution of each mechanism to overall performance

**Critical Ablations** (Test each independently):

```python
ablation_conditions = {
    'full_curriculum': {
        'description': 'Baseline with all mechanisms including Stage 0 bootstrap',
        'expected_performance': 100,  # Reference
    },
    'no_bootstrap_stage0': {
        'description': 'Skip Stage 0, use standard init (0.164) and parameters',
        'hypothesis': 'CATASTROPHIC: Weight collapse, silent cortex, no learning',
        'expected_drop': '70-80%',
    },
    'no_sensorimotor': {
        'description': 'Skip Stage 1, start with Stage 2 visual',
        'hypothesis': 'Reduced grounding, poorer transfer to abstract concepts',
        'expected_drop': '15-20%',
    },
    'no_critical_periods': {
        'description': 'Constant learning rate (no plasticity windows)',
        'hypothesis': 'Phonology/grammar harder to acquire in later stages',
        'expected_drop': '10-15%',
    },
    'no_executive_function': {
        'description': 'Remove EF tasks (inhibition, shifting, planning)',
        'hypothesis': 'Poor self-control, inflexible behavior, planning deficits',
        'expected_drop': '20-25%',
    },
    'no_attention': {
        'description': 'No bottom-up/top-down attention modulation',
        'hypothesis': 'Distractor interference, poor visual search',
        'expected_drop': '10-12%',
    },
    'no_interleaving': {
        'description': 'Blocked practice instead of interleaved',
        'hypothesis': 'Context-specific learning, poor discrimination',
        'expected_drop': '12-18%',
    },
    'no_testing_effect': {
        'description': 'No retrieval practice (only re-study)',
        'hypothesis': 'Weaker retention, more forgetting',
        'expected_drop': '8-12%',
    },
    'no_productive_failure': {
        'description': 'Immediate instruction (no struggle phase)',
        'hypothesis': 'Surface learning, poorer deep understanding',
        'expected_drop': '5-8%',
    },
    'no_social_learning': {
        'description': 'Individual learning only (no imitation/pedagogy)',
        'hypothesis': 'Slower acquisition, higher sample complexity',
        'expected_drop': '15-20%',
    },
    'no_oscillations': {
        'description': 'No theta-gamma coupling (rate-based only)',
        'hypothesis': 'Poor temporal binding, WM deficits',
        'expected_drop': '10-15%',
    },
    'no_dendritic_computation': {
        'description': 'Point neuron model (no dendritic branches)',
        'hypothesis': 'Impaired compositional reasoning, credit assignment',
        'expected_drop': '8-12%',
    },
    'no_consolidation': {
        'description': 'Continuous training (no sleep phases)',
        'hypothesis': 'Catastrophic forgetting, unstable learning',
        'expected_drop': '25-30%',
    },
    'no_ultradian_cycles': {
        'description': 'Single-mode consolidation (SWS only)',
        'hypothesis': 'Reduced schema extraction, less generalization',
        'expected_drop': '3-5%',
    },
    'no_scaffolding': {
        'description': 'Constant difficulty (no fading support)',
        'hypothesis': 'Frustration or boredom, suboptimal ZPD',
        'expected_drop': '5-8%',
    },
    'no_metacognition': {
        'description': 'No uncertainty estimates or abstention',
        'hypothesis': 'More hallucinations, poor calibration',
        'expected_drop': '10-15%',
    },
}
```

**Ablation Protocol**:
1. Train each condition through Stage 6 (sufficient for most mechanisms)
2. Evaluate on full test battery (all stages)
3. Compare to baseline on:
   - Final performance (accuracy)
   - Sample efficiency (steps to criterion)
   - Retention (backward transfer)
   - Generalization (novel tasks)
   - Catastrophic forgetting index

**Most Critical Ablations** (Predict largest drops):
1. **No bootstrap** (-70-80%): CATASTROPHIC (weight collapse)
2. **No consolidation** (-25-30%): Sleep is fundamental
3. **No executive function** (-20-25%): Self-control and flexibility essential
4. **No sensorimotor** (-15-20%): Grounding matters
5. **No social learning** (-15-20%): Human advantage
6. **No interleaving** (-12-18%): Learning structure critical

**Expected Outcome**: Full curriculum significantly outperforms all ablations, validating design.

---
