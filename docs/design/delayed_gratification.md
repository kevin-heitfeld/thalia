# Delayed Gratification Implementation Plan

**Version**: 2.0.0
**Status**: ✅ **Phases 1-3 COMPLETE** (Updated December 13, 2025)
**Original Date**: December 10, 2025
**Priority**: High (Core capability for AGI)

## Executive Summary

This document outlines the implementation plan for enhancing Thalia's delayed gratification capabilities - the ability to pursue long-term goals despite short-term costs. **UPDATE**: Most planned features are now implemented!

### ✅ **Implemented Capabilities (December 2025)**

**Phase 1: Multi-Step Credit Assignment**
- ✅ **TD(λ) Learning** (`src/thalia/regions/striatum/td_lambda.py`)
  - TDLambdaLearner with configurable λ and γ parameters
  - Bridge ~10 timesteps (5-10 seconds at dt=1ms)
  - Accumulating vs replacing trace modes
  - Integrated with striatum three-factor rule

**Phase 2: Model-Based Planning**
- ✅ **Dyna-style Planning** (`src/thalia/planning/dyna.py`)
  - DynaPlanner combines real experience with simulated planning
  - World model learning (transition dynamics)
  - Background planning sweeps (n_planning_steps configurable)
  - Priority sweeps for efficient updates

**Phase 3: Hierarchical Goals**
- ✅ **Goal Hierarchy Manager** (`src/thalia/regions/prefrontal_hierarchy.py`)
  - Goal stack with push/pop/peek operations
  - Goal decomposition into subgoals
  - Options learning and caching
  - Hyperbolic temporal discounting
  - Context-dependent value functions

**Additional Systems:**
- ✅ Eligibility traces (1000ms tau, ~1 second bridge)
- ✅ Working memory (PFC goal maintenance)
- ✅ VTA dopamine system (tonic + phasic)
- ✅ Counterfactual learning (imagined outcomes)
- ✅ Predictive coding (sensory predictions)
- ✅ Mental simulation coordinator (`src/thalia/planning/coordinator.py`)

### 📋 **Remaining Work**

**Performance Validation** (Implementation Complete ✅):
- 🔄 Validate TD(λ) performance on sensorimotor tasks (Stage -0.5)
- 🔄 Test Dyna planning on grammar tasks (Stage 2)
- 🔄 Verify hierarchical goals on essay writing (Stage 3)
- 🔄 Benchmark temporal credit assignment windows

**Recent Completions** (December 23, 2025):
- ✅ Cerebellum gap junction synchronization (IO neurons)
- ✅ Per-Purkinje cell dendritic learning (LTD/LTP)
- ✅ Gap junction bug fixes (initialization + error propagation)
- ✅ Comprehensive test coverage (11 tests, all passing)

**Curriculum Integration** (Ready for Use ✅):
- ✅ TD(λ) can be enabled via `use_td_lambda=True` in config
- ✅ Dyna planning available via `src/thalia/planning/dyna.py`
- ✅ Goal hierarchy available via `src/thalia/regions/prefrontal_hierarchy.py`
- 🔄 Add explicit curriculum triggers for automated activation

**Expected Impact:**
- Marshmallow test: Currently FAIL → With implementations: PASS (in appropriate contexts)
- Temporal horizon: 1 second → 10 seconds (TD(λ)) → minutes/hours (Dyna + Hierarchy)
- Novel situations: Pure model-free → Hybrid model-based/model-free
- Multi-goal learning: Sequential only → Parallel goal pursuit

## Curriculum Stage Analysis

### Stage Relevance Matrix

| Mechanism | Stage -0.5 | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 5 | Stage 6 |
|-----------|-----------|---------|---------|---------|---------|---------|---------|---------|
| **TD(λ)** | ⭐ Critical | ⭐⭐ Critical | ⭐⭐ Critical | ⭐⭐ Critical | ⭐ Important | ⭐ Important | Optional | Optional |
| **Model-Based Planning** | Optional | Optional | ⭐ Important | ⭐⭐ Critical | ⭐⭐⭐ Essential | ⭐⭐⭐ Essential | ⭐⭐ Critical | ⭐ Important |
| **Hierarchical Goals** | N/A | N/A | Optional | ⭐ Important | ⭐⭐ Critical | ⭐⭐⭐ Essential | ⭐⭐⭐ Essential | ⭐⭐⭐ Essential |
| **Goal-Conditioned Values** | N/A | N/A | ⭐ Important | ⭐⭐ Critical | ⭐⭐ Critical | ⭐⭐ Critical | ⭐ Important | Optional |
| **Hyperbolic Discounting** | Optional | Optional | Optional | Optional | ⭐ Important | ⭐⭐ Critical | ⭐ Important | Optional |

### Stage-Specific Rationale

#### Stage -0.5: Sensorimotor Grounding (Weeks 0-4)
**Why TD(λ) is Critical:**
- Sensorimotor prediction requires multi-step credit assignment
- "I moved left 500ms ago, now I see visual feedback" → 5-10 timesteps delay
- Cerebellum forward model learning needs temporal bridging
- Example: Push object → travels → hits wall → feedback (0.5-1 second delay)

**Current limitation**: One-step TD can't bridge action→outcome delays in physics simulation

**Implementation priority**: Phase 1, Week 1-2 (enable before Stage -0.5 training)

---

#### Stage 0: Sensory Foundations (Weeks 4-8)
**Why TD(λ) is Critical:**
- Phonological learning: Sound sequence → categorical perception (multi-step)
- Temporal patterns: A-B-C sequences require remembering earlier items
- Gaze following: Social cue → shift attention → verify (0.5-1 second)
- Example: Hear /p/ phoneme → categorize → store → compare to next phoneme

**Current limitation**: Eligibility traces (1s) barely sufficient, TD(λ) would strengthen

**Implementation priority**: Phase 1, must be ready by Stage 0 start

---

#### Stage 1: Working Memory (Weeks 8-16)
**Why TD(λ) is Critical:**
- N-back tasks: Remember item from 2-4 steps ago (1-2 seconds)
- Delayed gratification: "Wait for 1.5x reward" requires bridging ~1 second
- Go/no-go: Action → feedback delay
- Object tracking: Maintain representation across time

**Why Model-Based Planning is Important:**
- Working memory emerging → can simulate "what if I wait?"
- Simple forward models: "If I don't act now, what happens?"
- Foundation for later complex planning

**Why Goal-Conditioned Values are Important:**
- Multiple goals emerging (language goals, motor goals, inhibition goals)
- Need to switch between goal contexts rapidly
- Example: "Get red object" vs "Get blue object" - different value functions

**Implementation priority**:
- TD(λ): Phase 1, must be ready
- Model-based: Phase 2, introduce late Stage 1 (Week 14-16)
- Goal-conditioned: Phase 1, Week 8-10

---

#### Stage 2: Grammar & Composition (Weeks 16-30)
**Why TD(λ) is Critical:**
- Multi-step instructions: "Take ball, bring it, put here" (5-10 second sequences)
- Grammar learning: Earlier words affect later word choices
- Language switching: Previous language context affects current processing
- Set shifting (DCCS): Suppress old rule → activate new rule (multi-step EF)

**Why Model-Based Planning is Critical:**
- Must simulate sentence completions: "If I use SVO order, what comes next?"
- Task switching requires mental simulation: "What if I switch rules now?"
- Cross-lingual reasoning: Mentally translate concepts
- Example: Before speaking, simulate "Does this sentence make sense?"

**Why Goal-Conditioned Values are Critical:**
- Multiple languages = multiple goal contexts (English goal vs German goal vs Spanish goal)
- Each language has different grammar rules → different value landscapes
- Code-switching requires rapid goal context switching
- Must learn: V(state, action | goal=German) ≠ V(state, action | goal=Spanish)

**Why Hierarchical Goals are Important:**
- Multi-step instructions naturally hierarchical: "Take ball AND bring it AND put here"
- Grammar composition: Sentence = [subject + verb + object]
- Beginning of subgoal decomposition

**Implementation priority**:
- TD(λ): Already present from Phase 1
- Model-based: Phase 2, must be complete by Stage 2 start
- Goal-conditioned: Phase 1, must be ready
- Hierarchical: Phase 3, introduce Week 20-24

---

#### Stage 3: Reading & Writing (Weeks 30-46)
**Why Model-Based Planning is Essential:**
- Text generation REQUIRES simulation: "What's the next word? The next sentence?"
- Must plan narrative structure: Beginning → middle → end
- Reading comprehension: Build mental model of story world
- Tower of Hanoi (planning tasks): Must simulate action sequences mentally
- Example: "If I write X, then Y follows logically..."

**Why Hierarchical Goals are Critical:**
- Essay writing: Topic → paragraphs → sentences → words (4-level hierarchy)
- Story generation: Arc → scenes → events → actions
- Planning tasks: Goal → subgoals → actions
- Reading strategy: Understand passage → answer question (2-level)

**Why Goal-Conditioned Values are Critical:**
- Each language maintains separate value function
- Translation requires switching goal contexts mid-task
- Different reading goals: Summarize vs comprehend vs find-specific-info

**Why Hyperbolic Discounting is Important:**
- Metacognitive calibration requires realistic uncertainty modeling
- Confidence estimation: "Should I wait to answer (for higher confidence)?"
- Reading complex texts requires sustained attention despite difficulty

**Implementation priority**:
- Model-based: Phase 2, must be complete
- Hierarchical: Phase 3, must be functional by Week 35
- Hyperbolic: Phase 3, introduce Week 40-44
- Goal-conditioned: Already present from Phase 1

---

#### Stage 4: Abstract Reasoning (Weeks 46-70)
**Why Model-Based Planning is Essential:**
- Analogical reasoning: Simulate structure mappings mentally
- Mathematical word problems: Build mental model, simulate solutions
- Raven's matrices: Test hypothetical rules mentally
- Hypothesis testing: "If rule X, then pattern Y..."

**Why Hierarchical Goals are Essential:**
- Complex problem decomposition: Problem → subproblems → steps → operations
- Multi-step reasoning: Premise → inference → conclusion (3+ levels)
- Fluid reasoning requires goal stack management
- Metacognitive control: Monitor subgoals, adjust strategy

**Why Hyperbolic Discounting is Critical:**
- Metacognitive active learning: "Should I study this hard topic now?"
- Calibration under cognitive load: Stress increases impulsivity
- Self-directed learning requires balancing immediate vs long-term gains
- Example: "Do I tackle difficult problem (long-term gain) or easy one (immediate reward)?"

**Implementation priority**:
- All mechanisms must be mature and integrated
- Hyperbolic: Phase 3, must be calibrated by Week 55

---

#### Stage 5-6: Expert Knowledge & LLM-Level (Weeks 70-192)
**Why Hierarchical Goals are Essential:**
- Domain expertise requires deep goal hierarchies
- Long-form generation: Document → sections → paragraphs → sentences → words
- Multi-modal integration: Coordinate vision + language + reasoning goals
- Complex instruction following: Parse → plan → execute (hierarchical)

**Why Model-Based Planning is Important:**
- Few-shot learning: Simulate task from examples
- Instruction following: Mentally verify understanding before acting
- Long-form coherence: Maintain narrative model throughout generation

**Why mechanisms become "Optional":**
- By Stage 5-6, mechanisms are mature and automated
- Focus shifts to knowledge accumulation, not mechanism development
- However, they remain active and critical for performance
- "Optional" means "no new development needed", not "can disable"

**TD(λ) becomes less critical**: Most credit assignment learned in early stages, now applying cached knowledge

**Hyperbolic becomes less critical**: Calibration mature, no longer developing discounting curves

---

### Key Insights from Stage Analysis

1. **TD(λ) is foundational**: Must be present from Stage -0.5 onward
   - Sensorimotor learning REQUIRES multi-step credit
   - Early investment pays dividends throughout curriculum

2. **Model-based planning emerges in Stage 1, becomes critical in Stage 2-4**:
   - Stage 1: Simple "what if" simulation
   - Stage 2: Grammar and task switching require planning
   - Stage 3-4: Essential for complex reasoning and generation
   - Stage 5-6: Mature, used automatically

3. **Hierarchical goals track language complexity**:
   - Stage 1: No hierarchy needed (single goals)
   - Stage 2: Simple 2-level hierarchies (multi-step instructions)
   - Stage 3: 4-level hierarchies (essays, stories)
   - Stage 4-6: Deep hierarchies (complex problem decomposition)

4. **Goal-conditioned values match multilingual learning**:
   - Stage 1: Single language → simple goal conditioning
   - Stage 2: Three languages → critical for language switching
   - Stage 3-6: Multiple tasks per language → complex goal spaces

5. **Hyperbolic discounting supports metacognition**:
   - Not needed until Stage 3 (when calibration training begins)
   - Critical in Stage 4 (active learning, self-directed study)
   - Models realistic human impulsivity under cognitive load

## Implementation Phases

### Phase 1: Multi-Step Credit Assignment (Weeks 1-3)
**Priority**: Critical (blocks Stage -0.5 training)

**Components**:
1. TD(λ) implementation in striatum
2. Extended eligibility traces with lambda decay
3. N-step return computation
4. Goal-conditioned value functions

**Deliverables**: See `PHASE1_TD_LAMBDA.md`

---

### Phase 2: Model-Based Planning (Weeks 4-7)
**Priority**: Critical (needed by Stage 2)

**Components**:
1. World model in PFC (action-conditioned forward prediction)
2. Tree search / rollout implementation
3. Dyna-style planning (model-based + model-free integration)
4. Hippocampal episodic memory integration

**Deliverables**: See `PHASE2_MODEL_BASED.md`

---

### Phase 3: Hierarchical Goals (Weeks 8-11)
**Priority**: Important (needed by Stage 3)

**Components**:
1. Goal hierarchy data structures
2. Subgoal decomposition algorithms
3. Options framework for temporal abstraction
4. Hierarchical value propagation
5. Hyperbolic temporal discounting

**Deliverables**: See `PHASE3_HIERARCHICAL.md`

---

## Testing Strategy

Each phase includes:
1. **Unit tests**: Individual components (e.g., TD(λ) update rule)
2. **Integration tests**: Component interactions (e.g., world model + planning)
3. **Curriculum tests**: Stage-specific validation (e.g., delayed gratification in Stage 1)
4. **Ablation studies**: Measure impact of each mechanism

**Success Criteria** defined per phase (see phase-specific documents)

---

## Risk Mitigation

**Risk 1: Breaking existing curriculum tests**
- Mitigation: Run full test suite after each phase
- Keep old TD(0) as fallback option (config flag)

**Risk 2: Increased computational cost**
- Mitigation: Make planning depth configurable (start with 3 steps)
- Profile memory usage, optimize as needed

**Risk 3: Hyperparameter sensitivity**
- Mitigation: Use biologically-inspired defaults (λ=0.9, γ=0.99)
- Add hyperparameter robustness tests

**Risk 4: Phase dependencies**
- Mitigation: Phase 1 is self-contained, can proceed immediately
- Phase 2 depends on Phase 1 (goal-conditioned values needed)
- Phase 3 depends on Phase 2 (world model needed for goal simulation)

---

## Timeline Summary

| Phase | Duration | Start | Deliverable | Blocks Stage |
|-------|----------|-------|-------------|--------------|
| Phase 1 | 3 weeks | Immediate | TD(λ) + Goal-conditioned | Stage -0.5 |
| Phase 2 | 4 weeks | Week 4 | Model-based planning | Stage 2 |
| Phase 3 | 4 weeks | Week 8 | Hierarchical goals | Stage 3 |
| **Total** | **11 weeks** | - | Full delayed gratification system | - |

**Critical Path**: Phase 1 → Phase 2 → Phase 3 (sequential dependencies)

**Parallel Work Possible**:
- Documentation can be written alongside implementation
- Test infrastructure can be developed during Phase 1
- Curriculum stage mapping can be refined during Phases 2-3

---

## Next Steps

1. **Immediate**: Review and approve Phase 1 implementation plan
2. **Week 1**: Begin TD(λ) implementation in striatum
3. **Week 2**: Implement goal-conditioned value functions
4. **Week 3**: Testing and integration with existing curriculum
5. **Week 4**: Begin Phase 2 (world model design)

---

## References

**Neuroscience**:
- Schultz et al. (1997) - Dopamine signals TD errors
- Foster & Wilson (2006) - Hippocampal replay during rest
- Mattar & Daw (2018) - Prioritized experience replay
- Yagishita et al. (2014) - Eligibility traces and calcium dynamics

**Psychology**:
- Botvinick (2008) - Hierarchical RL and prefrontal organization
- Diamond (2013) - Executive function development
- Miller & Cohen (2001) - Prefrontal goal representation
- Zelazo (2006) - DCCS and cognitive flexibility

**Machine Learning**:
- Sutton & Barto (2018) - Reinforcement Learning (TD(λ), options)
- Andrychowicz et al. (2017) - Hindsight Experience Replay
- Ha & Schmidhuber (2018) - World Models
- Dayan & Hinton (1993) - Feudal reinforcement learning

---

**Document Structure**:
- `delayed_gratification_plan.md` (this file) - Overview and stage analysis
- `PHASE1_TD_LAMBDA.md` - Detailed Phase 1 implementation
- `PHASE2_MODEL_BASED.md` - Detailed Phase 2 implementation
- `PHASE3_HIERARCHICAL.md` - Detailed Phase 3 implementation

**Status**: ✅ Overview complete, ready for phase-specific planning
