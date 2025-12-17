# Architecture Documentation Index

**Quick reference guide to all architecture documentation**

## 🎯 Quick Start

**New to Thalia?** → [`ARCHITECTURE_OVERVIEW.md`](ARCHITECTURE_OVERVIEW.md)

**Looking for specific systems?** → Use navigation below ⬇️

---

## 📚 Active Documentation

### Core Architecture
| Document | Purpose | When to Read |
|----------|---------|--------------|
| [`ARCHITECTURE_OVERVIEW.md`](ARCHITECTURE_OVERVIEW.md) | Complete system architecture | **Start here** - Understanding whole system |
| [`CENTRALIZED_SYSTEMS.md`](CENTRALIZED_SYSTEMS.md) | Global coordination systems | Working with neuromodulators, oscillators, or goals |
| [`SUPPORTING_COMPONENTS.md`](SUPPORTING_COMPONENTS.md) | Infrastructure and utilities | Using managers, environments, or diagnostics |

### Hierarchical Planning
| Document | Purpose | When to Read |
|----------|---------|--------------|
| [`HIERARCHICAL_GOALS_COMPLETE.md`](HIERARCHICAL_GOALS_COMPLETE.md) | Goal system implementation | Implementing or using goal hierarchies |
| [`GOAL_HIERARCHY_IMPLEMENTATION_SUMMARY.md`](GOAL_HIERARCHY_IMPLEMENTATION_SUMMARY.md) | Implementation details | Integration points and method usage |

---

## 🗂️ Documentation by Topic

### Brain Regions
→ See [`ARCHITECTURE_OVERVIEW.md`](ARCHITECTURE_OVERVIEW.md) § "Brain Regions"
- Thalamus (sensory relay, attention gating)
- Cortex (sensory processing)
- Hippocampus (episodic memory)
- Prefrontal Cortex (working memory)
- Striatum (action selection)
- Cerebellum (motor learning)

### Neuromodulation
→ See [`CENTRALIZED_SYSTEMS.md`](CENTRALIZED_SYSTEMS.md) § "Neuromodulator Systems"
- VTA (dopamine)
- Locus Coeruleus (norepinephrine)
- Nucleus Basalis (acetylcholine)
- Coordination (DA-ACh, NE-ACh, DA-NE)

### Oscillations
→ See [`CENTRALIZED_SYSTEMS.md`](CENTRALIZED_SYSTEMS.md) § "Oscillator System"
- 5 brain rhythms (delta, theta, alpha, beta, gamma)
- 5 cross-frequency couplings
- Integration with regions

### Planning & Goals
→ See [`HIERARCHICAL_GOALS_COMPLETE.md`](HIERARCHICAL_GOALS_COMPLETE.md)
- Goal decomposition
- Options learning
- Temporal discounting
- Trial coordinator integration

### Memory Consolidation
→ See [`CENTRALIZED_SYSTEMS.md`](CENTRALIZED_SYSTEMS.md) § "Memory Consolidation System"
- Offline replay
- Experience storage
- Hindsight Experience Replay (HER)
- Coordinated learning during replay

### Attention Systems
→ See [`ARCHITECTURE_OVERVIEW.md`](ARCHITECTURE_OVERVIEW.md) § "Thalamus"
- Thalamus: Sensory relay and gating
- Bottom-up attention (stimulus-driven salience)
- Top-down attention (PFC → Cortex pathways)
- Developmental progression (reactive → proactive)
- Alpha-band gating and TRN inhibition

**Pathways**: `src/thalia/pathways/attention/`
- `attention.py` - AttentionMechanisms (bottom-up + top-down)
- `spiking_attention.py` - SpikingAttentionPathway (PFC → Cortex)
- `crossmodal_binding.py` - Cross-modal integration

### Language Processing
→ See `src/thalia/language/`
- Token encoding (text → spikes)
- Spike decoding (spikes → text)
- Oscillatory position encoding
- LanguageBrainInterface (unified brain integration)
- Multi-modal language processing

**See Also**: `../design/curriculum_strategy.md` for language training stages

### Social Learning
→ See `src/thalia/learning/social_learning.py`
- Imitation learning (2x learning rate from demonstration)
- Pedagogy detection (1.5x boost for teaching contexts)
- Joint attention (gaze-driven attention)
- Developmental progression through curriculum

**See Also**: `../design/curriculum_strategy.md` for social learning integration

### Metacognition
→ See `src/thalia/training/evaluation/metacognition.py`
- Confidence calibration and estimation
- Uncertainty quantification
- Expected Calibration Error (ECE) metrics
- Self-monitoring and cognitive control
- Binary uncertainty (Stage 1) → continuous confidence (Stage 4)

**See Also**: `../design/curriculum_strategy.md` for metacognitive development

### Action Selection
→ See [`SUPPORTING_COMPONENTS.md`](SUPPORTING_COMPONENTS.md) § "Decision Making"
- Selection strategies (softmax, greedy, UCB)
- Population coding
- Vote accumulation

### Environments
→ See [`SUPPORTING_COMPONENTS.md`](SUPPORTING_COMPONENTS.md) § "Environments"
- Sensorimotor wrapper (Gymnasium/MuJoCo)
- Task types (reaching, manipulation)
- Encoding/decoding

### Managers & Utilities
→ See [`SUPPORTING_COMPONENTS.md`](SUPPORTING_COMPONENTS.md) § "Managers System"
- BaseManager pattern
- Component registry
- Diagnostics
- I/O and checkpointing

---

## 📖 Related Documentation

### Design Specifications
**Location**: [`../design/`](../design/)

| Document | Purpose |
|----------|---------|
| `architecture.md` | Original detailed design |
| `neuron_models.md` | LIF/Conductance-LIF specifications |
| `curriculum_strategy.md` | Training curriculum stages |
| `checkpoint_format.md` | State serialization format |

### Implementation Patterns
**Location**: [`../patterns/`](../patterns/)

| Document | Purpose |
|----------|---------|
| `component-parity.md` | Region/pathway parity requirements |
| `state-management.md` | RegionState vs attributes |
| `mixins.md` | Available mixins and usage |

### Architecture Decisions
**Location**: [`../decisions/`](../decisions/)

See ADRs (Architecture Decision Records) for formal design decisions:
- ADR-001: Simulation backend (PyTorch)
- ADR-002: Numeric precision (float32)
- ADR-003: Clock-driven updates
- ADR-004: Boolean spikes
- ADR-005: No batch dimension
- And more...

---

## 🏛️ Archived Documentation

**Location**: `../archive/architecture/`
**Purpose**: Historical reference, preserved for context

| Document | Content | Current Reference |
|----------|---------|-------------------|
| `BRAIN_COORDINATION_INTEGRATION.md` | Original coordination integration | → `CENTRALIZED_SYSTEMS.md` |
| `NEUROMODULATOR_CENTRALIZATION_COMPLETE.md` | VTA/LC/NB extraction | → `CENTRALIZED_SYSTEMS.md` |
| `OSCILLATOR_INTEGRATION_COMPLETE.md` | Oscillator integration | → `CENTRALIZED_SYSTEMS.md` |

⚠️ **Note**: Archived docs contain outdated file paths. Use current references instead.

---

## 🔍 Finding Information

### By Component Type

**Brain Regions** → `ARCHITECTURE_OVERVIEW.md` § "Brain Regions"
**Pathways** → `ARCHITECTURE_OVERVIEW.md` § "Pathways"
**Neuromodulators** → `CENTRALIZED_SYSTEMS.md` § "Neuromodulator Systems"
**Oscillators** → `CENTRALIZED_SYSTEMS.md` § "Oscillator System"
**Goals** → `HIERARCHICAL_GOALS_COMPLETE.md`
**Consolidation** → `CENTRALIZED_SYSTEMS.md` § "Memory Consolidation System"
**Attention** → `ARCHITECTURE_OVERVIEW.md` § "Thalamus" + `src/thalia/pathways/attention/`
**Language** → `src/thalia/language/` + `../design/curriculum_strategy.md`
**Social Learning** → `src/thalia/learning/social_learning.py` + curriculum
**Metacognition** → `src/thalia/training/evaluation/metacognition.py`
**Managers** → `SUPPORTING_COMPONENTS.md` § "Managers System"
**Action Selection** → `SUPPORTING_COMPONENTS.md` § "Decision Making"
**Environments** → `SUPPORTING_COMPONENTS.md` § "Environments"

### By Task

**Understanding architecture** → Start with `ARCHITECTURE_OVERVIEW.md`
**Implementing new region** → `ARCHITECTURE_OVERVIEW.md` + `../patterns/component-parity.md`
**Adding neuromodulation** → `CENTRALIZED_SYSTEMS.md` § "Neuromodulator Systems"
**Implementing learning rule** → `ARCHITECTURE_OVERVIEW.md` § "Regional Specialization"
**Creating environments** → `SUPPORTING_COMPONENTS.md` § "Environments"
**Action selection** → `SUPPORTING_COMPONENTS.md` § "Decision Making"
**Goal decomposition** → `HIERARCHICAL_GOALS_COMPLETE.md`
**Debugging** → `SUPPORTING_COMPONENTS.md` § "Diagnostics"

### By Role

**New User** → `ARCHITECTURE_OVERVIEW.md`
**Developer** → `ARCHITECTURE_OVERVIEW.md` → topic-specific docs
**Researcher** → All active docs + `../design/` + `../decisions/`
**Maintainer** → This index + `../reviews/architecture-docs-update-2025-12-13.md`

---

## 📊 Documentation Coverage

### Fully Documented ✅
- DynamicBrain core (component-based architecture)
- All brain regions (thalamus, cortex, hippocampus, PFC, striatum, cerebellum)
- Neuromodulator systems (VTA, LC, NB)
- Oscillator system
- Goal hierarchy
- Memory consolidation (replay & offline learning)
- Attention systems (thalamus + attention pathways)
- Language processing (token encoding/decoding, integration)
- Social learning (imitation, pedagogy, joint attention)
- Metacognition (confidence calibration, uncertainty)
- Managers and utilities
- Action selection
- Environments

### Partially Documented ⚠️
- Event system (implementation complete, docs in progress)
- Memory consolidation (implementation in progress)

### Planned 📋
- Attention system (not yet implemented)
- Language processing (not yet implemented)

---

## 🔄 Keeping Documentation Current

### When to Update Docs

**Code changes** → Update relevant active doc
**New major system** → Create new doc + update this index
**Architectural decision** → Create ADR in `../decisions/`

### Update Checklist

- [ ] Code changes reflected in docs
- [ ] Cross-references still valid
- [ ] Examples still work
- [ ] Index updated (this file)
- [ ] README.md navigation updated

### What NOT to Update

❌ **Do NOT modify archived docs** - they're historical snapshots

---

## 💡 Tips

### For Reading
1. Start broad (`ARCHITECTURE_OVERVIEW.md`)
2. Drill down to specific topics
3. Check related patterns/decisions for context

### For Writing
1. Update existing docs when possible
2. Create new docs only for major systems
3. Keep examples practical and tested
4. Cross-reference related docs

### For Maintenance
1. Regular reviews (quarterly recommended)
2. Verify examples against current code
3. Archive outdated docs properly
4. Keep index up-to-date

---

## 📞 Getting Help

**Can't find what you need?**
1. Check this index
2. Search in active docs (Ctrl+F)
3. Check `../design/` for detailed specs
4. Check `../patterns/` for implementation guidance
5. Check `../decisions/` for design rationale

**Found incorrect information?**
1. Check if doc is archived (see Archived Documentation section)
2. If active doc: needs update
3. See update checklist above

---

**Last Updated**: December 13, 2025
**Status**: Complete ✅
**Next Review**: March 2026
