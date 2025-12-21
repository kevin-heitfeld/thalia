# Architecture Documentation

**Status**: 🟢 Current (December 21, 2025)

High-level architectural documentation for the Thalia framework.

---

## 🚀 Quick Navigation

**📑 Full Documentation Index** → [`INDEX.md`](INDEX.md) - Complete searchable index of all docs

**🎯 New to Thalia?** → Start with [`ARCHITECTURE_OVERVIEW.md`](ARCHITECTURE_OVERVIEW.md)

**🔍 Looking for something specific?** → Use the index or navigation below

---

## Essential Reading

### 📘 Start Here
- **[Architecture Overview](ARCHITECTURE_OVERVIEW.md)** - Complete system architecture guide
  - Core principles and design patterns
  - All brain regions and their functions
  - Data flow and integration points
  - Configuration and testing

### 🔧 Centralized Systems
- **[Centralized Systems](CENTRALIZED_SYSTEMS.md)** - Global coordination systems
  - Neuromodulator systems (VTA, LC, NB)
  - Oscillator manager (5 rhythms + cross-frequency coupling)
  - Goal hierarchy manager
  - Design patterns and integration

### 🛠️ Supporting Components
- **[Supporting Components](SUPPORTING_COMPONENTS.md)** - Infrastructure and utilities
  - Managers (standardized component pattern)
  - Action selection (decision making utilities)
  - Environments (task wrappers for Gymnasium/MuJoCo)
  - Diagnostics (monitoring and debugging)
  - I/O and training infrastructure

### 🎯 Hierarchical Planning
- **[Hierarchical Goals Complete](HIERARCHICAL_GOALS_COMPLETE.md)** - Full goal system implementation
  - Goal decomposition and stack management
  - Options learning and caching
  - Temporal discounting
  - Trial coordinator integration

- **[Goal Hierarchy Summary](GOAL_HIERARCHY_IMPLEMENTATION_SUMMARY.md)** - Implementation details
  - Method activation status
  - Integration points in trial coordinator
  - Usage examples

## Legacy Documentation (Archived)

These documents have been **moved to `../archive/architecture/`** and contain valuable historical context:

- **[Brain Coordination Integration](../archive/architecture/BRAIN_COORDINATION_INTEGRATION.md)** *(Now in: CENTRALIZED_SYSTEMS.md)*
  - Original neuromodulator coordination implementation notes

- **[Neuromodulator Centralization](../archive/architecture/NEUROMODULATOR_CENTRALIZATION_COMPLETE.md)** *(Now in: CENTRALIZED_SYSTEMS.md)*
  - VTA, LC, NB system extraction from Brain class

- **[Oscillator Integration Complete](../archive/architecture/OSCILLATOR_INTEGRATION_COMPLETE.md)** *(Now in: CENTRALIZED_SYSTEMS.md)*
  - All 5 oscillators + 5 cross-frequency couplings

**Note**: These docs contain outdated file paths but are preserved for historical reference.

## Directory Contents

This directory contains:
- **System architecture overviews** - High-level design and component relationships
- **Integration patterns** - How systems work together
- **Implementation status** - What's complete, what's in progress
- **Design decisions** - Why things are built the way they are

## Related Documentation

- **[Design Docs](../design/)** - Detailed design specifications
  - `architecture.md` - Original detailed design
  - `neuron_models.md` - Neuron dynamics specifications
  - `curriculum_strategy.md` - Training curriculum design
  - `checkpoint_format.md` - State serialization format

- **[Patterns](../patterns/)** - Common implementation patterns
  - `component-parity.md` - Region/pathway parity requirements
  - `state-management.md` - When to use RegionState vs attributes
  - `mixins.md` - Available mixins and their usage

- **[Decisions](../decisions/)** - Architecture Decision Records (ADRs)
  - Formal records of key architectural decisions
  - Rationale and trade-offs documented

## Quick Navigation

**New to Thalia?** Start with `ARCHITECTURE_OVERVIEW.md`

**Looking for specific systems?** Check `CENTRALIZED_SYSTEMS.md`

**Need implementation details?** See `HIERARCHICAL_GOALS_COMPLETE.md` for planning, or `../design/` for other systems

**Understanding design choices?** Read `../decisions/` ADRs

## Status

✅ Architecture documentation consolidated and up-to-date (December 13, 2025)
✅ All major systems documented (neuromodulators, oscillators, goals)
✅ Legacy docs preserved for historical reference
