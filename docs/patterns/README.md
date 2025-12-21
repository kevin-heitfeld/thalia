# Implementation Patterns

Common patterns and best practices for working with the Thalia codebase.

**Last Updated**: December 13, 2025

---

## Core Patterns (Read These First)

### ⭐ [Component Parity](./component-parity.md)
**Status**: 🟢 Active Design Pattern

Regions and pathways are equals - both implement the BrainComponent protocol.
- **Why it matters**: Pathways are NOT just "glue" - they are active learning components
- **Key rule**: When adding features to regions, MUST add to pathways too
- **Benefit**: Unified interface prevents forgetting pathways

### ⭐ [Learning Strategies](./learning-strategies.md)
**Status**: ✅ Implemented (v1.0, December 2025)

Pluggable learning rules (Hebbian, STDP, BCM, three-factor) via unified interface.
- **Why it matters**: Eliminates code duplication, enables easy experimentation
- **Migration**: Regions use `create_strategy()` instead of custom learning code
- **Replaces**: learning-strategy-pattern.md + learning-strategy-standardization.md (archived)

### ⭐ [State Management](./state-management.md)
**Status**: 🟢 Active Pattern

When to use RegionState vs direct attributes.
- **Why it matters**: Clear separation between config (immutable) and state (mutable)
- **Key pattern**: `self.state.attribute` for all dynamic state
- **Benefit**: Easier debugging, transparent adapters

---

## Component Design Patterns

### [Component Interface Enforcement](./component-interface-enforcement.md)
**Status**: ✅ Implemented (December 2025)

Abstract base class (`BrainComponentBase`) enforces complete interface at compile time.
- **What it does**: Forces all regions/pathways to implement required methods
- **Hierarchy**: BrainComponentBase → NeuralComponent → Regions/Pathways
- **Benefit**: Missing methods caught early, not at runtime

### [Component Standardization](./component-standardization.md)
**Status**: ✅ Implemented (Tier 2.1, December 2024)

Standardized naming for region sub-components (LearningComponent, HomeostasisComponent).
- **What it does**: Replaces inconsistent naming (Manager/Coordinator/Engine)
- **Pattern**: `{Region}{Component}` (e.g., `StriatumLearningComponent`)
- **Benefit**: Clear responsibilities, easier to implement new regions

**Relationship**: Interface Enforcement (abstract interface) + Standardization (naming conventions) work together

---

## Configuration & Validation

### [Configuration](./configuration.md)
**Status**: 🟢 Active Pattern (Updated December 2025)

Config hierarchy, organization, and declarative validation.
- **Organization**: When to extract config to separate file vs keep inline
- **Validation**: `ValidatedConfig` mixin with declarative rules
- **Best practices**: Parameter validation, cross-field checks

---

## Mixins

### [Mixins](./mixins.md)
**Status**: 🟢 Reference Document

Available mixins and their methods (Diagnostics, ActionSelection, Neuromodulator, etc.).
- **DiagnosticsMixin**: Health checks, firing rate analysis
- **ActionSelectionMixin**: Softmax, greedy, UCB selection
- **NeuromodulatorMixin**: Dopamine, ACh, NE handling
- **LearningStrategyMixin**: Strategy management

---

## Archived Documents

Moved to `docs/archive/patterns/` for historical reference:

- **neuromodulator-homeostasis-status.md** - Status document for Tier 2.12 feature (already implemented)
- **learning-strategy-pattern.md** - Superseded by consolidated learning-strategies.md
- **learning-strategy-standardization.md** - Superseded by consolidated learning-strategies.md

---

## Usage Guide

### For New Regions

1. **Read Component Parity** - Understand regions AND pathways are equal
2. **Inherit from NeuralComponent** - Gets interface enforcement automatically
3. **Use Learning Strategies** - Don't implement custom learning logic
4. **Follow State Management** - Use `self.state.attr` for mutable state
5. **Name components consistently** - `{Region}{Component}` pattern

### For New Pathways

1. **Same as regions!** - Component parity means same patterns apply
2. **Add learning strategy** - Even if you think pathways "don't learn"
3. **Implement full interface** - Same abstract methods as regions

### For Refactoring

1. **Check learning-strategies.md** - Can you replace custom logic with a strategy?
2. **Review component-parity.md** - Did you update both regions AND pathways?
3. **Validate config** - Use `ValidatedConfig` mixin for declarative rules

---

## Related Documentation

- **[Design Docs](../design/)** - Detailed design specifications
- **[Architecture](../architecture/)** - High-level system architecture
- **[Decisions](../decisions/)** - Architecture decision records (ADRs)

---

## Pattern Status Legend

- 🟢 **Active Pattern** - Current best practice, use for new code
- ✅ **Implemented** - Pattern implemented and production-ready
- 🟡 **In Progress** - Being adopted across codebase
- 📋 **Planned** - Documented but not yet implemented
- 🗄️ **Archived** - Historical reference, superseded by newer pattern

---

**Maintenance**: Update this README when:
- Adding new patterns
- Consolidating duplicate documentation
- Changing pattern status
- Archiving superseded patterns
