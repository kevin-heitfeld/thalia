---
agent: agent
---
You are an expert software architect specializing in neuroscience-inspired AI systems. Conduct a comprehensive architectural analysis of the Thalia codebase and provide structured refactoring recommendations.

**Scope:**
- Focus on `src/thalia/` directory (core, regions, learning, integration, sensory)
- Analyze module organization, learning rules, neuron models, and pathway implementations
- Review adherence to biological plausibility constraints (local learning, spike-based processing)

**Analysis Criteria:**
1. **File/Module Organization**: Are files logically grouped? Do directory names reflect their contents?
2. **Naming Consistency**: Do file names, class names, and region/pathway names accurately describe their purpose?
3. **Separation of Concerns**: Is learning logic properly separated from neuron dynamics, state management, and pathway routing?
4. **Pattern Adherence**: Does the code follow documented patterns (BrainComponent protocol, RegionState management, WeightInitializer registry, local learning rules)?
5. **Discoverability**: Can developers easily locate functionality based on naming and structure?
6. **Code Duplication**: Are there repeated code blocks, similar functions, or duplicated logic that should be consolidated into shared utilities or base classes?
7. **Antipattern Detection**: Identify antipatterns such as:
   - God objects/classes (excessive responsibilities)
   - Tight coupling between components
   - Circular dependencies
   - Magic numbers/strings without constants
   - Deep nesting (complexity > reasonable threshold)
   - Non-local learning rules (violates biological plausibility)
   - Global error signals or backpropagation
   - Analog firing rates instead of binary spikes in processing
8. **Pattern Improvements**: Identify patterns that work but could be replaced with better alternatives:
   - Repetitive learning rule patterns that could use strategy pattern
   - Manual state synchronization that could leverage mixins
   - Weight initialization patterns that should use WeightInitializer registry
   - Duplicated neuron update logic that could use base neuron classes
   - Multiple similar region implementations that could share base functionality

**Deliverable Format:**
Provide recommendations in three priority tiers:

**Tier 1 - High Impact, Low Disruption** (do first):
- Naming improvements that increase clarity without breaking references
- File relocations that better reflect current organization
- Code duplication elimination (extract shared utilities, create base classes)
- Magic number/string extraction to named constants (neuron time constants, threshold values)
- Simple antipattern fixes (e.g., using WeightInitializer instead of manual init)

**Tier 2 - Moderate Refactoring** (strategic improvements):
- Module consolidation or splitting for better cohesion
- Architectural pattern violations that should be corrected
- Region/pathway boundary adjustments
- Pattern replacements (e.g., mixin adoption, learning strategy pattern)
- Complexity reduction through decomposition
- Decoupling tightly coupled regions or pathways

**Tier 3 - Major Restructuring** (long-term considerations):
- Fundamental architectural changes (if any)
- Directory reorganization requiring widespread import updates
- Breaking API changes that improve design
- Large-scale pattern migrations (e.g., refactoring learning rules to unified interface)

**For each recommendation, specify:**
- Current state vs. proposed change
- Rationale (how this improves architecture)
- Impact (files affected, breaking change severity: low/medium/high)
- **For duplication**: Exact locations of duplicated code and proposed consolidation location
- **For antipatterns**: Specific antipattern name and concrete example from codebase
- **For pattern improvements**: Before/after pattern comparison with measurable benefits (readability, maintainability, performance)

**Constraints:**
- Maintain biological plausibility (local learning rules, spike-based processing, no backpropagation)
- Preserve the neuroscience-inspired architecture pattern
- Prioritize changes that improve developer experience and code discoverability

**Output & File Delivery:**
- Write the complete analysis to a markdown file so it is captured in the repository history.
- Target path: `docs/reviews/architecture-review-YYYY-MM-DD.md` (use current date, e.g., `architecture-review-2025-11-01.md`).
- To avoid response length limits, output the markdown file in small sections if needed.
- If `docs/reviews/` does not exist, create it.
- File structure:
   1. Title: `# Architecture Review – YYYY-MM-DD`
   2. Brief summary (3-5 bullets)
   3. Findings organized by Tier 1, Tier 2, Tier 3 (as specified above)
   4. Risk/impact assessment and suggested sequencing
   5. Appendix A: Affected files and links (relative paths)
   6. Appendix B: Detected duplications and locations
- Output only a short confirmation in chat (path + one-line summary); place full content in the markdown file.
