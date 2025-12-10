---
mode: agent
---
Conduct a comprehensive architectural analysis of the ai_chat codebase and provide structured refactoring recommendations.

**Scope:**
- Focus on `src/` directory (components, services, utils)
- Include HTML structure (`index.html`), CSS-in-TypeScript system (`src/styles/`), and TypeScript organization
- Analyze DOM class naming conventions for semantic clarity

**Note:** This project uses a CSS-in-TypeScript approach. Styles are defined in `*.styles.ts` files and generated at build time. See `src/styles/definitions.ts` for type definitions and `src/styles/tokens.ts` for design tokens.

**Analysis Criteria:**
1. **File/Module Organization**: Are files logically grouped? Do directory names reflect their contents?
2. **Naming Consistency**: Do file names, class names, and DOM classes accurately describe their purpose?
3. **Separation of Concerns**: Is business logic properly separated from UI, state management, and API layers?
4. **Pattern Adherence**: Does the code follow the documented service-oriented architecture (dependency injection, coordinator pattern)?
5. **Discoverability**: Can developers easily locate functionality based on naming and structure?
6. **Code Duplication**: Are there repeated code blocks, similar functions, or duplicated logic that should be consolidated into shared utilities or base classes?
7. **Antipattern Detection**: Identify antipatterns such as:
   - God objects/classes (excessive responsibilities)
   - Tight coupling between components
   - Circular dependencies
   - Magic numbers/strings without constants
   - Deep nesting (complexity > reasonable threshold)
   - Callback hell or promise chains that could use async/await
   - Direct DOM manipulation where declarative approaches would be clearer
8. **Pattern Improvements**: Identify patterns that work but could be replaced with better alternatives:
   - Repetitive event handler patterns that could use delegation
   - Manual state synchronization that could leverage observers
   - Imperative logic that could be declarative
   - String concatenation for HTML that could use template literals or DocumentFragment
   - Multiple similar conditionals that could use polymorphism or strategy pattern

**Deliverable Format:**
Provide recommendations in three priority tiers:

**Tier 1 - High Impact, Low Disruption** (do first):
- Naming improvements that increase clarity without breaking references
- File relocations that better reflect current organization
- DOM class renames for semantic HTML
- Code duplication elimination (extract shared utilities, create base classes)
- Magic number/string extraction to named constants
- Simple antipattern fixes (e.g., async/await over promise chains)

**Tier 2 - Moderate Refactoring** (strategic improvements):
- Module consolidation or splitting for better cohesion
- Architectural pattern violations that should be corrected
- Service/component boundary adjustments
- Pattern replacements (e.g., event delegation, observer patterns)
- Complexity reduction through decomposition
- Decoupling tightly coupled components

**Tier 3 - Major Restructuring** (long-term considerations):
- Fundamental architectural changes (if any)
- Directory reorganization requiring widespread import updates
- Breaking API changes that improve design
- Large-scale pattern migrations (e.g., imperative to declarative architecture)

**For each recommendation, specify:**
- Current state vs. proposed change
- Rationale (how this improves architecture)
- Impact (files affected, breaking change severity: low/medium/high)
- **For duplication**: Exact locations of duplicated code and proposed consolidation location
- **For antipatterns**: Specific antipattern name and concrete example from codebase
- **For pattern improvements**: Before/after pattern comparison with measurable benefits (readability, maintainability, performance)

**Constraints:**
- Respect existing conventions from `.github/copilot-instructions.md` (named exports, no default exports, types in `types.ts`)
- Maintain the zero-dependency philosophy
- Preserve the service-oriented architecture pattern
- Prioritize changes that improve developer experience and code discoverability

**Output & File Delivery:**
- Write the complete analysis to a markdown file so it is captured in the repository history.
- Target path: `docs/reviews/architecture-review-YYYY-MM-DD.md` (use current date, e.g., `architecture-review-2025-11-01.md`).
- If `docs/reviews/` does not exist, create it.
- File structure:
   1. Title: `# Architecture Review – YYYY-MM-DD`
   2. Brief summary (3-5 bullets)
   3. Findings organized by Tier 1, Tier 2, Tier 3 (as specified above)
   4. Risk/impact assessment and suggested sequencing
   5. Appendix A: Affected files and links (relative paths)
   6. Appendix B: Detected duplications and locations
- Output only a short confirmation in chat (path + one-line summary); place full content in the markdown file.
