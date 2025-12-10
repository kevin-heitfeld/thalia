---
mode: agent
---
# Documentation Cleanup and Consolidation

## Objectives
Consolidate related documentation, archive outdated content, and remove redundant files
while preserving docs referenced in code and developer workflows.

## Phase 1: Audit & Classify
Categorize each file in `/docs`:
- **Active**: Referenced in codebase, copilot-instructions.md, or core developer workflows
- **Outdated**: Completed phases, superseded docs, stale information (e.g., PHASE3_COMPLETE,
  prior session summaries)
- **Duplicate**: Overlaps significantly with another file; minimal unique value
- **Unnecessary**: Unreferenced; doesn't support current development

## Phase 2: Consolidation Rules
For overlapping topics, apply this decision matrix:

| Action | Apply When | Example |
|--------|-----------|---------|
| **Merge** | Multiple complementary files on one topic | Multiple audit reports → single findings doc |
| **Archive** | Historically relevant but inactive | PHASE3 completion, session summaries |
| **Delete** | Unreferenced + no historical value | Duplicate drafts, intermediate notes |
| **Preserve** | Serves distinct purpose (quick ref vs. guide) | Keep separate if each is independently useful |

## Phase 3: Execution Steps
1. Create mapping of files → actions (merge/archive/delete/preserve)
2. Find all doc references in codebase (PowerShell):
   ```powershell
   Get-ChildItem -Recurse -Include *.ts,*.md src,tests,.github | Select-String -Pattern "docs/"
   ```
3. Preserve all referenced files; update paths if moved
4. Verify links in copilot-instructions.md point to final locations
5. Execute consolidation; move completed work to `/docs/archive/`
6. Create `CONSOLIDATION_LOG.md` documenting all changes

## Phase 4: Validation
Before marking complete, verify:
- [ ] All codebase doc references point to valid files
- [ ] Links in copilot-instructions.md are functional
- [ ] DOCUMENTATION_INDEX.md reflects final structure
- [ ] File count reduced by 25-35% (quantifiable success metric)
- [ ] Each subdirectory has non-overlapping scope
