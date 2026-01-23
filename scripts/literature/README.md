# Literature Management System

## Overview

A comprehensive literature management system for research, supporting paper search, organization, content extraction, and citation management.

## Current Status: Fully Operational ✓

### Implemented Features

1. **Core Modules** (`src/literature/core/`)
   - `arxiv_api.py`: Enhanced arXiv client with rate limiting and retry logic
   - `database.py`: LiteratureDatabase class with collections, tags, and relationships
   - `config.py`: YAML configuration file loader

2. **Configuration System** (`config/`)
   - `queries/`: Search query configurations (vertex_operators.yaml, modular_invariance.yaml)
   - `collections/`: Paper collection definitions (priority_papers.yaml)

3. **CLI Interface** (`src/literature/cli/`)
   - `search.py`: arXiv search (direct and config-based)
   - `database.py`: Database management commands
   - `collect.py`: Automated literature collection
   - `download.py`: PDF and LaTeX source downloads
   - `extract.py`: Content extraction and full-text indexing
   - `__main__.py`: Unified entry point

4. **Content Extraction** (`src/literature/extractors/`)
   - Full-text extraction from PDFs and LaTeX sources
   - Equation extraction and rendering
   - Citation network analysis
   - Full-text search index with ~298 papers

## Quick Start

### Common Commands

#### Database Management

```powershell
# List all papers in database (305 papers)
python -m src.literature.cli db list

# Show comprehensive statistics
python -m src.literature.cli db stats

# List all collections (28 collections)
python -m src.literature.cli db collections

# Filter papers by collection
python -m src.literature.cli db list --collection modular_invariance

# Filter papers by status
python -m src.literature.cli db list --status to_review

# Filter by year
python -m src.literature.cli db list --year 2025

# Filter by tag
python -m src.literature.cli db list --tag "partition-functions"

# Sort papers
python -m src.literature.cli db list --sort-by year     # Sort by year (newest first)
python -m src.literature.cli db list --sort-by author   # Sort by first author
python -m src.literature.cli db list --sort-by status   # Sort by status
python -m src.literature.cli db list --sort-by arxiv    # Sort by arXiv ID

# Combine filters and sorting
python -m src.literature.cli db list --year 2024 --status reviewed --sort-by author

# Find papers that need metadata
python -m src.literature.cli db needs-metadata --limit 20

# Add paper to database
python -m src.literature.cli db add 2301.12345 --collection bootstrap_methods

# Show detailed paper information
python -m src.literature.cli db info 1403.4683

# Quick search across all papers
python -m src.literature.cli db find "komatsu"
python -m src.literature.cli db find "partition function" --max-results 20

# Update paper status
python -m src.literature.cli db update 2301.12345 reviewed --notes "Key paper on CLD"

# Update paper metadata
python -m src.literature.cli db update-metadata 1403.4683 --title "New Title" --authors "Author 1, Author 2" --year 2024

# Add tags to paper
python -m src.literature.cli db add-tags 1403.4683 "partition-functions,modular-forms,BRST"

# Remove tags from paper
python -m src.literature.cli db remove-tags 1403.4683 "old-tag,deprecated"

# Export collection to BibTeX for citations
python -m src.literature.cli db export references.bib --collection key_results
python -m src.literature.cli db export bootstrap.bib --collection conformal_bootstrap
```

#### arXiv Search

```powershell
# Direct search
python -m src.literature.cli search "linear dilaton CFT" --max-results 10

# Search using configuration file
python -m src.literature.cli search --config vertex_operators

# List available configs
python -m src.literature.cli search --list-configs

# Search specific category
python -m src.literature.cli search "bootstrap" --max-results 20
```

#### Content Extraction & Search

```powershell
# Extract content from a paper for full-text indexing
python -m src.literature.cli extract paper 2511.16280

# Extract content from entire collection
python -m src.literature.cli extract collection modular_invariance

# Search indexed content (full-text search)
python -m src.literature.cli search-content "partition function"

# Search for equations
python -m src.literature.cli search-equations "Z(\\tau)"

# Show index statistics
python -m src.literature.cli index-stats
```

#### Literature Collection

```powershell
# Run automated collection from config
python -m src.literature.cli collect run modular_invariance.yaml

# List available collection configs
python -m src.literature.cli collect list

# Dry run to preview results
python -m src.literature.cli collect run config.yaml --dry-run
```

#### Paper Downloads

```powershell
# Download PDF of a paper
python -m src.literature.cli download paper 2511.16280 --pdf

# Download LaTeX source
python -m src.literature.cli download paper 2511.16280 --latex

# Download all papers in collection
python -m src.literature.cli download collection modular_invariance

# Download from file list
python -m src.literature.cli download from-file paper_list.txt
```

### Using as Python Library

```python
from src.literature import ArxivClient, LiteratureDatabase

# Rate-limited arXiv searches
client = ArxivClient()
papers = client.search("conformal bootstrap", max_results=10)

# Database management
db = LiteratureDatabase()
for paper in papers:
    db.add_paper(paper, collection="bootstrap_methods")
db.save()

# Export to BibTeX
db.export_bibtex("bootstrap.bib", collection="bootstrap_methods")
```

### Configuration-Driven Searches

Create a YAML file in `config/queries/`:

```yaml
name: "My Research Topic"
collection: my_collection
category: hep-th

queries:
  - name: "main_search"
    query: "my search terms"
    max_results: 15
  - name: "related_search"
    query: "alternative terms"
    max_results: 10

output:
  directory: "docs/references/my_topic"
  formats: [json, markdown, bibtex]
```

Then run:
```powershell
python -m src.literature.cli search config my_research_topic
```

## Architecture

```
src/literature/
├── __init__.py          # Package exports
├── README.md            # This file
├── core/                # Core functionality
│   ├── arxiv_api.py     # Enhanced arXiv client with rate limiting
│   ├── database.py      # Database management
│   └── config.py        # YAML configuration loader
├── collectors/          # Automated collection utilities
├── extractors/          # Content extraction (PDF, LaTeX, equations)
│   └── search_index.py  # Full-text search index
└── cli/                 # Command-line interface
    ├── __main__.py      # Unified entry point
    ├── search.py        # arXiv search commands
    ├── database.py      # Database management commands
    ├── collect.py       # Literature collection commands
    ├── download.py      # PDF/LaTeX download commands
    └── extract.py       # Content extraction commands

config/
├── queries/             # Search configurations
│   ├── vertex_operators.yaml
│   └── modular_invariance.yaml
└── collections/         # Collection definitions
    └── priority_papers.yaml

docs/references/         # Database storage location
├── literature_database.json  # Main database
├── search_index.json         # Full-text search index
└── [subdirectories]/         # Organized paper notes
```

## Key Features

### 1. Comprehensive Database
- **Automatic categorization**: Papers tagged by topic and research area
- **Status tracking**: `to_review`, `reviewed`, `key_result`, `reference`
- **Rich metadata**: Title, authors, abstract, year, collections, tags, notes

### 2. Full-Text Search
- **Search index**: papers with extracted content
- **Content search**: Search paper text, equations, and citations
- **Equation search**: Find specific mathematical expressions
- **Fast retrieval**: Indexed for quick lookup

### 3. arXiv Integration
- **Rate limiting**: Automatic 3-second compliance with arXiv guidelines
- **Retry logic**: Exponential backoff for failed downloads
- **Metadata fetching**: Automatic title, author, abstract retrieval
- **PDF/LaTeX download**: Bulk download capabilities

### 4. Flexible Organization
- **Multiple collections**: Papers can belong to multiple research areas
- **Tagging system**: Flexible keyword tagging for cross-referencing
- **Hierarchical structure**: Organized by topic and subtopic
- **Citation export**: BibTeX export for paper writing

### 5. Bulk Operations
- **Bulk import**: Import papers from structured markdown files
- **Batch processing**: Extract content from entire collections
- **Automated organization**: Auto-categorize papers by folder structure
- **Progress tracking**: Detailed reports on import and extraction

### 6. Enhanced Database Schema
```json
{
  "papers": [
    {
      "arxiv": "2511.16280",
      "title": "Chiral Linear Dilaton...",
      "authors": ["Komatsu", "Maity"],
      "year": 2025,
      "abstract": "...",
      "collections": ["modular_invariance", "veneziano_uniqueness"],
      "tags": ["CLD", "BRST", "uniqueness"],
      "status": "key_result",
      "notes": "Proves BRST uniqueness of CLD"
    }
  ],
  "collections": {
    "modular_invariance": {
      "name": "Modular Invariance",
      "description": "Papers on modular invariance in string theory",
      "paper_count": 30
    }
  }
}
```

## Programmatic Usage

### Using as Python Library

```python
from src.literature import ArxivClient, LiteratureDatabase

# Rate-limited arXiv searches
client = ArxivClient()
papers = client.search("conformal bootstrap", max_results=10)

# Database management
db = LiteratureDatabase()
print(f"Total papers: {len(db.db['papers'])}")
print(f"Collections: {list(db.db['collections'].keys())}")

# Add papers
for paper in papers:
    db.add_paper(paper, collection="bootstrap_methods")
db.save()

# Query papers
bootstrap_papers = db.get_collection("conformal_bootstrap")
key_papers = [p for p in db.db['papers'] if p['status'] == 'key_result']

# Export to BibTeX
db.export_bibtex("references.bib", collection="key_results")

# Get statistics
stats = db.get_statistics()
print(f"Papers by status: {stats['by_status']}")
print(f"Papers by collection: {stats['by_collection']}")
```

## Workflow Examples

### Example 1: Finding Papers for New Research Topic

```powershell
# 1. Search arXiv for relevant papers
python -m src.literature.cli search search "worldsheet CFT" --max-results 20

# 2. Add important papers to database
python -m src.literature.cli db add 2301.12345 --collection worldsheet_cft

# 3. Download papers for reading
python -m src.literature.cli download paper 2301.12345 --pdf

# 4. Extract content for searchability
python -m src.literature.cli extract paper 2301.12345

# 5. Search for specific concepts
python -m src.literature.cli search-content "vertex operator"

# 6. Export citations for paper writing
python -m src.literature.cli db export worldsheet_refs.bib --collection worldsheet_cft
```

### Example 2: Organizing Existing Papers

```powershell
# 1. Bulk import papers from docs/references/
python scripts/bulk_import_papers.py

# 2. Review database statistics
python -m src.literature.cli db stats

# 3. List papers needing review
python -m src.literature.cli db list --status to_review

# 4. Update paper status after reading
python -m src.literature.cli db update 2511.16280 key_result --notes "Proves BRST uniqueness"

# 5. Create collection-specific bibliography
python -m src.literature.cli db export key_results.bib --status key_result
```

### Example 3: Building Full-Text Search Index

```powershell
# 1. Extract content from entire collection
python -m src.literature.cli extract collection modular_invariance

# 2. Check index statistics
python -m src.literature.cli index-stats

# 3. Search for specific equations
python -m src.literature.cli search-equations "\\mathcal{Z}(\\tau)"

# 4. Search for concepts across all papers
python -m src.literature.cli search-content "partition function modular"
```

## Complete CLI Command Reference

### Database Commands (`db`)

#### `db list`
List papers in the database with optional filtering.

```powershell
# List all papers
python -m src.literature.cli db list

# Filter by collection
python -m src.literature.cli db list --collection modular_invariance

# Filter by status
python -m src.literature.cli db list --status key_paper
```

#### `db info`
Show detailed information about a specific paper.

```powershell
python -m src.literature.cli db info 1403.4683
```

Output includes: arXiv ID, title, authors, year, status, collections, tags, notes, and abstract.

#### `db add`
Add a paper to the database by arXiv ID.

```powershell
# Add paper
python -m src.literature.cli db add 2301.12345

# Add paper to specific collection
python -m src.literature.cli db add 2301.12345 --collection bootstrap_methods
```

#### `db update`
Update the status of a paper.

```powershell
# Update status
python -m src.literature.cli db update 1403.4683 key_paper

# Update with notes
python -m src.literature.cli db update 1403.4683 reviewed --notes "Excellent reference for partition functions"
```

Valid status values:
- `key_paper` - Essential reference
- `reviewed` - Already read and analyzed
- `to_review` - Needs reading
- `classic` - Classic foundational paper
- `textbook` - Textbook or review article
- `reference` - Quick reference material
- `high_priority` - Priority reading

#### `db update-metadata`
Update paper metadata fields.

```powershell
# Update title
python -m src.literature.cli db update-metadata 1403.4683 --title "New Title"

# Update authors (comma-separated)
python -m src.literature.cli db update-metadata 1403.4683 --authors "Author 1, Author 2, Author 3"

# Update year
python -m src.literature.cli db update-metadata 1403.4683 --year 2024

# Update multiple fields at once
python -m src.literature.cli db update-metadata 1403.4683 \
    --title "Closed String Partition Functions" \
    --authors "H. S. Tan" \
    --year 2014
```

**Use cases:**
- Fix incorrect metadata from arXiv
- Add missing information
- Standardize author names
- Correct publication years

#### `db add-tags`
Add tags to a paper for better organization.

```powershell
# Add single tag
python -m src.literature.cli db add-tags 1403.4683 "modular-forms"

# Add multiple tags (comma-separated)
python -m src.literature.cli db add-tags 1403.4683 "partition-functions,BRST-cohomology,critical-dimension"
```

**Common tag categories:**
- Method tags: `bootstrap-methods`, `worldsheet-CFT`, `modular-invariance`
- Topic tags: `partition-functions`, `vertex-operators`, `BRST-cohomology`
- Dimension tags: `critical-dimension`, `D-26`, `higher-dimensions`
- Application tags: `CLD-backgrounds`, `string-amplitudes`, `compactifications`

#### `db remove-tags`
Remove tags from a paper.

```powershell
# Remove single tag
python -m src.literature.cli db remove-tags 1403.4683 "old-tag"

# Remove multiple tags
python -m src.literature.cli db remove-tags 1403.4683 "deprecated,incorrect,outdated"
```

#### `db stats`
Display comprehensive database statistics.

```powershell
python -m src.literature.cli db stats
```

Shows:
- Total paper count
- Papers by status
- Papers by collection
- Papers by year (recent years)

#### `db collections`
List all collections with paper counts.

```powershell
python -m src.literature.cli db collections
```

#### `db export`
Export papers to BibTeX format for citations.

```powershell
# Export all papers
python -m src.literature.cli db export all_papers.bib

# Export specific collection
python -m src.literature.cli db export modular_papers.bib --collection modular_invariance

# Export by status
python -m src.literature.cli db export key_papers.bib --status key_paper

# Combine filters
python -m src.literature.cli db export bootstrap_key.bib \
    --collection conformal_bootstrap \
    --status key_paper
```

#### `db find`
Quick search across all papers by keyword (searches titles, authors, tags, and arXiv IDs).

```powershell
# Find papers by author
python -m src.literature.cli db find "komatsu"

# Find papers by topic
python -m src.literature.cli db find "partition function"

# Find by arXiv ID fragment
python -m src.literature.cli db find "2511"

# Limit results
python -m src.literature.cli db find "bootstrap" --max-results 20
```

**Use cases:**
- "Who wrote about X?" → `db find "author name"`
- "Do we have paper Y?" → `db find "arxiv_id"`
- "Find all Z papers" → `db find "topic keyword"`

**Why use `find` vs `search-content`?**
- `find`: Fast keyword search in metadata (titles, authors, tags, IDs)
- `search-content`: Deep full-text search in paper content (slower, more comprehensive)

### Search Commands

#### `search`
Direct arXiv search. ⭐ **Simplified syntax** - no more double "search"!

```powershell
# Basic search
python -m src.literature.cli search "linear dilaton CFT"

# Limit results
python -m src.literature.cli search "bootstrap" --max-results 20

# Search by author
python -m src.literature.cli search "au:Komatsu"
```

#### `search --config`
Run predefined searches from configuration files.

```powershell
# Run configured search
python -m src.literature.cli search --config vertex_operators
python -m src.literature.cli search --config modular_invariance

# List available configs
python -m src.literature.cli search --list-configs
```

### Content Commands

#### `search-content`
Full-text search across extracted paper content.

```powershell
# Search for terms
python -m src.literature.cli search-content "partition function"

# Limit results
python -m src.literature.cli search-content "BRST cohomology" --max-results 10

# Find multiple terms
python -m src.literature.cli search-content "critical dimension 26"
```

Returns papers ranked by relevance score.

#### `search-equations`
Search for specific equations or mathematical expressions.

```powershell
# Search for LaTeX equation patterns
python -m src.literature.cli search-equations "Z(\\tau)"
python -m src.literature.cli search-equations "\\Delta"
```

#### `download paper`
Download PDF or LaTeX source for a paper.

```powershell
# Download PDF
python -m src.literature.cli download paper 1403.4683 --pdf

# Download LaTeX source
python -m src.literature.cli download paper 1403.4683 --source

# Download both
python -m src.literature.cli download paper 1403.4683 --pdf --source
```

PDFs saved to: `docs/papers/pdfs/`
Sources saved to: `docs/papers/sources/`

#### `download collection`
Bulk download for entire collections.

```powershell
# Download all PDFs in collection
python -m src.literature.cli download collection modular_invariance --pdf

# Download sources for collection
python -m src.literature.cli download collection key_results --source
```

#### `extract paper`
Extract content from a paper for full-text indexing.

```powershell
# Extract from downloaded paper
python -m src.literature.cli extract paper 1403.4683
```

Extracts:
- Full text content
- Equations (LaTeX)
- Citations and references
- Section structure

#### `extract collection`
Bulk extraction for collections.

```powershell
# Extract all papers in collection
python -m src.literature.cli extract collection modular_invariance
```

### Workflow Examples

#### New Paper Workflow
```powershell
# 1. Search for papers
python -m src.literature.cli search search "worldsheet CFT" --max-results 10

# 2. Add relevant paper to database
python -m src.literature.cli db add 2401.12345 --collection worldsheet_cft

# 3. Download PDF
python -m src.literature.cli download paper 2401.12345 --pdf

# 4. Extract content for searching
python -m src.literature.cli extract paper 2401.12345

# 5. Add tags
python -m src.literature.cli db add-tags 2401.12345 "worldsheet,CFT,modular-forms"

# 6. Update status after reading
python -m src.literature.cli db update 2401.12345 reviewed --notes "Key reference for worldsheet construction"
```

#### Literature Review Workflow
```powershell
# 1. Find relevant papers by topic
python -m src.literature.cli search-content "critical dimension" --max-results 20

# 2. Check specific paper details
python -m src.literature.cli db info 1403.4683

# 3. Mark as high priority
python -m src.literature.cli db update 1403.4683 high_priority

# 4. Export for citation
python -m src.literature.cli db export literature_review.bib --status high_priority
```

#### Collection Management Workflow
```powershell
# 1. List collection contents
python -m src.literature.cli db list --collection modular_invariance

# 2. Download all PDFs
python -m src.literature.cli download collection modular_invariance --pdf

# 3. Extract all content
python -m src.literature.cli extract collection modular_invariance

# 4. Search within collection content
python -m src.literature.cli search-content "modular transformation"

# 5. Export collection bibliography
python -m src.literature.cli db export modular_refs.bib --collection modular_invariance
```

### Database Cleanup

```powershell
# Remove duplicates (if any)
python -m src.literature.cli db deduplicate

# Validate arXiv IDs
python -m src.literature.cli db validate

# Update collection counts
python -m src.literature.cli db update-collections
```

## Future Enhancements

Potential improvements for the system:

1. **Automated Metadata Fetching**
   - Scheduled jobs to fetch missing metadata
   - Automatic updates for published papers

2. **Enhanced Search**
   - Relevance scoring for search results
   - Citation-based paper recommendations
   - Semantic search using embeddings

3. **Analysis Tools**
   - Citation network visualization
   - Research trend analysis
   - Collaboration network mapping

4. **Content Extraction**
   - Improved equation rendering
   - Table extraction from PDFs
   - Figure caption extraction
   - Theorem/lemma extraction

5. **Integration**
   - Jupyter notebook integration
   - LaTeX document integration
   - Reference manager export (Zotero, Mendeley)

## Documentation

- **This file**: Complete CLI and API reference
- `config/queries/*.yaml`: Search configuration examples

## Support & Development

**Current Status**: Production-ready, actively maintained

**Database Location**: `docs/references/literature_database.json`

**Search Index**: `docs/references/search_index.json`

**Last Major Update**: 2026-01-22

For questions or contributions, see the main project README.
