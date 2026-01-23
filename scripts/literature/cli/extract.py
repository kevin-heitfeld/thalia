"""
CLI commands for content extraction and search operations.
"""

from pathlib import Path

from ..core import LiteratureDatabase
from ..extractors import ContentExtractor, SearchIndex


def extract_paper(args):
    """Extract content from a single paper."""
    extractor = ContentExtractor()

    # Auto-detect PDF/LaTeX paths if not provided
    pdf_path = args.pdf
    latex_dir = args.latex

    if not pdf_path and not latex_dir:
        # Try to find files automatically
        arxiv_id = args.arxiv_id.replace("/", "_")

        # Check for PDF in standard locations
        pdf_locations = [
            Path("docs/papers/pdfs") / f"{arxiv_id}.pdf",
            Path("docs/papers/pdfs") / f"{args.arxiv_id}v1.pdf",
            Path("docs/papers/pdfs") / f"{args.arxiv_id}v2.pdf",
            Path("docs/papers/pdfs") / f"{args.arxiv_id}v3.pdf",
        ]

        for pdf_loc in pdf_locations:
            if pdf_loc.exists():
                pdf_path = str(pdf_loc)
                print(f"Found PDF: {pdf_path}")
                break

        # Check for LaTeX source
        latex_loc = Path("docs/papers/latex") / arxiv_id
        if latex_loc.exists():
            latex_dir = str(latex_loc)
            print(f"Found LaTeX: {latex_dir}")

    if not pdf_path and not latex_dir:
        print(f"Error: No PDF or LaTeX source found for {args.arxiv_id}")
        print("Please specify --pdf or --latex, or download the paper first:")
        print(f"  python -m src.literature.cli download paper {args.arxiv_id} --pdf")
        return

    content = extractor.extract_paper_content(pdf_path=pdf_path, latex_dir=latex_dir)

    # Display results
    if content["pdf_text"]:
        words = len(content["pdf_text"].split())
        print(f"\nPDF text: {words:,} words extracted")

    if content["equations"]:
        print(f"LaTeX equations: {len(content['equations'])} found")
        if args.show_equations:
            for i, eq in enumerate(content["equations"][:10], 1):
                print(f"\n  [{i}] Line {eq['line_number']} ({eq['environment']}):")
                print(f"      {eq['equation'][:80]}...")
            if len(content["equations"]) > 10:
                print(f"\n  ... and {len(content['equations']) - 10} more")

    if content["metadata"].get("title"):
        print(f"\nTitle: {content['metadata']['title']}")

    if content["metadata"].get("sections"):
        print(f"Sections: {len(content['metadata']['sections'])}")

    # Add to index if requested
    if args.index:
        index = SearchIndex(args.index_file)
        index.add_paper(
            arxiv_id=args.arxiv_id,
            pdf_text=content["pdf_text"],
            equations=content["equations"],
            metadata=content["metadata"],
        )
        print(f"\n✓ Added to search index: {args.index_file}")


def extract_collection(args):
    """Extract content from all papers in a collection."""
    db = LiteratureDatabase(args.database)
    extractor = ContentExtractor()
    index = SearchIndex(args.index_file)

    papers = db.get_collection(args.collection)

    print(f"\n{'=' * 80}")
    print(f"EXTRACTING CONTENT: {args.collection}")
    print(f"{'=' * 80}")
    print(f"Papers: {len(papers)}")
    print(f"{'=' * 80}\n")

    extracted_count = 0

    for i, paper in enumerate(papers, 1):
        arxiv_id = paper.get("arxiv", "")
        title = paper.get("title", "Unknown")[:60]

        print(f"\n[{i}/{len(papers)}] {arxiv_id}")
        print(f"    {title}")

        # Determine file paths
        pdf_path = args.pdf_dir / args.collection / f"{arxiv_id.replace('/', '_')}.pdf"
        latex_dir = args.latex_dir / arxiv_id.replace("/", "_")

        # Extract
        content = extractor.extract_paper_content(
            pdf_path=str(pdf_path) if pdf_path.exists() else None,
            latex_dir=str(latex_dir) if latex_dir.exists() else None,
        )

        # Add to index
        if content["pdf_text"] or content["equations"]:
            index.add_paper(
                arxiv_id=arxiv_id,
                pdf_text=content["pdf_text"],
                equations=content["equations"],
                metadata=content["metadata"],
            )
            extracted_count += 1

    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Papers processed: {len(papers)}")
    print(f"Papers indexed: {extracted_count}")
    print(f"Index saved: {args.index_file}")
    print(f"{'=' * 80}\n")


def extract_batch(args):
    """Batch extract content for multiple papers (prioritizing key papers)."""
    db = LiteratureDatabase(args.database)
    extractor = ContentExtractor()
    index = SearchIndex(args.index_file)

    # Determine which papers to extract
    if args.collection:
        papers = db.get_collection(args.collection)
        source_desc = f"collection '{args.collection}'"
    elif args.priority:
        # Get papers from priority collections first
        priority_collections = [
            "key_paper",
            "modular_invariance",
            "cld_amplitudes",
            "worldsheet_cft",
        ]
        papers = []
        for collection in priority_collections:
            coll_papers = db.get_collection(collection)
            if coll_papers:
                papers.extend(coll_papers)

        # Deduplicate
        seen = set()
        unique_papers = []
        for paper in papers:
            arxiv_id = paper.get("arxiv") or paper.get("arxiv_id")
            if arxiv_id and arxiv_id not in seen:
                seen.add(arxiv_id)
                unique_papers.append(paper)
        papers = unique_papers
        source_desc = "priority collections"
    else:
        # Get all papers
        all_papers = []
        for collection in db.get_collections():
            coll_papers = db.get_collection(collection)
            if coll_papers:
                all_papers.extend(coll_papers)

        # Deduplicate
        seen = set()
        papers = []
        for paper in all_papers:
            arxiv_id = paper.get("arxiv") or paper.get("arxiv_id")
            if arxiv_id and arxiv_id not in seen:
                seen.add(arxiv_id)
                papers.append(paper)
        source_desc = "all papers"

    # Apply limit
    if args.limit:
        papers = papers[: args.limit]

    print(f"\n{'=' * 80}")
    print(f"BATCH CONTENT EXTRACTION: {source_desc}")
    print(f"{'=' * 80}")
    print(f"Papers to process: {len(papers)}")
    print(f"{'=' * 80}\n")

    extracted_count = 0
    skipped_count = 0

    for i, paper in enumerate(papers, 1):
        arxiv_id = paper.get("arxiv") or paper.get("arxiv_id", "")
        title = paper.get("title", "Unknown")[:60]

        print(f"\n[{i}/{len(papers)}] {arxiv_id}")
        print(f"    {title}")

        # Determine file paths
        pdf_path = Path(args.pdf_dir) / f"{arxiv_id.replace('/', '_')}.pdf"
        latex_dir = Path(args.latex_dir) / arxiv_id.replace("/", "_")

        # Check if already indexed and skip if requested
        if args.skip_existing:
            existing_papers = index.list_papers()
            if arxiv_id in existing_papers:
                print("    ✓ Already indexed (skipping)")
                skipped_count += 1
                continue

        # Extract
        try:
            content = extractor.extract_paper_content(
                pdf_path=str(pdf_path) if pdf_path.exists() else None,
                latex_dir=str(latex_dir) if latex_dir.exists() else None,
            )

            # Add to index
            if content["pdf_text"] or content["equations"]:
                index.add_paper(
                    arxiv_id=arxiv_id,
                    pdf_text=content["pdf_text"],
                    equations=content["equations"],
                    metadata=content["metadata"],
                )
                print(
                    f"    ✓ Extracted ({len(content['pdf_text'])} chars, {len(content['equations'])} equations)"
                )
                extracted_count += 1
            else:
                print("    ✗ No content found")
        except Exception as e:  # noqa: BLE001
            print(f"    ✗ Error: {e}")

    print("\n" + "=" * 80)
    print("BATCH EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Papers processed:     {len(papers)}")
    print(f"Successfully indexed: {extracted_count}")
    print(f"Skipped (existing):   {skipped_count}")
    print(f"Index saved: {args.index_file}")
    print(f"{'=' * 80}\n")


def search_papers(args):
    """Search indexed papers."""
    index = SearchIndex(args.index_file)

    # Try to load database for better titles
    try:
        from ..core import LiteratureDatabase

        db = LiteratureDatabase()
        has_db = True
    except Exception:  # noqa: BLE001
        db = None
        has_db = False

    print("\n" + "=" * 80)
    print(f"SEARCHING: {args.query}")
    print("=" * 80 + "\n")

    results = index.search(
        query=args.query, max_results=args.max_results, search_equations=not args.no_equations
    )

    if not results:
        print("No results found.")
        return

    print(f"Found {len(results)} result(s):\n")

    for i, (arxiv_id, score) in enumerate(results, 1):
        paper_info = index.get_paper_info(arxiv_id)
        if not paper_info:
            continue

        # Get title and abstract from database first, fall back to index metadata
        title = None
        abstract = None
        if has_db and db.paper_exists(arxiv_id):
            db_paper = db.get_paper_info(arxiv_id)
            if db_paper:
                title = db_paper.get("title")
                abstract = db_paper.get("abstract")

        if not title:
            metadata = paper_info.get("metadata", {})
            title = metadata.get("title", "No title")

        if title is None:
            title = "No title"
        title = str(title)[:70]

        print(f"{i:2d}. {arxiv_id} (score: {score:.1f})")
        print(f"    {title}")

        if paper_info.get("word_count"):
            words = paper_info["word_count"]
            eqs = paper_info.get("num_equations", 0)
            print(f"    Words: {words:,}, Equations: {eqs}")

        # Show abstract snippet if available and --show-abstract is set
        if args.show_abstract and abstract:
            abstract_snippet = abstract[:200]
            if len(abstract) > 200:
                abstract_snippet += "..."
            print(f"    Abstract: {abstract_snippet}")

        if args.show_preview and paper_info.get("text_preview"):
            preview = paper_info["text_preview"][:150]
            print(f"    Preview: {preview}...")

        print()


def search_equations_cmd(args):
    """Search for equations matching pattern."""
    index = SearchIndex(args.index_file)

    print(f"\n{'=' * 80}")
    print(f"SEARCHING EQUATIONS: {args.pattern}")
    print(f"{'=' * 80}\n")

    results = index.search_equations(pattern=args.pattern, max_results=args.max_results)

    if not results:
        print("No equations found matching pattern.")
        return

    print(f"Found {len(results)} equation(s):\n")

    for i, result in enumerate(results, 1):
        arxiv_id = result["arxiv_id"]
        eq = result["equation"]

        print(f"{i:2d}. {arxiv_id} - Line {eq.get('line_number', '?')}")
        print(f"    Environment: {eq.get('environment', 'unknown')}")
        if eq.get("label"):
            print(f"    Label: {eq['label']}")
        print(f"    {eq['equation'][:100]}...")
        print()


def index_stats(args):
    """Show search index statistics."""
    index = SearchIndex(args.index_file)
    stats = index.get_statistics()

    # Get database stats for comparison
    db = LiteratureDatabase()
    db_stats = db.get_statistics()
    total_in_db = db_stats["total_papers"]

    print("\n" + "=" * 80)
    print("SEARCH INDEX STATISTICS")
    print("=" * 80)
    print(f"Index file: {args.index_file}")
    print(f"\nDatabase: {total_in_db} total papers")
    print(f"Index:    {stats['total_papers']} papers indexed")

    if stats["total_papers"] < total_in_db:
        missing = total_in_db - stats["total_papers"]
        print(f"\nNote: {missing} papers not yet indexed.")
        print("      Use 'extract paper <arxiv_id> --index' to add papers to search index.")
        print("      Full-text search requires PDF or LaTeX source extraction.")

    print("\nIndexed papers:")
    print(f"  With PDF text: {stats['papers_with_text']}")
    print(f"  With equations: {stats['papers_with_equations']}")
    print("\nIndex content:")
    print(f"  Words indexed: {stats['total_words_indexed']:,}")
    print(f"  Equation terms: {stats['total_equation_terms']:,}")
    print("=" * 80 + "\n")


def setup_extract_parser(subparsers):
    """Set up argument parser for extract commands."""
    extract_parser = subparsers.add_parser(
        "extract", help="Extract content from PDFs and LaTeX sources"
    )

    extract_subparsers = extract_parser.add_subparsers(
        dest="extract_command", help="Extract subcommands"
    )

    # extract paper
    paper_parser = extract_subparsers.add_parser(
        "paper", help="Extract content from a single paper"
    )
    paper_parser.add_argument("arxiv_id", help="ArXiv ID")
    paper_parser.add_argument("--pdf", help="Path to PDF file")
    paper_parser.add_argument("--latex", help="Path to LaTeX directory")
    paper_parser.add_argument(
        "--show-equations", action="store_true", help="Show extracted equations"
    )
    paper_parser.add_argument("--index", action="store_true", help="Add to search index")
    paper_parser.add_argument("--index-file", default="docs/references/search_index.json")
    paper_parser.set_defaults(func=extract_paper)

    # extract collection
    collection_parser = extract_subparsers.add_parser(
        "collection", help="Extract content from collection"
    )
    collection_parser.add_argument("collection", help="Collection name")
    collection_parser.add_argument("--database", default="docs/references/literature_database.json")
    collection_parser.add_argument("--pdf-dir", type=Path, default=Path("docs/papers/pdfs"))
    collection_parser.add_argument("--latex-dir", type=Path, default=Path("docs/papers/latex"))
    collection_parser.add_argument("--index", action="store_true", default=True)
    collection_parser.add_argument("--index-file", default="docs/references/search_index.json")
    collection_parser.set_defaults(func=extract_collection)

    # extract batch (multiple papers)
    batch_parser = extract_subparsers.add_parser(
        "batch", help="Batch extract content for multiple papers"
    )
    batch_parser.add_argument("--collection", help="Extract from specific collection")
    batch_parser.add_argument(
        "--priority", action="store_true", help="Extract priority collections only"
    )
    batch_parser.add_argument("--limit", type=int, help="Maximum papers to process")
    batch_parser.add_argument(
        "--skip-existing", action="store_true", help="Skip already indexed papers"
    )
    batch_parser.add_argument("--database", default="docs/references/literature_database.json")
    batch_parser.add_argument("--pdf-dir", default="docs/papers/pdfs")
    batch_parser.add_argument("--latex-dir", default="docs/papers/latex")
    batch_parser.add_argument("--index-file", default="docs/references/search_index.json")
    batch_parser.set_defaults(func=extract_batch)

    # search
    search_parser = subparsers.add_parser("search-content", help="Search indexed paper content")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--max-results", type=int, default=20)
    search_parser.add_argument("--no-equations", action="store_true", help="Don't search equations")
    search_parser.add_argument("--show-preview", action="store_true", help="Show text preview")
    search_parser.add_argument(
        "--show-abstract", action="store_true", help="Show abstract snippets"
    )
    search_parser.add_argument("--index-file", default="docs/references/search_index.json")
    search_parser.set_defaults(func=search_papers)

    # search equations
    eq_parser = subparsers.add_parser("search-equations", help="Search for equations by pattern")
    eq_parser.add_argument("pattern", help="LaTeX pattern (e.g., '\\\\frac')")
    eq_parser.add_argument("--max-results", type=int, default=20)
    eq_parser.add_argument("--index-file", default="docs/references/search_index.json")
    eq_parser.set_defaults(func=search_equations_cmd)

    # index stats
    stats_parser = subparsers.add_parser("index-stats", help="Show search index statistics")
    stats_parser.add_argument("--index-file", default="docs/references/search_index.json")
    stats_parser.set_defaults(func=index_stats)


def render_equations_cmd(args):
    """Render equations to images."""
    from ..extractors import EquationRenderer

    extractor = ContentExtractor()
    renderer = EquationRenderer(dpi=args.dpi, fontsize=args.fontsize)

    print(f"\n{'=' * 80}")
    print("RENDERING EQUATIONS")
    print(f"{'=' * 80}\n")

    # Extract equations
    if args.latex:
        equations = extractor.extract_latex_equations(args.latex)
        print(f"Found {len(equations)} equations in {args.latex}")
    elif args.latex_dir:
        # Find all .tex files and extract
        latex_path = Path(args.latex_dir)
        tex_files = list(latex_path.glob("**/*.tex"))
        equations = []
        for tex_file in tex_files:
            eqs = extractor.extract_latex_equations(str(tex_file))
            equations.extend(eqs)
        print(f"Found {len(equations)} total equations from {len(tex_files)} .tex files")
    else:
        print("Error: Provide --latex or --latex-dir")
        return

    # Render
    if equations:
        output_dir = Path(args.output_dir)
        print(f"\nRendering to: {output_dir}")

        if args.grid:
            # Render as grid
            eq_strs = [eq["equation"] for eq in equations[: args.max_equations]]
            success = renderer.render_equation_grid(
                eq_strs, str(output_dir / f"equations_grid.{args.format}"), cols=args.grid_cols
            )
            if success:
                print(f"✓ Created equation grid: equations_grid.{args.format}")
        else:
            # Render individually
            results = renderer.render_equations_batch(
                equations[: args.max_equations],
                str(output_dir),
                format=args.format,
                prefix=args.prefix,
            )
            print(f"\n✓ Rendered: {results['rendered']}")
            print(f"✗ Failed: {results['failed']}")

            if args.show_files and results["output_files"]:
                print("\nOutput files:")
                for f in results["output_files"][:10]:
                    print(f"  {f}")
                if len(results["output_files"]) > 10:
                    print(f"  ... and {len(results['output_files']) - 10} more")

    print(f"\n{'=' * 80}\n")


def extract_tables_cmd(args):
    """Extract tables from PDFs and LaTeX."""
    from ..extractors import TableExtractor

    extractor = TableExtractor()

    print(f"\n{'=' * 80}")
    print("EXTRACTING TABLES")
    print(f"{'=' * 80}\n")

    # Extract tables
    results = extractor.extract_all_tables(
        pdf_path=args.pdf if args.pdf else None, latex_path=args.latex if args.latex else None
    )

    pdf_tables = results["pdf_tables"]
    latex_tables = results["latex_tables"]

    total = len(pdf_tables) + len(latex_tables)
    print(f"Found {total} total tables:")
    print(f"  PDF: {len(pdf_tables)}")
    print(f"  LaTeX: {len(latex_tables)}")

    # Show details
    if args.show_details:
        if pdf_tables:
            print("\nPDF Tables:")
            for i, table in enumerate(pdf_tables[:5], 1):
                print(f"  [{i}] Page {table['page']+1}: {table['rows']}x{table['cols']}")
            if len(pdf_tables) > 5:
                print(f"  ... and {len(pdf_tables) - 5} more")

        if latex_tables:
            print("\nLaTeX Tables:")
            for i, table in enumerate(latex_tables[:5], 1):
                caption = table.get("caption", "No caption")[:50]
                print(f"  [{i}] Line {table['line_number']}: {caption}")
                if table.get("label"):
                    print(f"      Label: {table['label']}")
            if len(latex_tables) > 5:
                print(f"  ... and {len(latex_tables) - 5} more")

    # Export to CSV
    if args.export:
        output_dir = Path(args.output_dir)
        all_tables = pdf_tables + latex_tables

        results = extractor.export_tables_batch(all_tables, str(output_dir), prefix=args.prefix)

        print(f"\n✓ Exported: {results['exported']} tables")
        print(f"✗ Failed: {results['failed']}")
        print(f"Output directory: {output_dir}")

    print(f"\n{'=' * 80}\n")


def show_figures_cmd(args):
    """Show figure information with cross-references."""
    extractor = ContentExtractor()

    print(f"\n{'=' * 80}")
    print("FIGURE INFORMATION")
    print(f"{'=' * 80}\n")

    if not args.latex_dir:
        print("Error: Provide --latex-dir")
        return

    figures = extractor.extract_figures_info(args.latex_dir)

    print(f"Found {len(figures)} figures\n")

    for i, fig in enumerate(figures, 1):
        print(f"[{i}] {fig.get('filename', 'No file')}")
        if fig.get("caption"):
            caption = fig["caption"][:80] + "..." if len(fig["caption"]) > 80 else fig["caption"]
            print(f"    Caption: {caption}")
        if fig.get("label"):
            print(f"    Label: {fig['label']}")
        print(f"    Line: {fig['line_number']} in {fig['tex_file']}")
        print(f"    Exists: {'✓' if fig.get('exists') else '✗'}")
        print(f"    Referenced: {fig.get('reference_count', 0)} times")

        if args.show_references and fig.get("references"):
            print("    References:")
            for ref in fig["references"][:3]:
                print(f"      - {ref['file']}:{ref['line']}")
            if len(fig["references"]) > 3:
                print(f"      ... and {len(fig['references']) - 3} more")

        print()

    print(f"{'=' * 80}\n")


def setup_advanced_commands(subparsers):
    """Set up advanced extraction command parsers."""

    # render equations
    render_parser = subparsers.add_parser("render-equations", help="Render equations to images")
    render_parser.add_argument("--latex", help="Path to .tex file")
    render_parser.add_argument("--latex-dir", help="Path to directory with .tex files")
    render_parser.add_argument(
        "--output-dir", default="docs/papers/rendered_equations", help="Output directory"
    )
    render_parser.add_argument("--format", choices=["png", "svg"], default="png")
    render_parser.add_argument("--dpi", type=int, default=300, help="DPI for PNG output")
    render_parser.add_argument("--fontsize", type=int, default=14, help="Font size")
    render_parser.add_argument("--max-equations", type=int, help="Max equations to render")
    render_parser.add_argument("--prefix", default="eq", help="Filename prefix")
    render_parser.add_argument("--grid", action="store_true", help="Render as grid")
    render_parser.add_argument("--grid-cols", type=int, default=3, help="Grid columns")
    render_parser.add_argument("--show-files", action="store_true", help="Show output files")
    render_parser.set_defaults(func=render_equations_cmd)

    # extract tables
    tables_parser = subparsers.add_parser("extract-tables", help="Extract tables from PDFs/LaTeX")
    tables_parser.add_argument("--pdf", help="Path to PDF file")
    tables_parser.add_argument("--latex", help="Path to .tex file")
    tables_parser.add_argument("--export", action="store_true", help="Export to CSV")
    tables_parser.add_argument("--output-dir", default="extracted_tables", help="Output directory")
    tables_parser.add_argument("--prefix", default="table", help="Filename prefix")
    tables_parser.add_argument("--show-details", action="store_true", help="Show table details")
    tables_parser.set_defaults(func=extract_tables_cmd)

    # show figures
    figures_parser = subparsers.add_parser("show-figures", help="Show figure info and references")
    figures_parser.add_argument("latex_dir", help="Path to LaTeX directory")
    figures_parser.add_argument(
        "--show-references", action="store_true", help="Show all references"
    )
    figures_parser.set_defaults(func=show_figures_cmd)
