"""
Main entry point for literature management CLI.

Usage:
    python -m src.literature.cli search "quantum field theory"
    python -m src.literature.cli search --config vertex_operators
    python -m src.literature.cli search --list-configs
    python -m src.literature.cli db list
    python -m src.literature.cli db stats
    python -m src.literature.cli db find "komatsu"
    python -m src.literature.cli collect run modular_invariance.yaml
    python -m src.literature.cli download paper 2511.16280
"""

import argparse
import sys


def main():
    """Main entry point routing to subcommands."""
    parser = argparse.ArgumentParser(description="Literature Management System", prog="literature")

    subparsers = parser.add_subparsers(dest="module", help="Module to use")

    # Search module with full argument support
    search_parser = subparsers.add_parser("search", help="Search arXiv")
    search_parser.add_argument("query", nargs="?", help="Search query (omit to use --config)")
    search_parser.add_argument(
        "--max-results", type=int, default=10, help="Maximum results (default: 10)"
    )
    search_parser.add_argument("--add-to-db", action="store_true", help="Add results to database")
    search_parser.add_argument("--collection", help="Collection to add papers to")
    search_parser.add_argument(
        "--no-database", action="store_true", help="Don't check/update database"
    )
    search_parser.add_argument("--config", help="Use config file (name without .yaml)")
    search_parser.add_argument(
        "--list-configs", action="store_true", help="List available query configs"
    )

    # Database module
    subparsers.add_parser("db", help="Manage database")

    # Collection module
    subparsers.add_parser("collect", help="Run literature collections")

    # Download module
    subparsers.add_parser("download", help="Download PDFs and LaTeX sources")

    # Extract module
    subparsers.add_parser("extract", help="Extract content from papers")

    # Search content module
    subparsers.add_parser("search-content", help="Search indexed content")

    # Search equations module
    subparsers.add_parser("search-equations", help="Search equations")

    # Index stats module
    subparsers.add_parser("index-stats", help="Show index statistics")

    # Advanced extraction modules
    subparsers.add_parser("render-equations", help="Render equations to images")
    subparsers.add_parser("extract-tables", help="Extract tables from PDFs/LaTeX")
    subparsers.add_parser("show-figures", help="Show figure info and references")

    # Parse args - use parse_args for search, parse_known_args for others
    remaining = []  # Initialize for search case
    if len(sys.argv) > 1 and sys.argv[1] == "search":
        args = parser.parse_args()
    else:
        args, remaining = parser.parse_known_args()

    if args.module == "search":
        from .search import cmd_search, cmd_search_config

        if args.list_configs:
            from ..core import ConfigLoader

            loader = ConfigLoader()
            configs = loader.list_queries()
            print("\nAvailable query configurations:")
            for config in configs:
                print(f"  - {config}")
            print(f"\nTotal: {len(configs)} configurations")
        elif args.config:
            cmd_search_config(args)
        elif args.query:
            cmd_search(args)
        else:
            search_parser.print_help()
            print("\nError: Please provide a search query or use --config")
    elif args.module == "db":
        from .database import main as db_main

        sys.argv = ["literature-db"] + remaining
        db_main()
    elif args.module == "collect":
        from .collect import list_collections, run_collection

        # Parse collect subcommand
        collect_parser = argparse.ArgumentParser(prog="literature collect")
        collect_subparsers = collect_parser.add_subparsers(dest="collect_command")

        # run subcommand
        run_parser = collect_subparsers.add_parser("run", help="Run a collection")
        run_parser.add_argument("config", help="Configuration file")
        run_parser.add_argument("--dry-run", action="store_true")
        run_parser.add_argument("--max-total", type=int)

        # list subcommand
        collect_subparsers.add_parser("list", help="List collections")

        collect_args = collect_parser.parse_args(remaining)

        if collect_args.collect_command == "run":
            run_collection(collect_args)
        elif collect_args.collect_command == "list":
            list_collections(collect_args)
        else:
            collect_parser.print_help()
    elif args.module == "download":
        from .download import download_collection, download_from_list, download_paper

        # Parse download subcommand
        download_parser = argparse.ArgumentParser(prog="literature download")
        download_subparsers = download_parser.add_subparsers(dest="download_command")

        # paper subcommand
        paper_parser = download_subparsers.add_parser("paper", help="Download a single paper")
        paper_parser.add_argument("arxiv_id", help="ArXiv ID")
        paper_parser.add_argument("--pdf", action="store_true", default=True)
        paper_parser.add_argument("--latex", action="store_true")
        paper_parser.add_argument("--output-dir", default="docs/papers")

        # collection subcommand
        coll_parser = download_subparsers.add_parser("collection", help="Download collection")
        coll_parser.add_argument("collection", help="Collection name")
        coll_parser.add_argument("--database", default="docs/references/literature_database.json")
        coll_parser.add_argument("--pdf", action="store_true", default=True)
        coll_parser.add_argument("--latex", action="store_true")
        coll_parser.add_argument("--max-downloads", type=int)
        coll_parser.add_argument("--output-dir", default="docs/papers")

        # from-file subcommand
        file_parser = download_subparsers.add_parser("from-file", help="Download from file list")
        file_parser.add_argument("file", help="Text file with arXiv IDs")
        file_parser.add_argument("--pdf", action="store_true", default=True)
        file_parser.add_argument("--latex", action="store_true")
        file_parser.add_argument("--output-dir", default="docs/papers")

        download_args = download_parser.parse_args(remaining)

        if download_args.download_command == "paper":
            download_paper(download_args)
        elif download_args.download_command == "collection":
            download_collection(download_args)
        elif download_args.download_command == "from-file":
            download_from_list(download_args)
        else:
            download_parser.print_help()
    elif args.module == "extract":
        from .extract import (
            extract_collection,
            extract_paper,
            index_stats,
            search_equations_cmd,
            search_papers,
        )

        extract_parser = argparse.ArgumentParser(prog="literature extract")
        extract_subparsers = extract_parser.add_subparsers(dest="extract_command")

        # paper
        paper_p = extract_subparsers.add_parser("paper")
        paper_p.add_argument("arxiv_id")
        paper_p.add_argument("--pdf")
        paper_p.add_argument("--latex")
        paper_p.add_argument("--show-equations", action="store_true")
        paper_p.add_argument("--index", action="store_true")
        paper_p.add_argument("--index-file", default="docs/references/search_index.json")

        # collection
        coll_p = extract_subparsers.add_parser("collection")
        coll_p.add_argument("collection")
        coll_p.add_argument("--database", default="docs/references/literature_database.json")
        from pathlib import Path

        coll_p.add_argument("--pdf-dir", type=Path, default=Path("docs/papers/pdfs"))
        coll_p.add_argument("--latex-dir", type=Path, default=Path("docs/papers/latex"))
        coll_p.add_argument("--index", action="store_true", default=True)
        coll_p.add_argument("--index-file", default="docs/references/search_index.json")

        # batch
        batch_p = extract_subparsers.add_parser("batch")
        batch_p.add_argument("--collection")
        batch_p.add_argument("--priority", action="store_true")
        batch_p.add_argument("--limit", type=int)
        batch_p.add_argument("--skip-existing", action="store_true")
        batch_p.add_argument("--database", default="docs/references/literature_database.json")
        batch_p.add_argument("--pdf-dir", default="docs/papers/pdfs")
        batch_p.add_argument("--latex-dir", default="docs/papers/latex")
        batch_p.add_argument("--index-file", default="docs/references/search_index.json")

        extract_args = extract_parser.parse_args(remaining)

        if extract_args.extract_command == "paper":
            extract_paper(extract_args)
        elif extract_args.extract_command == "collection":
            extract_collection(extract_args)
        elif extract_args.extract_command == "batch":
            from .extract import extract_batch

            extract_batch(extract_args)
        else:
            extract_parser.print_help()
    elif args.module == "search-content":
        from .extract import search_papers

        parser2 = argparse.ArgumentParser(prog="literature search-content")
        parser2.add_argument("query")
        parser2.add_argument("--max-results", type=int, default=20)
        parser2.add_argument("--no-equations", action="store_true")
        parser2.add_argument("--show-preview", action="store_true")
        parser2.add_argument("--show-abstract", action="store_true")
        parser2.add_argument("--index-file", default="docs/references/search_index.json")
        args2 = parser2.parse_args(remaining)
        search_papers(args2)
    elif args.module == "search-equations":
        from .extract import search_equations_cmd

        parser2 = argparse.ArgumentParser(prog="literature search-equations")
        parser2.add_argument("pattern")
        parser2.add_argument("--max-results", type=int, default=20)
        parser2.add_argument("--index-file", default="docs/references/search_index.json")
        args2 = parser2.parse_args(remaining)
        search_equations_cmd(args2)
    elif args.module == "index-stats":
        from .extract import index_stats

        parser2 = argparse.ArgumentParser(prog="literature index-stats")
        parser2.add_argument("--index-file", default="docs/references/search_index.json")
        args2 = parser2.parse_args(remaining)
        index_stats(args2)
    elif args.module == "render-equations":
        from .extract import render_equations_cmd

        parser2 = argparse.ArgumentParser(prog="literature render-equations")
        parser2.add_argument("--latex", help="Path to .tex file")
        parser2.add_argument("--latex-dir", help="Path to directory with .tex files")
        parser2.add_argument("--output-dir", default="docs/papers/rendered_equations")
        parser2.add_argument("--format", choices=["png", "svg"], default="png")
        parser2.add_argument("--dpi", type=int, default=300)
        parser2.add_argument("--fontsize", type=int, default=14)
        parser2.add_argument("--max-equations", type=int)
        parser2.add_argument("--prefix", default="eq")
        parser2.add_argument("--grid", action="store_true")
        parser2.add_argument("--grid-cols", type=int, default=3)
        parser2.add_argument("--show-files", action="store_true")
        args2 = parser2.parse_args(remaining)
        render_equations_cmd(args2)
    elif args.module == "extract-tables":
        from .extract import extract_tables_cmd

        parser2 = argparse.ArgumentParser(prog="literature extract-tables")
        parser2.add_argument("--pdf", help="Path to PDF file")
        parser2.add_argument("--latex", help="Path to .tex file")
        parser2.add_argument("--export", action="store_true")
        parser2.add_argument("--output-dir", default="extracted_tables")
        parser2.add_argument("--prefix", default="table")
        parser2.add_argument("--show-details", action="store_true")
        args2 = parser2.parse_args(remaining)
        extract_tables_cmd(args2)
    elif args.module == "show-figures":
        from .extract import show_figures_cmd

        parser2 = argparse.ArgumentParser(prog="literature show-figures")
        parser2.add_argument("latex_dir", help="Path to LaTeX directory")
        parser2.add_argument("--show-references", action="store_true")
        args2 = parser2.parse_args(remaining)
        show_figures_cmd(args2)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
