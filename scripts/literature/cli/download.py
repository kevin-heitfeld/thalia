"""
CLI commands for download operations.
"""

from ..collectors.download_manager import DownloadManager
from ..core import LiteratureDatabase


def download_paper(args):
    """Download a single paper."""
    manager = DownloadManager(base_dir=args.output_dir)

    print(f"\nDownloading: {args.arxiv_id}")

    if args.pdf:
        pdf_path = manager.download_pdf(args.arxiv_id)
        if pdf_path:
            print(f"PDF saved to: {pdf_path}")

    if args.latex:
        latex_dir, tex_files = manager.download_latex(args.arxiv_id)
        if latex_dir:
            print(f"LaTeX extracted to: {latex_dir}")
            if tex_files:
                print(f"Found {len(tex_files)} .tex files:")
                for tex_file in tex_files:
                    print(f"  - {tex_file.name}")


def download_collection(args):
    """Download all papers from a collection."""
    manager = DownloadManager(base_dir=args.output_dir)
    database = LiteratureDatabase(args.database)

    stats = manager.download_from_database(
        database=database,
        collection=args.collection,
        download_pdfs=args.pdf,
        download_latex=args.latex,
        max_downloads=args.max_downloads,
    )

    return stats


def download_from_list(args):
    """Download papers from a text file list."""
    manager = DownloadManager(base_dir=args.output_dir)

    # Read arXiv IDs from file
    with open(args.file, "r") as f:
        arxiv_ids = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"\n{'=' * 80}")
    print(f"DOWNLOADING FROM FILE: {args.file}")
    print(f"{'=' * 80}")
    print(f"Papers to download: {len(arxiv_ids)}")
    print(f"{'=' * 80}\n")

    for i, arxiv_id in enumerate(arxiv_ids, 1):
        print(f"\n[{i}/{len(arxiv_ids)}] {arxiv_id}")

        if args.pdf:
            manager.download_pdf(arxiv_id)

        if args.latex:
            manager.download_latex(arxiv_id)

    stats = manager.get_statistics()

    print(f"\n{'=' * 80}")
    print("DOWNLOAD SUMMARY")
    print(f"{'=' * 80}")
    print(f"PDFs downloaded: {stats['pdfs_downloaded']}")
    print(f"PDFs failed: {stats['pdfs_failed']}")
    print(f"LaTeX downloaded: {stats['latex_downloaded']}")
    print(f"LaTeX failed: {stats['latex_failed']}")
    print(f"{'=' * 80}\n")


def setup_download_parser(subparsers):
    """
    Set up argument parser for download commands.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparser object from main argument parser
    """
    # Create download parser
    download_parser = subparsers.add_parser("download", help="Download PDFs and LaTeX sources")

    download_subparsers = download_parser.add_subparsers(
        dest="download_command", help="Download subcommands"
    )

    # download paper command
    paper_parser = download_subparsers.add_parser("paper", help="Download a single paper")
    paper_parser.add_argument("arxiv_id", help="ArXiv ID (e.g., 2511.16280)")
    paper_parser.add_argument(
        "--pdf", action="store_true", default=True, help="Download PDF (default: True)"
    )
    paper_parser.add_argument(
        "--latex", action="store_true", help="Download and extract LaTeX source"
    )
    paper_parser.add_argument(
        "--output-dir", default="docs/papers", help="Output directory (default: docs/papers)"
    )
    paper_parser.set_defaults(func=download_paper)

    # download collection command
    collection_parser = download_subparsers.add_parser(
        "collection", help="Download all papers from a collection"
    )
    collection_parser.add_argument("collection", help="Collection name (e.g., modular_invariance)")
    collection_parser.add_argument(
        "--database",
        default="docs/references/literature_database.json",
        help="Database path (default: docs/references/literature_database.json)",
    )
    collection_parser.add_argument(
        "--pdf", action="store_true", default=True, help="Download PDFs (default: True)"
    )
    collection_parser.add_argument("--latex", action="store_true", help="Download LaTeX sources")
    collection_parser.add_argument("--max-downloads", type=int, help="Maximum number to download")
    collection_parser.add_argument(
        "--output-dir", default="docs/papers", help="Output directory (default: docs/papers)"
    )
    collection_parser.set_defaults(func=download_collection)

    # download from-file command
    file_parser = download_subparsers.add_parser(
        "from-file", help="Download papers from a text file list"
    )
    file_parser.add_argument("file", help="Text file with arXiv IDs (one per line)")
    file_parser.add_argument(
        "--pdf", action="store_true", default=True, help="Download PDFs (default: True)"
    )
    file_parser.add_argument("--latex", action="store_true", help="Download LaTeX sources")
    file_parser.add_argument(
        "--output-dir", default="docs/papers", help="Output directory (default: docs/papers)"
    )
    file_parser.set_defaults(func=download_from_list)
