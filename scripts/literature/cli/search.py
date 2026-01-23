"""
Command-line interface for literature search operations.
"""

import argparse

from ..core import ArxivClient, ConfigLoader, LiteratureDatabase


def cmd_search(args):
    """Execute search command."""
    print(f"Searching arXiv: {args.query}")
    print(f"Max results: {args.max_results}")

    client = ArxivClient()
    db = LiteratureDatabase() if not args.no_database else None

    results = client.search(args.query, max_results=args.max_results)

    print(f"\nFound {len(results)} papers:\n")
    print("=" * 80)

    for i, paper in enumerate(results, 1):
        arxiv_id = paper.get_short_id()
        print(f"\n[{i}] {paper.title}")
        print(
            f"    Authors: {', '.join(str(a) for a in paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}"
        )
        print(f"    arXiv: {arxiv_id} ({paper.published.year})")
        print(f"    URL: {paper.entry_id}")

        if db and not args.no_database:
            if db.paper_exists(arxiv_id):
                print("    [Already in database]")
            elif args.add_to_db:
                collection = args.collection if args.collection else None
                db.add_paper(paper, collection=collection)
                print("    [Added to database]")

    if db and not args.no_database and args.add_to_db:
        db.save()
        print(
            f"\nDatabase updated with {len([r for r in results if not db.paper_exists(r.get_short_id())])} new papers"
        )


def cmd_search_config(args):
    """Execute search using configuration file."""
    print(f"Loading query configuration: {args.config}")

    loader = ConfigLoader()
    config = loader.load_query(args.config)

    print(f"Query: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"Collection: {config.get('collection', 'none')}\n")

    client = ArxivClient()
    db = LiteratureDatabase()
    collection = config.get("collection")

    for query_config in config["queries"]:
        query = query_config["query"]
        max_results = query_config.get("max_results", 10)

        print(f"\nExecuting: {query_config['name']}")
        print(f"  Query: {query}")
        print(f"  Max results: {max_results}")

        results = client.search(query, max_results=max_results)

        print(f"  Found: {len(results)} papers")

        added = 0
        for paper in results:
            if db.add_paper(paper, collection=collection):
                added += 1

        print(f"  Added: {added} new papers")

    db.save()
    print("\n✓ Search completed. Database updated.")


def create_parser():
    """Create argument parser for search commands."""
    parser = argparse.ArgumentParser(description="Search arXiv and manage literature database")

    # Direct search is now the default behavior with positional query
    parser.add_argument("query", nargs="?", help="Search query (omit to use --config)")
    parser.add_argument("--max-results", type=int, default=10, help="Maximum results (default: 10)")
    parser.add_argument("--add-to-db", action="store_true", help="Add results to database")
    parser.add_argument("--collection", help="Collection to add papers to")
    parser.add_argument("--no-database", action="store_true", help="Don't check/update database")

    # Config-based search
    parser.add_argument("--config", help="Use config file (name without .yaml)")

    # List available configs
    parser.add_argument("--list-configs", action="store_true", help="List available query configs")

    return parser


def main():
    """Main entry point for search CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if args.list_configs:
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
        parser.print_help()
        print("\nError: Please provide a search query or use --config")
        print("Examples:")
        print('  python -m src.literature.cli search "modular invariance"')
        print("  python -m src.literature.cli search --config vertex_operators")
        print("  python -m src.literature.cli search --list-configs")


if __name__ == "__main__":
    main()
