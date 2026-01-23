"""
CLI commands for literature collection operations.
"""

from ..collectors.search_engine import SmartSearchEngine
from ..core import ArxivClient, ConfigLoader, LiteratureDatabase


def run_collection(args):
    """Execute a collection from configuration file."""
    # Load configuration
    loader = ConfigLoader()

    # Try loading from both queries/ and collections/ directories
    try:
        config = loader.load_collection(args.config.replace(".yaml", ""))
    except FileNotFoundError:
        try:
            config = loader.load_query(args.config.replace(".yaml", ""))
        except FileNotFoundError:
            print(f"Error: Configuration file not found: {args.config}")
            print("Available configs:")
            for cfg in loader.list_queries():
                print(f"  - {cfg}")
            return

    # Extract settings
    collection_name = config.get("collection", args.config.replace(".yaml", ""))
    if isinstance(collection_name, dict):
        collection_name = collection_name.get("name", args.config.replace(".yaml", ""))

    db_path = config.get("output", {}).get("database", "docs/references/literature_database.json")

    # Initialize components
    client = ArxivClient()
    database = LiteratureDatabase(db_path)
    engine = SmartSearchEngine(client=client, database=database)

    # Prepare queries for SmartSearchEngine
    queries_list = []

    if "queries" in config:
        for query_config in config["queries"]:
            # Handle different YAML formats
            if isinstance(query_config, dict):
                if "query" in query_config:
                    # New format with weight
                    queries_list.append(
                        {
                            "query": query_config["query"],
                            "weight": query_config.get("weight", 1.0),
                            "max_results": query_config.get("max_results", 10),
                        }
                    )
                elif "name" in query_config:
                    # Old format with name field
                    queries_list.append(
                        {
                            "query": query_config.get("query", query_config["name"]),
                            "weight": query_config.get("weight", 1.0),
                            "max_results": query_config.get("max_results", 10),
                        }
                    )

    if not queries_list:
        print("Error: No queries found in configuration file")
        return

    print(f"\n{'=' * 80}")
    print(f"LITERATURE COLLECTION: {collection_name}")
    print(f"{'=' * 80}")
    print(f"Configuration: {args.config}")
    print(f"Database: {db_path}")
    print(f"Collection: {collection_name}")
    print(f"Queries: {len(queries_list)}")
    print(f"{'=' * 80}\n")

    # Execute collection
    if args.dry_run:
        print("DRY RUN MODE - No papers will be added\n")
        results = engine.search_with_scoring(queries_list)

        print(f"\nWould add {len(results)} papers:")
        for i, (arxiv_id, score, paper) in enumerate(results[:20], 1):
            print(f"{i:3d}. {arxiv_id} (score: {score:.2f}) - {paper.title[:60]}...")

        if len(results) > 20:
            print(f"     ... and {len(results) - 20} more")
    else:
        # Actually add to database
        stats = engine.search_and_add_to_database(
            queries_list, collection=collection_name, max_total=args.max_total
        )

        print(f"\n{'=' * 80}")
        print("COLLECTION COMPLETE")
        print(f"{'=' * 80}")
        print(f"  Total found: {stats['total_found']}")
        print(f"  Added: {stats['added']}")
        print(f"  Skipped: {stats['skipped']}")
        print(f"  Collection: {stats['collection']}")
        print(f"{'=' * 80}\n")


def list_collections(_args):
    """List available collection configurations."""
    loader = ConfigLoader()
    configs = loader.list_queries()

    print(f"\n{'=' * 80}")
    print("AVAILABLE COLLECTION CONFIGURATIONS")
    print(f"{'=' * 80}\n")

    for config_name in configs:
        try:
            config = loader.load_query(config_name)

            # Extract description
            description = "No description"
            if "description" in config:
                description = config["description"]
            elif "collection" in config and isinstance(config["collection"], dict):
                description = config["collection"].get("description", "No description")

            # Count queries
            num_queries = len(config.get("queries", []))

            print(f"  {config_name}")
            print(f"    Description: {description}")
            print(f"    Queries: {num_queries}")
            print()
        except Exception as e:
            print(f"  {config_name}")
            print(f"    Error loading: {e}")
            print()


def setup_collection_parser(subparsers):
    """
    Set up argument parser for collection commands.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparser object from main argument parser
    """
    # Create collection parser
    collection_parser = subparsers.add_parser(
        "collect", help="Execute literature collections from configuration files"
    )

    collection_subparsers = collection_parser.add_subparsers(
        dest="collection_command", help="Collection subcommands"
    )

    # collect run command
    run_parser = collection_subparsers.add_parser(
        "run", help="Run a collection from configuration file"
    )
    run_parser.add_argument(
        "config", help="Configuration file name (e.g., 'modular_invariance.yaml')"
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be collected without adding to database",
    )
    run_parser.add_argument(
        "--max-total", type=int, help="Maximum total papers to add (default: no limit)"
    )
    run_parser.set_defaults(func=run_collection)

    # collect list command
    list_parser = collection_subparsers.add_parser(
        "list", help="List available collection configurations"
    )
    list_parser.set_defaults(func=list_collections)
