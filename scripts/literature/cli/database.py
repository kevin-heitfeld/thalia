"""
Command-line interface for literature database management.
"""

import argparse
import time

from ..core import LiteratureDatabase
from ..core.arxiv_api import get_paper_by_id


def cmd_list(args):
    """List papers in database."""
    db = LiteratureDatabase()

    # Get papers from collection or all
    if args.collection:
        papers = db.get_collection(args.collection)
        if not papers:
            papers = []
    else:
        # Get all papers from all collections
        all_papers = []
        for collection in db.get_collections():
            collection_papers = db.get_collection(collection)
            if collection_papers:
                all_papers.extend(collection_papers)

        # Deduplicate by arxiv ID
        seen = set()
        papers = []
        for paper in all_papers:
            arxiv_id = paper.get("arxiv") or paper.get("arxiv_id")
            if arxiv_id and arxiv_id not in seen:
                seen.add(arxiv_id)
                papers.append(paper)

    # Apply status filter
    if args.status:
        papers = [p for p in papers if p.get("status") == args.status]

    # Apply year filter
    if hasattr(args, "year") and args.year:
        papers = [p for p in papers if p.get("year") == args.year]

    # Apply tag filter
    if hasattr(args, "tag") and args.tag:
        papers = [p for p in papers if args.tag in p.get("tags", [])]

    # Apply sorting
    if hasattr(args, "sort_by") and args.sort_by:
        if args.sort_by == "year":
            papers.sort(key=lambda p: p.get("year", 0), reverse=True)
        elif args.sort_by == "author":
            papers.sort(
                key=lambda p: (
                    p.get("authors", ["Unknown"])[0]
                    if isinstance(p.get("authors"), list)
                    else "Unknown"
                ).lower()
            )
        elif args.sort_by == "status":
            papers.sort(key=lambda p: p.get("status", ""))
        elif args.sort_by == "arxiv":
            papers.sort(key=lambda p: p.get("arxiv") or p.get("arxiv_id", ""))

    # Display header with count
    count = len(papers)
    paper_word = "paper" if count == 1 else "papers"

    filters_used = []
    if args.collection:
        filters_used.append(f"collection '{args.collection}'")
    if args.status:
        filters_used.append(f"status '{args.status}'")
    if hasattr(args, "year") and args.year:
        filters_used.append(f"year {args.year}")
    if hasattr(args, "tag") and args.tag:
        filters_used.append(f"tag '{args.tag}'")

    if filters_used:
        print(f"\nFound {count} {paper_word} with {', '.join(filters_used)}")
    else:
        print(f"\nTotal papers in database: {count}")
        if count > 50:
            print("(Tip: Use --collection, --status, --year, or --tag to filter results)")

    print("=" * 80)

    # Display papers
    if not papers:
        print("\nNo papers found matching criteria")
        print()
        return

    print()
    if args.collection:
        print(f"Papers in '{args.collection}':")
    elif filters_used:
        print("Papers matching filters:")
    else:
        print("All papers in database:")
    print("-" * 80)

    for paper in papers:
        arxiv_id = paper.get("arxiv") or paper.get("arxiv_id", "N/A")
        title = paper.get("title", "[No title]")

        # Handle authors (could be list or string)
        authors = paper.get("authors", [])
        if isinstance(authors, list):
            author_str = authors[0] if authors else "Unknown"
        else:
            author_str = str(authors) if authors else "Unknown"

        # Truncate long names
        if len(author_str) > 20:
            author_str = author_str[:17] + "..."

        year = paper.get("year", "N/A")
        if year == 0 or year is None:
            year = "N/A"

        # Ensure year is string for formatting
        year_str = str(year) if year != "N/A" else "N/A"

        status = paper.get("status", "N/A")

        # Truncate title
        if len(title) > 60:
            title = title[:57] + "..."

        print(f"{arxiv_id:15} | {author_str:20} | {year_str:4} | {status:15} | {title}")

    print()
    print(f"Total: {count} {paper_word}")
    print()


def cmd_add(args):
    """Add paper to database."""
    db = LiteratureDatabase()

    if db.add_paper_by_id(args.arxiv_id, collection=args.collection):
        db.save()
        print("✓ Paper added successfully")
    else:
        print("Paper already in database")


def cmd_update(args):
    """Update paper status."""
    db = LiteratureDatabase()

    if db.update_status(args.arxiv_id, args.status, notes=args.notes):
        db.save()
        print("✓ Paper updated successfully")


def cmd_update_metadata(args):
    """Update paper metadata."""
    db = LiteratureDatabase()

    # Parse authors if provided
    authors = None
    if args.authors:
        authors = [a.strip() for a in args.authors.split(",")]

    if db.update_metadata(
        args.arxiv_id,
        title=args.title,
        authors=authors,
        year=args.year,
        abstract=args.abstract,
    ):
        db.save()
        print("✓ Metadata updated successfully")


def cmd_add_tags(args):
    """Add tags to a paper."""
    db = LiteratureDatabase()

    tags = [t.strip() for t in args.tags.split(",")]
    if db.add_tags(args.arxiv_id, tags):
        db.save()
        print("✓ Tags added successfully")


def cmd_remove_tags(args):
    """Remove tags from a paper."""
    db = LiteratureDatabase()

    tags = [t.strip() for t in args.tags.split(",")]
    if db.remove_tags(args.arxiv_id, tags):
        db.save()
        print("✓ Tags removed successfully")


def cmd_info(args):
    """Show detailed information about a paper."""
    db = LiteratureDatabase()

    paper = db.get_paper_info(args.arxiv_id)
    if not paper:
        print(f"Paper {args.arxiv_id} not found")
        return

    print("\nPaper Information")
    print("=" * 80)
    print(f"arXiv ID:    {paper.get('arxiv', 'N/A')}")
    print(f"Title:       {paper.get('title', 'N/A')}")
    print(f"Authors:     {', '.join(paper.get('authors', []))}")
    print(f"Year:        {paper.get('year', 'N/A')}")
    print(f"Status:      {paper.get('status', 'N/A')}")
    print(f"Updated:     {paper.get('last_updated', 'N/A')}")

    if paper.get("collections"):
        print(f"Collections: {', '.join(paper.get('collections', []))}")

    if paper.get("tags"):
        print(f"Tags:        {', '.join(paper.get('tags', []))}")

    if paper.get("notes"):
        print(f"\nNotes:\n{paper.get('notes')}")

    if paper.get("abstract"):
        abstract = paper.get("abstract", "")
        if len(abstract) > 300:
            abstract = abstract[:297] + "..."
        print(f"\nAbstract:\n{abstract}")

    print()


def cmd_describe_collection(args):
    """Set or show collection description."""
    db = LiteratureDatabase()

    if args.description:
        # Set description
        db.set_collection_description(args.collection, args.description)
        db.save()
        print(f"✓ Description set for collection '{args.collection}'")
    else:
        # Show description
        description = db.get_collection_description(args.collection)
        if description:
            print(f"\nCollection: {args.collection}")
            print(f"Description: {description}")
        else:
            print(f"\nNo description set for collection '{args.collection}'")
            print("Use --description to set one.")


def cmd_stats(_args):
    """Show database statistics."""
    db = LiteratureDatabase()
    stats = db.get_statistics()

    print("\nDatabase Statistics")
    print("=" * 60)
    print(f"Total papers: {stats['total_papers']}")

    print("\nBy Status:")
    for status, count in sorted(stats["by_status"].items()):
        print(f"  {status:20s}: {count:4d}")

    print("\nBy Collection:")
    for collection, count in sorted(stats["by_collection"].items()):
        print(f"  {collection:30s}: {count:4d}")

    print("\nBy Year (recent):")
    years = sorted(stats["by_year"].items(), reverse=True)[:10]
    for year, count in years:
        print(f"  {year}: {count:4d}")


def cmd_export(args):
    """Export database to BibTeX."""
    db = LiteratureDatabase()

    # Validate collection exists if specified
    if args.collection:
        collections = db.get_collections()
        if args.collection not in collections:
            print("✗ Error: Collection '{}' not found".format(args.collection))
            print("\\nAvailable collections:")
            for coll in sorted(collections)[:10]:
                print(f"  - {coll}")
            if len(collections) > 10:
                print(f"  ... and {len(collections) - 10} more")
            print("\\nUse 'db collections' to see all collections")
            return

    count = db.export_bibtex(args.output, collection=args.collection, status=args.status)
    paper_word = "paper" if count == 1 else "papers"
    print(f"\u2713 Exported {count} {paper_word}")


def cmd_find(args):
    """Find papers by fuzzy search."""
    db = LiteratureDatabase()

    query = args.query.lower()
    results = []

    # Get all papers from all collections (deduplicate)
    seen = set()
    for collection in db.get_collections():
        for paper in db.get_collection(collection):
            arxiv_id = paper.get("arxiv")
            if arxiv_id and arxiv_id not in seen:
                seen.add(arxiv_id)
                # Search in title, authors, arxiv ID, and tags
                title = paper.get("title", "").lower()
                authors = " ".join(paper.get("authors", [])).lower()
                tags = " ".join(paper.get("tags", [])).lower()
                abstract = paper.get("abstract", "").lower()

                if (
                    query in title
                    or query in authors
                    or query in arxiv_id.lower()
                    or query in tags
                    or query in abstract
                ):
                    results.append(paper)

    if not results:
        print(f"No papers found matching '{args.query}'")
        return

    count = len(results)
    paper_word = "paper" if count == 1 else "papers"
    print(f"\nFound {count} {paper_word} matching '{args.query}':")
    print("=" * 80)

    for paper in results[: args.max_results]:
        arxiv_id = paper.get("arxiv", "N/A")
        title = paper.get("title", "Untitled")[:60]
        authors = ", ".join(paper.get("authors", [])[:2])
        if len(paper.get("authors", [])) > 2:
            authors += " et al."
        year = paper.get("year", "N/A")
        collections = ", ".join(paper.get("collections", [])[:3])

        print(f"\\n{arxiv_id:15s} | {authors[:30]:30s} | {year}")
        print(f"  Title: {title}")
        if collections:
            print(f"  Collections: {collections}")

    if len(results) > args.max_results:
        print("\\n... and {} more results".format(len(results) - args.max_results))
        print("Use --max-results to see more")


def cmd_needs_metadata(args):
    """Find papers that need metadata (missing title/author/year)."""
    db = LiteratureDatabase()

    # Get all papers
    all_papers = []
    for collection in db.get_collections():
        papers = db.get_collection(collection)
        if papers:
            all_papers.extend(papers)

    # Deduplicate by arxiv
    seen = set()
    unique_papers = []
    for paper in all_papers:
        arxiv_id = paper.get("arxiv") or paper.get("arxiv_id")  # Try both keys
        if arxiv_id and arxiv_id not in seen:
            seen.add(arxiv_id)
            unique_papers.append(paper)

    # Find papers with missing metadata
    needs_metadata = []
    for paper in unique_papers:
        title = paper.get("title", "")
        authors = paper.get("authors", [])
        if isinstance(authors, list):
            authors_str = authors[0] if authors else ""
        else:
            authors_str = str(authors)
        year = paper.get("year", 0)

        # Check if metadata is missing or placeholder
        if (
            not title
            or title.startswith("[To fetch")
            or title.startswith("[To be fetched")
            or not authors_str
            or authors_str == "Unknown"
            or authors_str == "[Unknown]"
            or not year
            or year == 0
        ):
            needs_metadata.append(paper)

    count = len(needs_metadata)
    paper_word = "paper" if count == 1 else "papers"

    if count == 0:
        print("\n✓ All papers have complete metadata!")
        return

    print(f"\nFound {count} {paper_word} needing metadata:")
    print("=" * 80)
    print()

    # Show limited results
    shown = min(count, args.limit)
    for i, paper in enumerate(needs_metadata[:shown], 1):
        arxiv_id = paper.get("arxiv") or paper.get("arxiv_id", "Unknown")
        title = paper.get("title", "No title")
        collections = paper.get("collections", [])

        print(f"{i:3d}. {arxiv_id}")
        print(f"     Title: {title[:70]}")
        if collections:
            print(f"     Collections: {', '.join(collections[:3])}")
        print()

    if count > shown:
        remaining = count - shown
        print(f"... and {remaining} more")
        print(f"Use --limit {count} to see all")

    print("\nTip: Use 'db info <arxiv_id>' to fetch metadata for individual papers")


def cmd_fetch_metadata(args):
    """Batch fetch metadata for papers needing it."""
    db = LiteratureDatabase()

    # Get papers needing metadata
    all_papers = []
    if args.collection:
        papers = db.get_collection(args.collection)
        if papers:
            all_papers = papers
    else:
        for collection in db.get_collections():
            collection_papers = db.get_collection(collection)
            if collection_papers:
                all_papers.extend(collection_papers)

    # Deduplicate
    seen = set()
    unique_papers = []
    for paper in all_papers:
        arxiv_id = paper.get("arxiv") or paper.get("arxiv_id")
        if arxiv_id and arxiv_id not in seen:
            seen.add(arxiv_id)
            unique_papers.append(paper)

    # Filter papers needing metadata
    needs_metadata = []
    for paper in unique_papers:
        title = paper.get("title", "")
        authors = paper.get("authors", [])
        if isinstance(authors, list):
            authors_str = authors[0] if authors else ""
        else:
            authors_str = str(authors)
        year = paper.get("year", 0)

        if (
            not title
            or title.startswith("[To fetch")
            or title.startswith("[To be fetched")
            or not authors_str
            or authors_str == "Unknown"
            or authors_str == "[Unknown]"
            or not year
            or year == 0
        ):
            needs_metadata.append(paper)

    if not needs_metadata:
        print("\n✓ No papers need metadata fetching!")
        return

    # Apply limit
    total_to_fetch = len(needs_metadata)
    if args.limit:
        needs_metadata = needs_metadata[: args.limit]

    print(
        f"\nFetching metadata for {len(needs_metadata)} papers (out of {total_to_fetch} total)..."
    )
    print("=" * 80)

    success_count = 0
    fail_count = 0

    for i, paper in enumerate(needs_metadata, 1):
        arxiv_id = paper.get("arxiv") or paper.get("arxiv_id")
        print(f"\n[{i}/{len(needs_metadata)}] Fetching {arxiv_id}...", end=" ")

        try:
            result = get_paper_by_id(arxiv_id)
            if result:
                # Update metadata
                authors = [author.name for author in result.authors]
                year = result.published.year
                title = result.title.replace("\n", " ").strip()
                abstract = result.summary.replace("\n", " ").strip()

                db.update_metadata(
                    arxiv_id, title=title, authors=authors, year=year, abstract=abstract
                )
                print(f"✓ {title[:50]}...")
                success_count += 1
            else:
                print("✗ Not found")
                fail_count += 1
        except Exception as e:
            print(f"✗ Error: {e}")
            fail_count += 1

        # Rate limiting (3 seconds between requests as per arXiv guidelines)
        if i < len(needs_metadata):
            time.sleep(3)

    # Save once at the end
    db.save()

    print("\n" + "=" * 80)
    print("\nMetadata fetching complete:")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  Total:   {len(needs_metadata)}")
    print()


def cmd_collections(_args):
    """List all collections."""
    db = LiteratureDatabase()
    collections = db.get_collections()

    print("\nCollections in database:")
    print("=" * 80)
    for collection in sorted(collections):
        papers = db.get_collection(collection)
        count = len(papers)
        paper_word = "paper" if count == 1 else "papers"

        # Get description if available
        description = db.get_collection_description(collection)
        if description:
            print(f"  {collection:30s}: {count:4d} {paper_word}")
            print(f"    → {description}")
        else:
            print(f"  {collection:30s}: {count:4d} {paper_word}")

    print(f"\nTotal: {len(collections)} collections")
    print("Tip: Use 'describe-collection' to add descriptions")


def create_parser():
    """Create argument parser for database commands."""
    parser = argparse.ArgumentParser(description="Manage literature database")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List papers
    list_parser = subparsers.add_parser("list", help="List papers")
    list_parser.add_argument("--collection", help="Filter by collection")
    list_parser.add_argument("--status", help="Filter by status")
    list_parser.add_argument("--year", type=int, help="Filter by publication year")
    list_parser.add_argument("--tag", help="Filter by tag")
    list_parser.add_argument(
        "--sort-by", choices=["year", "author", "status", "arxiv"], help="Sort results"
    )

    # Add paper
    add_parser = subparsers.add_parser("add", help="Add paper by arXiv ID")
    add_parser.add_argument("arxiv_id", help="arXiv ID")
    add_parser.add_argument("--collection", help="Add to collection")

    # Update paper
    update_parser = subparsers.add_parser("update", help="Update paper status")
    update_parser.add_argument("arxiv_id", help="arXiv ID")
    update_parser.add_argument("status", help="New status")
    update_parser.add_argument("--notes", help="Additional notes")

    # Update metadata
    update_meta_parser = subparsers.add_parser("update-metadata", help="Update paper metadata")
    update_meta_parser.add_argument("arxiv_id", help="arXiv ID")
    update_meta_parser.add_argument("--title", help="New title")
    update_meta_parser.add_argument("--authors", help="New authors (comma-separated)")
    update_meta_parser.add_argument("--year", type=int, help="New publication year")
    update_meta_parser.add_argument("--abstract", help="New abstract")

    # Add tags
    add_tags_parser = subparsers.add_parser("add-tags", help="Add tags to a paper")
    add_tags_parser.add_argument("arxiv_id", help="arXiv ID")
    add_tags_parser.add_argument("tags", help="Tags to add (comma-separated)")

    # Remove tags
    remove_tags_parser = subparsers.add_parser("remove-tags", help="Remove tags from a paper")
    remove_tags_parser.add_argument("arxiv_id", help="arXiv ID")
    remove_tags_parser.add_argument("tags", help="Tags to remove (comma-separated)")

    # Info
    info_parser = subparsers.add_parser("info", help="Show detailed paper information")
    info_parser.add_argument("arxiv_id", help="arXiv ID")

    # Statistics
    subparsers.add_parser("stats", help="Show database statistics")

    # Export
    export_parser = subparsers.add_parser("export", help="Export to BibTeX")
    export_parser.add_argument("output", help="Output .bib file")
    export_parser.add_argument("--collection", help="Export only this collection")
    export_parser.add_argument("--status", help="Export only papers with this status")

    # Find papers
    find_parser = subparsers.add_parser("find", help="Find papers by keyword")
    find_parser.add_argument("query", help="Search query (title, author, tags, arXiv ID)")
    find_parser.add_argument(
        "--max-results", type=int, default=10, help="Maximum results (default: 10)"
    )

    # Needs metadata
    needs_parser = subparsers.add_parser("needs-metadata", help="Find papers needing metadata")
    needs_parser.add_argument(
        "--limit", type=int, default=20, help="Maximum papers to show (default: 20)"
    )

    # Fetch metadata (batch)
    fetch_parser = subparsers.add_parser("fetch-metadata", help="Batch fetch metadata from arXiv")
    fetch_parser.add_argument("--collection", help="Fetch for specific collection only")
    fetch_parser.add_argument("--limit", type=int, help="Maximum papers to fetch (default: all)")

    # Describe collection
    desc_parser = subparsers.add_parser(
        "describe-collection", help="Set/show collection description"
    )
    desc_parser.add_argument("collection", help="Collection name")
    desc_parser.add_argument("--description", help="Description text to set")

    # Collections
    subparsers.add_parser("collections", help="List all collections")
    return parser


def main():
    """Main entry point for database CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "add":
        cmd_add(args)
    elif args.command == "update":
        cmd_update(args)
    elif args.command == "update-metadata":
        cmd_update_metadata(args)
    elif args.command == "add-tags":
        cmd_add_tags(args)
    elif args.command == "remove-tags":
        cmd_remove_tags(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "find":
        cmd_find(args)
    elif args.command == "needs-metadata":
        cmd_needs_metadata(args)
    elif args.command == "fetch-metadata":
        cmd_fetch_metadata(args)
    elif args.command == "describe-collection":
        cmd_describe_collection(args)
    elif args.command == "collections":
        cmd_collections(args)
    elif args.command == "collections":
        cmd_collections(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
