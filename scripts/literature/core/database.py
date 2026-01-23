"""
Enhanced Literature Database Management

This module provides a class-based interface for managing a literature database
with support for collections, relationships, and advanced querying.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

# Will import from sibling module
try:
    from .arxiv_api import generate_bibtex, get_paper_by_id
except ImportError:
    # Fallback for backward compatibility
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
    from arxiv_tools import generate_bibtex, get_paper_by_id


class LiteratureDatabase:
    """
    Enhanced literature database with collections and relationships.

    Examples
    --------
    >>> db = LiteratureDatabase()
    >>> db.add_paper_by_id("2301.12345", collection="vertex_operators")
    >>> papers = db.get_collection("vertex_operators")
    >>> db.export_bibtex("vertex_operators", "references.bib")
    """

    def __init__(self, database_path: Optional[str] = None):
        """
        Initialize database.

        Parameters
        ----------
        database_path : str, optional
            Path to database JSON file (default: docs/references/literature_database.json)
        """
        if database_path is None:
            # Default to project root / docs/references/literature_database.json
            self.db_path = (
                Path(__file__).parent.parent.parent.parent
                / "docs"
                / "references"
                / "literature_database.json"
            )
        else:
            self.db_path = Path(database_path)

        self.db = self.load()

    def load(self) -> Dict:
        """Load database from file."""
        if self.db_path.exists() and self.db_path.stat().st_size > 0:
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            # Initialize new database with v2.0 schema
            return {
                "metadata": {
                    "version": "2.0",
                    "last_updated": datetime.now().isoformat(),
                    "total_papers": 0,
                },
                "papers": [],
                "collections": {},
                "query_history": [],
            }

    def save(self) -> None:
        """Save database to file."""
        self.db["metadata"]["last_updated"] = datetime.now().isoformat()
        self.db["metadata"]["total_papers"] = len(self.db["papers"])

        # Create directory if needed
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.db, f, indent=2, ensure_ascii=False)

        print(f"Database saved to {self.db_path}")

    def paper_exists(self, arxiv_id: str) -> bool:
        """Check if paper already in database."""
        for paper in self.db["papers"]:
            if paper.get("arxiv") == arxiv_id:
                return True
        return False

    def get_paper(self, arxiv_id: str) -> Optional[Dict]:
        """Get paper entry by arXiv ID."""
        for paper in self.db["papers"]:
            if paper.get("arxiv") == arxiv_id:
                return paper
        return None

    def add_paper(self, paper_data: Dict, collection: Optional[str] = None) -> bool:
        """
        Add paper to database with automatic deduplication.

        Parameters
        ----------
        paper_data : dict
            Paper metadata (from arxiv.Result or dict)
        collection : str, optional
            Collection to add paper to

        Returns
        -------
        bool
            True if added, False if already exists
        """
        # Extract arxiv_id
        if hasattr(paper_data, "entry_id"):
            # arxiv.Result object
            arxiv_id = paper_data.entry_id.split("/")[-1].split("v")[0]
        else:
            # Dict
            arxiv_id = paper_data.get("arxiv", paper_data.get("arxiv_id"))

        if self.paper_exists(arxiv_id):
            print(f"  Paper {arxiv_id} already in database")
            if collection:
                self.add_to_collection(arxiv_id, collection)
            return False

        # Create entry
        if hasattr(paper_data, "entry_id"):
            # Create from arxiv.Result
            entry = self._create_entry_from_result(paper_data)
        else:
            # Already a dict
            entry = paper_data
            # Ensure required fields
            if "added_date" not in entry:
                entry["added_date"] = datetime.now().strftime("%Y-%m-%d")
            if "status" not in entry:
                entry["status"] = "to_review"

        # Add collection
        if "collections" not in entry:
            entry["collections"] = []
        if collection and collection not in entry["collections"]:
            entry["collections"].append(collection)
            self._ensure_collection_exists(collection)

        self.db["papers"].append(entry)
        print(f"  Added: {arxiv_id} - {entry.get('title', 'No title')[:60]}...")
        return True

    def add_paper_by_id(self, arxiv_id: str, collection: Optional[str] = None) -> bool:
        """
        Fetch and add paper by arXiv ID.

        Parameters
        ----------
        arxiv_id : str
            arXiv identifier
        collection : str, optional
            Collection to add to

        Returns
        -------
        bool
            True if successfully added
        """
        if self.paper_exists(arxiv_id):
            print(f"Paper {arxiv_id} already in database")
            if collection:
                self.add_to_collection(arxiv_id, collection)
            return False

        try:
            result = get_paper_by_id(arxiv_id)
            return self.add_paper(result, collection)
        except Exception as e:
            print(f"Error fetching paper {arxiv_id}: {e}")
            return False

    def add_to_collection(self, arxiv_id: str, collection: str) -> bool:
        """Add existing paper to a collection."""
        paper = self.get_paper(arxiv_id)
        if not paper:
            print(f"Paper {arxiv_id} not in database")
            return False

        if "collections" not in paper:
            paper["collections"] = []

        if collection not in paper["collections"]:
            paper["collections"].append(collection)
            self._ensure_collection_exists(collection)
            print(f"  Added {arxiv_id} to collection '{collection}'")
            return True
        else:
            print(f"  Paper {arxiv_id} already in collection '{collection}'")
            return False

    def get_collection(self, collection: str) -> List[Dict]:
        """Get all papers in a collection."""
        return [p for p in self.db["papers"] if collection in p.get("collections", [])]

    def get_collections(self) -> List[str]:
        """Get list of all collections."""
        collections: Set[str] = set()
        for paper in self.db["papers"]:
            collections.update(paper.get("collections", []))
        return sorted(collections)

    def update_status(self, arxiv_id: str, status: str, notes: Optional[str] = None) -> bool:
        """Update paper status and notes."""
        valid_statuses = [
            "reviewed",
            "pending_response",
            "to_review",
            "classic",
            "textbook",
            "reference",
            "high_priority",
            "key_paper",
        ]

        if status not in valid_statuses:
            print(f"Error: Invalid status. Must be one of: {valid_statuses}")
            return False

        paper = self.get_paper(arxiv_id)
        if not paper:
            print(f"Paper {arxiv_id} not found")
            return False

        paper["status"] = status
        if notes:
            paper["notes"] = notes
        paper["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        print(f"Updated {arxiv_id}: status={status}")
        return True

    def update_metadata(
        self,
        arxiv_id: str,
        title: Optional[str] = None,
        authors: Optional[List[str]] = None,
        year: Optional[int] = None,
        abstract: Optional[str] = None,
    ) -> bool:
        """
        Update paper metadata.

        Parameters
        ----------
        arxiv_id : str
            arXiv identifier
        title : str, optional
            New title
        authors : List[str], optional
            New author list
        year : int, optional
            New publication year
        abstract : str, optional
            New abstract

        Returns
        -------
        bool
            True if successfully updated
        """
        paper = self.get_paper(arxiv_id)
        if not paper:
            print(f"Paper {arxiv_id} not found")
            return False

        updated = False
        if title is not None:
            paper["title"] = title
            updated = True
        if authors is not None:
            paper["authors"] = authors
            updated = True
        if year is not None:
            paper["year"] = year
            updated = True
        if abstract is not None:
            paper["abstract"] = abstract
            updated = True

        if updated:
            paper["last_updated"] = datetime.now().strftime("%Y-%m-%d")
            print(f"Updated metadata for {arxiv_id}")
            return True
        else:
            print("No metadata fields specified for update")
            return False

    def add_tags(self, arxiv_id: str, tags: List[str]) -> bool:
        """
        Add tags to a paper.

        Parameters
        ----------
        arxiv_id : str
            arXiv identifier
        tags : List[str]
            Tags to add

        Returns
        -------
        bool
            True if successfully added
        """
        paper = self.get_paper(arxiv_id)
        if not paper:
            print(f"Paper {arxiv_id} not found")
            return False

        if "tags" not in paper:
            paper["tags"] = []

        new_tags = [tag for tag in tags if tag not in paper["tags"]]
        paper["tags"].extend(new_tags)
        paper["last_updated"] = datetime.now().strftime("%Y-%m-%d")

        print(f"Added {len(new_tags)} new tag(s) to {arxiv_id}")
        return True

    def remove_tags(self, arxiv_id: str, tags: List[str]) -> bool:
        """
        Remove tags from a paper.

        Parameters
        ----------
        arxiv_id : str
            arXiv identifier
        tags : List[str]
            Tags to remove

        Returns
        -------
        bool
            True if successfully removed
        """
        paper = self.get_paper(arxiv_id)
        if not paper:
            print(f"Paper {arxiv_id} not found")
            return False

        if "tags" not in paper:
            print(f"Paper {arxiv_id} has no tags")
            return False

        removed_count = 0
        for tag in tags:
            if tag in paper["tags"]:
                paper["tags"].remove(tag)
                removed_count += 1

        if removed_count > 0:
            paper["last_updated"] = datetime.now().strftime("%Y-%m-%d")
            print(f"Removed {removed_count} tag(s) from {arxiv_id}")
            return True
        else:
            print(f"No matching tags found for {arxiv_id}")
            return False

    def get_paper_info(self, arxiv_id: str) -> Optional[Dict]:
        """
        Get detailed information about a paper.

        Parameters
        ----------
        arxiv_id : str
            arXiv identifier

        Returns
        -------
        Dict or None
            Paper information dictionary
        """
        return self.get_paper(arxiv_id)

    def export_bibtex(
        self, output_file: str, collection: Optional[str] = None, status: Optional[str] = None
    ) -> int:
        """
        Export papers to BibTeX file.

        Parameters
        ----------
        output_file : str
            Output .bib file path
        collection : str, optional
            Only export papers from this collection
        status : str, optional
            Only export papers with this status

        Returns
        -------
        int
            Number of papers exported
        """
        papers = self.db["papers"]

        if collection:
            papers = [p for p in papers if collection in p.get("collections", [])]
        if status:
            papers = [p for p in papers if p.get("status") == status]

        with open(output_file, "w", encoding="utf-8") as f:
            for paper in papers:
                try:
                    bibtex = paper.get("bibtex")
                    if not bibtex:
                        # Generate if not stored
                        bibtex = generate_bibtex(paper["arxiv"])
                    f.write(bibtex + "\n\n")
                except Exception as e:
                    print(f"Warning: Could not export {paper['arxiv']}: {e}")

        print(f"Exported {len(papers)} papers to {output_file}")
        return len(papers)

    def get_statistics(self) -> Dict:
        """Get database statistics."""
        stats = {
            "total_papers": len(self.db["papers"]),
            "by_year": self._count_by_year(),
            "by_status": self._count_by_status(),
            "by_collection": self._count_by_collection(),
        }
        return stats

    def _create_entry_from_result(self, result) -> Dict:
        """Create database entry from arxiv.Result."""
        arxiv_id = result.entry_id.split("/")[-1].split("v")[0]

        # Extract categories
        if hasattr(result, "categories"):
            if result.categories and hasattr(result.categories[0], "term"):
                categories = [cat.term for cat in result.categories]
            else:
                categories = result.categories if isinstance(result.categories, list) else []
        else:
            categories = []

        entry = {
            "id": arxiv_id.replace(".", "_"),
            "arxiv": arxiv_id,
            "doi": result.doi if result.doi else None,
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "year": result.published.year,
            "categories": categories,
            "abstract": result.summary.replace("\n", " ").strip(),
            "url": result.entry_id,
            "pdf_url": result.pdf_url,
            "status": "to_review",
            "priority": "normal",
            "collections": [],
            "tags": [],
            "notes": "",
            "added_date": datetime.now().strftime("%Y-%m-%d"),
        }

        # Try to generate BibTeX
        try:
            entry["bibtex"] = generate_bibtex(arxiv_id)
        except Exception:
            entry["bibtex"] = None

        return entry

    def _ensure_collection_exists(self, collection: str):
        """Ensure collection entry exists in metadata."""
        if "collections" not in self.db:
            self.db["collections"] = {}
        if collection not in self.db["collections"]:
            self.db["collections"][collection] = {
                "created": datetime.now().strftime("%Y-%m-%d"),
                "description": "",
            }

    def _count_by_year(self) -> Dict[int, int]:
        """Count papers by year."""
        counts = {}
        for paper in self.db["papers"]:
            year = paper.get("year")
            if year:
                counts[year] = counts.get(year, 0) + 1
        return dict(sorted(counts.items()))

    def _count_by_status(self) -> Dict[str, int]:
        """Count papers by status."""
        counts = {}
        for paper in self.db["papers"]:
            status = paper.get("status", "unknown")
            counts[status] = counts.get(status, 0) + 1
        return counts

    def _count_by_collection(self) -> Dict[str, int]:
        """Count papers by collection."""
        counts = {}
        for paper in self.db["papers"]:
            for collection in paper.get("collections", []):
                counts[collection] = counts.get(collection, 0) + 1
        return counts

    def list_papers(self, collection: Optional[str] = None, status: Optional[str] = None) -> None:
        """Print formatted list of papers."""
        papers = self.db["papers"]

        if collection:
            papers = [p for p in papers if collection in p.get("collections", [])]
        if status:
            papers = [p for p in papers if p.get("status") == status]

        if collection and status:
            print(f"\nPapers in collection '{collection}' with status '{status}':")
        elif collection:
            print(f"\nPapers in collection '{collection}':")
        elif status:
            print(f"\nPapers with status '{status}':")
        else:
            print("\nAll papers in database:")

        print("-" * 80)

        for paper in papers:
            arxiv_id = paper.get("arxiv", "N/A")
            title = (paper.get("title") or "No title")[:60]
            authors = paper.get("authors", [])
            author_str = (authors[0] if authors else "Unknown")[:20]
            year = paper.get("year") or "N/A"
            status_str = paper.get("status") or "N/A"

            # Handle None values in format strings
            year_str = str(year) if year != "N/A" else "N/A"
            print(
                f"{arxiv_id:15s} | {author_str:20s} | {year_str:4s} | {status_str:15s} | {title}..."
            )

        print(f"\nTotal: {len(papers)} papers")

    def get_all_papers(self) -> List[Dict]:
        """Get all papers from database."""
        return self.db["papers"]

    def get_papers_by_collection(self, collection: str) -> List[Dict]:
        """
        Get all papers in a collection.

        Parameters
        ----------
        collection : str
            Collection name

        Returns
        -------
        list of dict
            List of paper dictionaries
        """
        return [p for p in self.db["papers"] if collection in p.get("collections", [])]

    def get_collection_description(self, collection: str) -> Optional[str]:
        """
        Get description for a collection.

        Parameters
        ----------
        collection : str
            Collection name

        Returns
        -------
        str or None
            Collection description if set
        """
        if "collection_descriptions" not in self.db:
            self.db["collection_descriptions"] = {}
        return self.db["collection_descriptions"].get(collection)

    def set_collection_description(self, collection: str, description: str) -> None:
        """
        Set description for a collection.

        Parameters
        ----------
        collection : str
            Collection name
        description : str
            Description text
        """
        if "collection_descriptions" not in self.db:
            self.db["collection_descriptions"] = {}
        self.db["collection_descriptions"][collection] = description
