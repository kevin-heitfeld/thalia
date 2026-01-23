"""
Smart Search Engine with Relevance Scoring

This module provides intelligent search capabilities including:
- Relevance scoring based on multiple criteria
- Deduplication across searches
- Citation-based expansion (future)
"""

from typing import Dict, List, Optional, Set

from ..core import ArxivClient, LiteratureDatabase


class SmartSearchEngine:
    """
    Intelligent search engine with relevance scoring.

    Features:
    - Multi-query searches with deduplication
    - Relevance scoring based on:
      * Query match strength (title vs abstract)
      * Citation count
      * Author prominence
      * Recency
    - Automatic integration with database

    Examples
    --------
    >>> engine = SmartSearchEngine()
    >>> results = engine.search_with_scoring([
    ...     {"query": "vertex operators", "weight": 2.0},
    ...     {"query": "linear dilaton", "weight": 1.5}
    ... ])
    >>> for paper_id, score in results[:10]:
    ...     print(f"{paper_id}: {score:.2f}")
    """

    def __init__(
        self, client: Optional[ArxivClient] = None, database: Optional[LiteratureDatabase] = None
    ):
        """
        Initialize search engine.

        Parameters
        ----------
        client : ArxivClient, optional
            ArXiv client (creates one if not provided)
        database : LiteratureDatabase, optional
            Literature database for deduplication
        """
        self.client = client if client else ArxivClient()
        self.database = database
        self._seen_papers: Set[str] = set()

    def search_with_scoring(
        self, queries: List[Dict], scoring_weights: Optional[Dict[str, float]] = None
    ) -> List[tuple]:
        """
        Execute multiple queries and score results by relevance.

        Parameters
        ----------
        queries : list of dict
            List of query configurations. Each dict should have:
            - query : str - Search query
            - weight : float - Query importance weight
            - max_results : int - Maximum results per query
        scoring_weights : dict, optional
            Weights for scoring factors:
            - title_match : float - Weight for title matches (default: 2.0)
            - recency : float - Weight for recent papers (default: 0.5)
            - citation_count : float - Weight for citations (default: 0.3)
            - author_count : float - Weight for # authors (default: 0.1)

        Returns
        -------
        list of (arxiv_id, score, paper) tuples
            Sorted by score (highest first)
        """
        if scoring_weights is None:
            scoring_weights = {
                "title_match": 2.0,
                "recency": 0.5,
                "citation_count": 0.0,  # Not available from arXiv API
                "author_count": 0.1,
                "query_weight": 1.0,
            }

        all_results = []

        for query_config in queries:
            query = query_config["query"]
            query_weight = query_config.get("weight", 1.0)
            max_results = query_config.get("max_results", 10)

            print(f"  Searching: {query}")

            # Execute search
            papers = self.client.search(query, max_results=max_results)

            for paper in papers:
                arxiv_id = paper.get_short_id()

                # Skip duplicates
                if arxiv_id in self._seen_papers:
                    continue

                # Skip if already in database
                if self.database and self.database.paper_exists(arxiv_id):
                    continue

                # Calculate relevance score
                score = self._calculate_relevance_score(paper, query, query_weight, scoring_weights)

                all_results.append((arxiv_id, score, paper))
                self._seen_papers.add(arxiv_id)

        # Sort by score
        all_results.sort(key=lambda x: x[1], reverse=True)

        return all_results

    def _calculate_relevance_score(
        self, paper, query: str, query_weight: float, weights: Dict[str, float]
    ) -> float:
        """Calculate relevance score for a paper."""
        score = 0.0

        # Query weight contribution
        score += query_weight * weights["query_weight"]

        # Title match (check if query terms appear in title)
        query_terms = query.lower().split()
        title = paper.title.lower()
        title_matches = sum(1 for term in query_terms if term in title)
        if title_matches > 0:
            title_score = (title_matches / len(query_terms)) * weights["title_match"]
            score += title_score

        # Recency (papers from last 5 years get boost)
        current_year = 2026
        paper_year = paper.published.year
        years_ago = current_year - paper_year
        if years_ago <= 5:
            recency_score = (1.0 - years_ago / 10.0) * weights["recency"]
            score += recency_score

        # Author count (more authors might indicate larger collaborations)
        # But diminishing returns
        author_count = len(paper.authors)
        author_score = min(author_count / 10.0, 1.0) * weights["author_count"]
        score += author_score

        return score

    def deduplicate_results(self, results: List) -> List:
        """
        Remove duplicates from results.

        Parameters
        ----------
        results : list
            List of arxiv.Result objects

        Returns
        -------
        list
            Deduplicated results
        """
        seen = set()
        deduped = []

        for result in results:
            arxiv_id = result.get_short_id()
            if arxiv_id not in seen:
                seen.add(arxiv_id)
                deduped.append(result)

        return deduped

    def search_and_add_to_database(
        self, queries: List[Dict], collection: str, max_total: Optional[int] = None
    ) -> Dict:
        """
        Search and automatically add results to database.

        Parameters
        ----------
        queries : list of dict
            Query configurations
        collection : str
            Collection to add papers to
        max_total : int, optional
            Maximum total papers to add (default: no limit)

        Returns
        -------
        dict
            Statistics about the operation
        """
        if not self.database:
            raise ValueError("Database required for this operation")

        # Execute scored search
        results = self.search_with_scoring(queries)

        # Limit total if requested
        if max_total:
            results = results[:max_total]

        # Add to database
        added_count = 0
        for arxiv_id, score, paper in results:
            if self.database.add_paper(paper, collection=collection):
                added_count += 1
                print(f"  Added: {arxiv_id} (score: {score:.2f})")

        # Save database
        self.database.save()

        return {
            "total_found": len(results),
            "added": added_count,
            "skipped": len(results) - added_count,
            "collection": collection,
        }

    def expand_from_citations(self, seed_papers: List[str]) -> List:
        """
        Expand search based on citations (future implementation).

        This would require integration with external APIs (Semantic Scholar,
        Crossref, etc.) as arXiv doesn't provide citation data.

        Parameters
        ----------
        seed_papers : list of str
            ArXiv IDs of seed papers

        Returns
        -------
        list
            Related papers
        """
        # Placeholder for future implementation
        print("Citation-based expansion not yet implemented")
        print("Requires integration with citation APIs (Semantic Scholar, etc.)")
        return []
