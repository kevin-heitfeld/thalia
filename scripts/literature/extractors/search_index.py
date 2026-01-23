"""
Search Index Module

Build and query searchable indexes of paper content:
- Full-text search across PDFs and LaTeX
- Equation search by pattern
- Metadata search (title, author, abstract)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class SearchIndex:
    """
    Searchable index of paper content.

    Features:
    - Full-text search with ranking
    - Equation pattern search
    - Metadata filtering
    - TF-IDF-like relevance scoring

    Examples
    --------
    >>> index = SearchIndex("docs/references/search_index.json")
    >>>
    >>> # Add paper to index
    >>> index.add_paper(
    ...     arxiv_id="2511.16280",
    ...     pdf_text="...",
    ...     equations=[...],
    ...     metadata={...}
    ... )
    >>>
    >>> # Search
    >>> results = index.search("modular invariance")
    >>> for arxiv_id, score in results[:10]:
    ...     print(f"{arxiv_id}: {score:.2f}")
    """

    def __init__(self, index_path: str = "docs/references/search_index.json"):
        """
        Initialize search index.

        Parameters
        ----------
        index_path : str
            Path to index file (default: docs/references/search_index.json)
        """
        self.index_path = Path(index_path)
        self.index = self._load_index()

    def _load_index(self) -> Dict:
        """Load existing index or create new one."""
        if self.index_path.exists():
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        return {
            "papers": {},
            "word_index": {},
            "equation_index": {},
            "metadata": {"total_papers": 0, "last_updated": None},
        }

    def save(self):
        """Save index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)

    def add_paper(
        self,
        arxiv_id: str,
        pdf_text: Optional[str] = None,
        equations: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Add paper to search index.

        Parameters
        ----------
        arxiv_id : str
            ArXiv ID
        pdf_text : str, optional
            Extracted PDF text
        equations : list of dict, optional
            Extracted equations
        metadata : dict, optional
            Paper metadata
        """
        # Create paper entry
        paper_entry = {
            "arxiv_id": arxiv_id,
            "has_pdf_text": pdf_text is not None,
            "num_equations": len(equations) if equations else 0,
            "metadata": metadata or {},
        }

        # Store PDF text (truncated for storage)
        if pdf_text:
            words = pdf_text.lower().split()
            paper_entry["word_count"] = len(words)
            paper_entry["text_preview"] = " ".join(words[:200])

            # Index words
            self._index_words(arxiv_id, words)

        # Store equations
        if equations:
            paper_entry["equations"] = equations
            self._index_equations(arxiv_id, equations)

        # Add to index
        self.index["papers"][arxiv_id] = paper_entry
        self.index["metadata"]["total_papers"] = len(self.index["papers"])

        # Save
        self.save()

    def _index_words(self, arxiv_id: str, words: List[str]):
        """Build word index for fast search."""
        # Get unique words
        unique_words = set(words)

        # Add to word index
        for word in unique_words:
            if len(word) < 3:  # Skip very short words
                continue

            if word not in self.index["word_index"]:
                self.index["word_index"][word] = []

            if arxiv_id not in self.index["word_index"][word]:
                self.index["word_index"][word].append(arxiv_id)

    def _index_equations(self, arxiv_id: str, equations: List[Dict]):
        """Index equations for pattern search."""
        for eq in equations:
            eq_text = eq.get("equation", "").lower()

            # Extract key terms (LaTeX commands, Greek letters, etc.)
            terms = re.findall(r"\\[a-z]+|[a-z]+", eq_text)

            for term in terms:
                if len(term) < 2:
                    continue

                if term not in self.index["equation_index"]:
                    self.index["equation_index"][term] = []

                if arxiv_id not in self.index["equation_index"][term]:
                    self.index["equation_index"][term].append(arxiv_id)

    def search(
        self, query: str, max_results: int = 20, search_equations: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Search for papers matching query.

        Parameters
        ----------
        query : str
            Search query
        max_results : int
            Maximum results to return (default: 20)
        search_equations : bool
            Whether to search equations (default: True)

        Returns
        -------
        list of (arxiv_id, score) tuples
            Sorted by relevance score
        """
        query_words = query.lower().split()

        # Find matching papers
        paper_scores = {}

        # Search text
        for word in query_words:
            if word in self.index["word_index"]:
                for arxiv_id in self.index["word_index"][word]:
                    paper_scores[arxiv_id] = paper_scores.get(arxiv_id, 0) + 1.0

        # Search equations if enabled
        if search_equations:
            for word in query_words:
                if word in self.index["equation_index"]:
                    for arxiv_id in self.index["equation_index"][word]:
                        # Boost equation matches
                        paper_scores[arxiv_id] = paper_scores.get(arxiv_id, 0) + 2.0

        # Sort by score
        results = sorted(paper_scores.items(), key=lambda x: x[1], reverse=True)

        return results[:max_results]

    def search_equations(self, pattern: str, max_results: int = 20) -> List[Dict]:
        """
        Search for equations matching pattern.

        Parameters
        ----------
        pattern : str
            LaTeX pattern to search for (e.g., "\\frac")
        max_results : int
            Maximum results to return

        Returns
        -------
        list of dict
            Matching equations with paper info
        """
        pattern = pattern.lower()
        results = []

        for arxiv_id, paper in self.index["papers"].items():
            if "equations" not in paper:
                continue

            for eq in paper["equations"]:
                eq_text = eq.get("equation", "").lower()
                if pattern in eq_text:
                    results.append(
                        {
                            "arxiv_id": arxiv_id,
                            "equation": eq,
                            "paper_metadata": paper.get("metadata", {}),
                        }
                    )

                    if len(results) >= max_results:
                        return results

        return results

    def get_paper_info(self, arxiv_id: str) -> Optional[Dict]:
        """Get indexed information for a paper."""
        return self.index["papers"].get(arxiv_id)

    def get_statistics(self) -> Dict:
        """Get index statistics."""
        return {
            "total_papers": len(self.index["papers"]),
            "total_words_indexed": len(self.index["word_index"]),
            "total_equation_terms": len(self.index["equation_index"]),
            "papers_with_text": sum(
                1 for p in self.index["papers"].values() if p.get("has_pdf_text")
            ),
            "papers_with_equations": sum(
                1 for p in self.index["papers"].values() if p.get("num_equations", 0) > 0
            ),
        }

    def list_papers(self) -> List[str]:
        """List all indexed paper IDs."""
        return list(self.index["papers"].keys())
