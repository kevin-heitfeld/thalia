"""
Literature Management System for Theoretical Physics Research

This package provides tools for:
- Searching arXiv for relevant papers
- Managing a literature database
- Downloading papers (PDF and LaTeX sources)
- Extracting formulas and key content
- Organizing papers into collections

Usage:
    from literature import ArxivClient, LiteratureDatabase

    client = ArxivClient()
    db = LiteratureDatabase()

    results = client.search("short term synaptic plasticity AND computational model")
    for paper in results:
        db.add_paper(paper, collection="STSP Models")
"""

__version__ = "2.0.0"

from .core.arxiv_api import ArxivClient, download_paper, generate_bibtex, search_arxiv
from .core.database import LiteratureDatabase

__all__ = [
    "ArxivClient",
    "LiteratureDatabase",
    "search_arxiv",
    "download_paper",
    "generate_bibtex",
]
