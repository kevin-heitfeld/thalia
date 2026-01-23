"""
Enhanced arXiv API Client

This module provides utilities for searching arXiv, downloading papers,
and generating BibTeX entries with rate limiting and retry logic.

IMPORTANT: arXiv API guidelines request no more than 1 request per 3 seconds.
This module enforces this automatically.
"""

import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

import arxiv


class ArxivClient:
    """
    Rate-limited arXiv API client.

    This client automatically enforces arXiv's rate limiting guidelines
    (3 seconds between requests) and provides retry logic for failed operations.

    Examples
    --------
    >>> client = ArxivClient()
    >>> papers = client.search("quantum field theory", max_results=10)
    >>> client.download_paper("2301.12345", "papers/")
    """

    def __init__(self, rate_limit_seconds: float = 3.0):
        """
        Initialize ArxivClient.

        Parameters
        ----------
        rate_limit_seconds : float, optional
            Minimum seconds between requests (default: 3.0 per arXiv guidelines)
        """
        self.rate_limit = rate_limit_seconds
        self.last_request_time = 0.0

    def _wait_for_rate_limit(self):
        """Enforce rate limiting per arXiv guidelines."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            wait_time = self.rate_limit - elapsed
            time.sleep(wait_time)
        self.last_request_time = time.time()

    def search(
        self, query: str, max_results: int = 10, sort_by: str = "relevance"
    ) -> List[arxiv.Result]:
        """
        Rate-limited search of arXiv.

        Parameters
        ----------
        query : str
            Search query (can use field prefixes like 'ti:', 'au:', 'abs:')
        max_results : int, optional
            Maximum results to return (default: 10)
        sort_by : str, optional
            Sort criterion: 'relevance', 'lastUpdatedDate', 'submittedDate'

        Returns
        -------
        list of arxiv.Result
            List of paper results
        """
        self._wait_for_rate_limit()
        return search_arxiv(query, max_results, sort_by)

    def get_paper(self, arxiv_id: str) -> arxiv.Result:
        """
        Retrieve a specific paper by arXiv ID (rate-limited).

        Parameters
        ----------
        arxiv_id : str
            arXiv identifier

        Returns
        -------
        arxiv.Result
            Paper metadata
        """
        self._wait_for_rate_limit()
        return get_paper_by_id(arxiv_id)

    def download_pdf(
        self, paper_id: str, directory: str = "docs/papers", max_retries: int = 3
    ) -> Optional[str]:
        """
        Download PDF with automatic retry on failure.

        Parameters
        ----------
        paper_id : str
            arXiv identifier
        directory : str, optional
            Output directory (default: 'docs/papers')
        max_retries : int, optional
            Maximum retry attempts (default: 3)

        Returns
        -------
        str or None
            Path to downloaded file, or None if failed
        """
        self._wait_for_rate_limit()

        for attempt in range(max_retries):
            try:
                return download_paper(paper_id, directory)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    print(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  Failed after {max_retries} attempts: {e}")
                    return None

    def download_latex(
        self, paper_id: str, directory: str = "docs/papers/latex_sources", max_retries: int = 3
    ) -> Optional[str]:
        """
        Download LaTeX source with retry logic.

        Parameters
        ----------
        paper_id : str
            arXiv identifier
        directory : str, optional
            Output directory (default: 'docs/papers/latex_sources')
        max_retries : int, optional
            Maximum retry attempts (default: 3)

        Returns
        -------
        str or None
            Path to downloaded tar.gz, or None if failed/unavailable
        """
        self._wait_for_rate_limit()

        for attempt in range(max_retries):
            try:
                return download_latex_source(paper_id, directory)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    print(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  Failed after {max_retries} attempts: {e}")
                    return None


# ============================================================================
# Legacy functions maintained for backward compatibility
# ============================================================================


def search_arxiv(query, max_results=10, sort_by="relevance"):
    """
    Search arXiv and return paper metadata.

    Parameters
    ----------
    query : str
        Search query (can use field prefixes like 'ti:', 'au:', 'abs:')
    max_results : int, optional
        Maximum number of results to return (default: 10)
    sort_by : str, optional
        Sort criterion: 'relevance', 'lastUpdatedDate', 'submittedDate'
        (default: 'relevance')

    Returns
    -------
    list of arxiv.Result
        List of paper results

    Examples
    --------
    >>> papers = search_arxiv("quantum field theory", max_results=5)
    >>> for paper in papers:
    ...     print(f"{paper.title} ({paper.published.year})")
    """
    sort_criteria = {
        "relevance": arxiv.SortCriterion.Relevance,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
        "submittedDate": arxiv.SortCriterion.SubmittedDate,
    }

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_criteria.get(sort_by, arxiv.SortCriterion.Relevance),
    )

    return list(search.results())


def search_by_author(author_name, max_results=10):
    """
    Search arXiv by author name.

    Parameters
    ----------
    author_name : str
        Author name to search for
    max_results : int, optional
        Maximum number of results (default: 10)

    Returns
    -------
    list of arxiv.Result
        List of papers by the author
    """
    query = f"au:{author_name}"
    return search_arxiv(query, max_results)


def search_by_category(category, keywords=None, max_results=20):
    """
    Search arXiv by subject category.

    Parameters
    ----------
    category : str
        arXiv category (e.g., 'hep-th', 'quant-ph', 'gr-qc', 'hep-ph')
    keywords : str, optional
        Additional keywords to refine search
    max_results : int, optional
        Maximum number of results (default: 20)

    Returns
    -------
    list of arxiv.Result
        List of papers in the category

    Examples
    --------
    >>> papers = search_by_category('hep-th', keywords='string theory')
    """
    if keywords:
        query = f"cat:{category} AND all:{keywords}"
    else:
        query = f"cat:{category}"

    return search_arxiv(query, max_results)


def get_paper_by_id(arxiv_id):
    """
    Retrieve a specific paper by arXiv ID.

    Parameters
    ----------
    arxiv_id : str
        arXiv identifier (e.g., '2301.12345' or 'hep-th/0612345')

    Returns
    -------
    arxiv.Result
        Paper metadata
    """
    search = arxiv.Search(id_list=[arxiv_id])
    return next(search.results())


def generate_bibtex(paper_id, entry_type="article"):
    """
    Generate BibTeX entry from arXiv ID.

    Parameters
    ----------
    paper_id : str
        arXiv identifier
    entry_type : str, optional
        BibTeX entry type (default: 'article')

    Returns
    -------
    str
        BibTeX entry

    Examples
    --------
    >>> bibtex = generate_bibtex('2301.12345')
    >>> print(bibtex)
    """
    paper = get_paper_by_id(paper_id)

    # Clean the ID for use as citation key
    citation_key = paper.get_short_id().replace("/", "_")

    # Format authors
    authors = " and ".join(str(author) for author in paper.authors)

    # Generate BibTeX
    bibtex = f"""@{entry_type}{{{citation_key},
  title     = {{{paper.title}}},
  author    = {{{authors}}},
  journal   = {{arXiv preprint arXiv:{paper.get_short_id()}}},
  year      = {{{paper.published.year}}},
  eprint    = {{{paper.get_short_id()}}},
  archivePrefix = {{arXiv}},
  primaryClass = {{{paper.primary_category}}},
  url       = {{{paper.entry_id}}},
  doi       = {{{paper.doi if paper.doi else ''}}},
  abstract  = {{{paper.summary}}}
}}"""

    return bibtex


def download_paper(paper_id, directory="docs/papers", filename=None):
    """
    Download PDF from arXiv.

    Parameters
    ----------
    paper_id : str
        arXiv identifier
    directory : str, optional
        Directory to save PDF (default: 'docs/papers')
    filename : str, optional
        Custom filename (default: uses arXiv ID)

    Returns
    -------
    str
        Path to downloaded file

    Examples
    --------
    >>> path = download_paper('2301.12345', directory='papers')
    >>> print(f"Downloaded to: {path}")
    """
    # Create directory if it doesn't exist
    Path(directory).mkdir(parents=True, exist_ok=True)

    # Get paper metadata
    paper = get_paper_by_id(paper_id)

    # Set filename
    if filename is None:
        filename = f"{paper.get_short_id().replace('/', '_')}.pdf"

    # Download
    filepath = paper.download_pdf(dirpath=directory, filename=filename)

    print(f"Downloaded: {paper.title}")
    print(f"Saved to: {filepath}")

    return filepath


def download_latex_source(paper_id, directory="docs/papers/latex_sources"):
    """
    Download LaTeX source from arXiv.

    Parameters
    ----------
    paper_id : str
        arXiv identifier
    directory : str, optional
        Directory to save source (default: 'docs/papers/latex_sources')

    Returns
    -------
    str or None
        Path to downloaded tar.gz file, or None if unavailable

    Examples
    --------
    >>> path = download_latex_source('2301.12345')
    >>> if path:
    ...     print(f"Downloaded LaTeX source to: {path}")
    """
    # Create directory if it doesn't exist
    Path(directory).mkdir(parents=True, exist_ok=True)

    # Get paper metadata
    paper = get_paper_by_id(paper_id)
    clean_id = paper.get_short_id().replace("/", "_")

    # Construct arXiv source URL
    source_url = f"https://arxiv.org/e-print/{paper.get_short_id()}"

    # Set output filename
    output_file = Path(directory) / f"{clean_id}_source.tar.gz"

    try:
        print(f"Downloading LaTeX source for: {paper.title}")
        print(f"From: {source_url}")

        # Download the source
        urllib.request.urlretrieve(source_url, output_file)

        print(f"✓ Saved to: {output_file}")
        return str(output_file)

    except urllib.error.HTTPError as e:
        if e.code == 403:
            print(f"✗ LaTeX source not available for {paper_id} (only PDF available)")
        else:
            print(f"✗ Error downloading source: {e}")
        return None
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return None


def extract_latex_source(tar_gz_path, extract_to=None):
    """
    Extract a downloaded LaTeX source archive.

    Parameters
    ----------
    tar_gz_path : str
        Path to the .tar.gz file
    extract_to : str, optional
        Directory to extract to (default: same directory as tar.gz)

    Returns
    -------
    str or None
        Path to extracted directory, or None if extraction failed
    """
    import tarfile

    tar_path = Path(tar_gz_path)

    if extract_to is None:
        extract_to = tar_path.parent / tar_path.stem.replace(".tar", "")
    else:
        extract_to = Path(extract_to)

    extract_to.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Extracting {tar_path.name}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_to)

        print(f"✓ Extracted to: {extract_to}")

        # List extracted files
        files = list(extract_to.iterdir())
        print(f"  Found {len(files)} files:")
        for f in sorted(files)[:10]:  # Show first 10
            print(f"    - {f.name}")
        if len(files) > 10:
            print(f"    ... and {len(files) - 10} more")

        return str(extract_to)

    except Exception as e:
        print(f"✗ Error extracting: {e}")
        return None
