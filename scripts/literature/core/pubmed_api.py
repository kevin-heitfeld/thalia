"""
PubMed API Client using NCBI Entrez

This module provides utilities for searching PubMed and retrieving paper metadata.
Uses NCBI Entrez E-utilities which are free but require rate limiting.

IMPORTANT: NCBI requests no more than 3 requests per second without an API key,
or 10 requests per second with an API key. This module uses conservative rate limiting.
"""

import time
from dataclasses import dataclass
from typing import List, Optional
from Bio import Entrez

# Set your email for NCBI (required by their guidelines)
# Users should set this or pass it to the client
Entrez.email = "your_email@example.com"


@dataclass
class PubMedResult:
    """
    Data class representing a PubMed paper result.

    Attributes
    ----------
    pmid : str
        PubMed ID
    title : str
        Paper title
    authors : List[str]
        List of author names
    abstract : str
        Paper abstract
    journal : str
        Journal name
    pub_date : str
        Publication date
    doi : Optional[str]
        Digital Object Identifier if available
    """
    pmid: str
    title: str
    authors: List[str]
    abstract: str
    journal: str
    pub_date: str
    doi: Optional[str] = None

    def __str__(self):
        """String representation of paper."""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        return f"{self.title}\n{authors_str} ({self.pub_date})\nPMID: {self.pmid}"


class PubMedClient:
    """
    Rate-limited PubMed API client using NCBI Entrez.

    This client automatically enforces NCBI rate limiting guidelines
    and provides structured access to PubMed literature.

    Examples
    --------
    >>> client = PubMedClient(email="researcher@university.edu")
    >>> papers = client.search("striatal plasticity dopamine", max_results=20)
    >>> for paper in papers:
    ...     print(paper.title, paper.pmid)
    """

    def __init__(
        self,
        email: str = "your_email@example.com",
        api_key: Optional[str] = None,
        rate_limit_seconds: float = 0.34
    ):
        """
        Initialize PubMedClient.

        Parameters
        ----------
        email : str
            Email address (required by NCBI)
        api_key : str, optional
            NCBI API key for higher rate limits
        rate_limit_seconds : float, optional
            Minimum seconds between requests (default: 0.34 = ~3 req/sec)
        """
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key
            self.rate_limit = 0.1  # 10 requests/second with API key
        else:
            self.rate_limit = rate_limit_seconds

        self.last_request_time = 0.0

    def _wait_for_rate_limit(self):
        """Enforce rate limiting per NCBI guidelines."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            wait_time = self.rate_limit - elapsed
            time.sleep(wait_time)
        self.last_request_time = time.time()

    def search(
        self,
        query: str,
        max_results: int = 20,
        sort: str = "relevance"
    ) -> List[PubMedResult]:
        """
        Search PubMed and return structured results.

        Parameters
        ----------
        query : str
            Search query (supports PubMed query syntax)
            Examples:
            - "striatal plasticity AND dopamine"
            - "basal ganglia[Title] AND action selection"
            - "credit assignment[Title/Abstract]"
        max_results : int, optional
            Maximum results to return (default: 20)
        sort : str, optional
            Sort order: 'relevance', 'pub_date', 'author'

        Returns
        -------
        list of PubMedResult
            List of paper results with metadata
        """
        # Search for PMIDs
        self._wait_for_rate_limit()
        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort=sort
        )
        record = Entrez.read(handle)
        handle.close()

        pmids = record["IdList"]
        if not pmids:
            return []

        # Fetch paper details
        self._wait_for_rate_limit()
        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(pmids),
            rettype="medline",
            retmode="xml"
        )
        records = Entrez.read(handle)
        handle.close()

        # Parse results
        papers = []
        for article in records.get("PubmedArticle", []):
            paper = self._parse_article(article)
            if paper:
                papers.append(paper)

        return papers

    def _parse_article(self, article) -> Optional[PubMedResult]:
        """Parse an article record into PubMedResult."""
        try:
            medline = article["MedlineCitation"]
            article_data = medline["Article"]

            # Extract PMID
            pmid = str(medline["PMID"])

            # Extract title
            title = article_data.get("ArticleTitle", "No title")

            # Extract authors
            authors = []
            author_list = article_data.get("AuthorList", [])
            for author in author_list:
                if "LastName" in author:
                    name = author["LastName"]
                    if "Initials" in author:
                        name = f"{author['LastName']} {author['Initials']}"
                    authors.append(name)

            # Extract abstract
            abstract_sections = article_data.get("Abstract", {}).get("AbstractText", [])
            if abstract_sections:
                if isinstance(abstract_sections, list):
                    abstract = " ".join(str(section) for section in abstract_sections)
                else:
                    abstract = str(abstract_sections)
            else:
                abstract = "No abstract available"

            # Extract journal
            journal = article_data.get("Journal", {}).get("Title", "Unknown journal")

            # Extract publication date
            pub_date_dict = article_data.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            year = pub_date_dict.get("Year", "")
            month = pub_date_dict.get("Month", "")
            pub_date = f"{month} {year}".strip() if month else year

            # Extract DOI if available
            doi = None
            article_ids = article.get("PubmedData", {}).get("ArticleIdList", [])
            for article_id in article_ids:
                if article_id.attributes.get("IdType") == "doi":
                    doi = str(article_id)
                    break

            return PubMedResult(
                pmid=pmid,
                title=title,
                authors=authors,
                abstract=abstract,
                journal=journal,
                pub_date=pub_date,
                doi=doi
            )

        except (KeyError, TypeError) as e:
            # Skip malformed records
            return None

    def get_paper_details(self, pmid: str) -> Optional[PubMedResult]:
        """
        Retrieve details for a specific PubMed ID.

        Parameters
        ----------
        pmid : str
            PubMed ID

        Returns
        -------
        PubMedResult or None
            Paper details if found
        """
        self._wait_for_rate_limit()
        handle = Entrez.efetch(
            db="pubmed",
            id=pmid,
            rettype="medline",
            retmode="xml"
        )
        records = Entrez.read(handle)
        handle.close()

        articles = records.get("PubmedArticle", [])
        if articles:
            return self._parse_article(articles[0])
        return None


def search_pubmed(
    query: str,
    max_results: int = 20,
    email: str = "your_email@example.com"
) -> List[PubMedResult]:
    """
    Convenience function for searching PubMed.

    Parameters
    ----------
    query : str
        Search query
    max_results : int, optional
        Maximum results (default: 20)
    email : str, optional
        Email for NCBI

    Returns
    -------
    list of PubMedResult
        Search results
    """
    client = PubMedClient(email=email)
    return client.search(query, max_results)
