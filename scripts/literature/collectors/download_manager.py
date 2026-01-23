"""
Unified Download Manager for Literature Management

Handles PDF and LaTeX source downloads with:
- Progress tracking
- Error handling and retry logic
- Automatic extraction of LaTeX sources
- Integration with database
"""

import tarfile
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..core import ArxivClient, LiteratureDatabase


class DownloadManager:
    """
    Unified manager for downloading PDFs and LaTeX sources.

    Features:
    - PDF downloads with progress tracking
    - LaTeX source download and extraction
    - Automatic retry on failures
    - Integration with literature database
    - Organized file structure

    Examples
    --------
    >>> manager = DownloadManager(base_dir="docs/papers")
    >>>
    >>> # Download PDF
    >>> pdf_path = manager.download_pdf("2511.16280")
    >>>
    >>> # Download and extract LaTeX
    >>> latex_dir, tex_files = manager.download_latex("2511.16280")
    >>>
    >>> # Batch download from database collection
    >>> results = manager.download_collection(
    ...     database=db,
    ...     collection="modular_invariance",
    ...     download_pdfs=True,
    ...     download_latex=False
    ... )
    """

    def __init__(
        self,
        base_dir: str = "docs/papers",
        client: Optional[ArxivClient] = None
    ):
        """
        Initialize download manager.

        Parameters
        ----------
        base_dir : str
            Base directory for downloads (default: "docs/papers")
        client : ArxivClient, optional
            ArXiv client (creates one if not provided)
        """
        self.base_dir = Path(base_dir)
        self.pdf_dir = self.base_dir / "pdfs"
        self.latex_dir = self.base_dir / "latex"

        # Create directories
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.latex_dir.mkdir(parents=True, exist_ok=True)

        self.client = client if client else ArxivClient()
        self._download_stats = {
            "pdfs_downloaded": 0,
            "pdfs_failed": 0,
            "latex_downloaded": 0,
            "latex_failed": 0,
        }

    def download_pdf(
        self,
        arxiv_id: str,
        subdirectory: Optional[str] = None,
        custom_filename: Optional[str] = None
    ) -> Optional[Path]:
        """
        Download PDF for a paper.

        Parameters
        ----------
        arxiv_id : str
            ArXiv ID (e.g., "2511.16280" or "2511.16280v1")
        subdirectory : str, optional
            Subdirectory within pdf_dir (e.g., "modular_invariance")
        custom_filename : str, optional
            Custom filename (default: uses arXiv ID)

        Returns
        -------
        Path or None
            Path to downloaded PDF, or None if failed
        """
        # Determine output directory
        if subdirectory:
            output_dir = self.pdf_dir / subdirectory
        else:
            output_dir = self.pdf_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get paper metadata
        try:
            papers = self.client.search(f"id:{arxiv_id}", max_results=1)
            if not papers:
                print(f"  ✗ Paper not found: {arxiv_id}")
                self._download_stats["pdfs_failed"] += 1
                return None

            paper = papers[0]
            clean_id = paper.get_short_id().replace("/", "_")

            # Determine filename
            if custom_filename:
                filename = custom_filename
            else:
                filename = f"{clean_id}.pdf"

            filepath = output_dir / filename

            # Skip if already exists
            if filepath.exists():
                print(f"  → {filename} (already exists)")
                return filepath

            # Download
            print(f"  Downloading: {paper.title[:60]}...", end=" ")
            paper.download_pdf(dirpath=str(output_dir), filename=filename)

            # Rate limiting
            time.sleep(3)

            print(f"✓")
            self._download_stats["pdfs_downloaded"] += 1
            return filepath

        except Exception as e:
            print(f"✗ Failed: {e}")
            self._download_stats["pdfs_failed"] += 1
            return None

    def download_latex(
        self,
        arxiv_id: str,
        subdirectory: Optional[str] = None,
        extract: bool = True
    ) -> Tuple[Optional[Path], List[Path]]:
        """
        Download and optionally extract LaTeX source.

        Parameters
        ----------
        arxiv_id : str
            ArXiv ID
        subdirectory : str, optional
            Subdirectory within latex_dir
        extract : bool
            Whether to extract the archive (default: True)

        Returns
        -------
        tuple
            (extraction_directory, list of .tex files) or (None, []) if failed
        """
        # Determine output directory
        if subdirectory:
            output_dir = self.latex_dir / subdirectory
        else:
            output_dir = self.latex_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Get paper metadata
            papers = self.client.search(f"id:{arxiv_id}", max_results=1)
            if not papers:
                print(f"  ✗ Paper not found: {arxiv_id}")
                self._download_stats["latex_failed"] += 1
                return None, []

            paper = papers[0]
            clean_id = paper.get_short_id().replace("/", "_")

            # Download source
            print(f"  Downloading LaTeX: {paper.title[:50]}...", end=" ")

            # Download source (arxiv library downloads without extension)
            paper.download_source(
                dirpath=str(output_dir), filename=f"{clean_id}_source"
            )

            # Rate limiting
            time.sleep(3)

            # The file might be downloaded as _source, _source.tar.gz, or _source.gz
            possible_files = [
                output_dir / f"{clean_id}_source",
                output_dir / f"{clean_id}_source.tar.gz",
                output_dir / f"{clean_id}_source.gz",
            ]

            temp_file = None
            for pf in possible_files:
                if pf.exists():
                    temp_file = pf
                    break

            if not temp_file:
                print("✗ Source file not found")
                self._download_stats["latex_failed"] += 1
                return None, []
                self._download_stats["latex_downloaded"] += 1
                return temp_file, []

            # Extract archive
            extract_dir = output_dir / clean_id
            extract_dir.mkdir(exist_ok=True)

            # Try to extract
            tex_files = self._extract_latex_source(temp_file, extract_dir)

            if tex_files:
                print(f"✓ ({len(tex_files)} .tex files)")
                self._download_stats["latex_downloaded"] += 1
                return extract_dir, tex_files
            else:
                print("✓ (no .tex files found)")
                self._download_stats["latex_downloaded"] += 1
                return extract_dir, []

        except Exception as e:
            print(f"✗ Failed: {e}")
            self._download_stats["latex_failed"] += 1
            return None, []

    def _extract_latex_source(
        self, archive_path: Path, extract_dir: Path
    ) -> List[Path]:
        """
        Extract LaTeX source from tar.gz or zip archive.

        Parameters
        ----------
        archive_path : Path
            Path to archive file
        extract_dir : Path
            Directory to extract to

        Returns
        -------
        list of Path
            List of .tex files found
        """
        tex_files = []

        try:
            # arXiv sources are typically gzipped tarballs
            # Try different extraction methods

            # Try as tar with any compression
            try:
                with tarfile.open(archive_path, "r:*") as tar:
                    tar.extractall(path=extract_dir)
            except tarfile.ReadError:
                # Maybe it's a zip
                try:
                    with zipfile.ZipFile(archive_path, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)
                except zipfile.BadZipFile:
                    # Last resort: try explicitly as gzipped tar
                    with tarfile.open(archive_path, "r:gz") as tar:
                        tar.extractall(path=extract_dir)
            # Find .tex files
            tex_files = list(extract_dir.glob("**/*.tex"))

        except Exception as e:
            print(f"  Warning: Extraction failed: {e}")

        return tex_files

    def download_from_database(
        self,
        database: LiteratureDatabase,
        collection: Optional[str] = None,
        arxiv_ids: Optional[List[str]] = None,
        download_pdfs: bool = True,
        download_latex: bool = False,
        max_downloads: Optional[int] = None
    ) -> Dict:
        """
        Download papers from database.

        Parameters
        ----------
        database : LiteratureDatabase
            Literature database
        collection : str, optional
            Collection name to download (downloads all if None)
        arxiv_ids : list of str, optional
            Specific arXiv IDs to download (overrides collection)
        download_pdfs : bool
            Whether to download PDFs (default: True)
        download_latex : bool
            Whether to download LaTeX sources (default: False)
        max_downloads : int, optional
            Maximum number to download (default: no limit)

        Returns
        -------
        dict
            Statistics about the download operation
        """
        # Get papers to download
        if arxiv_ids:
            papers_to_download = []
            for arxiv_id in arxiv_ids:
                paper = database.get_paper(arxiv_id)
                if paper:
                    papers_to_download.append(paper)
        elif collection:
            papers_to_download = database.get_papers_by_collection(collection)
        else:
            papers_to_download = database.get_all_papers()

        # Limit if requested
        if max_downloads:
            papers_to_download = papers_to_download[:max_downloads]

        print(f"\n{'=' * 80}")
        print(f"DOWNLOADING FROM DATABASE")
        print(f"{'=' * 80}")
        if collection:
            print(f"Collection: {collection}")
        print(f"Papers to download: {len(papers_to_download)}")
        print(f"Download PDFs: {download_pdfs}")
        print(f"Download LaTeX: {download_latex}")
        print(f"{'=' * 80}\n")

        # Reset stats
        self._download_stats = {
            "pdfs_downloaded": 0,
            "pdfs_failed": 0,
            "latex_downloaded": 0,
            "latex_failed": 0,
        }

        # Download
        for i, paper in enumerate(papers_to_download, 1):
            arxiv_id = paper.get("arxiv", paper.get("id", ""))
            title = paper.get("title", "Unknown")

            print(f"\n[{i}/{len(papers_to_download)}] {arxiv_id}")
            print(f"    {title[:70]}")

            # Create subdirectory if collection specified
            subdir = collection if collection else None

            if download_pdfs:
                self.download_pdf(arxiv_id, subdirectory=subdir)

            if download_latex:
                self.download_latex(arxiv_id, subdirectory=subdir)

        # Summary
        print(f"\n{'=' * 80}")
        print("DOWNLOAD SUMMARY")
        print(f"{'=' * 80}")
        if download_pdfs:
            print(f"PDFs:")
            print(f"  Downloaded: {self._download_stats['pdfs_downloaded']}")
            print(f"  Failed: {self._download_stats['pdfs_failed']}")
        if download_latex:
            print(f"LaTeX:")
            print(f"  Downloaded: {self._download_stats['latex_downloaded']}")
            print(f"  Failed: {self._download_stats['latex_failed']}")
        print(f"{'=' * 80}\n")

        return dict(self._download_stats)

    def get_statistics(self) -> Dict:
        """Get download statistics."""
        return dict(self._download_stats)
