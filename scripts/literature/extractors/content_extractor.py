"""
Content Extraction Module

Extract and process content from PDFs and LaTeX sources:
- PDF text extraction
- LaTeX equation parsing
- Figure and caption extraction
- Bibliography parsing
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

try:
    from pypdf import PdfReader

    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    import pdfplumber

    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


class ContentExtractor:
    """
    Extract content from PDFs and LaTeX sources.

    Features:
    - PDF text extraction (PyPDF2 or pdfplumber)
    - LaTeX equation extraction
    - Figure caption extraction
    - Bibliography parsing
    - Metadata extraction

    Examples
    --------
    >>> extractor = ContentExtractor()
    >>>
    >>> # Extract from PDF
    >>> text = extractor.extract_pdf_text("paper.pdf")
    >>>
    >>> # Extract equations from LaTeX
    >>> equations = extractor.extract_latex_equations("paper.tex")
    >>>
    >>> # Extract everything from paper directory
    >>> content = extractor.extract_paper_content(
    ...     pdf_path="pdfs/2511.16280v1.pdf",
    ...     latex_dir="latex/2511.16280v1"
    ... )
    """

    def __init__(self, use_pdfplumber: bool = True):
        """
        Initialize content extractor.

        Parameters
        ----------
        use_pdfplumber : bool
            Prefer pdfplumber over PyPDF2 if available (default: True)
        """
        self.use_pdfplumber = use_pdfplumber and HAS_PDFPLUMBER

        if not HAS_PYPDF and not HAS_PDFPLUMBER:
            print("Warning: Neither PyPDF2 nor pdfplumber installed")
            print("Install with: pip install PyPDF2 pdfplumber")

    def extract_pdf_text(self, pdf_path: str) -> Optional[str]:
        """
        Extract text from PDF.

        Parameters
        ----------
        pdf_path : str
            Path to PDF file

        Returns
        -------
        str or None
            Extracted text, or None if extraction failed
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            print(f"  ✗ PDF not found: {pdf_path}")
            return None

        # Try pdfplumber first, fallback to pypdf if it fails
        try:
            if self.use_pdfplumber:
                return self._extract_with_pdfplumber(pdf_path)
            elif HAS_PYPDF:
                return self._extract_with_pypdf2(pdf_path)
            else:
                print("  ✗ No PDF extraction library available")
                return None
        except Exception as e:
            print(f"  ⚠ pdfplumber failed: {str(e)[:100]}")
            if HAS_PYPDF:
                print(f"  → Trying pypdf instead...")
                try:
                    return self._extract_with_pypdf2(pdf_path)
                except Exception as e2:
                    print(f"  ✗ pypdf also failed: {str(e2)[:100]}")
                    return None
            return None

    def _extract_with_pdfplumber(self, pdf_path: Path) -> str:
        """Extract text using pdfplumber (better quality)."""
        import pdfplumber

        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)

        return "\n\n".join(text_parts)

    def _extract_with_pypdf2(self, pdf_path: Path) -> str:
        """Extract text using PyPDF2 (fallback)."""
        text_parts = []
        with open(pdf_path, "rb") as f:
            from pypdf import PdfReader

            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)

        return "\n\n".join(text_parts)

    def extract_latex_equations(
        self, tex_path: str, extract_labels: bool = True
    ) -> List[Dict[str, str]]:
        """
        Extract equations from LaTeX file.

        Parameters
        ----------
        tex_path : str
            Path to .tex file
        extract_labels : bool
            Whether to extract equation labels (default: True)

        Returns
        -------
        list of dict
            List of equations with metadata:
            - equation: LaTeX equation code
            - environment: Type (equation, align, etc.)
            - label: Label if present
            - line_number: Line in file
        """
        tex_path = Path(tex_path)

        if not tex_path.exists():
            return []

        try:
            with open(tex_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            print(f"  ✗ Failed to read {tex_path}: {e}")
            return []

        equations = []

        # Patterns for different equation environments
        patterns = [
            (r"\\begin\{equation\}(.*?)\\end\{equation\}", "equation"),
            (r"\\begin\{equation\*\}(.*?)\\end\{equation\*\}", "equation*"),
            (r"\\begin\{align\}(.*?)\\end\{align\}", "align"),
            (r"\\begin\{align\*\}(.*?)\\end\{align\*\}", "align*"),
            (r"\\begin\{eqnarray\}(.*?)\\end\{eqnarray\}", "eqnarray"),
            (r"\\\[(.*?)\\\]", "display"),
            (r"\$\$(.*?)\$\$", "display"),
        ]

        for pattern, env_type in patterns:
            for match in re.finditer(pattern, content, re.DOTALL):
                equation_text = match.group(1).strip()

                # Extract label if present
                label = None
                if extract_labels:
                    label_match = re.search(r"\\label\{([^}]+)\}", equation_text)
                    if label_match:
                        label = label_match.group(1)

                # Find line number
                line_num = content[: match.start()].count("\n") + 1

                equations.append(
                    {
                        "equation": equation_text,
                        "environment": env_type,
                        "label": label,
                        "line_number": line_num,
                        "full_text": match.group(0),
                    }
                )

        return equations

    def extract_latex_metadata(self, tex_path: str) -> Dict[str, any]:
        """
        Extract metadata from LaTeX file.

        Parameters
        ----------
        tex_path : str
            Path to .tex file

        Returns
        -------
        dict
            Metadata including:
            - title: Paper title
            - authors: List of authors
            - abstract: Abstract text
            - sections: List of section titles
        """
        tex_path = Path(tex_path)

        if not tex_path.exists():
            return {}

        try:
            with open(tex_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            return {}

        metadata = {"title": None, "authors": [], "abstract": None, "sections": []}

        # Extract title
        title_match = re.search(r"\\title\{([^}]+)\}", content)
        if title_match:
            metadata["title"] = title_match.group(1).strip()

        # Extract authors
        author_matches = re.findall(r"\\author\{([^}]+)\}", content)
        metadata["authors"] = [a.strip() for a in author_matches]

        # Extract abstract
        abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", content, re.DOTALL)
        if abstract_match:
            metadata["abstract"] = abstract_match.group(1).strip()

        # Extract sections
        section_matches = re.findall(r"\\section\{([^}]+)\}", content)
        metadata["sections"] = [s.strip() for s in section_matches]

        return metadata

    def extract_paper_content(
        self, pdf_path: Optional[str] = None, latex_dir: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Extract all content from a paper.

        Parameters
        ----------
        pdf_path : str, optional
            Path to PDF file
        latex_dir : str, optional
            Path to LaTeX directory

        Returns
        -------
        dict
            Comprehensive content including:
            - pdf_text: Extracted PDF text
            - equations: List of equations from LaTeX
            - metadata: LaTeX metadata
            - tex_files: List of .tex files found
        """
        content = {"pdf_text": None, "equations": [], "metadata": {}, "tex_files": []}

        # Extract PDF text
        if pdf_path:
            print(f"  Extracting PDF: {Path(pdf_path).name}...", end=" ")
            content["pdf_text"] = self.extract_pdf_text(pdf_path)
            if content["pdf_text"]:
                word_count = len(content["pdf_text"].split())
                print(f"✓ ({word_count:,} words)")
            else:
                print("✗")

        # Extract LaTeX content
        if latex_dir:
            latex_path = Path(latex_dir)
            if latex_path.exists():
                # Find .tex files
                tex_files = list(latex_path.glob("**/*.tex"))
                content["tex_files"] = [str(f.relative_to(latex_path)) for f in tex_files]

                if tex_files:
                    print(f"  Found {len(tex_files)} .tex file(s)")

                    # Extract from main file (usually the shortest filename)
                    main_tex = min(tex_files, key=lambda f: len(f.name))
                    print(f"  Processing: {main_tex.name}...", end=" ")

                    # Extract equations
                    equations = self.extract_latex_equations(str(main_tex))
                    content["equations"] = equations

                    # Extract metadata
                    metadata = self.extract_latex_metadata(str(main_tex))
                    content["metadata"] = metadata

                    print(f"✓ ({len(equations)} equations)")

        return content

    def extract_figures_info(self, latex_dir: str) -> List[Dict[str, str]]:
        """
        Extract figure information from LaTeX with enhanced cross-referencing.

        Parameters
        ----------
        latex_dir : str
            Path to LaTeX directory

        Returns
        -------
        list of dict
            Figure information including:
            - filename: Figure filename
            - caption: Figure caption
            - label: Figure label
            - line_number: Line number in source
            - tex_file: Source .tex file
            - references: List of places this figure is referenced
            - exists: Whether the figure file actually exists
        """
        latex_path = Path(latex_dir)
        if not latex_path.exists():
            return []

        figures = []
        all_references = {}  # Map label -> list of reference locations

        # First pass: find all \ref{} references
        tex_files = list(latex_path.glob("**/*.tex"))
        if not tex_files:
            return []

        # Collect all references
        for tex_file in tex_files:
            try:
                with open(tex_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Find all \ref{label} commands
                ref_pattern = r"\\ref\{([^}]+)\}"
                for match in re.finditer(ref_pattern, content):
                    label = match.group(1)
                    line_num = content[: match.start()].count("\n") + 1

                    if label not in all_references:
                        all_references[label] = []

                    all_references[label].append(
                        {"file": tex_file.name, "line": line_num}
                    )
            except Exception:
                continue

        # Second pass: extract figures with their references
        for tex_file in tex_files:
            try:
                with open(tex_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Find figure environments
                fig_pattern = r"\\begin\{figure\}(.*?)\\end\{figure\}"
                for match in re.finditer(fig_pattern, content, re.DOTALL):
                    fig_content = match.group(1)
                    line_number = content[: match.start()].count("\n") + 1

                    # Extract components
                    filename_match = re.search(
                        r"\\includegraphics(?:\[.*?\])?\{([^}]+)\}", fig_content
                    )
                    caption_match = re.search(r"\\caption\{([^}]+)\}", fig_content)
                    label_match = re.search(r"\\label\{([^}]+)\}", fig_content)

                    filename = filename_match.group(1) if filename_match else None
                    caption = caption_match.group(1) if caption_match else None
                    label = label_match.group(1) if label_match else None

                    # Check if figure file exists
                    fig_exists = False
                    if filename:
                        # Try common extensions if not specified
                        possible_paths = [
                            latex_path / filename,
                            latex_path / f"{filename}.pdf",
                            latex_path / f"{filename}.png",
                            latex_path / f"{filename}.jpg",
                            latex_path / f"{filename}.eps",
                        ]
                        fig_exists = any(p.exists() for p in possible_paths)

                    # Get references to this figure
                    references = all_references.get(label, []) if label else []

                    figures.append(
                        {
                            "filename": filename,
                            "caption": caption,
                            "label": label,
                            "line_number": line_number,
                            "tex_file": tex_file.name,
                            "references": references,
                            "reference_count": len(references),
                            "exists": fig_exists,
                        }
                    )
            except Exception:
                continue

        return figures
