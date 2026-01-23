"""
Table Extraction Module

Extract and structure tables from PDFs and LaTeX sources.
Uses pdfplumber for PDF table detection and regex for LaTeX parsing.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

try:
    import pdfplumber

    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


class TableExtractor:
    """
    Extract tables from PDFs and LaTeX files.

    Features:
    - PDF table detection with pdfplumber
    - LaTeX table parsing (tabular, table environments)
    - CSV export support
    - Caption and label extraction
    - Table metadata tracking

    Examples
    --------
    >>> extractor = TableExtractor()
    >>>
    >>> # Extract from PDF
    >>> tables = extractor.extract_pdf_tables("paper.pdf")
    >>> print(f"Found {len(tables)} tables")
    >>>
    >>> # Extract from LaTeX
    >>> latex_tables = extractor.extract_latex_tables("paper.tex")
    >>>
    >>> # Export to CSV
    >>> extractor.export_table_to_csv(tables[0], "table.csv")
    """

    def __init__(self):
        """Initialize table extractor."""
        pass

    def extract_pdf_tables(
        self, pdf_path: str, min_words: int = 3, min_rows: int = 2
    ) -> List[Dict]:
        """
        Extract tables from PDF using pdfplumber.

        Parameters
        ----------
        pdf_path : str
            Path to PDF file
        min_words : int, optional
            Minimum words per cell to consider valid (default: 3)
        min_rows : int, optional
            Minimum rows to consider valid table (default: 2)

        Returns
        -------
        list of dict
            List of extracted tables, each with:
            - page: Page number (0-indexed)
            - data: 2D list of cell contents
            - rows: Number of rows
            - cols: Number of columns
            - bbox: Bounding box (x0, y0, x1, y1)
        """
        if not HAS_PDFPLUMBER:
            raise ImportError(
                "pdfplumber is required for PDF table extraction. "
                "Install with: pip install pdfplumber"
            )

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return []

        tables = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract tables from page
                    page_tables = page.extract_tables()

                    for table in page_tables:
                        if not table or len(table) < min_rows:
                            continue

                        # Count valid cells
                        valid_cells = sum(
                            1
                            for row in table
                            for cell in row
                            if cell and len(str(cell).split()) >= min_words
                        )

                        # Skip if too few valid cells
                        if valid_cells < min_words * min_rows:
                            continue

                        # Get table info
                        rows = len(table)
                        cols = len(table[0]) if table else 0

                        tables.append(
                            {
                                "page": page_num,
                                "data": table,
                                "rows": rows,
                                "cols": cols,
                                "source": "pdf",
                            }
                        )

        except Exception as e:
            print(f"Error extracting tables from PDF: {e}")

        return tables

    def extract_latex_tables(self, tex_path: str) -> List[Dict]:
        """
        Extract tables from LaTeX file.

        Supports:
        - tabular environment
        - table environment (with caption)
        - longtable environment

        Parameters
        ----------
        tex_path : str
            Path to .tex file

        Returns
        -------
        list of dict
            List of extracted tables, each with:
            - environment: LaTeX environment (tabular, table, etc.)
            - content: Raw LaTeX table content
            - caption: Table caption (if available)
            - label: Table label (if available)
            - line_number: Starting line number
            - data: Parsed 2D list of cells (if parseable)
        """
        tex_path = Path(tex_path)
        if not tex_path.exists():
            return []

        try:
            with open(tex_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            return []

        tables = []

        # Pattern for table environments
        # Matches: \begin{table}...\end{table}, \begin{longtable}...\end{longtable}
        table_pattern = re.compile(
            r"\\begin\{(table|longtable)\*?\}(.*?)\\end\{\1\*?\}", re.DOTALL
        )

        for match in table_pattern.finditer(content):
            env = match.group(1)
            table_content = match.group(2)

            # Count line number
            line_number = content[: match.start()].count("\n") + 1

            # Extract caption
            caption_match = re.search(r"\\caption\{([^}]+)\}", table_content)
            caption = caption_match.group(1).strip() if caption_match else None

            # Extract label
            label_match = re.search(r"\\label\{([^}]+)\}", table_content)
            label = label_match.group(1).strip() if label_match else None

            # Extract tabular content
            tabular_match = re.search(
                r"\\begin\{tabular\}(\{[^}]+\}|\[[^\]]+\])*\{([^}]+)\}(.*?)\\end\{tabular\}",
                table_content,
                re.DOTALL,
            )

            if tabular_match:
                col_spec = tabular_match.group(2)
                tabular_content = tabular_match.group(3)

                # Try to parse table data
                data = self._parse_tabular_content(tabular_content)

                tables.append(
                    {
                        "environment": env,
                        "content": table_content.strip(),
                        "caption": caption,
                        "label": label,
                        "line_number": line_number,
                        "col_spec": col_spec,
                        "data": data,
                        "rows": len(data) if data else 0,
                        "cols": len(data[0]) if data else 0,
                        "source": "latex",
                    }
                )

        # Also find standalone tabular environments (not in table)
        standalone_pattern = re.compile(
            r"\\begin\{tabular\}(\{[^}]+\}|\[[^\]]+\])*\{([^}]+)\}(.*?)\\end\{tabular\}",
            re.DOTALL,
        )

        for match in standalone_pattern.finditer(content):
            # Skip if already captured as part of table environment
            if any(
                match.start() >= t["line_number"]
                and match.end() <= t["line_number"] + len(t["content"])
                for t in tables
            ):
                continue

            col_spec = match.group(2)
            tabular_content = match.group(3)
            line_number = content[: match.start()].count("\n") + 1

            data = self._parse_tabular_content(tabular_content)

            tables.append(
                {
                    "environment": "tabular",
                    "content": match.group(0).strip(),
                    "caption": None,
                    "label": None,
                    "line_number": line_number,
                    "col_spec": col_spec,
                    "data": data,
                    "rows": len(data) if data else 0,
                    "cols": len(data[0]) if data else 0,
                    "source": "latex",
                }
            )

        return tables

    def _parse_tabular_content(self, content: str) -> List[List[str]]:
        """
        Parse tabular content into 2D array.

        Parameters
        ----------
        content : str
            Tabular content (between \\begin{tabular} and \\end{tabular})

        Returns
        -------
        list of list of str
            2D array of cell contents
        """
        # Remove comments
        content = re.sub(r"%.*$", "", content, flags=re.MULTILINE)

        # Remove \hline, \cline, etc.
        content = re.sub(r"\\[hc]line.*$", "", content, flags=re.MULTILINE)

        # Remove \toprule, \midrule, \bottomrule (booktabs)
        content = re.sub(r"\\(top|mid|bottom)rule", "", content)

        # Split by row separator \\
        rows = content.split("\\\\")

        data = []
        for row in rows:
            row = row.strip()
            if not row:
                continue

            # Split by column separator &
            cells = [cell.strip() for cell in row.split("&")]

            # Clean cells (remove extra commands)
            cells = [re.sub(r"\\[a-zA-Z]+(\{[^}]*\}|\[[^\]]*\])*", "", cell) for cell in cells]

            if cells and any(cell for cell in cells):  # Skip empty rows
                data.append(cells)

        return data

    def extract_all_tables(self, pdf_path: Optional[str] = None, latex_path: Optional[str] = None) -> Dict[str, List[Dict]]:
        """
        Extract tables from both PDF and LaTeX sources.

        Parameters
        ----------
        pdf_path : str, optional
            Path to PDF file
        latex_path : str, optional
            Path to LaTeX file

        Returns
        -------
        dict
            Results with 'pdf_tables' and 'latex_tables' keys
        """
        results = {"pdf_tables": [], "latex_tables": []}

        if pdf_path:
            results["pdf_tables"] = self.extract_pdf_tables(pdf_path)

        if latex_path:
            results["latex_tables"] = self.extract_latex_tables(latex_path)

        return results

    def export_table_to_csv(self, table: Dict, output_path: str) -> bool:
        """
        Export table to CSV file.

        Parameters
        ----------
        table : dict
            Table dict with 'data' key
        output_path : str
            Output CSV file path

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if "data" not in table or not table["data"]:
            return False

        try:
            import csv

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(table["data"])

            return True

        except Exception as e:
            print(f"Error exporting table to CSV: {e}")
            return False

    def export_tables_batch(
        self, tables: List[Dict], output_dir: str, prefix: str = "table"
    ) -> Dict[str, any]:
        """
        Export multiple tables to CSV files.

        Parameters
        ----------
        tables : list of dict
            List of table dicts
        output_dir : str
            Output directory
        prefix : str, optional
            Filename prefix (default: 'table')

        Returns
        -------
        dict
            Results with exported count and file list
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {"exported": 0, "failed": 0, "output_files": []}

        for i, table in enumerate(tables, start=1):
            # Determine filename
            if table.get("label"):
                filename = f"{prefix}_{table['label']}.csv"
            else:
                filename = f"{prefix}_{i:04d}.csv"

            output_path = output_dir / filename

            success = self.export_table_to_csv(table, str(output_path))

            if success:
                results["exported"] += 1
                results["output_files"].append(str(output_path))
            else:
                results["failed"] += 1

        return results
