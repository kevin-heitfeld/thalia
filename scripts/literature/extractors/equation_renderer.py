"""
Equation Rendering Module

Render LaTeX equations to images (PNG, SVG) for visualization and documentation.
Uses matplotlib for rendering with proper font support.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib
    import matplotlib.pyplot as plt

    # Use non-interactive backend
    matplotlib.use("Agg")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class EquationRenderer:
    """
    Render LaTeX equations to images.

    Features:
    - PNG and SVG output formats
    - Configurable DPI and size
    - Batch rendering support
    - Automatic equation cleaning
    - Error handling for invalid equations

    Examples
    --------
    >>> renderer = EquationRenderer()
    >>>
    >>> # Render single equation
    >>> renderer.render_equation(
    ...     r"E = mc^2",
    ...     output_path="equation.png",
    ...     format="png"
    ... )
    >>>
    >>> # Render from extracted equations
    >>> equations = extractor.extract_latex_equations("paper.tex")
    >>> renderer.render_equations_batch(
    ...     equations,
    ...     output_dir="docs/papers/rendered_equations"
    ... )
    """

    def __init__(self, dpi: int = 300, fontsize: int = 14):
        """
        Initialize equation renderer.

        Parameters
        ----------
        dpi : int, optional
            Resolution for PNG output (default: 300)
        fontsize : int, optional
            Font size for equations (default: 14)
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for equation rendering. "
                "Install with: pip install matplotlib"
            )

        self.dpi = dpi
        self.fontsize = fontsize

    def clean_equation(self, equation: str) -> str:
        """
        Clean LaTeX equation for rendering.

        Removes:
        - Environment tags (begin/end)
        - Labels
        - Comments
        - Excessive whitespace

        Parameters
        ----------
        equation : str
            Raw LaTeX equation

        Returns
        -------
        str
            Cleaned equation ready for rendering
        """
        # Remove begin/end tags
        equation = re.sub(r"\\begin\{(equation|align|eqnarray|split|gathered)\*?\}", "", equation)
        equation = re.sub(r"\\end\{(equation|align|eqnarray|split|gathered)\*?\}", "", equation)

        # Remove labels
        equation = re.sub(r"\\label\{[^}]+\}", "", equation)

        # Remove comments
        equation = re.sub(r"%.*$", "", equation, flags=re.MULTILINE)

        # Remove display math delimiters
        equation = equation.replace(r"\[", "").replace(r"\]", "")
        equation = equation.replace("$$", "")

        # Clean whitespace
        equation = equation.strip()

        # Handle align environments - keep only first line or use cases
        if "&" in equation or "\\\\" in equation:
            # For aligned equations, split and take non-empty lines
            lines = [line.strip() for line in equation.split("\\\\")]
            lines = [line for line in lines if line and not line.startswith("%")]
            if lines:
                # Try to render all lines as aligned system
                equation = " \\\\ ".join(lines)

        return equation

    def render_equation(
        self,
        equation: str,
        output_path: str,
        format: str = "png",
        transparent: bool = True,
        clean: bool = True,
    ) -> bool:
        """
        Render a single equation to an image file.

        Parameters
        ----------
        equation : str
            LaTeX equation string
        output_path : str
            Output file path
        format : str, optional
            Output format ('png' or 'svg'), default: 'png'
        transparent : bool, optional
            Transparent background (default: True)
        clean : bool, optional
            Clean equation before rendering (default: True)

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if clean:
            equation = self.clean_equation(equation)

        if not equation or len(equation.strip()) == 0:
            return False

        try:
            # Create figure
            fig = plt.figure(figsize=(10, 2))
            fig.patch.set_alpha(0 if transparent else 1)

            # Render equation
            plt.text(
                0.5,
                0.5,
                f"${equation}$",
                fontsize=self.fontsize,
                ha="center",
                va="center",
            )
            plt.axis("off")

            # Tight layout
            plt.tight_layout(pad=0.1)

            # Save
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            plt.savefig(
                output_path,
                dpi=self.dpi,
                bbox_inches="tight",
                format=format,
                transparent=transparent,
            )
            plt.close()

            return True

        except Exception as e:
            print(f"Error rendering equation: {e}")
            plt.close()
            return False

    def render_equations_batch(
        self,
        equations: List[Dict],
        output_dir: str,
        format: str = "png",
        prefix: str = "eq",
        max_equations: Optional[int] = None,
    ) -> Dict[str, any]:
        """
        Render multiple equations from extracted equation list.

        Parameters
        ----------
        equations : list of dict
            List of equation dicts from ContentExtractor
            Each dict should have 'equation', 'line_number', 'label' keys
        output_dir : str
            Output directory for rendered images
        format : str, optional
            Output format ('png' or 'svg'), default: 'png'
        prefix : str, optional
            Filename prefix (default: 'eq')
        max_equations : int, optional
            Maximum number of equations to render (default: all)

        Returns
        -------
        dict
            Results with:
            - rendered: Number of successfully rendered equations
            - failed: Number of failed equations
            - output_files: List of output file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {"rendered": 0, "failed": 0, "output_files": []}

        equations_to_render = equations[:max_equations] if max_equations else equations

        for i, eq_dict in enumerate(equations_to_render, start=1):
            equation = eq_dict.get("equation", "")

            # Determine filename
            if eq_dict.get("label"):
                filename = f"{prefix}_{eq_dict['label']}.{format}"
            else:
                filename = f"{prefix}_{i:04d}.{format}"

            output_path = output_dir / filename

            # Render
            success = self.render_equation(equation, str(output_path), format=format)

            if success:
                results["rendered"] += 1
                results["output_files"].append(str(output_path))
            else:
                results["failed"] += 1

        return results

    def render_equation_grid(
        self,
        equations: List[str],
        output_path: str,
        cols: int = 3,
        format: str = "png",
    ) -> bool:
        """
        Render multiple equations in a grid layout.

        Parameters
        ----------
        equations : list of str
            List of LaTeX equation strings
        output_path : str
            Output file path
        cols : int, optional
            Number of columns (default: 3)
        format : str, optional
            Output format ('png' or 'svg'), default: 'png'

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if not equations:
            return False

        # Calculate grid dimensions
        rows = (len(equations) + cols - 1) // cols

        try:
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 2 * rows))
            fig.patch.set_alpha(0)

            # Flatten axes for easier iteration
            if rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()

            # Render each equation
            for idx, equation in enumerate(equations):
                ax = axes[idx]
                cleaned = self.clean_equation(equation)

                if cleaned:
                    ax.text(
                        0.5,
                        0.5,
                        f"${cleaned}$",
                        fontsize=self.fontsize,
                        ha="center",
                        va="center",
                    )
                ax.axis("off")

            # Hide unused subplots
            for idx in range(len(equations), len(axes)):
                axes[idx].axis("off")

            plt.tight_layout()

            # Save
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            plt.savefig(
                output_path, dpi=self.dpi, bbox_inches="tight", format=format, transparent=True
            )
            plt.close()

            return True

        except Exception as e:
            print(f"Error rendering equation grid: {e}")
            plt.close()
            return False
