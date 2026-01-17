"""
Test coverage analysis for Thalia components.

This script runs pytest with coverage to get actual coverage metrics
and analyzes test quality for each component.

Usage:
    python scripts/analyze_coverage.py

Output:
    - Console report
    - JSON file: coverage_results.json
"""

import ast
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set


class CoverageAnalyzer:
    """Analyze test coverage for components."""

    def __init__(self, src_dir: Path, tests_dir: Path, root_dir: Path):
        self.src_dir = src_dir
        self.tests_dir = tests_dir
        self.root_dir = root_dir
        self.results: Dict[str, Dict[str, Any]] = {}
        self.component_files: Dict[str, Path] = {}
        self.test_files: Dict[str, List[Path]] = defaultdict(list)
        self.coverage_data: Dict[str, float] = {}

    def find_components(self) -> None:
        """Find all component implementations."""
        # Find regions
        regions_dir = self.src_dir / "regions"
        if regions_dir.exists():
            for py_file in regions_dir.glob("*.py"):
                if not py_file.name.startswith("__"):
                    self.component_files[f"regions.{py_file.stem}"] = py_file

        # Find pathways
        pathways_dir = self.src_dir / "pathways"
        if pathways_dir.exists():
            for py_file in pathways_dir.glob("*.py"):
                if not py_file.name.startswith("__"):
                    self.component_files[f"pathways.{py_file.stem}"] = py_file

        # Find core components
        core_dir = self.src_dir / "core"
        if core_dir.exists():
            for py_file in core_dir.glob("*.py"):
                if not py_file.name.startswith("__"):
                    self.component_files[f"core.{py_file.stem}"] = py_file

        # Find learning strategies
        learning_dir = self.src_dir / "learning"
        if learning_dir.exists():
            for py_file in learning_dir.glob("*.py"):
                if not py_file.name.startswith("__"):
                    self.component_files[f"learning.{py_file.stem}"] = py_file

    def find_test_files(self) -> None:
        """Find all test files."""
        for test_file in self.tests_dir.rglob("test_*.py"):
            # Extract component name from imports
            components = self._extract_tested_components(test_file)
            for comp in components:
                self.test_files[comp].append(test_file)

    def _extract_tested_components(self, test_file: Path) -> Set[str]:
        """Extract which components are tested in a file."""
        components = set()

        try:
            with open(test_file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except (SyntaxError, UnicodeDecodeError):
            return components

        # Look for imports from thalia
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("thalia"):
                    # Extract module path
                    parts = node.module.split(".")
                    if len(parts) >= 2:
                        # e.g., "thalia.regions.cortex" -> "regions.cortex"
                        component_key = ".".join(parts[1:])
                        components.add(component_key)

        return components

    def analyze_test_quality(self, test_file: Path) -> Dict[str, Any]:
        """Analyze test file quality metrics."""
        try:
            with open(test_file, "r", encoding="utf-8") as f:
                content = f.read()
                tree = ast.parse(content)
        except (SyntaxError, UnicodeDecodeError):
            return {
                "test_count": 0,
                "fixture_count": 0,
                "has_edge_cases": False,
                "has_integration": False,
            }

        # Count test functions
        test_count = 0
        fixture_count = 0
        has_edge_cases = False
        has_integration = False

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for test functions
                if node.name.startswith("test_"):
                    test_count += 1

                    # Check for edge case tests
                    if any(
                        keyword in node.name.lower()
                        for keyword in [
                            "edge",
                            "zero",
                            "empty",
                            "invalid",
                            "error",
                            "silent",
                            "saturated",
                        ]
                    ):
                        has_edge_cases = True

                    # Check for integration tests
                    if "integration" in node.name.lower():
                        has_integration = True

                # Check for fixtures
                if any(
                    isinstance(dec, ast.Name) and dec.id == "fixture" for dec in node.decorator_list
                ):
                    fixture_count += 1
                elif any(
                    isinstance(dec, ast.Attribute) and dec.attr == "fixture"
                    for dec in node.decorator_list
                ):
                    fixture_count += 1

        return {
            "test_count": test_count,
            "fixture_count": fixture_count,
            "has_edge_cases": has_edge_cases,
            "has_integration": has_integration,
        }

    def run_pytest_coverage(self) -> None:
        """Run pytest with coverage to get actual coverage metrics."""
        print("=" * 60)
        print("RUNNING PYTEST WITH COVERAGE")
        print("=" * 60)
        print()
        print("This may take a few minutes...")
        print()

        # Run pytest with coverage
        json_report = self.root_dir / "coverage.json"

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "--cov=thalia",
            "--cov-report=json",
            "--cov-report=term-missing",
            "-q",  # Quiet mode
            str(self.tests_dir),
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.root_dir),
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                check=False,
            )

            if result.returncode != 0 and result.returncode != 1:
                # Return code 1 is OK (some tests failed), 0 is all passed
                print(f"Warning: pytest returned exit code {result.returncode}")
                print("STDERR:", result.stderr[:500])

            # Parse coverage JSON report
            if json_report.exists():
                with open(json_report, "r", encoding="utf-8") as f:
                    coverage_report = json.load(f)

                # Extract per-file coverage
                for file_path, file_data in coverage_report.get("files", {}).items():
                    # Convert absolute path to relative module path
                    file_path_obj = Path(file_path)
                    if "thalia" in file_path:
                        # Extract module name from path
                        parts = file_path_obj.parts
                        try:
                            thalia_idx = parts.index("thalia")
                            module_parts = parts[thalia_idx + 1 :]
                            if module_parts and module_parts[-1].endswith(".py"):
                                module_parts = list(module_parts[:-1]) + [module_parts[-1][:-3]]
                            module_name = ".".join(module_parts)

                            coverage_pct = file_data["summary"]["percent_covered"]
                            self.coverage_data[module_name] = coverage_pct
                        except (ValueError, IndexError):
                            continue

                print(f"Parsed coverage for {len(self.coverage_data)} files")
            else:
                print("Warning: coverage.json not found")

        except subprocess.TimeoutExpired:
            print("Error: pytest coverage timed out after 10 minutes")
        except FileNotFoundError:
            print("Error: pytest not found. Install with: pip install pytest pytest-cov")
        except Exception as e:
            print(f"Error running pytest: {e}")

        print()

    def analyze_component_coverage(self) -> None:
        """Analyze coverage for each component."""
        print("=" * 60)
        print("TEST COVERAGE ANALYSIS")
        print("=" * 60)
        print()

        for comp_name, comp_file in self.component_files.items():
            # Check if component has tests
            test_files = self.test_files.get(comp_name, [])
            has_tests = len(test_files) > 0

            # Analyze test quality
            test_metrics = {
                "total_tests": 0,
                "total_fixtures": 0,
                "has_edge_cases": False,
                "has_integration": False,
                "test_files": [],
            }

            for test_file in test_files:
                metrics = self.analyze_test_quality(test_file)
                test_metrics["total_tests"] += metrics["test_count"]
                test_metrics["total_fixtures"] += metrics["fixture_count"]
                test_metrics["has_edge_cases"] = (
                    test_metrics["has_edge_cases"] or metrics["has_edge_cases"]
                )
                test_metrics["has_integration"] = (
                    test_metrics["has_integration"] or metrics["has_integration"]
                )
                test_metrics["test_files"].append(str(test_file.relative_to(self.tests_dir)))

            # Count functions/classes in component
            try:
                with open(comp_file, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                functions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
                classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
            except (SyntaxError, UnicodeDecodeError):
                functions = 0
                classes = 0

            # Get actual coverage from pytest
            actual_coverage = self.coverage_data.get(comp_name, 0.0)

            self.results[comp_name] = {
                "has_tests": has_tests,
                "test_files": test_metrics["test_files"],
                "test_count": test_metrics["total_tests"],
                "fixture_count": test_metrics["total_fixtures"],
                "has_edge_cases": test_metrics["has_edge_cases"],
                "has_integration": test_metrics["has_integration"],
                "functions": functions,
                "classes": classes,
                "coverage_percent": actual_coverage,
            }

    def generate_report(self) -> None:
        """Generate coverage report."""
        print()
        print("=" * 60)
        print("COVERAGE SUMMARY")
        print("=" * 60)
        print()

        # Separate by category
        regions = {k: v for k, v in self.results.items() if k.startswith("regions.")}
        pathways = {k: v for k, v in self.results.items() if k.startswith("pathways.")}
        core = {k: v for k, v in self.results.items() if k.startswith("core.")}
        learning = {k: v for k, v in self.results.items() if k.startswith("learning.")}

        # Print category summaries
        for category_name, category_items in [
            ("Regions", regions),
            ("Pathways", pathways),
            ("Core", core),
            ("Learning", learning),
        ]:
            if not category_items:
                continue

            print(f"{category_name}:")
            print("-" * 60)
            print(f"{'Component':<30} {'Tests':<8} {'Edge':<8} {'Integ':<8} {'Cov%':<8}")
            print("-" * 60)

            for comp_name, metrics in sorted(category_items.items()):
                display_name = comp_name.split(".", 1)[1]  # Remove prefix
                edge_mark = "✓" if metrics["has_edge_cases"] else "✗"
                integ_mark = "✓" if metrics["has_integration"] else "✗"
                coverage_str = (
                    f"{metrics['coverage_percent']:.1f}%"
                    if metrics["coverage_percent"] > 0
                    else "0%"
                )

                print(
                    f"{display_name:<30} {metrics['test_count']:>7} {edge_mark:>7} "
                    f"{integ_mark:>7} {coverage_str:>7}"
                )
            print()

        # Overall statistics
        total_components = len(self.results)
        tested_components = sum(1 for m in self.results.values() if m["has_tests"])
        total_tests = sum(m["test_count"] for m in self.results.values())

        print("Overall Statistics:")
        print("-" * 60)
        print(f"Total components: {total_components}")
        print(
            f"Components with tests: {tested_components} ({tested_components/total_components*100:.1f}%)"
        )
        print(f"Total test functions: {total_tests}")
        print(
            f"Components with edge cases: {sum(1 for m in self.results.values() if m['has_edge_cases'])}"
        )
        print(
            f"Components with integration tests: {sum(1 for m in self.results.values() if m['has_integration'])}"
        )
        print()

        # Untested components
        untested = [k for k, v in self.results.items() if not v["has_tests"]]
        if untested:
            print("Untested Components:")
            print("-" * 60)
            for comp in sorted(untested):
                print(f"  - {comp}")
            print()

    def save_results(self, output_file: Path) -> None:
        """Save coverage results to JSON."""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to: {output_file}")


def main():
    """Run coverage analysis."""
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    src_dir = root_dir / "src" / "thalia"
    tests_dir = root_dir / "tests"
    output_file = root_dir / "coverage_results.json"

    analyzer = CoverageAnalyzer(src_dir, tests_dir, root_dir)

    # Find components and tests
    print("Scanning components...")
    analyzer.find_components()
    print(f"Found {len(analyzer.component_files)} components")

    print("Scanning test files...")
    analyzer.find_test_files()
    print(f"Found {len(analyzer.test_files)} tested components")
    print()

    # Run pytest with coverage
    analyzer.run_pytest_coverage()

    # Analyze coverage
    analyzer.analyze_component_coverage()

    # Generate report
    analyzer.generate_report()

    # Save results
    analyzer.save_results(output_file)


if __name__ == "__main__":
    main()
