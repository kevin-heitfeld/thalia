"""
Performance profiling for Thalia components.

This script measures execution time, memory usage, and complexity
for all regions and pathways in the framework.

Usage:
    python scripts/profile_components.py

Output:
    - Console report
    - JSON file: profile_results.json
"""

import ast
import json

# Add src to path
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict

import torch

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent / "src"))

from thalia.config import GlobalConfig
from thalia.core.brain_builder import BrainBuilder


class ComponentProfiler:
    """Profile component performance metrics."""

    def __init__(self, src_dir: Path):
        self.src_dir = src_dir
        self.results: Dict[str, Dict[str, Any]] = {}

    def profile_execution(
        self, component_name: str, component_fn: Any, n_iterations: int = 100
    ) -> Dict[str, float]:
        """Profile execution time and memory usage.

        Args:
            component_name: Name for display
            component_fn: Callable that takes no args and executes the component
            n_iterations: Number of iterations to profile
        """
        # Warm-up
        for _ in range(10):
            _ = component_fn()

        # Memory tracking
        tracemalloc.start()
        mem_before = tracemalloc.get_traced_memory()[0]

        # Time execution
        times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            _ = component_fn()
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms

        mem_after = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        # Calculate statistics
        times_sorted = sorted(times)
        return {
            "mean_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "median_ms": times_sorted[len(times_sorted) // 2],
            "p95_ms": times_sorted[int(len(times_sorted) * 0.95)],
            "p99_ms": times_sorted[int(len(times_sorted) * 0.99)],
            "memory_kb": (mem_after - mem_before) / 1024,
        }

    def analyze_code_complexity(self, file_path: Path) -> Dict[str, int]:
        """Analyze code complexity metrics."""
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError:
                return {"lines": 0, "functions": 0, "classes": 0, "complexity": 0}

        lines = sum(1 for _ in open(file_path, "r", encoding="utf-8"))
        functions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))

        # Cyclomatic complexity (simplified: count branches)
        complexity = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return {
            "lines": lines,
            "functions": functions,
            "classes": classes,
            "complexity": complexity,
        }

    def profile_brain_components(self) -> None:
        """Profile all components in a default brain."""
        print("=" * 60)
        print("COMPONENT PERFORMANCE PROFILING")
        print("=" * 60)
        print()

        # Create test brain
        config = GlobalConfig(device="cpu", dt_ms=1.0)
        brain = BrainBuilder.preset("default", config)

        print(f"Profiling {len(brain.components)} components via brain.forward()...")
        print("(Measuring full forward pass performance)")
        print()

        # Warm-up
        sensory_input = torch.rand(128, device=config.device) > 0.5
        for _ in range(10):
            brain.forward(sensory_input, n_timesteps=1)
        brain.reset_state()

        # Profile full forward pass
        times = []
        for _ in range(100):
            sensory_input = torch.rand(128, device=config.device) > 0.5
            start = time.perf_counter()
            brain.forward(sensory_input, n_timesteps=1)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)
            brain.reset_state()

        times_sorted = sorted(times)
        full_forward_metrics = {
            "mean_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "median_ms": times_sorted[len(times_sorted) // 2],
            "p95_ms": times_sorted[int(len(times_sorted) * 0.95)],
            "p99_ms": times_sorted[int(len(times_sorted) * 0.99)],
        }

        print(f"Full forward pass: {full_forward_metrics['mean_ms']:.3f} ms (mean)")
        print()

        # Store results for each component
        for comp_name, component in brain.components.items():
            component_type = type(component).__name__
            module_path = Path(component.__class__.__module__.replace(".", "/") + ".py")
            file_path = self.src_dir.parent / "src" / module_path

            if file_path.exists():
                code_metrics = self.analyze_code_complexity(file_path)
            else:
                code_metrics = {"lines": 0, "functions": 0, "classes": 0, "complexity": 0}

            # Store code metrics only (execution profiled at brain level)
            self.results[comp_name] = {
                "type": component_type,
                "execution": None,  # Not profiled individually
                "code": code_metrics,
            }

        # Store full forward pass metrics
        self.results["_full_forward_pass"] = {
            "type": "DynamicBrain.forward",
            "execution": full_forward_metrics,
            "code": {"lines": 0, "functions": 0, "classes": 0, "complexity": 0},
        }

        print()
        print("Profiling complete!")

    def profile_region_implementations(self) -> None:
        """Profile all region implementations."""
        print()
        print("=" * 60)
        print("REGION IMPLEMENTATION ANALYSIS")
        print("=" * 60)
        print()

        regions_dir = self.src_dir / "regions"
        if not regions_dir.exists():
            print("Regions directory not found")
            return

        region_files = list(regions_dir.glob("*.py"))
        print(f"Analyzing {len(region_files)} region files...")
        print()

        for region_file in region_files:
            if region_file.name.startswith("__"):
                continue

            region_name = region_file.stem
            print(f"Analyzing {region_name}...")

            code_metrics = self.analyze_code_complexity(region_file)

            self.results[f"region_{region_name}"] = {
                "type": "region_implementation",
                "execution": None,  # Not profiled in isolation
                "code": code_metrics,
            }

    def profile_pathway_implementations(self) -> None:
        """Profile all pathway implementations."""
        print()
        print("=" * 60)
        print("PATHWAY IMPLEMENTATION ANALYSIS")
        print("=" * 60)
        print()

        pathways_dir = self.src_dir / "pathways"
        if not pathways_dir.exists():
            print("Pathways directory not found")
            return

        pathway_files = list(pathways_dir.glob("*.py"))
        print(f"Analyzing {len(pathway_files)} pathway files...")
        print()

        for pathway_file in pathway_files:
            if pathway_file.name.startswith("__"):
                continue

            pathway_name = pathway_file.stem
            print(f"Analyzing {pathway_name}...")

            code_metrics = self.analyze_code_complexity(pathway_file)

            self.results[f"pathway_{pathway_name}"] = {
                "type": "pathway_implementation",
                "execution": None,  # Not profiled in isolation
                "code": code_metrics,
            }

    def generate_report(self) -> None:
        """Generate performance report."""
        print()
        print("=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        print()

        # Full forward pass timing
        if "_full_forward_pass" in self.results:
            exec_m = self.results["_full_forward_pass"]["execution"]
            print("Full Forward Pass (1 timestep):")
            print("-" * 60)
            print(f"{'Metric':<20} {'Value':<12}")
            print("-" * 60)
            print(f"{'Mean':<20} {exec_m['mean_ms']:>11.3f} ms")
            print(f"{'Median':<20} {exec_m['median_ms']:>11.3f} ms")
            print(f"{'Min':<20} {exec_m['min_ms']:>11.3f} ms")
            print(f"{'Max':<20} {exec_m['max_ms']:>11.3f} ms")
            print(f"{'P95':<20} {exec_m['p95_ms']:>11.3f} ms")
            print(f"{'P99':<20} {exec_m['p99_ms']:>11.3f} ms")
            print()

        # Code complexity
        brain_components = {
            k: v
            for k, v in self.results.items()
            if not k.startswith("region_")
            and not k.startswith("pathway_")
            and k != "_full_forward_pass"
        }

        if brain_components:
            print("Brain Components - Code Complexity:")
            print("-" * 60)
            print(f"{'Component':<30} {'Lines':<8} {'Funcs':<8} {'Classes':<8} {'Complex':<8}")
            print("-" * 60)

            for comp_name, metrics in sorted(brain_components.items()):
                if metrics["code"]["lines"] > 0:
                    code_m = metrics["code"]
                    print(
                        f"{comp_name:<30} {code_m['lines']:>7} {code_m['functions']:>7} "
                        f"{code_m['classes']:>7} {code_m['complexity']:>7}"
                    )
            print()

        # Implementation complexity
        print("Implementation Files - Code Complexity:")
        print("-" * 60)
        print(f"{'Component':<30} {'Lines':<8} {'Funcs':<8} {'Classes':<8} {'Complex':<8}")
        print("-" * 60)

        for comp_name, metrics in sorted(self.results.items()):
            if (comp_name.startswith("region_") or comp_name.startswith("pathway_")) and metrics[
                "code"
            ]["lines"] > 0:
                code_m = metrics["code"]
                display_name = comp_name.replace("region_", "").replace("pathway_", "")
                print(
                    f"{display_name:<30} {code_m['lines']:>7} {code_m['functions']:>7} "
                    f"{code_m['classes']:>7} {code_m['complexity']:>7}"
                )
        print()

    def save_results(self, output_file: Path) -> None:
        """Save profiling results to JSON."""
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to: {output_file}")


def main():
    """Run profiling suite."""
    script_dir = Path(__file__).parent
    src_dir = script_dir.parent / "src" / "thalia"
    output_file = script_dir.parent / "profile_results.json"

    profiler = ComponentProfiler(src_dir)

    # Profile brain components
    profiler.profile_brain_components()

    # Analyze region implementations
    profiler.profile_region_implementations()

    # Analyze pathway implementations
    profiler.profile_pathway_implementations()

    # Generate report
    profiler.generate_report()

    # Save results
    profiler.save_results(output_file)


if __name__ == "__main__":
    main()
