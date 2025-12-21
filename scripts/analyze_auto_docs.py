"""
Analyze Thalia codebase to identify documentation that can be auto-generated.

This script scans the codebase and identifies:
1. Registry-based components (regions, pathways) - can generate component catalogs
2. Strategy factory functions - can generate learning strategy reference
3. Config classes - can generate configuration reference
4. Public API functions - can generate API reference

Output: Report of auto-generation opportunities with examples.
"""

import ast
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class RegistryItem:
    """Component registered in ComponentRegistry."""
    name: str
    aliases: List[str]
    class_name: str
    file_path: str
    docstring: str
    config_class: str


@dataclass
class StrategyFactory:
    """Factory function for creating learning strategies."""
    name: str
    parameters: List[str]
    return_type: str
    docstring: str
    file_path: str


@dataclass
class ConfigClass:
    """Configuration dataclass."""
    name: str
    fields: List[Tuple[str, str, str]]  # (name, type, default)
    docstring: str
    file_path: str


class AutoDocAnalyzer:
    """Analyze codebase for auto-documentation opportunities."""

    def __init__(self, src_dir: Path):
        self.src_dir = src_dir
        self.regions: List[RegistryItem] = []
        self.pathways: List[RegistryItem] = []
        self.strategies: List[StrategyFactory] = []
        self.configs: List[ConfigClass] = []

    def analyze(self):
        """Run full analysis."""
        print("Analyzing Thalia codebase for auto-documentation opportunities...\n")

        # 1. Find all @register_region decorators
        self._find_registered_components()

        # 2. Find all create_*_strategy functions
        self._find_strategy_factories()

        # 3. Find all Config dataclasses
        self._find_config_classes()

        # 4. Generate report
        self._generate_report()

    def _find_registered_components(self):
        """Find all @register_region and @register_pathway decorators."""
        for py_file in self.src_dir.rglob("*.py"):
            if "test" in str(py_file) or "__pycache__" in str(py_file):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                if "@register_region" in content:
                    self._extract_region_registration(py_file, content)
                if "@register_pathway" in content:
                    self._extract_pathway_registration(py_file, content)
            except Exception as e:
                print(f"Warning: Could not process {py_file}: {e}")

    def _extract_region_registration(self, file_path: Path, content: str):
        """Extract region registration details."""
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call):
                            if hasattr(decorator.func, 'id') and decorator.func.id == 'register_region':
                                # Extract name from first argument
                                name = ast.literal_eval(decorator.args[0]) if decorator.args else None
                                aliases = []
                                config_class = None

                                # Extract keyword arguments
                                for keyword in decorator.keywords:
                                    if keyword.arg == 'aliases':
                                        aliases = ast.literal_eval(keyword.value)
                                    elif keyword.arg == 'config_class':
                                        if hasattr(keyword.value, 'id'):
                                            config_class = keyword.value.id

                                docstring = ast.get_docstring(node) or "No docstring"

                                self.regions.append(RegistryItem(
                                    name=name or node.name,
                                    aliases=aliases,
                                    class_name=node.name,
                                    file_path=str(file_path.relative_to(self.src_dir.parent)),
                                    docstring=docstring.split('\n')[0][:100],  # First line, truncated
                                    config_class=config_class or "Unknown"
                                ))
        except Exception as e:
            print(f"Warning: Could not parse {file_path}: {e}")

    def _extract_pathway_registration(self, file_path: Path, content: str):
        """Extract pathway registration details."""
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call):
                            if hasattr(decorator.func, 'id') and decorator.func.id == 'register_pathway':
                                name = ast.literal_eval(decorator.args[0]) if decorator.args else None
                                aliases = []
                                config_class = None

                                for keyword in decorator.keywords:
                                    if keyword.arg == 'aliases':
                                        aliases = ast.literal_eval(keyword.value)
                                    elif keyword.arg == 'config_class':
                                        if hasattr(keyword.value, 'id'):
                                            config_class = keyword.value.id

                                docstring = ast.get_docstring(node) or "No docstring"

                                self.pathways.append(RegistryItem(
                                    name=name or node.name,
                                    aliases=aliases,
                                    class_name=node.name,
                                    file_path=str(file_path.relative_to(self.src_dir.parent)),
                                    docstring=docstring.split('\n')[0][:100],
                                    config_class=config_class or "Unknown"
                                ))
        except Exception as e:
            print(f"Warning: Could not parse {file_path}: {e}")

    def _find_strategy_factories(self):
        """Find all create_*_strategy and create_strategy functions."""
        strategy_files = [
            self.src_dir / "learning" / "strategy_registry.py",
            self.src_dir / "learning" / "rules" / "strategies.py",
        ]

        for py_file in strategy_files:
            if not py_file.exists():
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if node.name.startswith("create_") and "strategy" in node.name.lower():
                            params = [arg.arg for arg in node.args.args]
                            return_type = "LearningStrategy"  # Default
                            docstring = ast.get_docstring(node) or "No docstring"

                            self.strategies.append(StrategyFactory(
                                name=node.name,
                                parameters=params,
                                return_type=return_type,
                                docstring=docstring.split('\n')[0][:100],
                                file_path=str(py_file.relative_to(self.src_dir.parent))
                            ))
            except Exception as e:
                print(f"Warning: Could not parse {py_file}: {e}")

    def _find_config_classes(self):
        """Find all Config dataclasses."""
        for py_file in self.src_dir.rglob("config.py"):
            if "test" in str(py_file) or "__pycache__" in str(py_file):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        if node.name.endswith("Config"):
                            # Check if it's a dataclass
                            is_dataclass = any(
                                isinstance(d, ast.Name) and d.id == 'dataclass'
                                for d in node.decorator_list
                            )

                            if is_dataclass:
                                fields = []
                                for item in node.body:
                                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                                        field_name = item.target.id
                                        field_type = ast.unparse(item.annotation) if hasattr(ast, 'unparse') else "Any"
                                        default = ast.unparse(item.value) if item.value else "Required"
                                        fields.append((field_name, field_type, default))

                                docstring = ast.get_docstring(node) or "No docstring"

                                self.configs.append(ConfigClass(
                                    name=node.name,
                                    fields=fields[:5],  # First 5 fields for sample
                                    docstring=docstring.split('\n')[0][:100],
                                    file_path=str(py_file.relative_to(self.src_dir.parent))
                                ))
            except Exception as e:
                print(f"Warning: Could not parse {py_file}: {e}")

    def _generate_report(self):
        """Generate auto-documentation opportunities report."""
        print("=" * 80)
        print("AUTO-DOCUMENTATION OPPORTUNITIES REPORT")
        print("=" * 80)
        print()

        # 1. Component Registry Catalog
        print("1. COMPONENT REGISTRY CATALOG")
        print("-" * 80)
        print(f"Found {len(self.regions)} registered regions and {len(self.pathways)} registered pathways")
        print()
        print("Auto-generate: docs/api/COMPONENT_CATALOG.md")
        print()
        print("Example output:")
        print()
        print("### Registered Regions")
        print()
        if self.regions:
            for region in self.regions[:3]:  # Show first 3
                print(f"#### `{region.name}`")
                print(f"- **Class**: `{region.class_name}`")
                print(f"- **Aliases**: {', '.join(region.aliases) if region.aliases else 'None'}")
                print(f"- **Config**: `{region.config_class}`")
                print(f"- **File**: `{region.file_path}`")
                print(f"- **Description**: {region.docstring}")
                print()
        print(f"... and {len(self.regions) - 3} more regions" if len(self.regions) > 3 else "")
        print()

        # 2. Learning Strategy Reference
        print("2. LEARNING STRATEGY REFERENCE")
        print("-" * 80)
        print(f"Found {len(self.strategies)} strategy factory functions")
        print()
        print("Auto-generate: docs/api/LEARNING_STRATEGIES_API.md")
        print()
        print("Example output:")
        print()
        if self.strategies:
            for strategy in self.strategies[:3]:  # Show first 3
                print(f"### `{strategy.name}()`")
                print(f"- **Parameters**: {', '.join(strategy.parameters)}")
                print(f"- **Returns**: `{strategy.return_type}`")
                print(f"- **File**: `{strategy.file_path}`")
                print(f"- **Description**: {strategy.docstring}")
                print()
        print(f"... and {len(self.strategies) - 3} more strategies" if len(self.strategies) > 3 else "")
        print()

        # 3. Configuration Reference
        print("3. CONFIGURATION REFERENCE")
        print("-" * 80)
        print(f"Found {len(self.configs)} configuration dataclasses")
        print()
        print("Auto-generate: docs/api/CONFIGURATION_REFERENCE.md")
        print()
        print("Example output:")
        print()
        if self.configs:
            for config in self.configs[:2]:  # Show first 2
                print(f"### `{config.name}`")
                print(f"- **File**: `{config.file_path}`")
                print(f"- **Description**: {config.docstring}")
                print()
                print("**Fields**:")
                for name, type_, default in config.fields:
                    print(f"- `{name}: {type_}` = `{default}`")
                print()
        print(f"... and {len(self.configs) - 2} more configs" if len(self.configs) > 2 else "")
        print()

        # Summary
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print()
        print("Recommended auto-generated documentation:")
        print()
        print("1. ✅ COMPONENT_CATALOG.md")
        print(f"   - {len(self.regions)} regions")
        print(f"   - {len(self.pathways)} pathways")
        print("   - Always up-to-date with @register_region/@register_pathway decorators")
        print()
        print("2. ✅ LEARNING_STRATEGIES_API.md")
        print(f"   - {len(self.strategies)} factory functions")
        print("   - Parameter signatures from AST")
        print("   - Always matches actual function signatures")
        print()
        print("3. ✅ CONFIGURATION_REFERENCE.md")
        print(f"   - {len(self.configs)} config classes")
        print("   - All fields with types and defaults")
        print("   - Generated from dataclass definitions")
        print()
        print("Benefits:")
        print("- ✅ Always synchronized with code")
        print("- ✅ No manual maintenance required")
        print("- ✅ Catches undocumented components")
        print("- ✅ Consistent formatting")
        print()
        print("Recommendation: Create scripts/generate_api_docs.py to build these")
        print()


def main():
    """Run analysis."""
    src_dir = Path(__file__).parent.parent / "src" / "thalia"

    if not src_dir.exists():
        print(f"Error: Source directory not found: {src_dir}")
        return

    analyzer = AutoDocAnalyzer(src_dir)
    analyzer.analyze()


if __name__ == "__main__":
    main()
