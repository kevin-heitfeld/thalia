"""
Generate API documentation from Thalia codebase.

This script auto-generates:
1. COMPONENT_CATALOG.md - All registered regions and pathways
2. LEARNING_STRATEGIES_API.md - All learning strategy factory functions
3. CONFIGURATION_REFERENCE.md - All configuration dataclasses

Run this script whenever components are added/modified to keep docs synchronized.
"""

import ast
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
from datetime import datetime


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


class APIDocGenerator:
    """Generate API documentation from codebase."""

    def __init__(self, src_dir: Path, docs_dir: Path):
        self.src_dir = src_dir
        self.docs_dir = docs_dir
        self.api_dir = docs_dir / "api"
        self.regions: List[RegistryItem] = []
        self.pathways: List[RegistryItem] = []
        self.strategies: List[StrategyFactory] = []
        self.configs: List[ConfigClass] = []

    def generate(self):
        """Generate all API documentation."""
        print("Generating API documentation from Thalia codebase...\n")

        # Ensure api directory exists
        self.api_dir.mkdir(exist_ok=True)

        # Extract data from code
        self._find_registered_components()
        self._find_strategy_factories()
        self._find_config_classes()

        # Generate documentation files
        self._generate_component_catalog()
        self._generate_learning_strategies_api()
        self._generate_configuration_reference()

        print("\n✅ API documentation generated successfully!")
        print(f"   Location: {self.api_dir.relative_to(self.docs_dir.parent)}")

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

                                self.regions.append(RegistryItem(
                                    name=name or node.name,
                                    aliases=aliases,
                                    class_name=node.name,
                                    file_path=str(file_path.relative_to(self.src_dir.parent)),
                                    docstring=docstring.split('\n')[0],  # First line only
                                    config_class=config_class or "None"
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
                                    docstring=docstring.split('\n')[0],
                                    config_class=config_class or "None"
                                ))
        except Exception as e:
            print(f"Warning: Could not parse {file_path}: {e}")

    def _find_strategy_factories(self):
        """Find all create_*_strategy functions."""
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
                            return_type = "LearningStrategy"
                            docstring = ast.get_docstring(node) or "No docstring"

                            self.strategies.append(StrategyFactory(
                                name=node.name,
                                parameters=params,
                                return_type=return_type,
                                docstring=docstring.split('\n')[0],
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
                                        default = ast.unparse(item.value) if item.value else "**Required**"
                                        fields.append((field_name, field_type, default))

                                docstring = ast.get_docstring(node) or "No docstring"

                                self.configs.append(ConfigClass(
                                    name=node.name,
                                    fields=fields,
                                    docstring=docstring.split('\n')[0],
                                    file_path=str(py_file.relative_to(self.src_dir.parent))
                                ))
            except Exception as e:
                print(f"Warning: Could not parse {py_file}: {e}")

    def _generate_component_catalog(self):
        """Generate COMPONENT_CATALOG.md."""
        output_file = self.api_dir / "COMPONENT_CATALOG.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with output_file.open("w", encoding="utf-8") as f:
            f.write("# Component Catalog\n\n")
            f.write("> **Auto-generated documentation** - Do not edit manually!\n")
            f.write(f"> Last updated: {timestamp}\n")
            f.write("> Generated from: `scripts/generate_api_docs.py`\n\n")

            f.write("This document catalogs all registered brain regions and pathways ")
            f.write("in the Thalia component registry.\n\n")

            # Table of contents
            f.write("## Contents\n\n")
            f.write("- [Registered Regions](#registered-regions)\n")
            f.write("- [Registered Pathways](#registered-pathways)\n\n")

            # Regions
            f.write("## Registered Regions\n\n")
            f.write(f"Total: {len(self.regions)} regions\n\n")

            # Sort regions alphabetically
            for region in sorted(self.regions, key=lambda r: r.name):
                f.write(f"### `{region.name}`\n\n")
                f.write(f"**Class**: `{region.class_name}`\n\n")

                if region.aliases:
                    f.write(f"**Aliases**: `{', '.join(region.aliases)}`\n\n")

                f.write(f"**Config Class**: `{region.config_class}`\n\n")
                f.write(f"**Source**: `{region.file_path}`\n\n")
                f.write(f"**Description**: {region.docstring}\n\n")

                f.write("---\n\n")

            # Pathways
            f.write("## Registered Pathways\n\n")
            f.write(f"Total: {len(self.pathways)} pathways\n\n")

            for pathway in sorted(self.pathways, key=lambda p: p.name):
                f.write(f"### `{pathway.name}`\n\n")
                f.write(f"**Class**: `{pathway.class_name}`\n\n")

                if pathway.aliases:
                    f.write(f"**Aliases**: `{', '.join(pathway.aliases)}`\n\n")

                f.write(f"**Config Class**: `{pathway.config_class}`\n\n")
                f.write(f"**Source**: `{pathway.file_path}`\n\n")
                f.write(f"**Description**: {pathway.docstring}\n\n")

                f.write("---\n\n")

        print(f"✅ Generated: {output_file.relative_to(self.docs_dir.parent)}")

    def _generate_learning_strategies_api(self):
        """Generate LEARNING_STRATEGIES_API.md."""
        output_file = self.api_dir / "LEARNING_STRATEGIES_API.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with output_file.open("w", encoding="utf-8") as f:
            f.write("# Learning Strategies API\n\n")
            f.write("> **Auto-generated documentation** - Do not edit manually!\n")
            f.write(f"> Last updated: {timestamp}\n")
            f.write("> Generated from: `scripts/generate_api_docs.py`\n\n")

            f.write("This document catalogs all learning strategy factory functions ")
            f.write("available in Thalia.\n\n")

            f.write(f"Total: {len(self.strategies)} factory functions\n\n")

            f.write("## Factory Functions\n\n")

            for strategy in sorted(self.strategies, key=lambda s: s.name):
                f.write(f"### `{strategy.name}()`\n\n")
                f.write(f"**Returns**: `{strategy.return_type}`\n\n")
                f.write(f"**Source**: `{strategy.file_path}`\n\n")

                if strategy.parameters:
                    f.write("**Parameters**:\n\n")
                    for param in strategy.parameters:
                        f.write(f"- `{param}`\n")
                    f.write("\n")

                f.write(f"**Description**: {strategy.docstring}\n\n")

                f.write("---\n\n")

        print(f"✅ Generated: {output_file.relative_to(self.docs_dir.parent)}")

    def _generate_configuration_reference(self):
        """Generate CONFIGURATION_REFERENCE.md."""
        output_file = self.api_dir / "CONFIGURATION_REFERENCE.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with output_file.open("w", encoding="utf-8") as f:
            f.write("# Configuration Reference\n\n")
            f.write("> **Auto-generated documentation** - Do not edit manually!\n")
            f.write(f"> Last updated: {timestamp}\n")
            f.write("> Generated from: `scripts/generate_api_docs.py`\n\n")

            f.write("This document catalogs all configuration dataclasses in Thalia.\n\n")

            f.write(f"Total: {len(self.configs)} configuration classes\n\n")

            f.write("## Configuration Classes\n\n")

            for config in sorted(self.configs, key=lambda c: c.name):
                f.write(f"### `{config.name}`\n\n")
                f.write(f"**Source**: `{config.file_path}`\n\n")
                f.write(f"**Description**: {config.docstring}\n\n")

                if config.fields:
                    f.write("**Fields**:\n\n")
                    f.write("| Field | Type | Default |\n")
                    f.write("|-------|------|-------|\n")
                    for name, type_, default in config.fields:
                        # Escape pipe characters in default values
                        default_escaped = default.replace("|", "\\|")
                        f.write(f"| `{name}` | `{type_}` | `{default_escaped}` |\n")
                    f.write("\n")

                f.write("---\n\n")

        print(f"✅ Generated: {output_file.relative_to(self.docs_dir.parent)}")


def main():
    """Run API documentation generation."""
    script_dir = Path(__file__).parent
    src_dir = script_dir.parent / "src" / "thalia"
    docs_dir = script_dir.parent / "docs"

    if not src_dir.exists():
        print(f"Error: Source directory not found: {src_dir}")
        return

    if not docs_dir.exists():
        print(f"Error: Docs directory not found: {docs_dir}")
        return

    generator = APIDocGenerator(src_dir, docs_dir)
    generator.generate()


if __name__ == "__main__":
    main()
