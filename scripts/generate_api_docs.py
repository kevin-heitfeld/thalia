"""
Generate API documentation from Thalia codebase.

Run this script whenever components are added/modified to keep docs synchronized.
"""

import ast
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from thalia.brain import NeuralRegionRegistry
from thalia.components.neurons import NeuronFactory


@dataclass
class RegistryItem:
    """Component registered in NeuralRegionRegistry."""

    name: str
    aliases: List[str]
    class_name: str
    file_path: str
    docstring: str
    config_class: str
    line_number: int
    config_file_path: Optional[str] = None
    config_line_number: Optional[int] = None


@dataclass
class StrategyFactory:
    """Factory function for creating learning strategies."""

    name: str
    parameters: List[Tuple[str, str, str]]  # (name, type, default)
    return_type: str
    docstring: str
    file_path: str
    line_number: int = 0
    examples: Optional[List[str]] = None
    used_by: Optional[List[str]] = None  # Components using this strategy

    def __post_init__(self):
        if self.examples is None:
            self.examples = []
        if self.used_by is None:
            self.used_by = []


@dataclass
class ConfigClass:
    """Configuration dataclass."""

    name: str
    fields: List[Tuple[str, str, str]]  # (name, type, default)
    docstring: str
    file_path: str
    line_number: int = 0
    used_by: Optional[List[Tuple[str, str, int]]] = None  # (region_name, file_path, line_number)
    examples: Optional[List[str]] = None

    def __post_init__(self):
        if self.used_by is None:
            self.used_by = []
        if self.examples is None:
            self.examples = []


@dataclass
class ModuleExport:
    """Public export from a module."""

    module_name: str
    export_name: str
    export_type: str  # "class", "function", "constant"
    file_path: str


@dataclass
class TypeAliasInfo:
    """Type alias definition."""

    name: str
    definition: str
    description: str
    file_path: str
    category: str  # "Component", "Routing", "State", etc.


@dataclass
class ComponentRelation:
    """Relationship between components in preset architectures."""

    source: str
    target: str
    preset_name: str
    source_port: Optional[str]
    target_port: Optional[str]


@dataclass
class NeuronFactoryInfo:
    """Neuron factory function."""

    name: str
    parameters: List[Tuple[str, str, str]]  # (name, type, default)
    return_type: str
    docstring: str
    file_path: str
    examples: List[str]  # Code examples
    line_number: int = 0


class APIDocGenerator:
    """Generate API documentation from codebase."""

    def __init__(self, src_dir: Path, docs_dir: Path):
        self.src_dir = src_dir
        self.docs_dir = docs_dir
        self.api_dir = docs_dir / "api"
        self.regions: List[RegistryItem] = []
        self.strategies: List[StrategyFactory] = []
        self.configs: List[ConfigClass] = []
        self.module_exports: Dict[str, List[ModuleExport]] = {}  # module_path -> exports
        self.type_aliases: List[TypeAliasInfo] = []
        self.component_relations: List[ComponentRelation] = []
        self.neuron_factories: List[NeuronFactoryInfo] = []
        self.dependencies: Dict[str, List[str]] = {}  # module -> imported modules

    def generate(self):
        """Generate all API documentation."""
        print("Generating API documentation from Thalia codebase...\n")

        # Ensure api directory exists
        self.api_dir.mkdir(exist_ok=True)

        # Extract data from code
        self._find_strategy_factories()
        self._find_config_classes()
        self._find_module_exports()
        self._find_neuron_factories()

        # Analyze dependencies and architecture
        self._analyze_dependencies()

        # Generate documentation files
        self._generate_all_registrations_doc()
        self._generate_configuration_reference()
        self._generate_learning_strategies_api()
        self._generate_module_exports_reference()
        self._generate_neuron_factories_reference()

        print("\n✅ API documentation generated successfully!")
        print(f"   Location: {self.api_dir.relative_to(self.docs_dir.parent)}")

    # =================================================================
    # Internal methods for documentation generation
    # =================================================================

    def _find_strategy_factories(self):
        """Find all create_*_strategy functions."""
        strategy_files = [
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
                            # Extract full parameter information
                            parameters = []
                            for arg in node.args.args:
                                arg_name = arg.arg
                                arg_type = "Any"
                                arg_default = ""

                                # Get type annotation
                                if arg.annotation:
                                    try:
                                        arg_type = ast.unparse(arg.annotation)
                                    except (AttributeError, TypeError):
                                        pass

                                # Get default value
                                defaults_offset = len(node.args.args) - len(node.args.defaults)
                                arg_idx = node.args.args.index(arg)
                                if arg_idx >= defaults_offset:
                                    default_idx = arg_idx - defaults_offset
                                    try:
                                        arg_default = ast.unparse(node.args.defaults[default_idx])
                                    except (AttributeError, TypeError, IndexError):
                                        pass

                                parameters.append((arg_name, arg_type, arg_default))

                            # Extract return type
                            return_type = "LearningStrategy"
                            if node.returns:
                                try:
                                    return_type = ast.unparse(node.returns)
                                except (AttributeError, TypeError):
                                    pass

                            docstring = ast.get_docstring(node) or "No docstring"

                            # Extract examples from docstring
                            examples = self._extract_examples_from_docstring(docstring)

                            self.strategies.append(
                                StrategyFactory(
                                    name=node.name,
                                    parameters=parameters,
                                    return_type=return_type,
                                    docstring=docstring.split("\n")[0],
                                    file_path=str(py_file.relative_to(self.src_dir.parent)),
                                    line_number=node.lineno,
                                    examples=examples,
                                )
                            )
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
                                isinstance(d, ast.Name) and d.id == "dataclass"
                                for d in node.decorator_list
                            )

                            if is_dataclass:
                                fields = []
                                for item in node.body:
                                    if isinstance(item, ast.AnnAssign) and isinstance(
                                        item.target, ast.Name
                                    ):
                                        field_name = item.target.id
                                        field_type = ast.unparse(item.annotation)

                                        # Parse default value intelligently
                                        if item.value:
                                            default_str = ast.unparse(item.value)
                                            # Clean up field() calls
                                            if "field(default" in default_str:
                                                # Extract just the default part
                                                if "default_factory=" in default_str:
                                                    # e.g., field(default_factory=dict) -> {}
                                                    if "dict" in default_str:
                                                        default = "{}"
                                                    elif "list" in default_str:
                                                        default = "[]"
                                                    elif "set" in default_str:
                                                        default = "set()"
                                                    else:
                                                        default = "(factory)"
                                                elif "default=" in default_str:
                                                    # Extract actual default value
                                                    import re

                                                    match = re.search(
                                                        r"default=([^,)]+)", default_str
                                                    )
                                                    default = (
                                                        match.group(1) if match else default_str
                                                    )
                                                else:
                                                    default = "field(...)"
                                            else:
                                                default = default_str
                                        else:
                                            default = "**Required**"

                                        fields.append((field_name, field_type, default))

                                docstring = ast.get_docstring(node) or "No docstring"

                                self.configs.append(
                                    ConfigClass(
                                        name=node.name,
                                        fields=fields,
                                        docstring=docstring.split("\n")[0],
                                        file_path=str(py_file.relative_to(self.src_dir.parent)),
                                        line_number=node.lineno,
                                    )
                                )
            except Exception as e:
                print(f"Warning: Could not parse {py_file}: {e}")

    def _find_module_exports(self):
        """Find all public exports from __init__.py files."""
        for init_file in self.src_dir.rglob("__init__.py"):
            if "__pycache__" in str(init_file):
                continue

            try:
                content = init_file.read_text(encoding="utf-8")
                tree = ast.parse(content)

                exports = []
                all_list = []

                # Find __all__ definition
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id == "__all__":
                                if isinstance(node.value, ast.List):
                                    all_list = [
                                        (
                                            elt.value
                                            if isinstance(elt, ast.Constant)
                                            else None
                                        )
                                        for elt in node.value.elts
                                    ]
                                    all_list = [x for x in all_list if x is not None]

                if all_list:
                    # Determine module path relative to thalia
                    module_path = str(init_file.parent.relative_to(self.src_dir.parent))
                    module_name = module_path.replace("\\", ".").replace("/", ".")

                    for export in all_list[:20]:  # Limit to first 20 to avoid huge lists
                        exports.append(
                            ModuleExport(
                                module_name=module_name,
                                export_name=export,
                                export_type="unknown",  # Would need import resolution
                                file_path=str(init_file.relative_to(self.src_dir.parent)),
                            )
                        )

                    if exports:
                        self.module_exports[module_name] = exports

            except Exception as e:
                print(f"Warning: Could not parse {init_file}: {e}")

    def _find_neuron_factories(self):
        """Find all neuron factory functions (create_*_neurons)."""
        factory_file = self.src_dir / "components" / "neurons" / "neuron_factory.py"

        if not factory_file.exists():
            return

        try:
            content = factory_file.read_text(encoding="utf-8")
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name.startswith("create_") and (
                        "neurons" in node.name or "neuron" in node.name
                    ):
                        # Extract function signature
                        parameters = []
                        for arg in node.args.args:
                            arg_name = arg.arg
                            arg_type = "Any"
                            arg_default = ""

                            # Get type annotation
                            if arg.annotation:
                                try:
                                    arg_type = ast.unparse(arg.annotation)
                                except (AttributeError, TypeError):
                                    pass

                            # Get default value
                            defaults_offset = len(node.args.args) - len(node.args.defaults)
                            arg_idx = node.args.args.index(arg)
                            if arg_idx >= defaults_offset:
                                default_idx = arg_idx - defaults_offset
                                try:
                                    arg_default = ast.unparse(node.args.defaults[default_idx])
                                except (AttributeError, TypeError, IndexError):
                                    pass

                            parameters.append((arg_name, arg_type, arg_default))

                        # Get return type
                        return_type = "ConductanceLIF"
                        if node.returns:
                            try:
                                return_type = ast.unparse(node.returns)
                            except (AttributeError, TypeError):
                                pass

                        # Extract docstring and examples
                        docstring = ast.get_docstring(node) or "No docstring"
                        examples = []

                        # Extract example code blocks from docstring
                        if docstring:
                            lines = docstring.split("\n")
                            in_example = False
                            example_lines = []

                            for line in lines:
                                if "Examples:" in line or ">>>" in line:
                                    in_example = True
                                    if ">>>" in line:
                                        example_lines = [line]
                                    continue

                                if in_example:
                                    if line.strip() and not line.strip().startswith(
                                        ("Args:", "Returns:", "Raises:")
                                    ):
                                        example_lines.append(line)
                                    elif (
                                        line.strip().startswith(("Args:", "Returns:", "Raises:"))
                                        and example_lines
                                    ):
                                        examples.append("\n".join(example_lines))
                                        example_lines = []
                                        in_example = False

                            if example_lines:
                                examples.append("\n".join(example_lines))

                        self.neuron_factories.append(
                            NeuronFactoryInfo(
                                name=node.name,
                                parameters=parameters,
                                return_type=return_type,
                                docstring=docstring.split("\n")[0],  # First line
                                file_path=str(factory_file.relative_to(self.src_dir.parent)),
                                examples=examples,
                            )
                        )

        except Exception as e:
            print(f"Warning: Could not parse {factory_file}: {e}")

    # =================================================================
    # Cross-reference and dependency analysis
    # =================================================================

    def _analyze_dependencies(self):
        """Analyze import dependencies between modules."""
        self.dependencies = {}  # module -> list of imported modules

        for py_file in self.src_dir.rglob("*.py"):
            if "__pycache__" in str(py_file) or "test" in str(py_file):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)

                module_path = py_file.relative_to(self.src_dir.parent)
                module_name = (
                    str(module_path).replace("\\", ".").replace("/", ".").replace(".py", "")
                )

                imports = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name.startswith("thalia"):
                                imports.add(
                                    alias.name.split(".")[1]
                                    if len(alias.name.split(".")) > 1
                                    else alias.name
                                )
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and node.module.startswith("thalia"):
                            parts = node.module.split(".")
                            if len(parts) > 1:
                                imports.add(parts[1])

                if imports:
                    self.dependencies[module_name] = list(imports)

            except Exception:
                continue

    # =================================================================
    # Documentation generation methods
    # =================================================================

    def _generate_all_registrations_doc(self):
        """Generate ALL_REGISTRATIONS.md."""
        output_file = self.api_dir / "ALL_REGISTRATIONS.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with output_file.open("w", encoding="utf-8") as f:
            f.write("# All Registrations\n\n")
            f.write("> **Auto-generated documentation** - Do not edit manually!\n")
            f.write(f"> Last updated: {timestamp}\n")
            f.write("> Generated from: `scripts/generate_api_docs.py`\n\n")

            f.write("## NeuronFactory\n\n")
            for name in NeuronFactory.list_neuron_types():
                f.write(f"- {name}\n")

            f.write("\n## NeuralRegionRegistry\n\n")
            for registry_name in NeuralRegionRegistry.list_regions():
                f.write(f"- {registry_name}\n")

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

            f.write(f"Total: **{len(self.strategies)}** factory functions\n\n")

            # Add complexity metrics
            total_params = sum(len(s.parameters) for s in self.strategies)
            avg_params = total_params / len(self.strategies) if self.strategies else 0
            strategies_with_examples = sum(1 for s in self.strategies if s.examples)

            f.write("## 📊 Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| **Total Strategies** | {len(self.strategies)} |\n")
            f.write(f"| **Average Parameters** | {avg_params:.1f} |\n")
            f.write(f"| **With Examples** | {strategies_with_examples}/{len(self.strategies)} |\n")
            f.write(
                f"| **With Usage Info** | {sum(1 for s in self.strategies if s.used_by)}/{len(self.strategies)} |\n"
            )
            f.write("\n")

            # Add badges
            f.write(
                f"![Strategies](https://img.shields.io/badge/Factories-{len(self.strategies)}-blue) "
            )
            f.write("![Learning](https://img.shields.io/badge/Type-Learning-orange) ")
            f.write("![Pluggable](https://img.shields.io/badge/Architecture-Pluggable-green)\n\n")

            # Add strategy type diagram
            f.write("## 🔄 Strategy Types\n\n")
            f.write("```mermaid\n")
            f.write("graph LR\n")
            f.write("    A[Learning Strategies] --> B[Local Rules]\n")
            f.write("    A --> C[Neuromodulated]\n")
            f.write("    A --> D[Multi-factor]\n")
            f.write("    B --> B1[STDP]\n")
            f.write("    B --> B2[Hebbian]\n")
            f.write("    B --> B3[BCM]\n")
            f.write("    C --> C1[Three-factor]\n")
            f.write("    D --> D1[Composite]\n")
            f.write("```\n\n")

            f.write("## 📚 Factory Functions\n\n")

            for strategy in sorted(self.strategies, key=lambda s: s.name):
                # Make function name clickable
                func_link = self._make_source_link(
                    strategy.file_path,
                    line_number=strategy.line_number,
                    display_text=f"`{strategy.name}()`",
                )
                f.write(f"### {func_link}\n\n")
                f.write(f"**Returns**: `{strategy.return_type}`\n\n")
                f.write(f"**Source**: {self._make_source_link(strategy.file_path)}\n\n")

                if strategy.parameters:
                    f.write("**Parameters**:\n\n")
                    f.write("| Parameter | Type | Default |\n")
                    f.write("|-----------|------|----------|\n")
                    for param_name, param_type, param_default in strategy.parameters:
                        # Check if parameter is a config and add cross-link
                        param_display = param_type
                        if "Config" in param_type:
                            # Find matching config
                            config_name = (
                                param_type.replace("Optional[", "").replace("]", "").strip()
                            )
                            matching_config = next(
                                (c for c in self.configs if c.name == config_name), None
                            )
                            if matching_config:
                                config_link = f"[`{config_name}`](CONFIGURATION_REFERENCE.md#{config_name.lower()})"
                                param_display = param_display.replace(config_name, config_link)

                        f.write(f"| `{param_name}` | {param_display} | `{param_default}` |\n")
                    f.write("\n")

                if strategy.examples:
                    f.write("**Examples**:\n\n")
                    for example in strategy.examples:
                        f.write(f"```python\n{example}\n```\n\n")

                f.write(f"**Description**: {strategy.docstring}\n\n")

                # Show typical usage
                if strategy.used_by:
                    f.write("**Typically Used By**: ")
                    f.write(", ".join(f"`{region}`" for region in strategy.used_by))
                    f.write("\n\n")

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
                # Make class name clickable
                class_link = self._make_source_link(
                    config.file_path,
                    line_number=config.line_number,
                    display_text=f"`{config.name}`",
                )
                f.write(f"### {class_link}\n\n")
                f.write(f"**Source**: {self._make_source_link(config.file_path)}\n\n")
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

                # Show which components use this config
                if config.used_by:
                    f.write("**Used By**:\n\n")
                    for comp_name, comp_file, comp_line in config.used_by:
                        comp_link = self._make_source_link(
                            comp_file, line_number=comp_line, display_text=comp_name
                        )
                        f.write(f"- {comp_link}\n")
                    f.write("\n")

                f.write("---\n\n")

        print(f"✅ Generated: {output_file.relative_to(self.docs_dir.parent)}")

    def _generate_module_exports_reference(self):
        """Generate MODULE_EXPORTS.md."""
        output_file = self.api_dir / "MODULE_EXPORTS.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with output_file.open("w", encoding="utf-8") as f:
            f.write("# Module Exports Reference\n\n")
            f.write("> **Auto-generated documentation** - Do not edit manually!\n")
            f.write(f"> Last updated: {timestamp}\n")
            f.write("> Generated from: `scripts/generate_api_docs.py`\n\n")

            f.write("This document catalogs all public exports (`__all__`) from Thalia modules. ")
            f.write("These are the recommended imports for external code.\n\n")

            total_exports = sum(len(exports) for exports in self.module_exports.values())
            f.write(f"Total: {len(self.module_exports)} modules, {total_exports} exports\n\n")

            # Add table of contents for navigation
            f.write("## 📑 Table of Contents\n\n")
            f.write("Quick jump to module:\n\n")
            for i, module_name in enumerate(sorted(self.module_exports.keys())):
                # Create anchor-safe module name
                anchor = module_name.replace(".", "")
                f.write(f"- [{module_name}](#{anchor})")
                # Add line break every 3 modules for readability
                if (i + 1) % 3 == 0:
                    f.write("\n")
                else:
                    f.write(" | ")
            f.write("\n\n")

            f.write("## Module Exports\n\n")

            for module_name in sorted(self.module_exports.keys()):
                exports = self.module_exports[module_name]
                f.write(f"### `{module_name}`\n\n")
                f.write(f"**Source**: {self._make_source_link(exports[0].file_path)}\n\n")
                f.write(f"**Exports** ({len(exports)}):\n\n")

                for export in exports:
                    f.write(f"- `{export.export_name}`\n")

                f.write("\n**Usage**:\n\n")
                f.write(f"```python\nfrom {module_name} import {exports[0].export_name}\n```\n\n")
                f.write("---\n\n")

        print(f"✅ Generated: {output_file.relative_to(self.docs_dir.parent)}")

    def _generate_neuron_factories_reference(self):
        """Generate NEURON_FACTORIES_REFERENCE.md."""
        output_file = self.api_dir / "NEURON_FACTORIES_REFERENCE.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with output_file.open("w", encoding="utf-8") as f:
            f.write("# Neuron Factories Reference\n\n")
            f.write("> **Auto-generated documentation** - Do not edit manually!\n")
            f.write(f"> Last updated: {timestamp}\n")
            f.write("> Generated from: `scripts/generate_api_docs.py`\n\n")

            f.write("This document catalogs all neuron factory functions for creating ")
            f.write("pre-configured neuron populations with biologically-motivated parameters.\n\n")

            f.write(f"Total: **{len(self.neuron_factories)}** factory functions\n\n")

            # Add badges
            f.write("![Factories](https://img.shields.io/badge/Factories-4-blue) ")
            f.write("![Tested](https://img.shields.io/badge/Status-Tested-success) ")
            f.write(
                "![Biological](https://img.shields.io/badge/Type-Biologically--Accurate-orange)\n\n"
            )

            f.write("## 💡 Why Use Neuron Factories?\n\n")
            f.write("Neuron factories provide:\n")
            f.write("- ✅ **Biological realism**: Parameters based on neuroscience literature\n")
            f.write("- ✅ **Consistency**: Standard configurations across the codebase\n")
            f.write("- ✅ **Customization**: Override defaults for specialized regions\n")
            f.write("- ✅ **Documentation**: Clear intent (pyramidal vs interneuron vs relay)\n\n")

            # Add neuron type comparison diagram
            f.write("## 🔬 Neuron Type Comparison\n\n")
            f.write("```mermaid\n")
            f.write("graph LR\n")
            f.write("    A[Neuron Factories] --> B[Pyramidal]\n")
            f.write("    A --> C[Relay]\n")
            f.write("    A --> D[TRN]\n")
            f.write("    A --> E[Cortical Layers]\n")
            f.write("    B --> B1[τ=20ms]\n")
            f.write("    C --> C1[Fast transmission]\n")
            f.write("    D --> D1[Inhibitory]\n")
            f.write("    E --> E1[L4/L2-3/L5/L6]\n")
            f.write("```\n\n")

            f.write("## 📚 Factory Functions\n\n")

            for factory in sorted(self.neuron_factories, key=lambda f: f.name):
                # Add complexity badge
                complexity = (
                    "🟢 Simple"
                    if len(factory.parameters) <= 3
                    else "🟡 Moderate" if len(factory.parameters) <= 5 else "🔴 Advanced"
                )
                # Make function name clickable
                func_link = self._make_source_link(
                    factory.file_path,
                    line_number=factory.line_number,
                    display_text=f"`{factory.name}()`",
                )
                f.write(f"### {func_link} {complexity}\n\n")
                f.write(f"**Returns**: `{factory.return_type}`  \n")
                f.write(f"**Source**: {self._make_source_link(factory.file_path)}\n\n")

                f.write(f"**Description**: {factory.docstring}\n\n")

                if factory.parameters:
                    f.write("**Parameters**:\n\n")
                    f.write("| Parameter | Type | Default | Description |\n")
                    f.write("|-----------|------|---------|-------------|\n")

                    for param_name, param_type, param_default in factory.parameters:
                        default_str = param_default if param_default else "—"
                        if param_name == "**overrides":
                            f.write(
                                "| `**overrides` | `Any` | — | Custom parameters to override defaults |\n"
                            )
                        else:
                            f.write(f"| `{param_name}` | `{param_type}` | `{default_str}` | |\n")

                    f.write("\n")

                if factory.examples:
                    f.write("**Examples**:\n\n")
                    for example in factory.examples:
                        f.write("```python\n")
                        f.write(example.strip())
                        f.write("\n```\n\n")

                f.write("---\n\n")

        print(f"✅ Generated: {output_file.relative_to(self.docs_dir.parent)}")

    # =================================================================
    # Extraction Helpers
    # =================================================================

    def _extract_examples_from_docstring(self, docstring: str) -> List[str]:
        """Extract code examples from docstring.

        Args:
            docstring: Full docstring text

        Returns:
            List of code example strings
        """
        examples = []
        if not docstring:
            return examples

        lines = docstring.split("\n")
        in_example = False
        example_lines = []

        for line in lines:
            # Look for example markers
            if any(
                marker in line.lower() for marker in ["example:", "examples:", ">>>", "```python"]
            ):
                in_example = True
                if "```python" in line:
                    example_lines = []
                elif ">>>" in line:
                    example_lines = [line]
                continue

            # End of example
            if in_example and (
                "```" in line or line.strip().startswith(("Args:", "Returns:", "Raises:", "Note:"))
            ):
                if example_lines:
                    examples.append("\n".join(example_lines).strip())
                    example_lines = []
                in_example = False
                continue

            # Collect example lines
            if in_example and line.strip():
                example_lines.append(line)

        # Capture final example if docstring ends
        if example_lines:
            examples.append("\n".join(example_lines).strip())

        return examples

    # =================================================================
    # Generic AST Helpers
    # =================================================================

    def _make_source_link(
        self, file_path: str, line_number: Optional[int] = None, display_text: Optional[str] = None
    ) -> str:
        """Convert file path to clickable relative link.

        Args:
            file_path: Path relative to src/ directory (e.g., "thalia/core/brain.py")
            line_number: Optional line number to link to specific location
            display_text: Optional custom text to display (default: file_path)

        Returns:
            Markdown link to source file from docs/api/ directory
        """
        # From docs/api/, we need to go up two levels to reach project root
        # Then navigate to src/ and the file
        relative_path = f"../../src/{file_path}"
        # Add line number if provided
        if line_number:
            relative_path += f"#L{line_number}"
        # Normalize path separators for URLs (use forward slashes)
        relative_path = relative_path.replace("\\", "/")
        # Determine display text
        if display_text:
            text = display_text
        else:
            text = file_path.replace("\\", "/")
        return f"[`{text}`]({relative_path})"


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
