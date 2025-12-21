"""
Generate API documentation from Thalia codebase.

This script auto-generates:
1. COMPONENT_CATALOG.md - All registered regions and pathways
2. LEARNING_STRATEGIES_API.md - All learning strategy factory functions
3. CONFIGURATION_REFERENCE.md - All configuration dataclasses
4. DATASETS_REFERENCE.md - Dataset classes and factory functions
5. DIAGNOSTICS_REFERENCE.md - Diagnostic monitor classes
6. EXCEPTIONS_REFERENCE.md - Custom exception hierarchy
7. MODULE_EXPORTS.md - Public exports from __init__.py files
8. MIXINS_REFERENCE.md - Mixin classes used by NeuralRegion
9. CONSTANTS_REFERENCE.md - Module-level constants and defaults
10. PROTOCOLS_REFERENCE.md - Protocol/interface definitions
11. USAGE_EXAMPLES.md - Code examples from docstrings
12. CHECKPOINT_FORMAT.md - Checkpoint file structure and format
13. TYPE_ALIASES.md - Type alias definitions
14. COMPONENT_RELATIONSHIPS.md - Component connections in preset architectures

Run this script whenever components are added/modified to keep docs synchronized.
"""

import ast
from pathlib import Path
from typing import List, Tuple, Dict, Optional
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


@dataclass
class DatasetInfo:
    """Dataset class or factory function."""
    name: str
    class_or_function: str
    parameters: List[str]
    docstring: str
    file_path: str
    is_factory: bool


@dataclass
class MonitorInfo:
    """Diagnostic monitor class."""
    name: str
    class_name: str
    docstring: str
    file_path: str
    methods: List[str]


@dataclass
class ExceptionInfo:
    """Custom exception class."""
    name: str
    base_class: str
    docstring: str
    file_path: str


@dataclass
class ModuleExport:
    """Public export from a module."""
    module_name: str
    export_name: str
    export_type: str  # "class", "function", "constant"
    file_path: str


@dataclass
class MixinInfo:
    """Mixin class providing functionality."""
    name: str
    docstring: str
    methods: List[Tuple[str, str]]  # (method_name, signature)
    file_path: str


@dataclass
class ConstantInfo:
    """Module-level constant."""
    name: str
    value: str
    docstring: str
    file_path: str
    category: str


@dataclass
class ProtocolInfo:
    """Protocol/interface definition."""
    name: str
    docstring: str
    methods: List[Tuple[str, str]]  # (method_name, signature)
    file_path: str
    is_runtime_checkable: bool


@dataclass
class UsageExample:
    """Code usage example from docstrings or training scripts."""
    title: str
    code: str
    description: str
    source_file: str
    category: str  # "component", "training", "diagnostic", etc.


@dataclass
class StateField:
    """Field in a state dictionary."""
    key: str
    type_hint: str
    description: str
    required: bool
    example: str


@dataclass
class CheckpointStructure:
    """Checkpoint state structure for a component."""
    component_type: str
    file_path: str
    top_level_keys: List[StateField]
    nested_structures: Dict[str, List[StateField]]  # key -> fields
    docstring: str


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
    pathway_type: str
    preset_name: str
    source_port: Optional[str]
    target_port: Optional[str]


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
        self.datasets: List[DatasetInfo] = []
        self.monitors: List[MonitorInfo] = []
        self.exceptions: List[ExceptionInfo] = []
        self.module_exports: Dict[str, List[ModuleExport]] = {}  # module_path -> exports
        self.mixins: List[MixinInfo] = []
        self.constants: List[ConstantInfo] = []
        self.protocols: List[ProtocolInfo] = []
        self.usage_examples: List[UsageExample] = []
        self.checkpoint_structures: List[CheckpointStructure] = []
        self.type_aliases: List[TypeAliasInfo] = []
        self.component_relations: List[ComponentRelation] = []

    def generate(self):
        """Generate all API documentation."""
        print("Generating API documentation from Thalia codebase...\n")

        # Ensure api directory exists
        self.api_dir.mkdir(exist_ok=True)

        # Extract data from code
        self._find_registered_components()
        self._find_strategy_factories()
        self._find_config_classes()
        self._find_datasets()
        self._find_monitors()
        self._find_exceptions()
        self._find_module_exports()
        self._find_mixins()
        self._find_constants()
        self._find_protocols()
        self._find_usage_examples()
        self._find_checkpoint_structures()
        self._find_type_aliases()
        self._find_component_relations()

        # Generate documentation files
        self._generate_component_catalog()
        self._generate_learning_strategies_api()
        self._generate_configuration_reference()
        self._generate_datasets_reference()
        self._generate_diagnostics_reference()
        self._generate_exceptions_reference()
        self._generate_module_exports_reference()
        self._generate_mixins_reference()
        self._generate_constants_reference()
        self._generate_protocols_reference()
        self._generate_usage_examples_reference()
        self._generate_checkpoint_format_reference()
        self._generate_type_aliases_reference()
        self._generate_component_relationships()

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

    def _find_datasets(self):
        """Find all dataset classes and factory functions."""
        dataset_dir = self.src_dir / "datasets"
        if not dataset_dir.exists():
            return

        for py_file in dataset_dir.rglob("*.py"):
            if "test" in str(py_file) or "__pycache__" in str(py_file):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    # Find Dataset classes
                    if isinstance(node, ast.ClassDef) and node.name.endswith("Dataset"):
                        docstring = ast.get_docstring(node) or "No docstring"
                        self.datasets.append(DatasetInfo(
                            name=node.name,
                            class_or_function="class",
                            parameters=[],
                            docstring=docstring.split('\n')[0],
                            file_path=str(py_file.relative_to(self.src_dir.parent)),
                            is_factory=False
                        ))

                    # Find create_* factory functions
                    if isinstance(node, ast.FunctionDef) and node.name.startswith("create_stage"):
                        params = [arg.arg for arg in node.args.args]
                        docstring = ast.get_docstring(node) or "No docstring"
                        self.datasets.append(DatasetInfo(
                            name=node.name,
                            class_or_function="function",
                            parameters=params,
                            docstring=docstring.split('\n')[0],
                            file_path=str(py_file.relative_to(self.src_dir.parent)),
                            is_factory=True
                        ))
            except Exception as e:
                print(f"Warning: Could not parse {py_file}: {e}")

    def _find_monitors(self):
        """Find all diagnostic monitor classes."""
        diagnostics_dir = self.src_dir / "diagnostics"
        if not diagnostics_dir.exists():
            return

        for py_file in diagnostics_dir.rglob("*.py"):
            if "test" in str(py_file) or "__pycache__" in str(py_file):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name.endswith("Monitor"):
                        docstring = ast.get_docstring(node) or "No docstring"
                        methods = [
                            item.name for item in node.body
                            if isinstance(item, ast.FunctionDef) and not item.name.startswith('_')
                        ]
                        self.monitors.append(MonitorInfo(
                            name=node.name,
                            class_name=node.name,
                            docstring=docstring.split('\n')[0],
                            file_path=str(py_file.relative_to(self.src_dir.parent)),
                            methods=methods[:5]  # First 5 public methods
                        ))
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
                                        elt.s if isinstance(elt, ast.Str)
                                        else elt.value if isinstance(elt, ast.Constant)
                                        else None
                                        for elt in node.value.elts
                                    ]
                                    all_list = [x for x in all_list if x is not None]

                if all_list:
                    # Determine module path relative to thalia
                    module_path = str(init_file.parent.relative_to(self.src_dir.parent))
                    module_name = module_path.replace("\\", ".").replace("/", ".")

                    for export in all_list[:20]:  # Limit to first 20 to avoid huge lists
                        exports.append(ModuleExport(
                            module_name=module_name,
                            export_name=export,
                            export_type="unknown",  # Would need import resolution
                            file_path=str(init_file.relative_to(self.src_dir.parent))
                        ))

                    if exports:
                        self.module_exports[module_name] = exports

            except Exception as e:
                print(f"Warning: Could not parse {init_file}: {e}")

    def _find_mixins(self):
        """Find all mixin classes used by NeuralRegion."""
        # NeuralRegion uses 4 mixins
        mixin_files = [
            ("NeuromodulatorMixin", self.src_dir / "neuromodulation" / "mixin.py"),
            ("GrowthMixin", self.src_dir / "mixins" / "growth_mixin.py"),
            ("ResettableMixin", self.src_dir / "mixins" / "resettable_mixin.py"),
            ("DiagnosticsMixin", self.src_dir / "mixins" / "diagnostics_mixin.py"),
        ]

        for mixin_name, mixin_file in mixin_files:
            if not mixin_file.exists():
                continue

            try:
                content = mixin_file.read_text(encoding="utf-8")
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == mixin_name:
                        docstring = ast.get_docstring(node) or "No docstring"
                        methods = []

                        for item in node.body:
                            if isinstance(item, ast.FunctionDef) and not item.name.startswith('_'):
                                # Extract method signature
                                params = [arg.arg for arg in item.args.args if arg.arg != 'self']
                                signature = f"{item.name}({', '.join(params)})"
                                methods.append((item.name, signature))

                        self.mixins.append(MixinInfo(
                            name=mixin_name,
                            docstring=docstring.split('\n')[0],
                            methods=methods[:10],  # First 10 methods
                            file_path=str(mixin_file.relative_to(self.src_dir.parent))
                        ))
                        break

            except Exception as e:
                print(f"Warning: Could not parse {mixin_file}: {e}")

    def _find_constants(self):
        """Find all module-level constants."""
        constant_files = [
            self.src_dir / "neuromodulation" / "constants.py",
            self.src_dir / "training" / "datasets" / "constants.py",
            self.src_dir / "training" / "visualization" / "constants.py",
        ]

        for constants_file in constant_files:
            if not constants_file.exists():
                continue

            try:
                content = constants_file.read_text(encoding="utf-8")
                tree = ast.parse(content)
                lines = content.split('\n')

                current_category = "General"

                for i, node in enumerate(tree.body):
                    # Track category from comments
                    if i < len(lines):
                        line = lines[i] if i < len(lines) else ""
                        if "==============" in line or "----------" in line:
                            # Look for category in nearby lines
                            for j in range(max(0, i-2), min(len(lines), i+3)):
                                if "#" in lines[j] and not lines[j].strip().startswith("# ="):
                                    current_category = lines[j].strip("# ").strip()
                                    break

                    # Extract constant assignments
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                const_name = target.id
                                # Only uppercase constants
                                if const_name.isupper() and len(const_name) > 2:
                                    try:
                                        value = ast.literal_eval(node.value)
                                        value_str = str(value)

                                        # Get inline comment as docstring
                                        doc = ""
                                        if i < len(lines):
                                            line = lines[i]
                                            if "#" in line:
                                                doc = line.split("#", 1)[1].strip()

                                        self.constants.append(ConstantInfo(
                                            name=const_name,
                                            value=value_str,
                                            docstring=doc or "No description",
                                            file_path=str(constants_file.relative_to(self.src_dir.parent)),
                                            category=current_category
                                        ))
                                    except:
                                        pass  # Skip complex expressions

            except Exception as e:
                print(f"Warning: Could not parse {constants_file}: {e}")

    def _find_protocols(self):
        """Find all Protocol classes defining interfaces."""
        protocol_files = [
            self.src_dir / "core" / "protocols" / "neural.py",
            self.src_dir / "core" / "protocols" / "component.py",
        ]

        for protocol_file in protocol_files:
            if not protocol_file.exists():
                continue

            try:
                content = protocol_file.read_text(encoding="utf-8")
                tree = ast.parse(content)

                # Track if next class is runtime_checkable
                is_runtime_checkable = False

                for node in tree.body:
                    # Check for @runtime_checkable decorator
                    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Name):
                        if node.value.id == "runtime_checkable":
                            is_runtime_checkable = True
                            continue

                    # Check if it's a Protocol class
                    if isinstance(node, ast.ClassDef):
                        # Check if it inherits from Protocol
                        is_protocol = any(
                            (hasattr(base, 'id') and base.id == 'Protocol') or
                            (hasattr(base, 'attr') and base.attr == 'Protocol')
                            for base in node.bases
                        )

                        if is_protocol:
                            docstring = ast.get_docstring(node) or "No docstring"
                            methods = []

                            for item in node.body:
                                if isinstance(item, ast.FunctionDef) and not item.name.startswith('_'):
                                    # Extract method signature
                                    params = []
                                    for arg in item.args.args:
                                        if arg.arg != 'self':
                                            # Get type annotation if present
                                            if arg.annotation:
                                                try:
                                                    type_str = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else "Any"
                                                    params.append(f"{arg.arg}: {type_str}")
                                                except:
                                                    params.append(arg.arg)
                                            else:
                                                params.append(arg.arg)

                                    signature = f"{item.name}({', '.join(params)})"
                                    methods.append((item.name, signature))

                            self.protocols.append(ProtocolInfo(
                                name=node.name,
                                docstring=docstring.split('\n')[0],
                                methods=methods[:10],  # First 10 methods
                                file_path=str(protocol_file.relative_to(self.src_dir.parent)),
                                is_runtime_checkable=is_runtime_checkable
                            ))
                            is_runtime_checkable = False

            except Exception as e:
                print(f"Warning: Could not parse {protocol_file}: {e}")

    def _find_usage_examples(self):
        """Find usage examples from docstrings and training scripts."""
        # Extract from module docstrings
        example_files = [
            (self.src_dir / "core" / "neural_region.py", "component"),
            (self.src_dir / "learning" / "strategy_registry.py", "learning"),
            (self.src_dir / "diagnostics" / "health_monitor.py", "diagnostic"),
        ]

        # Add training script
        training_script = self.src_dir.parent / "training" / "thalia_birth_sensorimotor.py"
        if training_script.exists():
            example_files.append((training_script, "training"))

        for file_path, category in example_files:
            if not file_path.exists():
                continue

            try:
                content = file_path.read_text(encoding="utf-8")

                # Extract code blocks from docstrings
                in_example = False
                example_lines = []
                example_title = ""

                for line in content.split('\n'):
                    # Look for Usage: or Example: sections
                    if 'Usage:' in line or 'Example:' in line:
                        example_title = line.strip().strip('"""').strip('#').strip()
                        in_example = True
                        continue

                    # Look for code blocks
                    if in_example and ('```python' in line or '>>>' in line):
                        example_lines = []
                        continue

                    if in_example and ('```' in line or '"""' in line) and example_lines:
                        # Save the example
                        code = '\n'.join(example_lines).strip()
                        if code and len(code) > 20:  # Non-trivial example
                            self.usage_examples.append(UsageExample(
                                title=example_title or f"Usage from {file_path.name}",
                                code=code,
                                description="",
                                source_file=str(file_path.relative_to(self.src_dir.parent)),
                                category=category
                            ))
                        in_example = False
                        example_lines = []
                        continue

                    if in_example and line.strip() and not line.strip().startswith('#'):
                        example_lines.append(line)

            except Exception as e:
                print(f"Warning: Could not parse {file_path}: {e}")

    def _find_exceptions(self):
        """Find all custom exception classes."""
        errors_file = self.src_dir / "core" / "errors.py"
        if not errors_file.exists():
            return

        try:
            content = errors_file.read_text(encoding="utf-8")
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if it inherits from Exception or ThaliaError
                    for base in node.bases:
                        base_name = base.id if hasattr(base, 'id') else str(base)
                        if 'Error' in node.name or 'Exception' in node.name:
                            docstring = ast.get_docstring(node) or "No docstring"
                            self.exceptions.append(ExceptionInfo(
                                name=node.name,
                                base_class=base_name,
                                docstring=docstring.split('\n')[0],
                                file_path=str(errors_file.relative_to(self.src_dir.parent))
                            ))
                            break
        except Exception as e:
            print(f"Warning: Could not parse {errors_file}: {e}")

    def _generate_datasets_reference(self):
        """Generate DATASETS_REFERENCE.md."""
        output_file = self.api_dir / "DATASETS_REFERENCE.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with output_file.open("w", encoding="utf-8") as f:
            f.write("# Datasets Reference\n\n")
            f.write("> **Auto-generated documentation** - Do not edit manually!\n")
            f.write(f"> Last updated: {timestamp}\n")
            f.write("> Generated from: `scripts/generate_api_docs.py`\n\n")

            f.write("This document catalogs all dataset classes and factory functions ")
            f.write("for curriculum training stages.\n\n")

            # Separate classes and factories
            classes = [d for d in self.datasets if not d.is_factory]
            factories = [d for d in self.datasets if d.is_factory]

            f.write(f"Total: {len(classes)} dataset classes, {len(factories)} factory functions\n\n")

            # Dataset Classes
            if classes:
                f.write("## Dataset Classes\n\n")
                for dataset in sorted(classes, key=lambda d: d.name):
                    f.write(f"### `{dataset.name}`\n\n")
                    f.write(f"**Source**: `{dataset.file_path}`\n\n")
                    f.write(f"**Description**: {dataset.docstring}\n\n")
                    f.write("---\n\n")

            # Factory Functions
            if factories:
                f.write("## Factory Functions\n\n")
                for factory in sorted(factories, key=lambda d: d.name):
                    f.write(f"### `{factory.name}()`\n\n")
                    f.write(f"**Source**: `{factory.file_path}`\n\n")
                    if factory.parameters:
                        f.write("**Parameters**:\n\n")
                        for param in factory.parameters:
                            f.write(f"- `{param}`\n")
                        f.write("\n")
                    f.write(f"**Description**: {factory.docstring}\n\n")
                    f.write("---\n\n")

        print(f"✅ Generated: {output_file.relative_to(self.docs_dir.parent)}")

    def _generate_diagnostics_reference(self):
        """Generate DIAGNOSTICS_REFERENCE.md."""
        output_file = self.api_dir / "DIAGNOSTICS_REFERENCE.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with output_file.open("w", encoding="utf-8") as f:
            f.write("# Diagnostics Reference\n\n")
            f.write("> **Auto-generated documentation** - Do not edit manually!\n")
            f.write(f"> Last updated: {timestamp}\n")
            f.write("> Generated from: `scripts/generate_api_docs.py`\n\n")

            f.write("This document catalogs all diagnostic monitor classes ")
            f.write("for system health and performance monitoring.\n\n")

            f.write(f"Total: {len(self.monitors)} monitors\n\n")

            f.write("## Monitor Classes\n\n")

            for monitor in sorted(self.monitors, key=lambda m: m.name):
                f.write(f"### `{monitor.name}`\n\n")
                f.write(f"**Source**: `{monitor.file_path}`\n\n")
                f.write(f"**Description**: {monitor.docstring}\n\n")

                if monitor.methods:
                    f.write("**Key Methods**:\n\n")
                    for method in monitor.methods:
                        f.write(f"- `{method}()`\n")
                    f.write("\n")

                f.write("---\n\n")

        print(f"✅ Generated: {output_file.relative_to(self.docs_dir.parent)}")

    def _generate_exceptions_reference(self):
        """Generate EXCEPTIONS_REFERENCE.md."""
        output_file = self.api_dir / "EXCEPTIONS_REFERENCE.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with output_file.open("w", encoding="utf-8") as f:
            f.write("# Exceptions Reference\n\n")
            f.write("> **Auto-generated documentation** - Do not edit manually!\n")
            f.write(f"> Last updated: {timestamp}\n")
            f.write("> Generated from: `scripts/generate_api_docs.py`\n\n")

            f.write("This document catalogs all custom exception classes in Thalia.\n\n")

            f.write(f"Total: {len(self.exceptions)} exception classes\n\n")

            f.write("## Exception Hierarchy\n\n")

            for exception in sorted(self.exceptions, key=lambda e: e.name):
                f.write(f"### `{exception.name}`\n\n")
                f.write(f"**Inherits from**: `{exception.base_class}`\n\n")
                f.write(f"**Source**: `{exception.file_path}`\n\n")
                f.write(f"**Description**: {exception.docstring}\n\n")
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

            f.write("## Module Exports\n\n")

            for module_name in sorted(self.module_exports.keys()):
                exports = self.module_exports[module_name]
                f.write(f"### `{module_name}`\n\n")
                f.write(f"**Source**: `{exports[0].file_path}`\n\n")
                f.write(f"**Exports** ({len(exports)}):\n\n")

                for export in exports:
                    f.write(f"- `{export.export_name}`\n")

                f.write("\n**Usage**:\n\n")
                f.write(f"```python\nfrom {module_name} import {exports[0].export_name}\n```\n\n")
                f.write("---\n\n")

        print(f"✅ Generated: {output_file.relative_to(self.docs_dir.parent)}")

    def _generate_mixins_reference(self):
        """Generate MIXINS_REFERENCE.md."""
        output_file = self.api_dir / "MIXINS_REFERENCE.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with output_file.open("w", encoding="utf-8") as f:
            f.write("# Mixins Reference\n\n")
            f.write("> **Auto-generated documentation** - Do not edit manually!\n")
            f.write(f"> Last updated: {timestamp}\n")
            f.write("> Generated from: `scripts/generate_api_docs.py`\n\n")

            f.write("This document catalogs all mixin classes used by `NeuralRegion`. ")
            f.write("These mixins provide standard functionality to all brain regions.\n\n")

            f.write(f"Total: {len(self.mixins)} mixins\n\n")

            f.write("## NeuralRegion Composition\n\n")
            f.write("```python\n")
            f.write("class NeuralRegion(nn.Module,\n")
            for i, mixin in enumerate(self.mixins):
                comma = "," if i < len(self.mixins) - 1 else ""
                f.write(f"                   {mixin.name}{comma}\n")
            f.write("    ):\n")
            f.write("    # ...\n")
            f.write("```\n\n")

            f.write("## Mixin Classes\n\n")

            for mixin in sorted(self.mixins, key=lambda m: m.name):
                f.write(f"### `{mixin.name}`\n\n")
                f.write(f"**Source**: `{mixin.file_path}`\n\n")
                f.write(f"**Description**: {mixin.docstring}\n\n")

                if mixin.methods:
                    f.write("**Public Methods**:\n\n")
                    for _, signature in mixin.methods:
                        f.write(f"- `{signature}`\n")
                    f.write("\n")

                f.write("---\n\n")

        print(f"✅ Generated: {output_file.relative_to(self.docs_dir.parent)}")

    def _generate_constants_reference(self):
        """Generate CONSTANTS_REFERENCE.md."""
        output_file = self.api_dir / "CONSTANTS_REFERENCE.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with output_file.open("w", encoding="utf-8") as f:
            f.write("# Constants Reference\n\n")
            f.write("> **Auto-generated documentation** - Do not edit manually!\n")
            f.write(f"> Last updated: {timestamp}\n")
            f.write("> Generated from: `scripts/generate_api_docs.py`\n\n")

            f.write("This document catalogs all module-level constants in Thalia. ")
            f.write("These include biological time constants, default values, and thresholds.\n\n")

            f.write(f"Total: {len(self.constants)} constants\n\n")

            # Group by category
            from collections import defaultdict
            by_category = defaultdict(list)
            for const in self.constants:
                by_category[const.category].append(const)

            f.write("## Constants by Category\n\n")

            for category in sorted(by_category.keys()):
                constants = by_category[category]
                f.write(f"### {category}\n\n")

                # Create table
                f.write("| Constant | Value | Description |\n")
                f.write("|----------|-------|-------------|\n")

                for const in sorted(constants, key=lambda c: c.name):
                    # Escape pipe characters
                    value_escaped = const.value.replace("|", "\\|")
                    doc_escaped = const.docstring.replace("|", "\\|")
                    f.write(f"| `{const.name}` | `{value_escaped}` | {doc_escaped} |\n")

                f.write("\n")

                # Show source file
                if constants:
                    f.write(f"**Source**: `{constants[0].file_path}`\n\n")

                f.write("---\n\n")

        print(f"✅ Generated: {output_file.relative_to(self.docs_dir.parent)}")

    def _generate_protocols_reference(self):
        """Generate PROTOCOLS_REFERENCE.md."""
        output_file = self.api_dir / "PROTOCOLS_REFERENCE.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with output_file.open("w", encoding="utf-8") as f:
            f.write("# Protocols Reference\n\n")
            f.write("> **Auto-generated documentation** - Do not edit manually!\n")
            f.write(f"> Last updated: {timestamp}\n")
            f.write("> Generated from: `scripts/generate_api_docs.py`\n\n")

            f.write("This document catalogs all Protocol classes defining interfaces ")
            f.write("for duck-typed components in Thalia.\n\n")

            f.write(f"Total: {len(self.protocols)} protocols\n\n")

            f.write("## Protocol Classes\n\n")

            for protocol in sorted(self.protocols, key=lambda p: p.name):
                f.write(f"### `{protocol.name}`\n\n")

                if protocol.is_runtime_checkable:
                    f.write("**Runtime Checkable**: ✅ Yes (can use `isinstance()`)\n\n")
                else:
                    f.write("**Runtime Checkable**: ❌ No (static type checking only)\n\n")

                f.write(f"**Source**: `{protocol.file_path}`\n\n")
                f.write(f"**Description**: {protocol.docstring}\n\n")

                if protocol.methods:
                    f.write("**Required Methods**:\n\n")
                    f.write("```python\n")
                    for _, signature in protocol.methods:
                        f.write(f"def {signature}:\n")
                        f.write(f"    ...\n\n")
                    f.write("```\n\n")

                f.write("---\n\n")

        print(f"✅ Generated: {output_file.relative_to(self.docs_dir.parent)}")

    def _generate_usage_examples_reference(self):
        """Generate USAGE_EXAMPLES.md."""
        output_file = self.api_dir / "USAGE_EXAMPLES.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with output_file.open("w", encoding="utf-8") as f:
            f.write("# Usage Examples\n\n")
            f.write("> **Auto-generated documentation** - Do not edit manually!\n")
            f.write(f"> Last updated: {timestamp}\n")
            f.write("> Generated from: `scripts/generate_api_docs.py`\n\n")

            f.write("This document catalogs usage examples extracted from docstrings ")
            f.write("and training scripts.\n\n")

            f.write(f"Total: {len(self.usage_examples)} examples\n\n")

            # Group by category
            from collections import defaultdict
            by_category = defaultdict(list)
            for example in self.usage_examples:
                by_category[example.category].append(example)

            f.write("## Examples by Category\n\n")

            for category in sorted(by_category.keys()):
                examples = by_category[category]
                f.write(f"### {category.title()}\n\n")

                for example in examples:
                    f.write(f"#### {example.title}\n\n")
                    f.write(f"**Source**: `{example.source_file}`\n\n")

                    if example.description:
                        f.write(f"{example.description}\n\n")

                    f.write("```python\n")
                    f.write(example.code)
                    f.write("\n```\n\n")
                    f.write("---\n\n")

        print(f"✅ Generated: {output_file.relative_to(self.docs_dir.parent)}")

    def _find_checkpoint_structures(self):
        """Extract checkpoint state structures from get_full_state() methods."""
        # Check key files that define checkpoint structure
        key_files = [
            self.src_dir / "core" / "dynamic_brain.py",
            self.src_dir / "core" / "neural_region.py",
            self.src_dir / "pathways" / "axonal_projection.py",
            self.src_dir / "io" / "checkpoint.py",
        ]

        for py_file in key_files:
            if not py_file.exists():
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == "get_full_state":
                        # Extract state dict structure
                        state_fields = self._extract_state_dict_structure(node, content)
                        if state_fields:
                            component_type = self._get_containing_class(node, tree) or "TopLevel"
                            docstring = ast.get_docstring(node) or ""

                            self.checkpoint_structures.append(CheckpointStructure(
                                component_type=component_type,
                                file_path=str(py_file.relative_to(self.src_dir)),
                                top_level_keys=state_fields,
                                nested_structures={},
                                docstring=docstring,
                            ))
            except Exception:
                continue  # Skip files with parse errors

    def _extract_state_dict_structure(self, func_node: ast.FunctionDef, content: str) -> List[StateField]:
        """Extract state dict keys from get_full_state() method."""
        fields = []
        seen_keys = set()

        # Look for state = {...} initialization
        for node in ast.walk(func_node):
            if isinstance(node, ast.Assign):
                # Check for state = {...}
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "state":
                        if isinstance(node.value, ast.Dict):
                            # Extract keys from dict literal
                            for key_node, value_node in zip(node.value.keys, node.value.values):
                                if isinstance(key_node, ast.Constant):
                                    key = key_node.value
                                elif isinstance(key_node, ast.Str):
                                    key = key_node.s
                                else:
                                    continue

                                if key not in seen_keys:
                                    seen_keys.add(key)
                                    type_hint = self._infer_type_from_value(value_node)
                                    fields.append(StateField(
                                        key=key,
                                        type_hint=type_hint,
                                        description="",
                                        required=True,
                                        example="",
                                    ))

                # Also check for state["key"] = value patterns
                for target in node.targets:
                    if isinstance(target, ast.Subscript):
                        if isinstance(target.value, ast.Name) and target.value.id == "state":
                            # Get the key
                            if isinstance(target.slice, ast.Constant):
                                key = target.slice.value
                            elif isinstance(target.slice, ast.Str):
                                key = target.slice.s
                            else:
                                continue

                            if key not in seen_keys:
                                seen_keys.add(key)
                                value_node = node.value
                                type_hint = self._infer_type_from_value(value_node)
                                fields.append(StateField(
                                    key=key,
                                    type_hint=type_hint,
                                    description="",
                                    required=True,
                                    example="",
                                ))

        return fields

    def _infer_type_from_value(self, node: ast.AST) -> str:
        """Infer type from AST value node."""
        if isinstance(node, ast.Dict):
            return "Dict[str, Any]"
        elif isinstance(node, ast.List):
            return "List[Any]"
        elif isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, int):
                return "int"
            elif isinstance(value, float):
                return "float"
            elif isinstance(value, str):
                return "str"
            elif isinstance(value, bool):
                return "bool"
            return "Any"
        elif isinstance(node, ast.Num):  # Python 3.7 compat
            return "int" if isinstance(node.n, int) else "float"
        elif isinstance(node, ast.Str):
            return "str"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                method_name = node.func.attr
                if method_name in ("get_full_state", "get_state"):
                    return "Dict[str, Any]"
                elif method_name == "to_dict":
                    return "Dict[str, Any]"
                return "Any"
            elif isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in ("dict", "Dict"):
                    return "Dict[str, Any]"
                elif func_name in ("list", "List"):
                    return "List[Any]"
        elif isinstance(node, ast.ListComp):
            return "List[Any]"
        elif isinstance(node, ast.DictComp):
            return "Dict[str, Any]"
        return "Any"

    def _get_containing_class(self, func_node: ast.FunctionDef, tree: ast.AST) -> str:
        """Get the class containing this function."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if item == func_node:
                        return node.name
        return ""

    def _generate_checkpoint_format_reference(self):
        """Generate CHECKPOINT_FORMAT.md."""
        output_file = self.api_dir / "CHECKPOINT_FORMAT.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with output_file.open("w", encoding="utf-8") as f:
            f.write("# Checkpoint Format Reference\n\n")
            f.write("> **Auto-generated documentation** - Do not edit manually!\n")
            f.write(f"> Last updated: {timestamp}\n")
            f.write("> Generated from: `scripts/generate_api_docs.py`\n\n")

            f.write("This document describes the checkpoint file format used by Thalia. ")
            f.write("Checkpoints are created using `brain.save_checkpoint()` and restored with `brain.load_checkpoint()`.\n\n")

            f.write("## Overview\n\n")
            f.write("Thalia checkpoints use a hierarchical structure:\n\n")
            f.write("```\n")
            f.write("checkpoint.thalia\n")
            f.write("├── metadata (timestamp, versions, sizes)\n")
            f.write("├── regions (component states)\n")
            f.write("├── pathways (connection states)\n")
            f.write("├── oscillators (rhythm generator states)\n")
            f.write("├── neuromodulators (dopamine, acetylcholine, etc.)\n")
            f.write("└── config (brain configuration)\n")
            f.write("```\n\n")

            f.write("## Top-Level Structure\n\n")
            f.write("The checkpoint is a dictionary with these top-level keys:\n\n")

            # Find DynamicBrain structure
            brain_struct = next((s for s in self.checkpoint_structures if s.component_type == "DynamicBrain"), None)

            if brain_struct:
                f.write("### Keys from `DynamicBrain.get_full_state()`\n\n")
                f.write("| Key | Type | Description |\n")
                f.write("|-----|------|-------------|\n")

                for field in brain_struct.top_level_keys:
                    f.write(f"| `{field.key}` | `{field.type_hint}` | {field.description or 'Component state'} |\n")

                f.write("\n")
                f.write(f"**Source**: `{brain_struct.file_path}`\n\n")

            f.write("## Component State Structure\n\n")
            f.write("Each component (region or pathway) stores its state in the checkpoint.\n\n")

            # Find NeuralRegion structure
            region_struct = next((s for s in self.checkpoint_structures if s.component_type == "NeuralRegion"), None)

            if region_struct:
                f.write("### NeuralRegion State (Base Class)\n\n")
                f.write("All regions include these fields:\n\n")

                f.write("| Key | Type | Description |\n")
                f.write("|-----|------|-------------|\n")

                for field in region_struct.top_level_keys:
                    f.write(f"| `{field.key}` | `{field.type_hint}` | {field.description or 'Base region data'} |\n")

                f.write("\n")
                f.write(f"**Source**: `{region_struct.file_path}`\n\n")

            # Find AxonalProjection structure
            pathway_struct = next((s for s in self.checkpoint_structures if s.component_type == "AxonalProjection"), None)

            if pathway_struct:
                f.write("### AxonalProjection State (Pathways)\n\n")
                f.write("All pathways include these fields:\n\n")

                f.write("| Key | Type | Description |\n")
                f.write("|-----|------|-------------|\n")

                for field in pathway_struct.top_level_keys:
                    f.write(f"| `{field.key}` | `{field.type_hint}` | {field.description or 'Pathway data'} |\n")

                f.write("\n")
                f.write(f"**Source**: `{pathway_struct.file_path}`\n\n")

            f.write("## File Format Details\n\n")
            f.write("Checkpoints can be saved in multiple formats:\n\n")
            f.write("1. **PyTorch Format** (`.pt`, `.pth`, `.ckpt`) - Standard PyTorch `torch.save()` format\n")
            f.write("2. **Binary Format** (`.thalia`, `.thalia.zst`) - Custom binary format with compression\n\n")

            f.write("### Compression Support\n\n")
            f.write("- `.zst` extension → Zstandard compression\n")
            f.write("- `.lz4` extension → LZ4 compression\n")
            f.write("- No extension → Uncompressed\n\n")

            f.write("### Precision Policies\n\n")
            f.write("Checkpoints support mixed precision:\n\n")
            f.write("- `fp32` - Full precision (default)\n")
            f.write("- `fp16` - Half precision (smaller files, some accuracy loss)\n")
            f.write("- `mixed` - fp16 for weights, fp32 for critical state\n\n")

            f.write("## Usage Examples\n\n")
            f.write("```python\n")
            f.write("# Save checkpoint\n")
            f.write("brain.save_checkpoint(\n")
            f.write('    "checkpoints/epoch_100.ckpt",\n')
            f.write('    metadata={"epoch": 100, "loss": 0.42}\n')
            f.write(")\n\n")
            f.write("# Load checkpoint\n")
            f.write('brain.load_checkpoint("checkpoints/epoch_100.ckpt")\n\n')
            f.write("# Save with compression\n")
            f.write('brain.save_checkpoint("checkpoints/epoch_100.thalia.zst", compression="zstd")\n\n')
            f.write("# Save with mixed precision\n")
            f.write('brain.save_checkpoint("checkpoints/epoch_100.ckpt", precision_policy="fp16")\n')
            f.write("```\n\n")

            f.write("## Validation\n\n")
            f.write("Use `CheckpointManager.validate()` to check checkpoint integrity:\n\n")
            f.write("```python\n")
            f.write("from thalia.io import CheckpointManager\n\n")
            f.write("manager = CheckpointManager(brain)\n")
            f.write('is_valid, error = manager.validate("checkpoints/epoch_100.ckpt")\n')
            f.write("if not is_valid:\n")
            f.write('    print(f"Checkpoint invalid: {error}")\n')
            f.write("```\n\n")

            f.write("## Version Compatibility\n\n")
            f.write("Checkpoints include version metadata for compatibility checking:\n\n")
            f.write("- `checkpoint_format_version` - Format version (2.0+)\n")
            f.write("- `thalia_version` - Thalia library version\n")
            f.write("- `pytorch_version` - PyTorch version used\n\n")

            f.write("The checkpoint manager can migrate old formats when loading.\n\n")

        print(f"✅ Generated: {output_file.relative_to(self.docs_dir.parent)}")

    def _find_type_aliases(self):
        """Extract type alias definitions from code."""
        # Check key files
        key_files = [
            self.src_dir / "core" / "dynamic_brain.py",
            self.src_dir / "core" / "brain_builder.py",
            self.src_dir / "pathways" / "axonal_projection.py",
            self.src_dir / "io" / "compression.py",
            self.src_dir / "synapses" / "spillover.py",
        ]

        # Also scan copilot instructions for documented aliases
        copilot_file = self.docs_dir.parent / ".github" / "copilot-instructions.md"
        if copilot_file.exists():
            self._extract_aliases_from_copilot(copilot_file)

        for py_file in key_files:
            if not py_file.exists():
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    # Look for: TypeName = Type[...] or TypeName = Literal[...]
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                name = target.id
                                # Check if it's a type alias (uppercase first letter)
                                if name[0].isupper() and name not in ("True", "False", "None"):
                                    definition = self._get_source_segment(content, node.value)
                                    if definition and any(kw in definition for kw in ["Dict", "List", "Tuple", "Literal", "Callable", "Optional", "Union"]):
                                        self.type_aliases.append(TypeAliasInfo(
                                            name=name,
                                            definition=definition,
                                            description="",
                                            file_path=str(py_file.relative_to(self.src_dir)),
                                            category=self._categorize_type_alias(name),
                                        ))
            except Exception:
                continue

    def _extract_aliases_from_copilot(self, copilot_file: Path):
        """Extract type aliases documented in copilot instructions."""
        content = copilot_file.read_text(encoding="utf-8")
        lines = content.split("\n")

        in_glossary = False
        for i, line in enumerate(lines):
            if "## Type Alias Glossary" in line:
                in_glossary = True
                continue

            if in_glossary:
                # Stop at next major heading
                if line.startswith("## ") and "Type Alias" not in line:
                    break

                # Look for: TypeName = TypeDefinition  # comment
                if " = " in line and not line.strip().startswith("#"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        name = parts[0].strip()
                        rest = parts[1].split("#", 1)
                        definition = rest[0].strip()
                        description = rest[1].strip() if len(rest) > 1 else ""

                        if name and definition and name[0].isupper():
                            self.type_aliases.append(TypeAliasInfo(
                                name=name,
                                definition=definition,
                                description=description,
                                file_path=".github/copilot-instructions.md",
                                category=self._infer_category_from_context(lines, i),
                            ))

    def _get_source_segment(self, content: str, node: ast.AST) -> str:
        """Extract source code for an AST node."""
        try:
            return ast.unparse(node)
        except Exception:
            # Fallback for older Python or complex nodes
            return ""

    def _categorize_type_alias(self, name: str) -> str:
        """Categorize type alias by name patterns."""
        if "Graph" in name or "Topology" in name:
            return "Component Organization"
        elif "Source" in name or "Input" in name or "Output" in name or "Port" in name:
            return "Routing"
        elif "State" in name or "Checkpoint" in name:
            return "State Management"
        elif "Config" in name or "Spec" in name:
            return "Configuration"
        elif "Dict" in name or "Metadata" in name:
            return "Data Structures"
        return "Other"

    def _infer_category_from_context(self, lines: List[str], current_idx: int) -> str:
        """Infer category from surrounding comment lines."""
        # Look backwards for comment lines
        for i in range(current_idx - 1, max(0, current_idx - 10), -1):
            line = lines[i].strip()
            if line.startswith("#"):
                comment = line.lstrip("#").strip()
                if comment and not comment.startswith("("):
                    return comment
        return "Other"

    def _find_component_relations(self):
        """Extract component relationships from preset architectures."""
        brain_builder_file = self.src_dir / "core" / "brain_builder.py"

        if not brain_builder_file.exists():
            return

        try:
            content = brain_builder_file.read_text(encoding="utf-8")
            tree = ast.parse(content)

            # Look for preset builder functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name.startswith("_build_"):
                        preset_name = node.name.replace("_build_", "")
                        self._extract_connections_from_preset(node, preset_name)
        except Exception:
            pass

    def _extract_connections_from_preset(self, func_node: ast.FunctionDef, preset_name: str):
        """Extract connection calls from preset builder function."""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                # Look for builder.connect() calls
                if isinstance(node.func, ast.Attribute) and node.func.attr == "connect":
                    # Extract source and target (always first 2 positional args)
                    if len(node.args) >= 2:
                        source = self._get_string_value(node.args[0])
                        target = self._get_string_value(node.args[1])

                        # pathway_type can be positional (3rd arg) or keyword
                        pathway_type = None
                        if len(node.args) >= 3:
                            pathway_type = self._get_string_value(node.args[2])

                        source_port = None
                        target_port = None

                        # Check for keyword arguments
                        for keyword in node.keywords:
                            if keyword.arg == "pathway_type" and not pathway_type:
                                pathway_type = self._get_string_value(keyword.value)
                            elif keyword.arg == "source_port":
                                source_port = self._get_string_value(keyword.value)
                            elif keyword.arg == "target_port":
                                target_port = self._get_string_value(keyword.value)

                        if source and target and pathway_type:
                            self.component_relations.append(ComponentRelation(
                                source=source,
                                target=target,
                                pathway_type=pathway_type,
                                preset_name=preset_name,
                                source_port=source_port,
                                target_port=target_port,
                            ))

    def _get_string_value(self, node: ast.AST) -> Optional[str]:
        """Extract string value from AST node."""
        if isinstance(node, ast.Constant):
            return str(node.value) if isinstance(node.value, str) else None
        elif isinstance(node, ast.Str):
            return str(node.s) if isinstance(node.s, str) else None
        return None

    def _generate_type_aliases_reference(self):
        """Generate TYPE_ALIASES.md."""
        output_file = self.api_dir / "TYPE_ALIASES.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with output_file.open("w", encoding="utf-8") as f:
            f.write("# Type Aliases Reference\n\n")
            f.write("> **Auto-generated documentation** - Do not edit manually!\n")
            f.write(f"> Last updated: {timestamp}\n")
            f.write("> Generated from: `scripts/generate_api_docs.py`\n\n")

            f.write("This document catalogs all type aliases used in Thalia for clearer type hints.\n\n")

            f.write(f"Total: {len(self.type_aliases)} type aliases\n\n")

            # Group by category
            from collections import defaultdict
            by_category = defaultdict(list)
            for alias in self.type_aliases:
                by_category[alias.category].append(alias)

            f.write("## Type Aliases by Category\n\n")

            for category in sorted(by_category.keys()):
                aliases = by_category[category]
                f.write(f"### {category}\n\n")

                for alias in sorted(aliases, key=lambda a: a.name):
                    f.write(f"#### `{alias.name}`\n\n")
                    f.write(f"**Definition**: `{alias.definition}`\n\n")

                    if alias.description:
                        f.write(f"**Description**: {alias.description}\n\n")

                    f.write(f"**Source**: `{alias.file_path}`\n\n")
                    f.write("---\n\n")

        print(f"✅ Generated: {output_file.relative_to(self.docs_dir.parent)}")

    def _generate_component_relationships(self):
        """Generate COMPONENT_RELATIONSHIPS.md."""
        output_file = self.api_dir / "COMPONENT_RELATIONSHIPS.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with output_file.open("w", encoding="utf-8") as f:
            f.write("# Component Relationships\n\n")
            f.write("> **Auto-generated documentation** - Do not edit manually!\n")
            f.write(f"> Last updated: {timestamp}\n")
            f.write("> Generated from: `scripts/generate_api_docs.py`\n\n")

            f.write("This document shows how components connect in preset brain architectures.\n\n")

            if not self.component_relations:
                f.write("No preset architectures found.\n\n")
                print(f"✅ Generated: {output_file.relative_to(self.docs_dir.parent)}")
                return

            # Group by preset
            from collections import defaultdict
            by_preset = defaultdict(list)
            for relation in self.component_relations:
                by_preset[relation.preset_name].append(relation)

            f.write(f"Total: {len(by_preset)} preset architectures\n\n")

            for preset_name in sorted(by_preset.keys()):
                relations = by_preset[preset_name]
                f.write(f"## Preset: `{preset_name}`\n\n")

                # Build component list
                components = set()
                for rel in relations:
                    components.add(rel.source)
                    components.add(rel.target)

                f.write(f"**Components**: {len(components)}\n\n")
                f.write(f"**Connections**: {len(relations)}\n\n")

                # Show connections table
                f.write("### Connections\n\n")
                f.write("| Source | Source Port | → | Target | Target Port | Pathway Type |\n")
                f.write("|--------|-------------|---|--------|-------------|-------------|\n")

                for rel in relations:
                    src_port = rel.source_port or "default"
                    tgt_port = rel.target_port or "default"
                    f.write(f"| `{rel.source}` | `{src_port}` | → | `{rel.target}` | `{tgt_port}` | `{rel.pathway_type}` |\n")

                f.write("\n")

                # Generate mermaid diagram
                f.write("### Architecture Diagram\n\n")
                f.write("```mermaid\n")
                f.write("graph TB\n")

                for rel in relations:
                    # Create edge label - use simple text without special characters for mermaid
                    label = rel.pathway_type
                    if rel.source_port and rel.source_port != "default":
                        label = f"{rel.source_port} {label}"
                    if rel.target_port and rel.target_port != "default":
                        label = f"{label} to {rel.target_port}"

                    f.write(f"    {rel.source} -->|{label}| {rel.target}\n")

                f.write("```\n\n")
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
