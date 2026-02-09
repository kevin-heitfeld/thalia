"""
Generate API documentation from Thalia codebase.

Run this script whenever components are added/modified to keep docs synchronized.
"""

import ast
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent / "src"))

from thalia.brain import NeuralRegionRegistry
from thalia.components.neurons import NeuronFactory
from thalia.components.synapses import WeightInitializer
from thalia.learning import LearningStrategyRegistry


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
class DatasetInfo:
    """Dataset class or factory function."""

    name: str
    class_or_function: str
    parameters: List[Tuple[str, str, str]]  # (name, type, default)
    docstring: str
    file_path: str
    is_factory: bool
    line_number: int = 0
    examples: Optional[List[str]] = None
    stage: Optional[str] = None  # Which curriculum stage

    def __post_init__(self):
        if self.examples is None:
            self.examples = []


@dataclass
class ExceptionInfo:
    """Custom exception class."""

    name: str
    base_class: str
    docstring: str
    file_path: str
    line_number: int = 0


@dataclass
class ModuleExport:
    """Public export from a module."""

    module_name: str
    export_name: str
    export_type: str  # "class", "function", "constant"
    file_path: str


@dataclass
class ConstantInfo:
    """Module-level constant."""

    name: str
    value: str
    docstring: str
    file_path: str
    category: str
    biological_range: Optional[str] = None  # e.g., "10-30ms"
    references: Optional[List[str]] = None  # e.g., ["Bi & Poo (1998)"]

    def __post_init__(self):
        if self.references is None:
            self.references = []


@dataclass
class StateField:
    """Field in a state dictionary."""

    key: str
    type_hint: str
    description: str
    required: bool
    example: str


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


@dataclass
class EnumInfo:
    """Enumeration type definition."""

    name: str
    docstring: str
    values: List[Tuple[str, str]]  # (member_name, member_value_or_comment)
    file_path: str
    enum_type: str  # "Enum", "IntEnum", "StrEnum"
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
        self.datasets: List[DatasetInfo] = []
        self.exceptions: List[ExceptionInfo] = []
        self.module_exports: Dict[str, List[ModuleExport]] = {}  # module_path -> exports
        self.constants: List[ConstantInfo] = []
        self.type_aliases: List[TypeAliasInfo] = []
        self.component_relations: List[ComponentRelation] = []
        self.enumerations: List[EnumInfo] = []
        self.neuron_factories: List[NeuronFactoryInfo] = []
        self.dependencies: Dict[str, List[str]] = {}  # module -> imported modules
        self.profile_data: Dict[str, Any] = {}  # Performance profiling data
        self.coverage_data: Dict[str, Any] = {}  # Test coverage data

        # Load profiling data if available
        profile_file = src_dir.parent.parent / "profile_results.json"
        if profile_file.exists():
            with open(profile_file, "r", encoding="utf-8") as f:
                self.profile_data = json.load(f)

        # Load coverage data if available
        coverage_file = src_dir.parent.parent / "coverage_results.json"
        if coverage_file.exists():
            with open(coverage_file, "r", encoding="utf-8") as f:
                self.coverage_data = json.load(f)

    def generate(self):
        """Generate all API documentation."""
        print("Generating API documentation from Thalia codebase...\n")

        # Ensure api directory exists
        self.api_dir.mkdir(exist_ok=True)

        # Extract data from code
        self._find_strategy_factories()
        self._find_config_classes()
        self._find_datasets()
        self._find_exceptions()
        self._find_module_exports()
        self._find_constants()
        self._find_enumerations()
        self._find_neuron_factories()

        # Build cross-references
        self._build_cross_references()

        # Analyze dependencies and architecture
        self._analyze_dependencies()

        # Generate documentation files
        self._generate_all_registrations_doc()
        self._generate_configuration_reference()
        self._generate_constants_reference()
        self._generate_datasets_reference()
        self._generate_diagnostics_reference()
        self._generate_enumerations_reference()
        self._generate_exceptions_reference()
        self._generate_learning_strategies_api()
        self._generate_mixins_reference()
        self._generate_module_exports_reference()
        self._generate_neuron_factories_reference()

        print("\n✅ API documentation generated successfully!")
        print(f"   Location: {self.api_dir.relative_to(self.docs_dir.parent)}")

    # =================================================================
    # Internal methods for documentation generation
    # =================================================================

    def _find_config_class_location(
        self, config_class_name: str
    ) -> Tuple[Optional[str], Optional[int]]:
        """Find the file and line number where a config class is defined.

        Args:
            config_class_name: Name of the config class to find

        Returns:
            Tuple of (file_path, line_number) or (None, None) if not found
        """
        # Search in config directory
        config_dir = self.src_dir / "config"
        if config_dir.exists():
            for py_file in config_dir.rglob("*.py"):
                if "__pycache__" in str(py_file):
                    continue
                try:
                    content = py_file.read_text(encoding="utf-8")
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef) and node.name == config_class_name:
                            return str(py_file.relative_to(self.src_dir.parent)), node.lineno
                except Exception:
                    continue

        # Search in all Python files as fallback
        for py_file in self.src_dir.rglob("*.py"):
            if "__pycache__" in str(py_file) or "test" in str(py_file):
                continue
            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == config_class_name:
                        return str(py_file.relative_to(self.src_dir.parent)), node.lineno
            except Exception:
                continue

        return None, None

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
                                        arg_type = (
                                            ast.unparse(arg.annotation)
                                            if hasattr(ast, "unparse")
                                            else "Any"
                                        )
                                    except (AttributeError, TypeError):
                                        pass

                                # Get default value
                                defaults_offset = len(node.args.args) - len(node.args.defaults)
                                arg_idx = node.args.args.index(arg)
                                if arg_idx >= defaults_offset:
                                    default_idx = arg_idx - defaults_offset
                                    try:
                                        arg_default = (
                                            ast.unparse(node.args.defaults[default_idx])
                                            if hasattr(ast, "unparse")
                                            else ""
                                        )
                                    except (AttributeError, TypeError, IndexError):
                                        pass

                                parameters.append((arg_name, arg_type, arg_default))

                            # Extract return type
                            return_type = "LearningStrategy"
                            if node.returns:
                                try:
                                    return_type = (
                                        ast.unparse(node.returns)
                                        if hasattr(ast, "unparse")
                                        else "LearningStrategy"
                                    )
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
                                        field_type = (
                                            ast.unparse(item.annotation)
                                            if hasattr(ast, "unparse")
                                            else "Any"
                                        )

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
                        self.datasets.append(
                            DatasetInfo(
                                name=node.name,
                                class_or_function="class",
                                parameters=[],
                                docstring=docstring.split("\n")[0],
                                file_path=str(py_file.relative_to(self.src_dir.parent)),
                                is_factory=False,
                                line_number=node.lineno,
                            )
                        )

                    # Find create_* factory functions
                    if isinstance(node, ast.FunctionDef) and node.name.startswith("create_stage"):
                        # Extract full parameter info (name, type, default)
                        parameters = []
                        defaults = [None] * (
                            len(node.args.args) - len(node.args.defaults)
                        ) + node.args.defaults
                        for arg, default in zip(node.args.args, defaults):
                            arg_name = arg.arg
                            arg_type = ast.unparse(arg.annotation) if arg.annotation else "Any"
                            arg_default = ast.unparse(default) if default else "**Required**"
                            parameters.append((arg_name, arg_type, arg_default))

                        docstring = ast.get_docstring(node) or "No docstring"
                        examples = self._extract_examples_from_docstring(docstring)

                        self.datasets.append(
                            DatasetInfo(
                                name=node.name,
                                class_or_function="function",
                                parameters=parameters,
                                docstring=docstring.split("\n")[0],
                                file_path=str(py_file.relative_to(self.src_dir.parent)),
                                is_factory=True,
                                line_number=node.lineno,
                                examples=examples,
                            )
                        )
            except Exception as e:
                print(f"Warning: Could not parse {py_file}: {e}")

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
                        base_name = base.id if hasattr(base, "id") else str(base)
                        if "Error" in node.name or "Exception" in node.name:
                            docstring = ast.get_docstring(node) or "No docstring"
                            self.exceptions.append(
                                ExceptionInfo(
                                    name=node.name,
                                    base_class=base_name,
                                    docstring=docstring.split("\n")[0],
                                    file_path=str(errors_file.relative_to(self.src_dir.parent)),
                                    line_number=node.lineno,
                                )
                            )
                            break
        except Exception as e:
            print(f"Warning: Could not parse {errors_file}: {e}")

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

    def _find_constants(self):
        """Find all module-level constants with enhanced biological documentation."""
        constant_files = [
            self.src_dir / "neuromodulation" / "constants.py",
            self.src_dir / "components" / "neurons" / "neuron_constants.py",
            self.src_dir / "regulation" / "learning_constants.py",
            self.src_dir / "regulation" / "homeostasis_constants.py",
            self.src_dir / "regulation" / "region_constants.py",
            self.src_dir / "training" / "datasets" / "constants.py",
            self.src_dir / "training" / "visualization" / "constants.py",
        ]

        for constants_file in constant_files:
            if not constants_file.exists():
                continue

            try:
                content = constants_file.read_text(encoding="utf-8")
                tree = ast.parse(content)
                lines = content.split("\n")

                current_category = "General"

                for i, node in enumerate(tree.body):
                    # Track category from comments
                    if i < len(lines):
                        line = lines[i] if i < len(lines) else ""
                        if "==============" in line or "----------" in line:
                            # Look for category in nearby lines
                            for j in range(max(0, i - 2), min(len(lines), i + 3)):
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

                                        # Get docstring (string literal after assignment)
                                        docstring = ""
                                        biological_range = None
                                        references = []

                                        # Check if next node is a docstring
                                        node_idx = tree.body.index(node)
                                        if node_idx + 1 < len(tree.body):
                                            next_node = tree.body[node_idx + 1]
                                            if isinstance(next_node, ast.Expr) and isinstance(
                                                next_node.value, ast.Constant
                                            ):
                                                if isinstance(next_node.value.value, str):
                                                    full_docstring = next_node.value.value
                                                    docstring = full_docstring.split("\n")[
                                                        0
                                                    ]  # First line

                                                    # Extract biological range (look for patterns like "10-30ms" or "range: X-Y")
                                                    import re

                                                    range_match = re.search(
                                                        r"(\d+[\-–]\d+\s*(?:ms|mV|Hz))",
                                                        full_docstring,
                                                        re.IGNORECASE,
                                                    )
                                                    if range_match:
                                                        biological_range = range_match.group(1)

                                                    # Extract references (Author et al. (YEAR) or Author (YEAR))
                                                    ref_matches = re.findall(
                                                        r"([A-Z][a-z]+(?: (?:et al\.|& [A-Z][a-z]+))? \(\d{4}\))",
                                                        full_docstring,
                                                    )
                                                    references = list(
                                                        set(ref_matches)
                                                    )  # Deduplicate

                                        # Fallback to inline comment
                                        if not docstring and i < len(lines):
                                            line = lines[i]
                                            if "#" in line:
                                                docstring = line.split("#", 1)[1].strip()

                                        self.constants.append(
                                            ConstantInfo(
                                                name=const_name,
                                                value=value_str,
                                                docstring=docstring or "No description",
                                                file_path=str(
                                                    constants_file.relative_to(self.src_dir.parent)
                                                ),
                                                category=current_category,
                                                biological_range=biological_range,
                                                references=references,
                                            )
                                        )
                                    except (ValueError, SyntaxError):
                                        pass  # Skip complex expressions

            except Exception as e:
                print(f"Warning: Could not parse {constants_file}: {e}")

    def _find_enumerations(self):
        """Extract all Enum class definitions."""
        for py_file in self.src_dir.rglob("*.py"):
            if "test" in str(py_file) or "__pycache__" in str(py_file):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")

                # Quick check before parsing
                if "Enum)" not in content:
                    continue

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Check if it inherits from Enum, IntEnum, or StrEnum
                        enum_type = None
                        for base in node.bases:
                            if isinstance(base, ast.Name):
                                if base.id in ("Enum", "IntEnum", "StrEnum"):
                                    enum_type = base.id
                                    break

                        if enum_type:
                            # Extract enum members
                            values = []
                            for item in node.body:
                                if isinstance(item, ast.Assign):
                                    # Get member name
                                    for target in item.targets:
                                        if isinstance(target, ast.Name):
                                            member_name = target.id

                                            # Try to get comment or value
                                            comment = ""
                                            if isinstance(item.value, ast.Call):
                                                # auto() or similar
                                                comment = "auto()"
                                            elif isinstance(item.value, ast.Constant):
                                                comment = repr(item.value.value)

                                            # Look for inline comments in source
                                            try:
                                                lines = content.split("\n")
                                                for line in lines:
                                                    if (
                                                        member_name in line
                                                        and "=" in line
                                                        and "#" in line
                                                    ):
                                                        comment_part = line.split("#", 1)[1].strip()
                                                        if comment_part:
                                                            comment = comment_part
                                                        break
                                            except Exception:
                                                pass

                                            values.append((member_name, comment))

                            if values:  # Only add if we found members
                                docstring = ast.get_docstring(node) or ""
                                # Ensure path starts with thalia/
                                rel_path = str(py_file.relative_to(self.src_dir.parent))
                                if not rel_path.startswith("thalia/"):
                                    rel_path = "thalia/" + str(py_file.relative_to(self.src_dir))

                                self.enumerations.append(
                                    EnumInfo(
                                        name=node.name,
                                        docstring=docstring,
                                        values=values,
                                        file_path=rel_path,
                                        enum_type=enum_type,
                                        line_number=node.lineno,
                                    )
                                )
            except Exception:
                continue

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
                                    arg_type = (
                                        ast.unparse(arg.annotation)
                                        if hasattr(ast, "unparse")
                                        else "Any"
                                    )
                                except (AttributeError, TypeError):
                                    pass

                            # Get default value
                            defaults_offset = len(node.args.args) - len(node.args.defaults)
                            arg_idx = node.args.args.index(arg)
                            if arg_idx >= defaults_offset:
                                default_idx = arg_idx - defaults_offset
                                try:
                                    arg_default = (
                                        ast.unparse(node.args.defaults[default_idx])
                                        if hasattr(ast, "unparse")
                                        else ""
                                    )
                                except (AttributeError, TypeError, IndexError):
                                    pass

                            parameters.append((arg_name, arg_type, arg_default))

                        # Get return type
                        return_type = "ConductanceLIF"
                        if node.returns:
                            try:
                                return_type = (
                                    ast.unparse(node.returns)
                                    if hasattr(ast, "unparse")
                                    else "ConductanceLIF"
                                )
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

    def _build_cross_references(self):
        """Build cross-reference network between components.

        This analyzes the extracted data to identify relationships:
        - Which regions use which configs
        - Which regions use which learning strategies
        - Which datasets are used in which curriculum stages
        """
        # Map configs to regions that use them
        for region in self.regions:
            if region.config_class:
                # Find the config in our list
                for config in self.configs:
                    if config.name == region.config_class:
                        config.used_by.append((region.name, region.file_path, region.line_number))

        # Categorize datasets by stage
        for dataset in self.datasets:
            if "stage0" in dataset.name.lower():
                dataset.stage = "Stage 0 (Temporal)"
            elif "stage1" in dataset.name.lower():
                dataset.stage = "Stage 1 (Visual)"
            elif "stage2" in dataset.name.lower():
                dataset.stage = "Stage 2 (Grammar)"
            elif "stage3" in dataset.name.lower():
                dataset.stage = "Stage 3 (Reading)"
            elif "phonological" in dataset.name.lower():
                dataset.stage = "Stage 0 (Phonology)"

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

            f.write("\n## WeightInitializer\n\n")
            for name in WeightInitializer.list_initializers():
                f.write(f"- {name}\n")

            f.write("\n## LearningStrategyRegistry\n\n")
            for name in LearningStrategyRegistry.list_strategies(include_aliases=True):
                f.write(f"- {name}\n")

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

            f.write(
                f"Total: {len(classes)} dataset classes, {len(factories)} factory functions\n\n"
            )

            # Add statistics by stage
            f.write("## 📊 Distribution by Stage\n\n")
            from collections import Counter

            stage_counts = Counter(d.stage for d in self.datasets if d.stage)
            f.write("| Stage | Count |\n")
            f.write("|-------|-------|\n")
            for stage in [
                "Stage 0 (Temporal)",
                "Stage 0 (Phonology)",
                "Stage 1 (Visual)",
                "Stage 2 (Grammar)",
                "Stage 3 (Reading)",
            ]:
                count = stage_counts.get(stage, 0)
                f.write(f"| {stage} | {count} |\n")
            f.write("\n")

            # Add curriculum stage diagram
            f.write("## 📚 Curriculum Stages\n\n")
            f.write("```mermaid\n")
            f.write("graph LR\n")
            f.write("    S0[Stage 0: Temporal] --> S1[Stage 1: Visual]\n")
            f.write("    S1 --> S2[Stage 2: Grammar]\n")
            f.write("    S2 --> S3[Stage 3: Reading]\n")
            f.write("    S0 -.-> P[Phonology]\n")
            f.write("```\n\n")

            # Dataset Classes
            if classes:
                f.write("## Dataset Classes\n\n")
                for dataset in sorted(classes, key=lambda d: d.name):
                    # Make class name clickable
                    class_link = self._make_source_link(
                        dataset.file_path,
                        line_number=dataset.line_number,
                        display_text=f"`{dataset.name}`",
                    )
                    f.write(f"### {class_link}\n\n")
                    f.write(f"**Source**: {self._make_source_link(dataset.file_path)}\n\n")

                    if dataset.stage:
                        f.write(f"**Curriculum Stage**: {dataset.stage}\n\n")

                    f.write(f"**Description**: {dataset.docstring}\n\n")
                    f.write("---\n\n")

            # Factory Functions (grouped by stage)
            if factories:
                f.write("## Factory Functions\n\n")

                # Group by stage
                from collections import defaultdict

                by_stage = defaultdict(list)
                for factory in factories:
                    stage = factory.stage or "Other"
                    by_stage[stage].append(factory)

                # Output by stage in order
                stage_order = [
                    "Stage 0 (Temporal)",
                    "Stage 0 (Phonology)",
                    "Stage 1 (Visual)",
                    "Stage 2 (Grammar)",
                    "Stage 3 (Reading)",
                    "Other",
                ]

                for stage in stage_order:
                    if stage in by_stage:
                        f.write(f"### {stage}\n\n")
                        for factory in sorted(by_stage[stage], key=lambda d: d.name):
                            # Make function name clickable
                            func_link = self._make_source_link(
                                factory.file_path,
                                line_number=factory.line_number,
                                display_text=f"`{factory.name}()`",
                            )
                            f.write(f"#### {func_link}\n\n")
                            f.write(f"**Source**: {self._make_source_link(factory.file_path)}\n\n")

                            if factory.parameters:
                                f.write("**Parameters**:\n\n")
                                f.write("| Parameter | Type | Default |\n")
                                f.write("|-----------|------|----------|\n")
                                for param_name, param_type, param_default in factory.parameters:
                                    f.write(
                                        f"| `{param_name}` | `{param_type}` | `{param_default}` |\n"
                                    )
                                f.write("\n")

                            f.write(f"**Description**: {factory.docstring}\n\n")

                            if factory.examples:
                                f.write("**Examples**:\n\n")
                                for example in factory.examples:
                                    f.write(f"```python\n{example}\n```\n\n")
                            f.write("---\n\n")
                        f.write("\n")  # Extra space between stages

            # Add dataset usage tips
            f.write("## 💡 Dataset Usage Tips\n\n")
            f.write("### Curriculum Training\n\n")
            f.write("1. **Follow stage order**: Start with Stage 0, progress sequentially\n")
            f.write("2. **Validate each stage**: Ensure basic performance before advancing\n")
            f.write("3. **Adjust difficulty**: Use dataset parameters to control complexity\n")
            f.write("4. **Monitor learning**: Use diagnostic monitors to track progress\n\n")

            f.write("### Common Issues\n\n")
            f.write("⚠️ **Encoding mismatch**: Ensure `device` matches brain device\n\n")
            f.write("⚠️ **Language mismatch**: Verify `Language` enum for multilingual datasets\n\n")
            f.write(
                "⚠️ **Insufficient timesteps**: Use at least 50-100 timesteps for temporal data\n\n"
            )

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

            f.write(f"Total: **{len(self.monitors)}** monitors\n\n")

            # Add badges
            f.write(
                f"![Monitors](https://img.shields.io/badge/Monitors-{len(self.monitors)}-blue) "
            )
            f.write("![Diagnostics](https://img.shields.io/badge/Type-Diagnostics-yellow) ")
            f.write("![Real--time](https://img.shields.io/badge/Mode-Real--time-green)\n\n")

            f.write("## 🔍 Monitor Classes\n\n")

            for monitor in sorted(self.monitors, key=lambda m: m.name):
                # Make class name clickable
                class_link = self._make_source_link(
                    monitor.file_path,
                    line_number=monitor.line_number,
                    display_text=f"`{monitor.name}`",
                )
                f.write(f"### {class_link}\n\n")
                f.write(f"**Source**: {self._make_source_link(monitor.file_path)}\n\n")
                f.write(f"**Description**: {monitor.docstring}\n\n")

                if monitor.methods:
                    f.write("**Key Methods**:\n\n")
                    for _, signature, line_num in monitor.methods:
                        method_link = self._make_source_link(
                            monitor.file_path, line_number=line_num, display_text=signature
                        )
                        f.write(f"- {method_link}\n")
                    f.write("\n")

                if monitor.examples:
                    f.write("**Examples**:\n\n")
                    for example in monitor.examples:
                        f.write(f"```python\n{example}\n```\n\n")

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

            # Add usage guidance
            f.write("## 🎯 When to Use Each Exception\n\n")
            exception_guidance = {
                "ConfigurationError": "Invalid configuration values or missing required config",
            }

            f.write("| Exception | Use Case |\n")
            f.write("|-----------|----------|\n")
            for exc_name, guidance in exception_guidance.items():
                # Check if this exception exists in our list
                if any(e.name == exc_name for e in self.exceptions):
                    f.write(f"| `{exc_name}` | {guidance} |\n")
            f.write("\n")

            f.write("## Exception Hierarchy\n\n")

            for exception in sorted(self.exceptions, key=lambda e: e.name):
                # Make class name clickable
                class_link = self._make_source_link(
                    exception.file_path,
                    line_number=exception.line_number,
                    display_text=f"`{exception.name}`",
                )
                f.write(f"### {class_link}\n\n")
                f.write(f"**Inherits from**: `{exception.base_class}`\n\n")
                f.write(f"**Source**: {self._make_source_link(exception.file_path)}\n\n")
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
                # Make class name clickable
                class_link = self._make_source_link(
                    mixin.file_path, line_number=mixin.line_number, display_text=f"`{mixin.name}`"
                )
                f.write(f"### {class_link}\n\n")
                f.write(f"**Source**: {self._make_source_link(mixin.file_path)}\n\n")
                f.write(f"**Description**: {mixin.docstring}\n\n")

                if mixin.methods:
                    f.write("**Public Methods**:\n\n")
                    for _, signature, line_num in mixin.methods:
                        method_link = self._make_source_link(
                            mixin.file_path, line_number=line_num, display_text=signature
                        )
                        f.write(f"- {method_link}\n")
                    f.write("\n")

                f.write("---\n\n")

        print(f"✅ Generated: {output_file.relative_to(self.docs_dir.parent)}")

    def _generate_constants_reference(self):
        """Generate CONSTANTS_REFERENCE.md with enhanced biological documentation."""
        output_file = self.api_dir / "CONSTANTS_REFERENCE.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with output_file.open("w", encoding="utf-8") as f:
            f.write("# Constants Reference\n\n")
            f.write("> **Auto-generated documentation** - Do not edit manually!\n")
            f.write(f"> Last updated: {timestamp}\n")
            f.write("> Generated from: `scripts/generate_api_docs.py`\n\n")

            f.write("This document catalogs all module-level constants with biological context, ")
            f.write("ranges, and scientific references.\n\n")

            f.write(f"Total: **{len(self.constants)}** constants\n\n")

            # Add badges
            f.write(
                f"![Constants](https://img.shields.io/badge/Constants-{len(self.constants)}-blue) "
            )
            f.write("![Biological](https://img.shields.io/badge/Type-Biological-orange) ")
            f.write("![References](https://img.shields.io/badge/Citations-Scientific-green)\n\n")

            # Group by category
            from collections import defaultdict

            by_category = defaultdict(list)
            for const in self.constants:
                by_category[const.category].append(const)

            # Add visual category overview
            f.write("## 📊 Category Overview\n\n")
            f.write("```mermaid\n")
            f.write("pie title Constants by Category\n")
            for category in sorted(list(by_category.keys())[:8]):  # Top 8 categories
                count = len(by_category[category])
                safe_cat = category.replace('"', "").replace("(", "").replace(")", "")[:30]
                f.write(f'    "{safe_cat}" : {count}\n')
            f.write("```\n\n")

            f.write("## 📑 Categories\n\n")
            for category in sorted(by_category.keys()):
                f.write(f"- [{category}](#{category.lower().replace(' ', '-').replace('/', '')})\n")
            f.write("\n")

            # Generate each category
            for category in sorted(by_category.keys()):
                constants = sorted(by_category[category], key=lambda c: c.name)
                f.write(f"## {category}\n\n")

                # Create table with enhanced columns
                f.write("| Constant | Value | Biological Range | Description |\n")
                f.write("|----------|-------|------------------|-------------|\n")

                for const in constants:
                    # Truncate long values
                    value_display = (
                        const.value[:17] + "..." if len(const.value) > 20 else const.value
                    )
                    bio_range = const.biological_range if const.biological_range else "—"

                    # Truncate long descriptions
                    desc = (
                        const.docstring[:77] + "..."
                        if len(const.docstring) > 80
                        else const.docstring
                    )

                    # Escape pipes
                    value_display = value_display.replace("|", "\\|")
                    desc = desc.replace("|", "\\|")

                    f.write(f"| `{const.name}` | `{value_display}` | {bio_range} | {desc} |\n")

                f.write("\n")

                # Add detailed documentation for constants with references
                constants_with_refs = [c for c in constants if c.references]
                if constants_with_refs:
                    f.write("### Detailed Documentation\n\n")
                    for const in constants_with_refs:
                        f.write(f"#### `{const.name}`\n\n")
                        f.write(f"**Value**: `{const.value}`")
                        if const.biological_range:
                            f.write(f" (Biological range: {const.biological_range})")
                        f.write("\n\n")
                        f.write(f"{const.docstring}\n\n")
                        if const.references:
                            f.write("**References**: " + ", ".join(const.references) + "\n\n")
                        f.write(f"**Source**: `{self._make_source_link(const.file_path)}`\n\n")

                f.write("---\n\n")

            # Add bibliography section if we have references
            all_refs = set()
            for const in self.constants:
                all_refs.update(const.references)

            if all_refs:
                f.write("## References\n\n")
                f.write("Scientific references cited in constant definitions:\n\n")
                for ref in sorted(all_refs):
                    f.write(f"- {ref}\n")
                f.write("\n")

        print(f"✅ Generated: {output_file.relative_to(self.docs_dir.parent)}")

    def _generate_enumerations_reference(self):
        """Generate ENUMERATIONS_REFERENCE.md."""
        output_file = self.api_dir / "ENUMERATIONS_REFERENCE.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with output_file.open("w", encoding="utf-8") as f:
            f.write("# Enumerations Reference\n\n")
            f.write("> **Auto-generated documentation** - Do not edit manually!\n")
            f.write(f"> Last updated: {timestamp}\n")
            f.write("> Generated from: `scripts/generate_api_docs.py`\n\n")

            f.write("This document catalogs all enumeration types used in Thalia.\n\n")

            f.write(f"Total: {len(self.enumerations)} enumerations\n\n")

            # Group by category (infer from file path)
            from collections import defaultdict

            by_category = defaultdict(list)

            for enum in self.enumerations:
                # Infer category from path
                parts = enum.file_path.split("\\")
                if len(parts) > 1:
                    category = parts[0].replace("_", " ").title()
                else:
                    category = "Core"
                by_category[category].append(enum)

            # Add table of contents
            f.write("## 📑 Table of Contents\n\n")
            f.write("Jump to category:\n\n")
            for category in sorted(by_category.keys()):
                count = len(by_category[category])
                # Create anchor-safe category name
                anchor = category.lower().replace(" ", "-").replace("/", "")
                f.write(f"- [{category}](#{anchor}) ({count} enums)\n")
            f.write("\n")

            f.write("## Enumerations by Category\n\n")

            for category in sorted(by_category.keys()):
                enums = by_category[category]
                f.write(f"### {category}\n\n")

                for enum in sorted(enums, key=lambda e: e.name):
                    # Make enum name clickable
                    enum_link = self._make_source_link(
                        enum.file_path, line_number=enum.line_number, display_text=f"`{enum.name}`"
                    )
                    f.write(f"#### {enum_link} ({enum.enum_type})\n\n")

                    if enum.docstring:
                        f.write(f"{enum.docstring}\n\n")

                    f.write(f"**Source**: {self._make_source_link(enum.file_path)}\n\n")

                    f.write("**Members**:\n\n")
                    for member_name, comment in enum.values:
                        if comment:
                            f.write(f"- `{member_name}` — {comment}\n")
                        else:
                            f.write(f"- `{member_name}`\n")

                    f.write("\n---\n\n")

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

    def _extract_region_registration(self, file_path: Path, content: str):
        """Extract region registration details."""
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call):
                            if (
                                hasattr(decorator.func, "id")
                                and decorator.func.id == "register_region"
                            ):
                                name = (
                                    ast.literal_eval(decorator.args[0]) if decorator.args else None
                                )
                                aliases = []
                                config_class = None

                                for keyword in decorator.keywords:
                                    if keyword.arg == "aliases":
                                        aliases = ast.literal_eval(keyword.value)
                                    elif keyword.arg == "config_class":
                                        if hasattr(keyword.value, "id"):
                                            config_class = keyword.value.id

                                docstring = ast.get_docstring(node) or "No docstring"

                                # Try to find config class definition in the same file
                                config_file = None
                                config_line = None
                                if config_class:
                                    for n in ast.walk(tree):
                                        if isinstance(n, ast.ClassDef) and n.name == config_class:
                                            config_file = str(
                                                file_path.relative_to(self.src_dir.parent)
                                            )
                                            config_line = n.lineno
                                            break
                                    # If not found in same file, search elsewhere
                                    if not config_file:
                                        config_file, config_line = self._find_config_class_location(
                                            config_class
                                        )

                                self.regions.append(
                                    RegistryItem(
                                        name=name or node.name,
                                        aliases=aliases,
                                        class_name=node.name,
                                        file_path=str(file_path.relative_to(self.src_dir.parent)),
                                        docstring=docstring.split("\n")[0],  # First line only
                                        config_class=config_class or "None",
                                        line_number=node.lineno,
                                        config_file_path=config_file,
                                        config_line_number=config_line,
                                    )
                                )
        except Exception as e:
            print(f"Warning: Could not parse {file_path}: {e}")

    def _extract_state_dict_structure(
        self, func_node: ast.FunctionDef, content: str
    ) -> List[StateField]:
        """Extract state dict keys."""
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
                                    fields.append(
                                        StateField(
                                            key=key,
                                            type_hint=type_hint,
                                            description="",
                                            required=True,
                                            example="",
                                        )
                                    )

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
                                fields.append(
                                    StateField(
                                        key=key,
                                        type_hint=type_hint,
                                        description="",
                                        required=True,
                                        example="",
                                    )
                                )

        return fields

    def _extract_aliases_from_typing_module(self, typing_file: Path):
        """Extract type aliases from thalia.typing module."""
        try:
            content = typing_file.read_text(encoding="utf-8")
            tree = ast.parse(content)
            lines = content.split("\n")

            for node in ast.walk(tree):
                # Look for section comments like # ============ Component Organization ============
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                    if isinstance(node.value.value, str):
                        # Check if this is a section header comment
                        pass

                # Look for: TypeName = Type[...]
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            name = target.id
                            # Skip private variables and non-type-aliases
                            if name.startswith("_") or name in ("TYPE_CHECKING",):
                                continue

                            # Check if it's a type alias (uppercase first letter)
                            if name[0].isupper() and name not in ("True", "False", "None"):
                                definition = self._get_source_segment(content, node.value)
                                if definition:
                                    # Extract docstring (next line after assignment)
                                    description = self._extract_docstring_after_assignment(
                                        lines, node.lineno
                                    )

                                    # Infer category from surrounding comments
                                    category = self._infer_category_from_typing_module(
                                        lines, node.lineno
                                    )

                                    self.type_aliases.append(
                                        TypeAliasInfo(
                                            name=name,
                                            definition=definition,
                                            description=description,
                                            file_path="thalia/typing.py",
                                            category=category,
                                        )
                                    )
        except Exception as e:
            print(f"Warning: Could not parse thalia.typing: {e}")

    def _extract_docstring_after_assignment(self, lines: List[str], lineno: int) -> str:
        """Extract docstring from the line(s) after an assignment."""
        # lineno is 1-based, convert to 0-based
        idx = lineno
        if idx < len(lines):
            next_line = lines[idx].strip()
            # Check for triple-quoted docstring
            if next_line.startswith('"""'):
                docstring_lines = [next_line[3:]]
                idx += 1
                while idx < len(lines):
                    line = lines[idx].strip()
                    if line.endswith('"""'):
                        docstring_lines.append(line[:-3])
                        break
                    docstring_lines.append(line)
                    idx += 1
                return " ".join(docstring_lines).strip()
        return ""

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

                        source_port = None
                        target_port = None

                        # Check for keyword arguments
                        for keyword in node.keywords:
                            if keyword.arg == "source_port":
                                source_port = self._get_string_value(keyword.value)
                            elif keyword.arg == "target_port":
                                target_port = self._get_string_value(keyword.value)

                        if source and target:
                            self.component_relations.append(
                                ComponentRelation(
                                    source=source,
                                    target=target,
                                    preset_name=preset_name,
                                    source_port=source_port,
                                    target_port=target_port,
                                )
                            )

    # =================================================================
    # Type inference methods
    # =================================================================

    def _infer_category_from_typing_module(self, lines: List[str], lineno: int) -> str:
        """Infer category from section comment above assignment."""
        # Look backwards for # ==== Section Name ==== comments
        for i in range(lineno - 2, max(0, lineno - 20), -1):
            line = lines[i].strip()
            if line.startswith("# ===") and "===" in line:
                # Extract category name between the ===
                category = line.replace("#", "").replace("=", "").strip()
                return category
        return "Other"

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
        elif isinstance(node, ast.Constant):
            # Handle all constant types
            if isinstance(node.value, (int, float, str, bool)):
                return type(node.value).__name__
            return "Any"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                method_name = node.func.attr
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

    # =================================================================
    # Generic AST Helpers
    # =================================================================

    def _get_source_segment(self, content: str, node: ast.AST) -> str:
        """Extract source code for an AST node."""
        try:
            return ast.unparse(node)
        except Exception:
            # Fallback for older Python or complex nodes
            return ""

    def _get_string_value(self, node: ast.AST) -> Optional[str]:
        """Extract string value from AST node."""
        if isinstance(node, ast.Constant):
            return str(node.value) if isinstance(node.value, str) else None
        elif isinstance(node, ast.Str):
            return str(node.s) if isinstance(node.s, str) else None
        return None

    def _get_containing_class(self, func_node: ast.FunctionDef, tree: ast.AST) -> str:
        """Get the class containing this function."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if item == func_node:
                        return node.name
        return ""

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
