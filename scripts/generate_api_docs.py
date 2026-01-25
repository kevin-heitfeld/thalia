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
15. ENUMERATIONS_REFERENCE.md - All enumeration types
16. STATE_CLASSES_REFERENCE.md - State classes for checkpointing
17. NEURON_FACTORIES_REFERENCE.md - Neuron factory functions
18. COMPUTE_FUNCTIONS_REFERENCE.md - Utility compute functions (NEW)
19. VISUALIZATION_REFERENCE.md - Visualization/plotting functions (NEW)
20. API_INDEX.md - Comprehensive searchable index
21. DEPENDENCY_GRAPH.md - Module dependency visualization
22. ARCHITECTURE_GUIDE.md - System architecture diagrams

Run this script whenever components are added/modified to keep docs synchronized.
"""

import ast
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RegistryItem:
    """Component registered in ComponentRegistry."""

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
    used_by: Optional[List[Tuple[str, str, int]]] = None  # (component_name, file_path, line_number)
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
class MonitorInfo:
    """Diagnostic monitor class."""

    name: str
    class_name: str
    docstring: str
    file_path: str
    methods: List[Tuple[str, str, int]]  # (method_name, signature, line_number)
    line_number: int = 0
    examples: Optional[List[str]] = None

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
class MixinInfo:
    """Mixin class providing functionality."""

    name: str
    docstring: str
    methods: List[Tuple[str, str, int]]  # (method_name, signature, line_number)
    file_path: str
    line_number: int = 0
    examples: Optional[List[str]] = None

    def __post_init__(self):
        if self.examples is None:
            self.examples = []


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
class ProtocolInfo:
    """Protocol/interface definition."""

    name: str
    docstring: str
    methods: List[Tuple[str, str]]  # (method_name, signature)
    file_path: str
    is_runtime_checkable: bool
    line_number: int = 0


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


@dataclass
class NeuronFactory:
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


@dataclass
class StateClassInfo:
    """State class for regions/pathways."""

    name: str
    base_class: str  # "RegionState", "BaseRegionState", "PathwayState"
    docstring: str
    fields: List[Tuple[str, str, str]]  # (name, type, default)
    state_version: int
    file_path: str
    component_type: str  # "region" or "pathway"
    line_number: int = 0


@dataclass
class ComputeFunction:
    """Utility compute function."""

    name: str
    parameters: List[Tuple[str, str, str]]  # (name, type, default)
    return_type: str
    docstring: str
    file_path: str
    category: str  # "oscillator", "neuromodulation", "sizing", "diagnostic"
    biological_context: Optional[str] = None
    line_number: int = 0
    examples: Optional[List[str]] = None

    def __post_init__(self):
        if self.examples is None:
            self.examples = []


@dataclass
class VisualizationFunction:
    """Visualization/plotting function."""

    name: str
    parameters: List[Tuple[str, str, str]]  # (name, type, default)
    return_type: str
    docstring: str
    file_path: str
    category: str  # "topology", "training", "diagnostics", "critical_period"
    line_number: int = 0
    examples: Optional[List[str]] = None

    def __post_init__(self):
        if self.examples is None:
            self.examples = []


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
        self.enumerations: List[EnumInfo] = []
        self.state_classes: List[StateClassInfo] = []
        self.neuron_factories: List[NeuronFactory] = []
        self.compute_functions: List[ComputeFunction] = []
        self.visualization_functions: List[VisualizationFunction] = []
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
        self._find_enumerations()
        self._find_state_classes()
        self._find_neuron_factories()
        self._find_compute_functions()
        self._find_visualization_functions()

        # Build cross-references
        self._build_cross_references()

        # Analyze dependencies and architecture
        self._analyze_dependencies()

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
        self._generate_enumerations_reference()
        self._generate_state_classes_reference()
        self._generate_neuron_factories_reference()
        self._generate_compute_functions_reference()
        self._generate_visualization_reference()

        # Generate comprehensive index
        self._generate_api_index()

        # Generate architecture visualizations
        self._generate_dependency_graph()
        self._generate_architecture_guide()

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

    def _extract_pathway_registration(self, file_path: Path, content: str):
        """Extract pathway registration details."""
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call):
                            if (
                                hasattr(decorator.func, "id")
                                and decorator.func.id == "register_pathway"
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

                                self.pathways.append(
                                    RegistryItem(
                                        name=name or node.name,
                                        aliases=aliases,
                                        class_name=node.name,
                                        file_path=str(file_path.relative_to(self.src_dir.parent)),
                                        docstring=docstring.split("\n")[0],
                                        config_class=config_class or "None",
                                        line_number=node.lineno,
                                        config_file_path=config_file,
                                        config_line_number=config_line,
                                    )
                                )
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

        # Categorize strategies by their typical use
        strategy_categories = {
            "create_cortex_strategy": ["cortex", "visual_cortex", "auditory_cortex"],
            "create_striatum_strategy": ["striatum"],
            "create_hippocampus_strategy": ["hippocampus"],
            "create_cerebellum_strategy": ["cerebellum"],
            "create_prefrontal_strategy": ["prefrontal", "pfc"],
        }

        for strategy in self.strategies:
            if strategy.name in strategy_categories:
                strategy.used_by = strategy_categories[strategy.name]

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

    def _generate_component_catalog(self):
        """Generate COMPONENT_CATALOG.md."""
        output_file = self.api_dir / "COMPONENT_CATALOG.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with output_file.open("w", encoding="utf-8") as f:
            f.write("# Component Catalog\n\n")
            f.write("> **Auto-generated documentation** - Do not edit manually!\n")
            f.write(f"> Last updated: {timestamp}\n")
            f.write("> Generated from: `scripts/generate_api_docs.py`\n\n")

            # Add badges
            f.write(
                f"![Components](https://img.shields.io/badge/Regions-{len(self.regions)}-blue) "
            )
            f.write(
                f"![Pathways](https://img.shields.io/badge/Pathways-{len(self.pathways)}-green) "
            )
            f.write(
                "![Auto-Generated](https://img.shields.io/badge/Status-Auto--Generated-brightgreen)\n\n"
            )

            f.write("This document catalogs all registered brain regions and pathways ")
            f.write("in the Thalia component registry.\n\n")

            # Add statistics section
            f.write("## 📊 Statistics\n\n")
            f.write("| Metric | Count |\n")
            f.write("|--------|-------|\n")
            f.write(f"| **Total Regions** | {len(self.regions)} |\n")
            f.write(f"| **Total Pathways** | {len(self.pathways)} |\n")
            regions_with_config = sum(1 for r in self.regions if r.config_class != "None")
            f.write(f"| **Regions with Custom Config** | {regions_with_config} |\n")
            regions_with_aliases = sum(1 for r in self.regions if r.aliases)
            f.write(f"| **Components with Aliases** | {regions_with_aliases} |\n")
            f.write("\n")

            # Add performance metrics if available
            if self.profile_data:
                f.write("## ⚡ Performance Metrics\n\n")

                # Full forward pass timing
                if "_full_forward_pass" in self.profile_data:
                    exec_data = self.profile_data["_full_forward_pass"]["execution"]
                    f.write("**Full Forward Pass (1 timestep)**:\n\n")
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    f.write(f"| Mean | {exec_data['mean_ms']:.2f} ms |\n")
                    f.write(f"| Median | {exec_data['median_ms']:.2f} ms |\n")
                    f.write(f"| P95 | {exec_data['p95_ms']:.2f} ms |\n")
                    f.write(f"| Min | {exec_data['min_ms']:.2f} ms |\n")
                    f.write(f"| Max | {exec_data['max_ms']:.2f} ms |\n")
                    f.write("\n")

                # Code complexity summary
                f.write("**Component Implementation Complexity**:\n\n")
                f.write("| Component | Type | Lines | Functions | Classes | Complexity |\n")
                f.write("|-----------|------|-------|-----------|---------|------------|\n")

                # Get complexity data for implementation files - separate by type
                regions_data = [
                    (k, v)
                    for k, v in self.profile_data.items()
                    if k.startswith("region_") and v["code"]["lines"] > 0
                ]
                pathways_data = [
                    (k, v)
                    for k, v in self.profile_data.items()
                    if k.startswith("pathway_") and v["code"]["lines"] > 0
                ]

                if regions_data:
                    f.write("| **REGIONS** | | | | | |\n")
                    for key, data in sorted(
                        regions_data, key=lambda x: -x[1]["code"]["complexity"]
                    )[:5]:
                        code = data["code"]
                        display_name = key.replace("region_", "")
                        f.write(
                            f"| `{display_name}` | Region | {code['lines']} | {code['functions']} | {code['classes']} | {code['complexity']} |\n"
                        )
                    if len(regions_data) > 5:
                        f.write(f"| *...+{len(regions_data)-5} more* | | | | | |\n")

                if pathways_data:
                    f.write("| **PATHWAYS** | | | | | |\n")
                    for key, data in sorted(
                        pathways_data, key=lambda x: -x[1]["code"]["complexity"]
                    )[:3]:
                        code = data["code"]
                        display_name = key.replace("pathway_", "")
                        f.write(
                            f"| `{display_name}` | Pathway | {code['lines']} | {code['functions']} | {code['classes']} | {code['complexity']} |\n"
                        )
                    if len(pathways_data) > 3:
                        f.write(f"| *...+{len(pathways_data)-3} more* | | | | | |\n")

                f.write("\n")
                f.write("*Generated by: `scripts/profile_components.py`*\n\n")

            # Add test coverage metrics if available
            if self.coverage_data:
                f.write("## 🧪 Test Coverage\n\n")

                # Overall statistics
                total_components = len(self.coverage_data)
                tested_components = sum(1 for c in self.coverage_data.values() if c["has_tests"])
                total_tests = sum(c["test_count"] for c in self.coverage_data.values())
                with_edge_cases = sum(1 for c in self.coverage_data.values() if c["has_edge_cases"])
                with_integration = sum(
                    1 for c in self.coverage_data.values() if c["has_integration"]
                )
                avg_coverage = sum(
                    c["coverage_percent"] for c in self.coverage_data.values() if c["has_tests"]
                ) / max(tested_components, 1)

                f.write("**Overall Test Statistics**:\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                f.write(f"| Total Components | {total_components} |\n")
                f.write(
                    f"| Components with Tests | {tested_components} ({tested_components/total_components*100:.1f}%) |\n"
                )
                f.write(f"| Total Test Functions | {total_tests} |\n")
                f.write(f"| Average Coverage | {avg_coverage:.1f}% |\n")
                f.write(f"| Components with Edge Case Tests | {with_edge_cases} |\n")
                f.write(f"| Components with Integration Tests | {with_integration} |\n")
                f.write("\n")

                # Helper function to get test file path
                def get_test_link(comp_name: str) -> str:
                    """Generate relative path to test file."""
                    comp_type, name = comp_name.split(".", 1)
                    if comp_type == "regions":
                        return f"../../tests/unit/regions/test_{name}.py"
                    elif comp_type == "pathways":
                        return f"../../tests/unit/pathways/test_{name}.py"
                    elif comp_type == "core":
                        return f"../../tests/unit/core/test_{name}.py"
                    elif comp_type == "learning":
                        return f"../../tests/unit/learning/test_{name}.py"
                    return ""

                # Highlight well-tested components
                well_tested = [
                    (k, v)
                    for k, v in self.coverage_data.items()
                    if v["test_count"] >= 50 or (v["has_edge_cases"] and v["has_integration"])
                ]
                if well_tested:
                    f.write("**Well-Tested Components** (50+ tests OR edge+integration tests):\n\n")
                    for comp_name, data in sorted(well_tested, key=lambda x: -x[1]["test_count"])[
                        :5
                    ]:
                        display_name = comp_name.split(".", 1)[1]
                        tests = data["test_count"]
                        edge = "✓" if data["has_edge_cases"] else ""
                        integ = "✓" if data["has_integration"] else ""
                        badges = " ".join(
                            filter(None, [edge and "🎯Edge", integ and "🔗Integration"])
                        )
                        test_link = get_test_link(comp_name)
                        f.write(f"- `{display_name}`: [{tests} tests]({test_link}) {badges}\n")
                    f.write("\n")

                # Highlight components needing tests
                needs_tests = [k for k, v in self.coverage_data.items() if not v["has_tests"]]
                if needs_tests:
                    f.write(f"**Components Needing Tests** ({len(needs_tests)} components):\n\n")
                    f.write("<details>\n<summary>Click to expand list</summary>\n\n")
                    for comp_name in sorted(needs_tests):
                        display_name = comp_name.split(".", 1)[1]
                        comp_type = comp_name.split(".", 1)[0]
                        f.write(f"- `{display_name}` ({comp_type})\n")
                    f.write("\n</details>\n\n")

                # Coverage by component type (collapsed)
                f.write(
                    "<details>\n<summary>📋 <b>Detailed Coverage by Component Type</b> (click to expand)</summary>\n\n"
                )
                f.write("\n| Component | Tests | Coverage | Edge Cases | Integration |\n")
                f.write("|-----------|-------|----------|------------|-------------|\n")

                # Group by type
                for component_type in ["regions", "pathways", "core", "learning"]:
                    relevant_items = {
                        k: v
                        for k, v in self.coverage_data.items()
                        if k.startswith(f"{component_type}.")
                    }
                    if relevant_items:
                        # Add type header
                        f.write(f"| **{component_type.upper()}** | | | | |\n")
                        for comp_name in sorted(relevant_items.keys()):
                            data = relevant_items[comp_name]
                            display_name = comp_name.split(".", 1)[1]
                            tests = data["test_count"]
                            # Add link to test file if tests exist
                            if tests > 0:
                                test_link = get_test_link(comp_name)
                                test_display = f"[{tests}]({test_link})"
                            else:
                                test_display = str(tests)
                            coverage = (
                                f"{data['coverage_percent']:.1f}%"
                                if data["coverage_percent"] > 0
                                else "-"
                            )
                            edge = "✓" if data["has_edge_cases"] else "✗"
                            integ = "✓" if data["has_integration"] else "✗"
                            f.write(
                                f"| `{display_name}` | {test_display} | {coverage} | {edge} | {integ} |\n"
                            )

                f.write("\n</details>\n\n")
                f.write("*Generated by: `scripts/analyze_coverage.py`*\n\n")

                # Note about coverage percentages
                if avg_coverage == 0:
                    f.write(
                        "> **Note**: Coverage percentages show 0% because pytest timed out during execution. "
                    )
                    f.write(
                        "Test counts and quality indicators (edge cases, integration tests) are accurate.\n\n"
                    )  # Add component hierarchy diagram
            f.write("## 🧩 Component Hierarchy\n\n")
            f.write("```mermaid\n")
            f.write("graph TB\n")
            f.write("    Brain[DynamicBrain]\n")
            f.write("    Brain --> Regions[Neural Regions]\n")
            f.write("    Brain --> Pathways[Axonal Pathways]\n")
            for region in sorted(self.regions[:5], key=lambda r: r.name):  # Show first 5
                safe_name = region.name.replace("-", "_").replace(" ", "_")
                f.write(f"    Regions --> {safe_name}[{region.name}]\n")
            if len(self.regions) > 5:
                f.write(f"    Regions --> More[... +{len(self.regions)-5} more]\n")
            for pathway in sorted(self.pathways[:3], key=lambda p: p.name):  # Show first 3
                safe_name = pathway.name.replace("-", "_").replace(" ", "_")
                f.write(f"    Pathways --> {safe_name}[{pathway.name}]\n")
            if len(self.pathways) > 3:
                f.write(f"    Pathways --> MoreP[... +{len(self.pathways)-3} more]\n")
            f.write("```\n\n")

            # Table of contents
            f.write("## 📑 Contents\n\n")
            f.write("- [Registered Regions](#registered-regions)\n")
            f.write("- [Registered Pathways](#registered-pathways)\n\n")

            # Regions
            f.write("## Registered Regions\n\n")
            f.write(f"Total: **{len(self.regions)}** regions\n\n")

            # Sort regions alphabetically
            for region in sorted(self.regions, key=lambda r: r.name):
                f.write(f"### `{region.name}`\n\n")
                class_link = self._make_source_link(
                    region.file_path, region.line_number, region.class_name
                )
                f.write(f"**Class**: {class_link}\n\n")

                if region.aliases:
                    f.write(f"**Aliases**: `{', '.join(region.aliases)}`\n\n")

                # Config class with link if available
                if (
                    region.config_class != "None"
                    and region.config_file_path
                    and region.config_line_number
                ):
                    config_link = self._make_source_link(
                        region.config_file_path, region.config_line_number, region.config_class
                    )
                    f.write(f"**Config Class**: {config_link}\n\n")
                else:
                    f.write(f"**Config Class**: `{region.config_class}`\n\n")
                f.write(f"**Source**: {self._make_source_link(region.file_path)}\n\n")
                f.write(f"**Description**: {region.docstring}\n\n")

                f.write("---\n\n")

            # Pathways
            f.write("## Registered Pathways\n\n")
            f.write(f"Total: **{len(self.pathways)}** pathways\n\n")

            for pathway in sorted(self.pathways, key=lambda p: p.name):
                f.write(f"### `{pathway.name}`\n\n")
                class_link = self._make_source_link(
                    pathway.file_path, pathway.line_number, pathway.class_name
                )
                f.write(f"**Class**: {class_link}\n\n")

                if pathway.aliases:
                    f.write(f"**Aliases**: `{', '.join(pathway.aliases)}`\n\n")

                # Config class with link if available
                if (
                    pathway.config_class != "None"
                    and pathway.config_file_path
                    and pathway.config_line_number
                ):
                    config_link = self._make_source_link(
                        pathway.config_file_path, pathway.config_line_number, pathway.config_class
                    )
                    f.write(f"**Config Class**: {config_link}\n\n")
                else:
                    f.write(f"**Config Class**: `{pathway.config_class}`\n\n")
                f.write(f"**Source**: {self._make_source_link(pathway.file_path)}\n\n")
                f.write(f"**Description**: {pathway.docstring}\n\n")

                f.write("---\n\n")

            # Add See Also section
            f.write("## See Also\n\n")
            f.write(
                "- [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) - Configuration classes for components\n"
            )
            f.write(
                "- [CONSTANTS_REFERENCE.md](CONSTANTS_REFERENCE.md) - Biological constants used by components\n"
            )
            f.write(
                "- [NEURON_FACTORIES_REFERENCE.md](NEURON_FACTORIES_REFERENCE.md) - Neuron populations used by regions\n"
            )
            f.write(
                "- [LEARNING_STRATEGIES_API.md](LEARNING_STRATEGIES_API.md) - Learning rules used by components\n"
            )
            f.write(
                "- [STATE_CLASSES_REFERENCE.md](STATE_CLASSES_REFERENCE.md) - State classes for checkpointing\n\n"
            )

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

            # Add quick reference table
            f.write("## 📋 Quick Reference\n\n")
            f.write("| Strategy | Best For | Modulator | Biological Accuracy |\n")
            f.write("|----------|----------|-----------|--------------------|\n")
            f.write("| `create_cortex_strategy()` | Cortical learning | Optional | ⭐⭐⭐⭐⭐ |\n")
            f.write("| `create_striatum_strategy()` | Reward learning | Dopamine | ⭐⭐⭐⭐⭐ |\n")
            f.write("| `create_hippocampus_strategy()` | Memory | Acetylcholine | ⭐⭐⭐⭐ |\n")
            f.write("| `create_cerebellum_strategy()` | Motor learning | None | ⭐⭐⭐⭐ |\n")
            f.write(
                "| `create_prefrontal_strategy()` | Executive control | Dopamine | ⭐⭐⭐⭐ |\n\n"
            )

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

            # Add See Also section
            f.write("## See Also\n\n")
            f.write(
                "- [COMPONENT_CATALOG.md](COMPONENT_CATALOG.md) - Components using these strategies\n"
            )
            f.write(
                "- [CONSTANTS_REFERENCE.md](CONSTANTS_REFERENCE.md) - Learning rate constants\n"
            )
            f.write(
                "- [PROTOCOLS_REFERENCE.md](PROTOCOLS_REFERENCE.md) - LearningStrategy protocol\n"
            )
            f.write("- [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) - Examples of strategy usage\n\n")

            # Add best practices section
            f.write("## 💡 Best Practices\n\n")
            f.write("### Choosing a Learning Strategy\n\n")
            f.write(
                "1. **For cortical regions**: Use `create_cortex_strategy()` with STDP+BCM composite\n"
            )
            f.write(
                "2. **For reward learning**: Use `create_striatum_strategy()` with dopamine modulation\n"
            )
            f.write(
                "3. **For memory formation**: Use `create_hippocampus_strategy()` with one-shot capability\n"
            )
            f.write(
                "4. **For motor learning**: Use `create_cerebellum_strategy()` with error correction\n"
            )
            f.write(
                "5. **For executive function**: Use `create_prefrontal_strategy()` with gated plasticity\n\n"
            )

            f.write("### Parameter Tuning Tips\n\n")
            f.write("- **Learning rate**: Start with default values, reduce if unstable\n")
            f.write("- **Time constants**: Match biological ranges (10-100ms for STDP)\n")
            f.write("- **Modulator sensitivity**: Tune based on task reward structure\n")
            f.write("- **Testing**: Always validate with curriculum training stages\n\n")

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

            # Add configuration tips section
            f.write("## 💡 Configuration Best Practices\n\n")
            f.write("### Common Patterns\n\n")
            f.write(
                "1. **Start with defaults**: All configs have biologically-motivated defaults\n"
            )
            f.write("2. **Override selectively**: Only change what's needed for your task\n")
            f.write("3. **Validate early**: Use config validation before training\n")
            f.write("4. **Document changes**: Keep notes on why you changed defaults\n\n")

            f.write("### Performance Considerations\n\n")
            f.write("- **Layer sizes**: Larger = more capacity but slower training\n")
            f.write("- **Sparsity**: Higher sparsity = faster but less connectivity\n")
            f.write("- **Learning rates**: Too high = instability, too low = slow learning\n")
            f.write("- **Time constants**: Should match biological ranges (ms scale)\n\n")

            f.write("### Common Pitfalls\n\n")
            f.write("⚠️ **Too small networks**: Use at least 64 neurons per region\n\n")
            f.write(
                "⚠️ **Mismatched time scales**: Keep tau values in biological range (5-200ms)\n\n"
            )
            f.write("⚠️ **Extreme learning rates**: Stay within 0.0001-0.01 range\n\n")
            f.write("⚠️ **Disabled plasticity**: Ensure learning strategies are enabled\n\n")

            # Add related documentation section
            f.write("## 📚 Related Documentation\n\n")
            f.write(
                "- [COMPONENT_CATALOG.md](COMPONENT_CATALOG.md) - Components using these configs\n"
            )
            f.write(
                "- [LEARNING_STRATEGIES_API.md](LEARNING_STRATEGIES_API.md) - Learning rules and parameters\n"
            )
            f.write("- [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) - Configuration examples\n")
            f.write(
                "- [ENUMERATIONS_REFERENCE.md](ENUMERATIONS_REFERENCE.md) - Enum types used in configs\n\n"
            )

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
                        examples = self._extract_examples_from_docstring(docstring)

                        # Extract method signatures with line numbers
                        methods = []
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                                params = [arg.arg for arg in item.args.args if arg.arg != "self"]
                                signature = f"{item.name}({', '.join(params)})"
                                methods.append((item.name, signature, item.lineno))

                        self.monitors.append(
                            MonitorInfo(
                                name=node.name,
                                class_name=node.name,
                                docstring=docstring.split("\n")[0],
                                file_path=str(py_file.relative_to(self.src_dir.parent)),
                                methods=methods[:5],  # First 5 public methods
                                line_number=node.lineno,
                                examples=examples,
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

                        examples = self._extract_examples_from_docstring(docstring)

                        for item in node.body:
                            if isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                                # Extract method signature with line number
                                params = [arg.arg for arg in item.args.args if arg.arg != "self"]
                                signature = f"{item.name}({', '.join(params)})"
                                methods.append((item.name, signature, item.lineno))

                        self.mixins.append(
                            MixinInfo(
                                name=mixin_name,
                                docstring=docstring.split("\n")[0],
                                methods=methods[:10],  # First 10 methods
                                file_path=str(mixin_file.relative_to(self.src_dir.parent)),
                                line_number=node.lineno,
                                examples=examples,
                            )
                        )
                        break

            except Exception as e:
                print(f"Warning: Could not parse {mixin_file}: {e}")

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
                            NeuronFactory(
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

    def _find_compute_functions(self):
        """Find all compute_* utility functions."""
        compute_files = [
            (self.src_dir / "utils" / "oscillator_utils.py", "oscillator"),
            (self.src_dir / "neuromodulation" / "constants.py", "neuromodulation"),
            (self.src_dir / "config" / "region_sizes.py", "sizing"),
        ]

        for py_file, category in compute_files:
            if not py_file.exists():
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name.startswith("compute_"):
                        # Extract parameters
                        parameters = []
                        for arg in node.args.args:
                            arg_name = arg.arg
                            arg_type = "Any"
                            arg_default = ""

                            if arg.annotation:
                                try:
                                    arg_type = (
                                        ast.unparse(arg.annotation)
                                        if hasattr(ast, "unparse")
                                        else "Any"
                                    )
                                except (AttributeError, TypeError):
                                    pass

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
                        return_type = "Any"
                        if node.returns:
                            try:
                                return_type = (
                                    ast.unparse(node.returns) if hasattr(ast, "unparse") else "Any"
                                )
                            except (AttributeError, TypeError):
                                pass

                        # Extract docstring
                        docstring = ast.get_docstring(node) or "No docstring"

                        # Extract biological context (references in docstring)
                        biological_context = None
                        if (
                            "References:" in docstring
                            or "Biological" in docstring
                            or "neuroscience" in docstring.lower()
                        ):
                            lines = docstring.split("\n")
                            for i, line in enumerate(lines):
                                if "References:" in line or "Biological" in line:
                                    biological_context = "\n".join(
                                        lines[i : min(i + 3, len(lines))]
                                    )
                                    break

                        # Extract examples
                        examples = self._extract_examples_from_docstring(docstring)

                        self.compute_functions.append(
                            ComputeFunction(
                                name=node.name,
                                parameters=parameters,
                                return_type=return_type,
                                docstring=docstring.split("\n")[0],  # First line
                                file_path=str(py_file.relative_to(self.src_dir.parent)),
                                category=category,
                                biological_context=biological_context,
                                line_number=node.lineno,
                                examples=examples,
                            )
                        )

            except Exception as e:
                print(f"Warning: Could not parse {py_file}: {e}")

    def _find_visualization_functions(self):
        """Find all visualization/plotting functions."""
        viz_files = [
            (self.src_dir / "visualization" / "network_graph.py", "topology"),
            (self.src_dir / "training" / "visualization" / "monitor.py", "training"),
            (self.src_dir / "training" / "visualization" / "live_diagnostics.py", "diagnostics"),
            (
                self.src_dir / "training" / "visualization" / "critical_period_plots.py",
                "critical_period",
            ),
        ]

        for py_file, category in viz_files:
            if not py_file.exists():
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Look for visualization functions (plot_, visualize_, quick_)
                        if any(
                            node.name.startswith(prefix)
                            for prefix in ["plot_", "visualize_", "export_", "quick_"]
                        ):
                            # Extract parameters
                            parameters = []
                            for arg in node.args.args:
                                arg_name = arg.arg
                                arg_type = "Any"
                                arg_default = ""

                                if arg.annotation:
                                    try:
                                        arg_type = (
                                            ast.unparse(arg.annotation)
                                            if hasattr(ast, "unparse")
                                            else "Any"
                                        )
                                    except (AttributeError, TypeError):
                                        pass

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
                            return_type = "None"
                            if node.returns:
                                try:
                                    return_type = (
                                        ast.unparse(node.returns)
                                        if hasattr(ast, "unparse")
                                        else "None"
                                    )
                                except (AttributeError, TypeError):
                                    pass

                            # Extract docstring and examples
                            docstring = ast.get_docstring(node) or "No docstring"
                            examples = self._extract_examples_from_docstring(docstring)

                            self.visualization_functions.append(
                                VisualizationFunction(
                                    name=node.name,
                                    parameters=parameters,
                                    return_type=return_type,
                                    docstring=docstring.split("\n")[0],  # First line
                                    file_path=str(py_file.relative_to(self.src_dir.parent)),
                                    category=category,
                                    line_number=node.lineno,
                                    examples=examples,
                                )
                            )

            except Exception as e:
                print(f"Warning: Could not parse {py_file}: {e}")

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
                            (hasattr(base, "id") and base.id == "Protocol")
                            or (hasattr(base, "attr") and base.attr == "Protocol")
                            for base in node.bases
                        )

                        if is_protocol:
                            docstring = ast.get_docstring(node) or "No docstring"
                            methods = []

                            for item in node.body:
                                if isinstance(item, ast.FunctionDef) and not item.name.startswith(
                                    "_"
                                ):
                                    # Extract method signature
                                    params = []
                                    for arg in item.args.args:
                                        if arg.arg != "self":
                                            # Get type annotation if present
                                            if arg.annotation:
                                                try:
                                                    type_str = (
                                                        ast.unparse(arg.annotation)
                                                        if hasattr(ast, "unparse")
                                                        else "Any"
                                                    )
                                                    params.append(f"{arg.arg}: {type_str}")
                                                except (AttributeError, TypeError):
                                                    params.append(arg.arg)
                                            else:
                                                params.append(arg.arg)

                                    signature = f"{item.name}({', '.join(params)})"
                                    methods.append((item.name, signature))

                            self.protocols.append(
                                ProtocolInfo(
                                    name=node.name,
                                    docstring=docstring.split("\n")[0],
                                    methods=methods[:10],  # First 10 methods
                                    file_path=str(protocol_file.relative_to(self.src_dir.parent)),
                                    is_runtime_checkable=is_runtime_checkable,
                                    line_number=node.lineno,
                                )
                            )
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
                lines = content.split("\n")

                # Extract code blocks from docstrings
                in_docstring = False
                in_example_block = False
                example_lines = []
                example_title = ""

                for line in lines:
                    # Detect docstring boundaries
                    if '"""' in line or "'''" in line:
                        in_docstring = not in_docstring
                        continue

                    if not in_docstring:
                        continue

                    # Look for Usage: or Example: headers
                    if any(marker in line for marker in ["Usage:", "Example:", "Examples:"]):
                        example_title = line.strip().replace("=", "").strip()
                        continue

                    # Detect code block start
                    if "```python" in line or (
                        line.strip().startswith(">>>") and not in_example_block
                    ):
                        in_example_block = True
                        example_lines = []
                        continue

                    # Detect code block end
                    if in_example_block and (
                        "```" in line
                        or (
                            not line.strip().startswith(">>>")
                            and not line.strip().startswith("...")
                            and example_lines
                        )
                    ):
                        if len(example_lines) >= 3:  # At least 3 lines for meaningful example
                            # Clean up the code
                            code = "\n".join(example_lines).strip()
                            # Filter out malformed examples
                            if (
                                len(code) > 30
                                and code.count("\n") >= 2
                                and not code.startswith("...")
                                and "(" in code
                            ):  # Has function calls

                                # Clean title
                                clean_title = (
                                    example_title
                                    if example_title
                                    else f"Example from {file_path.name}"
                                )
                                if not clean_title.startswith(("Usage", "Example")):
                                    clean_title = f"Example: {clean_title}"

                                self.usage_examples.append(
                                    UsageExample(
                                        title=clean_title,
                                        code=code,
                                        description="",
                                        source_file=str(file_path.relative_to(self.src_dir.parent)),
                                        category=category,
                                    )
                                )

                        in_example_block = False
                        example_lines = []
                        example_title = ""
                        continue

                    # Collect code lines
                    if in_example_block:
                        # Remove >>> and ... prompts
                        clean_line = line.replace(">>> ", "").replace("... ", "")
                        if clean_line.strip():
                            example_lines.append(clean_line)

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

            # Add monitoring workflow diagram
            f.write("## 📊 Monitoring Workflow\n\n")
            f.write("```mermaid\n")
            f.write("graph LR\n")
            f.write("    A[Brain Training] --> B[HealthMonitor]\n")
            f.write("    A --> C[CriticalityMonitor]\n")
            f.write("    A --> D[MetacognitiveMonitor]\n")
            f.write("    A --> E[TrainingMonitor]\n")
            f.write("    B --> F[Health Reports]\n")
            f.write("    C --> F\n")
            f.write("    D --> F\n")
            f.write("    E --> G[Training Metrics]\n")
            f.write("```\n\n")

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

            # Add monitoring best practices
            f.write("## 💡 Monitoring Best Practices\n\n")
            f.write("### When to Use Each Monitor\n\n")
            f.write("- **HealthMonitor**: Every training run (catches pathological states)\n")
            f.write("- **CriticalityMonitor**: When tuning network connectivity\n")
            f.write("- **MetacognitiveMonitor**: For confidence estimation and active learning\n")
            f.write("- **TrainingMonitor**: For visualization and metric tracking\n\n")

            f.write("### Interpreting Results\n\n")
            f.write("✅ **Healthy network**: Firing rates 0.01-0.1, weights stable, no NaN\n\n")
            f.write("⚠️ **Warning signs**: Extreme firing rates, rapid weight changes\n\n")
            f.write("❌ **Critical issues**: NaN values, zero activity, runaway excitation\n\n")

            f.write("### Performance Tips\n\n")
            f.write("- Check health every 10-100 steps (not every step)\n")
            f.write("- Store history for trend analysis\n")
            f.write("- Use thresholds to trigger adaptive responses\n")
            f.write("- Log detailed diagnostics only when issues detected\n\n")

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
                "ComponentError": "Issues with component registration or initialization",
                "GrowthError": "Problems during network growth operations",
                "StateError": "Invalid state transitions or state management issues",
                "CheckpointError": "Problems loading or saving checkpoints",
                "ValidationError": "Input validation failures",
                "CompatibilityError": "Version or compatibility mismatches",
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

            # Add usage guide
            f.write("## Usage Guide\n\n")
            f.write("Constants should be imported directly from their source modules:\n\n")
            f.write("```python\n")
            f.write("# Neuron parameters\n")
            f.write("from thalia.components.neurons.neuron_constants import (\n")
            f.write("    TAU_MEM_STANDARD,\n")
            f.write("    V_THRESHOLD_STANDARD,\n")
            f.write(")\n\n")
            f.write("# Neuromodulation\n")
            f.write("from thalia.neuromodulation.constants import (\n")
            f.write("    DA_PHASIC_DECAY_PER_MS,\n")
            f.write("    ACH_ENCODING_LEVEL,\n")
            f.write(")\n")
            f.write("```\n\n")

            # Cross-reference to configs
            f.write("## See Also\n\n")
            f.write(
                "- [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) - Config classes that use these constants\n"
            )
            f.write(
                "- [COMPONENT_CATALOG.md](COMPONENT_CATALOG.md) - Components using these parameters\n"
            )
            f.write(
                "- [NEURON_FACTORIES_REFERENCE.md](NEURON_FACTORIES_REFERENCE.md) - Pre-configured neuron populations\n\n"
            )

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
                # Make class name clickable
                class_link = self._make_source_link(
                    protocol.file_path,
                    line_number=protocol.line_number,
                    display_text=f"`{protocol.name}`",
                )
                f.write(f"### {class_link}\n\n")

                if protocol.is_runtime_checkable:
                    f.write("**Runtime Checkable**: ✅ Yes (can use `isinstance()`)\n\n")
                else:
                    f.write("**Runtime Checkable**: ❌ No (static type checking only)\n\n")

                f.write(f"**Source**: {self._make_source_link(protocol.file_path)}\n\n")
                f.write(f"**Description**: {protocol.docstring}\n\n")

                if protocol.methods:
                    f.write("**Required Methods**:\n\n")
                    f.write("```python\n")
                    for _, signature in protocol.methods:
                        f.write(f"def {signature}:\n")
                        f.write("    ...\n\n")
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

                            self.checkpoint_structures.append(
                                CheckpointStructure(
                                    component_type=component_type,
                                    file_path="thalia/" + str(py_file.relative_to(self.src_dir)),
                                    top_level_keys=state_fields,
                                    nested_structures={},
                                    docstring=docstring,
                                )
                            )
            except Exception:
                continue  # Skip files with parse errors

    def _extract_state_dict_structure(
        self, func_node: ast.FunctionDef, content: str
    ) -> List[StateField]:
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

            # Add cross-reference to design doc
            f.write(
                "> **📚 For complete binary format specification, version compatibility, and implementation details, "
            )
            f.write("see [checkpoint_format.md](../design/checkpoint_format.md)**\n\n")

            f.write(
                "This document provides a quick reference for checkpoint usage and the state dictionary structure "
            )
            f.write("returned by `brain.get_full_state()`.\n\n")

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

            f.write("## Top-Level State Structure\n\n")
            f.write(
                "The checkpoint is a dictionary with these top-level keys returned by `DynamicBrain.get_full_state()`:\n\n"
            )

            # Find DynamicBrain structure
            brain_struct = next(
                (s for s in self.checkpoint_structures if s.component_type == "DynamicBrain"), None
            )

            if brain_struct:
                f.write("| Key | Type | Description |\n")
                f.write("|-----|------|-------------|\n")

                for field in brain_struct.top_level_keys:
                    desc = field.description or "Component state"
                    # Improve descriptions
                    if field.key == "global_config":
                        desc = "Global configuration"
                    elif field.key == "current_time":
                        desc = "Current simulation time"
                    elif field.key == "topology":
                        desc = "Brain topology graph"
                    elif field.key == "regions":
                        desc = "All region states"
                    elif field.key == "pathways":
                        desc = "All pathway states"
                    elif field.key == "oscillators":
                        desc = "Oscillator states"
                    elif field.key == "neuromodulators":
                        desc = "Neuromodulator levels"
                    elif field.key == "config":
                        desc = "Brain configuration"
                    elif field.key == "growth_history":
                        desc = "Growth event log"

                    f.write(f"| `{field.key}` | `{field.type_hint}` | {desc} |\n")

                f.write("\n")
                f.write(f"**Source**: {self._make_source_link(brain_struct.file_path)}\n\n")

            f.write("## Component State Structure\n\n")
            f.write("Each component (region or pathway) stores its state in the checkpoint.\n\n")

            # Find NeuralRegion structure
            region_struct = next(
                (s for s in self.checkpoint_structures if s.component_type == "NeuralRegion"), None
            )

            if region_struct:
                f.write("### NeuralRegion State (Base Class)\n\n")
                f.write("All regions include these base fields:\n\n")

                f.write("| Key | Type | Description |\n")
                f.write("|-----|------|-------------|\n")

                for field in region_struct.top_level_keys:
                    desc = field.description or "Base region data"
                    # Improve descriptions
                    if field.key == "type":
                        desc = "Region type identifier"
                    elif field.key == "n_neurons":
                        desc = "Number of neurons"
                    elif field.key == "n_input":
                        desc = "Input dimension"
                    elif field.key == "n_output":
                        desc = "Output dimension"
                    elif field.key == "device":
                        desc = "Device (CPU/GPU)"
                    elif field.key == "dt_ms":
                        desc = "Timestep in milliseconds"
                    elif field.key == "default_learning_strategy":
                        desc = "Default learning strategy"
                    elif field.key == "input_sources":
                        desc = "Input source names"
                    elif field.key == "synaptic_weights":
                        desc = "Weight matrices per source"
                    elif field.key == "plasticity_enabled":
                        desc = "Learning enabled flag"

                    f.write(f"| `{field.key}` | `{field.type_hint}` | {desc} |\n")

                f.write("\n")
                f.write(f"**Source**: {self._make_source_link(region_struct.file_path)}\n\n")

            # Find AxonalProjection structure
            pathway_struct = next(
                (s for s in self.checkpoint_structures if s.component_type == "AxonalProjection"),
                None,
            )

            if pathway_struct:
                f.write("### AxonalProjection State (Pathways)\n\n")
                f.write("All pathways include these fields:\n\n")

                f.write("| Key | Type | Description |\n")
                f.write("|-----|------|-------------|\n")

                for field in pathway_struct.top_level_keys:
                    f.write(
                        f"| `{field.key}` | `{field.type_hint}` | {field.description or 'Pathway data'} |\n"
                    )

                f.write("\n")
                f.write(f"**Source**: {self._make_source_link(pathway_struct.file_path)}\n\n")

            f.write("## File Formats\n\n")
            f.write("Checkpoints can be saved in two formats:\n\n")
            f.write(
                "1. **PyTorch Format** (`.pt`, `.pth`, `.ckpt`) - Standard PyTorch `torch.save()` format (default)\n"
            )
            f.write(
                "2. **Binary Format** (`.thalia`, `.thalia.zst`) - Custom binary format with compression (advanced)\n\n"
            )

            f.write("## Usage Examples\n\n")
            f.write("```python\n")
            f.write("# Save checkpoint (PyTorch format - default)\n")
            f.write("brain.save_checkpoint(\n")
            f.write('    "checkpoints/epoch_100.ckpt",\n')
            f.write('    metadata={"epoch": 100, "loss": 0.42}\n')
            f.write(")\n\n")
            f.write("# Load checkpoint\n")
            f.write('brain.load_checkpoint("checkpoints/epoch_100.ckpt")\n\n')
            f.write("# Save with compression (binary format)\n")
            f.write(
                'brain.save_checkpoint("checkpoints/epoch_100.thalia.zst", compression="zstd")\n\n'
            )
            f.write("# Save with mixed precision\n")
            f.write(
                'brain.save_checkpoint("checkpoints/epoch_100.ckpt", precision_policy="fp16")\n'
            )
            f.write("```\n\n")

            f.write("## Validation\n\n")
            f.write("```python\n")
            f.write("from thalia.io import CheckpointManager\n\n")
            f.write("manager = CheckpointManager(brain)\n")
            f.write('is_valid, error = manager.validate("checkpoints/epoch_100.ckpt")\n')
            f.write("if not is_valid:\n")
            f.write('    print(f"Checkpoint invalid: {error}")\n')
            f.write("```\n\n")

            f.write("## See Also\n\n")
            f.write(
                "- **[Checkpoint Format Specification](../design/checkpoint_format.md)** - Complete binary format details, byte layouts, compression algorithms\n"
            )
            f.write(
                "- **[Curriculum Strategy](../design/curriculum_strategy.md)** - Training stages and checkpoint usage in curriculum training\n"
            )
            f.write(
                "- **[GETTING_STARTED_CURRICULUM](../GETTING_STARTED_CURRICULUM.md)** - Tutorial including checkpoint management\n\n"
            )

        print(f"✅ Generated: {output_file.relative_to(self.docs_dir.parent)}")

    def _find_type_aliases(self):
        """Extract type alias definitions from code."""
        # Extract from thalia.typing module (canonical source)
        typing_file = self.src_dir / "typing.py"
        if typing_file.exists():
            self._extract_aliases_from_typing_module(typing_file)
        else:
            print("Warning: thalia/typing.py not found - no type aliases extracted")

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

    def _get_source_segment(self, content: str, node: ast.AST) -> str:
        """Extract source code for an AST node."""
        try:
            return ast.unparse(node)
        except Exception:
            # Fallback for older Python or complex nodes
            return ""

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
                            self.component_relations.append(
                                ComponentRelation(
                                    source=source,
                                    target=target,
                                    pathway_type=pathway_type,
                                    preset_name=preset_name,
                                    source_port=source_port,
                                    target_port=target_port,
                                )
                            )

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

            f.write(
                "This document catalogs all type aliases used in Thalia for clearer type hints.\n\n"
            )

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

                    f.write(f"**Source**: {self._make_source_link(alias.file_path)}\n\n")
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
                    src_port = rel.source_port
                    tgt_port = rel.target_port
                    f.write(
                        f"| `{rel.source}` | `{src_port}` | → | `{rel.target}` | `{tgt_port}` | `{rel.pathway_type}` |\n"
                    )

                f.write("\n")

                # Generate mermaid diagram
                f.write("### Architecture Diagram\n\n")
                f.write("```mermaid\n")
                f.write("graph TB\n")

                for rel in relations:
                    # Create edge label - use simple text without special characters for mermaid
                    label = rel.pathway_type
                    if rel.source_port:
                        label = f"{rel.source_port} {label}"
                    if rel.target_port:
                        label = f"{label} to {rel.target_port}"

                    f.write(f"    {rel.source} -->|{label}| {rel.target}\n")

                f.write("```\n\n")
                f.write("---\n\n")

        print(f"✅ Generated: {output_file.relative_to(self.docs_dir.parent)}")

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

    def _find_state_classes(self):
        """Find all state classes (RegionState, PathwayState subclasses)."""
        for py_file in self.src_dir.rglob("*.py"):
            if "test" in str(py_file) or "__pycache__" in str(py_file):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                # Quick check for state-related classes
                if (
                    "RegionState" not in content
                    and "PathwayState" not in content
                    and "BaseRegionState" not in content
                ):
                    continue

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Check if inherits from state base classes
                        base_class = None
                        for base in node.bases:
                            base_name = base.id if hasattr(base, "id") else ""
                            if base_name in [
                                "RegionState",
                                "BaseRegionState",
                                "PathwayState",
                                "NeuralComponentState",
                            ]:
                                base_class = base_name
                                break

                        if not base_class:
                            continue

                        # Extract fields
                        fields = []
                        state_version = 1
                        for item in node.body:
                            # Look for STATE_VERSION
                            if isinstance(item, ast.AnnAssign) and isinstance(
                                item.target, ast.Name
                            ):
                                if item.target.id == "STATE_VERSION":
                                    if item.value:
                                        try:
                                            state_version = ast.literal_eval(item.value)
                                        except (ValueError, SyntaxError):
                                            pass
                                else:
                                    # Regular field
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

                                                match = re.search(r"default=([^,)]+)", default_str)
                                                default = match.group(1) if match else default_str
                                            else:
                                                default = "field(...)"
                                        else:
                                            default = default_str
                                    else:
                                        default = "None"

                                    fields.append((field_name, field_type, default))
                            elif isinstance(item, ast.Assign):
                                # Handle ClassVar[int] = 1 style
                                for target in item.targets:
                                    if (
                                        isinstance(target, ast.Name)
                                        and target.id == "STATE_VERSION"
                                    ):
                                        try:
                                            state_version = ast.literal_eval(item.value)
                                        except (ValueError, SyntaxError):
                                            pass

                        docstring = ast.get_docstring(node) or "No docstring"

                        # Determine component type
                        component_type = "pathway" if "Pathway" in node.name else "region"

                        self.state_classes.append(
                            StateClassInfo(
                                name=node.name,
                                base_class=base_class,
                                docstring=docstring.split("\n")[0],
                                fields=fields,
                                state_version=state_version,
                                file_path=str(py_file.relative_to(self.src_dir.parent)),
                                component_type=component_type,
                                line_number=node.lineno,
                            )
                        )

            except Exception as e:
                print(f"Warning: Could not parse {py_file} for state classes: {e}")

    def _generate_state_classes_reference(self):
        """Generate STATE_CLASSES_REFERENCE.md."""
        output_file = self.api_dir / "STATE_CLASSES_REFERENCE.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with output_file.open("w", encoding="utf-8") as f:
            f.write("# State Classes Reference\n\n")
            f.write("> **Auto-generated documentation** - Do not edit manually!\n")
            f.write(f"> Last updated: {timestamp}\n")
            f.write("> Generated from: `scripts/generate_api_docs.py`\n\n")

            f.write("This document catalogs all state classes used for serialization ")
            f.write("in Thalia's checkpoint system. State classes inherit from `RegionState`, ")
            f.write("`BaseRegionState`, or `PathwayState`.\n\n")

            f.write(f"Total: {len(self.state_classes)} state classes\n\n")

            # Group by component type
            region_states = [s for s in self.state_classes if s.component_type == "region"]
            pathway_states = [s for s in self.state_classes if s.component_type == "pathway"]

            f.write("## Overview\n\n")
            f.write(
                "State classes provide serialization support for checkpoints. Each state class:\n\n"
            )
            f.write(
                "- Inherits from a base state class (`RegionState`, `BaseRegionState`, or `PathwayState`)\n"
            )
            f.write("- Implements `to_dict()` and `from_dict()` for serialization\n")
            f.write("- Includes `STATE_VERSION` for migration support\n")
            f.write("- Contains only mutable state (not configuration or learned parameters)\n\n")

            f.write("## State Class Hierarchy\n\n")
            f.write("```\n")
            f.write("RegionState (Protocol)\n")
            f.write("├── BaseRegionState (dataclass)\n")
            f.write("│   ├── PrefrontalState\n")
            f.write("│   ├── ThalamicRelayState\n")
            f.write("│   └── ... (other region states)\n")
            f.write("│\n")
            f.write("└── PathwayState (Protocol)\n")
            f.write("    └── AxonalProjectionState\n")
            f.write("```\n\n")

            # Region States
            f.write("## Region State Classes\n\n")
            f.write(f"Total region states: {len(region_states)}\n\n")

            for state in sorted(region_states, key=lambda s: s.name):
                # Make class name clickable
                class_link = self._make_source_link(
                    state.file_path, line_number=state.line_number, display_text=f"`{state.name}`"
                )
                f.write(f"### {class_link}\n\n")
                f.write(f"**Base Class**: `{state.base_class}`  \n")
                f.write(f"**Version**: {state.state_version}  \n")
                f.write(f"**Source**: {self._make_source_link(state.file_path)}\n\n")

                if state.docstring != "No docstring":
                    f.write(f"**Description**: {state.docstring}\n\n")

                if state.fields:
                    f.write("**Fields**:\n\n")
                    f.write("| Field | Type | Default |\n")
                    f.write("|-------|------|----------|\n")
                    for field_name, field_type, default in state.fields:
                        # Truncate long types/defaults
                        field_type_display = (
                            field_type if len(field_type) < 50 else field_type[:47] + "..."
                        )
                        default_display = default if len(default) < 30 else default[:27] + "..."
                        f.write(
                            f"| `{field_name}` | `{field_type_display}` | `{default_display}` |\n"
                        )
                    f.write("\n")

                f.write("---\n\n")

            # Pathway States
            if pathway_states:
                f.write("## Pathway State Classes\n\n")
                f.write(f"Total pathway states: {len(pathway_states)}\n\n")

                for state in sorted(pathway_states, key=lambda s: s.name):
                    # Make class name clickable
                    class_link = self._make_source_link(
                        state.file_path,
                        line_number=state.line_number,
                        display_text=f"`{state.name}`",
                    )
                    f.write(f"### {class_link}\n\n")
                    f.write(f"**Base Class**: `{state.base_class}`  \n")
                    f.write(f"**Version**: {state.state_version}  \n")
                    f.write(f"**Source**: {self._make_source_link(state.file_path)}\n\n")

                    if state.docstring != "No docstring":
                        f.write(f"**Description**: {state.docstring}\n\n")

                    if state.fields:
                        f.write("**Fields**:\n\n")
                        f.write("| Field | Type | Default |\n")
                        f.write("|-------|------|----------|\n")
                        for field_name, field_type, default in state.fields:
                            field_type_display = (
                                field_type if len(field_type) < 50 else field_type[:47] + "..."
                            )
                            default_display = default if len(default) < 30 else default[:27] + "..."
                            f.write(
                                f"| `{field_name}` | `{field_type_display}` | `{default_display}` |\n"
                            )
                        f.write("\n")

                    f.write("---\n\n")

            # Usage guide
            f.write("## Usage Guide\n\n")
            f.write("### Creating a New State Class\n\n")
            f.write("```python\n")
            f.write("from dataclasses import dataclass\n")
            f.write("from typing import Optional\n")
            f.write("import torch\n")
            f.write("from thalia.core.region_state import BaseRegionState\n\n")
            f.write("@dataclass\n")
            f.write("class MyRegionState(BaseRegionState):\n")
            f.write('    """State for MyRegion."""\n')
            f.write("    STATE_VERSION: int = 1\n\n")
            f.write("    # Add your state fields\n")
            f.write("    custom_spikes: Optional[torch.Tensor] = None\n")
            f.write("    custom_membrane: Optional[torch.Tensor] = None\n")
            f.write("```\n\n")

            f.write("### Serialization\n\n")
            f.write(
                "State classes automatically inherit `to_dict()` and `from_dict()` methods:\n\n"
            )
            f.write("```python\n")
            f.write("# Save state\n")
            f.write("state_dict = region.get_state().to_dict()\n\n")
            f.write("# Load state\n")
            f.write("loaded_state = MyRegionState.from_dict(state_dict, device='cpu')\n")
            f.write("region.load_state(loaded_state)\n")
            f.write("```\n\n")

            f.write("### Version Migration\n\n")
            f.write(
                "When adding new fields, increment `STATE_VERSION` and add migration logic:\n\n"
            )
            f.write("```python\n")
            f.write("@dataclass\n")
            f.write("class MyRegionState(BaseRegionState):\n")
            f.write("    STATE_VERSION: int = 2  # Incremented from 1\n\n")
            f.write("    # New field in v2\n")
            f.write("    new_field: Optional[torch.Tensor] = None\n\n")
            f.write("    @classmethod\n")
            f.write("    def _migrate_from_v1(cls, data: Dict[str, Any]) -> Dict[str, Any]:\n")
            f.write('        """Migrate v1 state to v2."""\n')
            f.write("        data['new_field'] = None  # Initialize with default\n")
            f.write("        return data\n")
            f.write("```\n\n")

            f.write("**See Also**:\n")
            f.write("- `docs/patterns/state-management.md` - State management patterns\n")
            f.write("- `docs/api/CHECKPOINT_FORMAT.md` - Checkpoint file format\n\n")

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

            # Add usage guide
            f.write("## Usage Patterns\n\n")

            f.write("### Simple Usage\n\n")
            f.write("```python\n")
            f.write("from thalia.components.neurons import create_pyramidal_neurons\n\n")
            f.write("# Create standard pyramidal neurons\n")
            f.write("neurons = create_pyramidal_neurons(\n")
            f.write("    n_neurons=128,\n")
            f.write("    device=device\n")
            f.write(")\n")
            f.write("```\n\n")

            f.write("### Custom Configuration\n\n")
            f.write("```python\n")
            f.write("from thalia.components.neurons import create_pyramidal_neurons\n\n")
            f.write("# Override defaults for specialized regions\n")
            f.write("ca3_neurons = create_pyramidal_neurons(\n")
            f.write("    n_neurons=32,\n")
            f.write("    device=device,\n")
            f.write("    adapt_increment=0.1,  # Strong adaptation for CA3\n")
            f.write("    tau_adapt=100.0,\n")
            f.write(")\n")
            f.write("```\n\n")

            f.write("### Layer-Specific Neurons\n\n")
            f.write("```python\n")
            f.write("from thalia.components.neurons import create_cortical_layer_neurons\n\n")
            f.write("# Create neurons for specific cortical layers\n")
            f.write('l4 = create_cortical_layer_neurons(512, "L4", device)\n')
            f.write('l23 = create_cortical_layer_neurons(256, "L2/3", device)\n')
            f.write('l5 = create_cortical_layer_neurons(128, "L5", device)\n')
            f.write("```\n\n")

            f.write("### Registry-Based Creation (NEW)\n\n")
            f.write("```python\n")
            f.write("from thalia.components.neurons import NeuronFactory\n\n")
            f.write("# Dynamic neuron creation by type name\n")
            f.write('pyramidal = NeuronFactory.create("pyramidal", n_neurons=100, device=device)\n')
            f.write('relay = NeuronFactory.create("relay", n_neurons=64, device=device)\n')
            f.write('l23 = NeuronFactory.create("cortical_layer", 256, device, layer="L2/3")\n\n')
            f.write("# List available types\n")
            f.write("available = NeuronFactory.list_types()\n")
            f.write("print(available)  # ['cortical_layer', 'pyramidal', 'relay', 'trn']\n\n")
            f.write("# Check if type exists\n")
            f.write('if NeuronFactory.has_type("pyramidal"):\n')
            f.write('    neurons = NeuronFactory.create("pyramidal", 100, device)\n')
            f.write("```\n\n")

            # Cross-reference
            f.write("## See Also\n\n")
            f.write(
                "- [CONSTANTS_REFERENCE.md](CONSTANTS_REFERENCE.md) - Biological constants used by factories\n"
            )
            f.write(
                "- [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) - Neuron configuration classes\n"
            )
            f.write(
                "- [COMPONENT_CATALOG.md](COMPONENT_CATALOG.md) - Regions using these neurons\n\n"
            )

        print(f"✅ Generated: {output_file.relative_to(self.docs_dir.parent)}")

    def _generate_compute_functions_reference(self):
        """Generate COMPUTE_FUNCTIONS_REFERENCE.md."""
        output_file = self.api_dir / "COMPUTE_FUNCTIONS_REFERENCE.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with output_file.open("w", encoding="utf-8") as f:
            f.write("# Compute Functions Reference\n\n")
            f.write("> **Auto-generated documentation** - Do not edit manually!\n")
            f.write(f"> Last updated: {timestamp}\n")
            f.write("> Generated from: `scripts/generate_api_docs.py`\n\n")

            f.write("This document catalogs all `compute_*` utility functions that implement ")
            f.write("biological computations throughout the Thalia codebase.\n\n")

            f.write(f"Total: **{len(self.compute_functions)}** compute functions\n\n")

            # Add badges
            f.write(
                f"![Functions](https://img.shields.io/badge/Functions-{len(self.compute_functions)}-blue) "
            )
            f.write(
                "![Biological](https://img.shields.io/badge/Type-Biologically--Grounded-orange) "
            )
            f.write("![Utils](https://img.shields.io/badge/Category-Utilities-green)\n\n")

            # Group by category
            by_category = {}
            for func in self.compute_functions:
                if func.category not in by_category:
                    by_category[func.category] = []
                by_category[func.category].append(func)

            # Add quick reference
            f.write("## 📋 Quick Reference by Category\n\n")
            f.write("| Category | Functions | Purpose |\n")
            f.write("|----------|-----------|----------|\n")
            f.write(
                f"| **Oscillator** | {len(by_category.get('oscillator', []))} | Phase-based modulation and coupling |\n"
            )
            f.write(
                f"| **Neuromodulation** | {len(by_category.get('neuromodulation', []))} | Dopamine, ACh effect computation |\n"
            )
            f.write(
                f"| **Sizing** | {len(by_category.get('sizing', []))} | Region size calculations |\n\n"
            )

            # Category overview diagram
            f.write("## 🔬 Function Categories\n\n")
            f.write("```mermaid\n")
            f.write("graph LR\n")
            f.write("    A[Compute Functions] --> B[Oscillator Utils]\n")
            f.write("    A --> C[Neuromodulation]\n")
            f.write("    A --> D[Sizing]\n")
            f.write("    B --> B1[Theta-Gamma<br/>Coupling]\n")
            f.write("    B --> B2[Phase<br/>Gating]\n")
            f.write("    C --> C1[Dopamine<br/>Effects]\n")
            f.write("    C --> C2[ACh<br/>Modulation]\n")
            f.write("    D --> D1[Optimal<br/>Sizes]\n")
            f.write("```\n\n")

            # Generate sections by category
            category_names = {
                "oscillator": "🌊 Oscillator Modulation Functions",
                "neuromodulation": "💊 Neuromodulation Functions",
                "sizing": "📏 Region Sizing Functions",
            }

            for category in ["oscillator", "neuromodulation", "sizing"]:
                if category not in by_category:
                    continue

                f.write(f"## {category_names.get(category, category.title())}\n\n")

                for func in sorted(by_category[category], key=lambda f: f.name):
                    # Make function name clickable
                    func_link = self._make_source_link(
                        func.file_path,
                        line_number=func.line_number,
                        display_text=f"`{func.name}()`",
                    )
                    f.write(f"### {func_link}\n\n")
                    f.write(f"**Returns**: `{func.return_type}`  \n")
                    f.write(f"**Source**: {self._make_source_link(func.file_path)}\n\n")

                    f.write(f"**Description**: {func.docstring}\n\n")

                    if func.parameters:
                        f.write("**Parameters**:\n\n")
                        f.write("| Parameter | Type | Default |\n")
                        f.write("|-----------|------|----------|\n")
                        for param_name, param_type, param_default in func.parameters:
                            default_str = param_default if param_default else "-"
                            f.write(f"| `{param_name}` | `{param_type}` | `{default_str}` |\n")
                        f.write("\n")

                    # Add biological context if available
                    if func.biological_context:
                        f.write("**Biological Context**:\n\n")
                        f.write(f"{func.biological_context}\n\n")

                    if func.examples:
                        f.write("**Examples**:\n\n")
                        for example in func.examples:
                            f.write("```python\n")
                            f.write(example.strip())
                            f.write("\n```\n\n")

                    f.write("---\n\n")

            # Usage patterns
            f.write("## Usage Patterns\n\n")

            f.write("### Oscillator Modulation Pattern\n\n")
            f.write("```python\n")
            f.write("from thalia.utils.oscillator_utils import (\n")
            f.write("    compute_theta_encoding_retrieval,\n")
            f.write("    compute_gamma_phase_gate\n")
            f.write(")\n\n")
            f.write("# Theta-based encoding/retrieval switching\n")
            f.write("theta_phase = oscillators['theta'].phase\n")
            f.write(
                "encoding_strength, retrieval_strength = compute_theta_encoding_retrieval(theta_phase)\n\n"
            )
            f.write("# Gamma phase gating\n")
            f.write("gamma_phase = oscillators['gamma'].phase\n")
            f.write("gate = compute_gamma_phase_gate(gamma_phase, window_deg=60.0)\n")
            f.write("```\n\n")

            f.write("### Neuromodulation Pattern\n\n")
            f.write("```python\n")
            f.write("from thalia.neuromodulation.constants import (\n")
            f.write("    compute_dopamine_effect,\n")
            f.write("    compute_acetylcholine_effect\n")
            f.write(")\n\n")
            f.write("# Apply dopamine modulation\n")
            f.write("da_level = self.get_neuromodulator_level('dopamine')\n")
            f.write("modulated_lr = base_lr * compute_dopamine_effect(da_level)\n\n")
            f.write("# Apply acetylcholine modulation\n")
            f.write("ach_level = self.get_neuromodulator_level('acetylcholine')\n")
            f.write("encoding_bias = compute_acetylcholine_effect(ach_level)\n")
            f.write("```\n\n")

            # Cross-references
            f.write("## See Also\n\n")
            f.write(
                "- [CONSTANTS_REFERENCE.md](CONSTANTS_REFERENCE.md) - Biological constants used by compute functions\n"
            )
            f.write(
                "- [COMPONENT_CATALOG.md](COMPONENT_CATALOG.md) - Regions that use these functions\n"
            )
            f.write("- [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) - Real-world usage examples\n\n")

        print(f"✅ Generated: {output_file.relative_to(self.docs_dir.parent)}")

    def _generate_visualization_reference(self):
        """Generate VISUALIZATION_REFERENCE.md."""
        output_file = self.api_dir / "VISUALIZATION_REFERENCE.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with output_file.open("w", encoding="utf-8") as f:
            f.write("# Visualization Functions Reference\n\n")
            f.write("> **Auto-generated documentation** - Do not edit manually!\n")
            f.write(f"> Last updated: {timestamp}\n")
            f.write("> Generated from: `scripts/generate_api_docs.py`\n\n")

            f.write("This document catalogs all visualization and plotting functions ")
            f.write("for analyzing brain topology, training progress, and diagnostic metrics.\n\n")

            f.write(f"Total: **{len(self.visualization_functions)}** visualization functions\n\n")

            # Add badges
            f.write(
                f"![Functions](https://img.shields.io/badge/Functions-{len(self.visualization_functions)}-blue) "
            )
            f.write("![Viz](https://img.shields.io/badge/Type-Visualization-purple) ")
            f.write("![Analysis](https://img.shields.io/badge/Purpose-Analysis-green)\n\n")

            # Group by category
            by_category = {}
            for func in self.visualization_functions:
                if func.category not in by_category:
                    by_category[func.category] = []
                by_category[func.category].append(func)

            # Add quick reference
            f.write("## 📋 Quick Reference by Use Case\n\n")
            f.write("| Use Case | Functions | Purpose |\n")
            f.write("|----------|-----------|----------|\n")
            f.write(
                f"| **Topology** | {len(by_category.get('topology', []))} | Network structure visualization |\n"
            )
            f.write(
                f"| **Training** | {len(by_category.get('training', []))} | Training progress monitoring |\n"
            )
            f.write(
                f"| **Diagnostics** | {len(by_category.get('diagnostics', []))} | Real-time health metrics |\n"
            )
            f.write(
                f"| **Critical Periods** | {len(by_category.get('critical_period', []))} | Developmental windows |\n\n"
            )

            # Visualization workflow diagram
            f.write("## 📊 Visualization Workflow\n\n")
            f.write("```mermaid\n")
            f.write("graph TD\n")
            f.write("    A[Analysis Need] --> B{What to analyze?}\n")
            f.write("    B -->|Structure| C[Topology Viz]\n")
            f.write("    B -->|Progress| D[Training Viz]\n")
            f.write("    B -->|Health| E[Diagnostics Viz]\n")
            f.write("    B -->|Development| F[Critical Periods]\n")
            f.write("    C --> C1[visualize_brain_topology]\n")
            f.write("    D --> D1[TrainingMonitor]\n")
            f.write("    E --> E1[LiveDiagnostics]\n")
            f.write("    F --> F1[plot_critical_period_windows]\n")
            f.write("```\n\n")

            # Generate sections by category
            category_names = {
                "topology": "🧠 Network Topology Visualization",
                "training": "📈 Training Progress Visualization",
                "diagnostics": "🔬 Diagnostic Monitoring",
                "critical_period": "🌱 Critical Period Analysis",
            }

            for category in ["topology", "training", "diagnostics", "critical_period"]:
                if category not in by_category:
                    continue

                f.write(f"## {category_names.get(category, category.title())}\n\n")

                for func in sorted(by_category[category], key=lambda f: f.name):
                    # Make function name clickable
                    func_link = self._make_source_link(
                        func.file_path,
                        line_number=func.line_number,
                        display_text=f"`{func.name}()`",
                    )
                    f.write(f"### {func_link}\n\n")
                    f.write(f"**Returns**: `{func.return_type}`  \n")
                    f.write(f"**Source**: {self._make_source_link(func.file_path)}\n\n")

                    f.write(f"**Description**: {func.docstring}\n\n")

                    if func.parameters:
                        f.write("**Parameters**:\n\n")
                        f.write("| Parameter | Type | Default |\n")
                        f.write("|-----------|------|----------|\n")
                        for param_name, param_type, param_default in func.parameters:
                            # Shorten long types
                            if len(param_type) > 40:
                                param_type = param_type[:37] + "..."
                            default_str = param_default if param_default else "-"
                            if len(default_str) > 30:
                                default_str = default_str[:27] + "..."
                            f.write(f"| `{param_name}` | `{param_type}` | `{default_str}` |\n")
                        f.write("\n")

                    if func.examples:
                        f.write("**Examples**:\n\n")
                        for example in func.examples:
                            f.write("```python\n")
                            f.write(example.strip())
                            f.write("\n```\n\n")

                    f.write("---\n\n")

            # Usage patterns
            f.write("## Usage Patterns\n\n")

            f.write("### Quick Topology Visualization\n\n")
            f.write("```python\n")
            f.write("from thalia.visualization.network_graph import visualize_brain_topology\n\n")
            f.write("# Create topology visualization\n")
            f.write("fig = visualize_brain_topology(\n")
            f.write("    brain=brain,\n")
            f.write("    layout='hierarchical',\n")
            f.write("    show_weights=True\n")
            f.write(")\n")
            f.write("fig.savefig('brain_topology.png')\n")
            f.write("```\n\n")

            f.write("### Training Progress Monitoring\n\n")
            f.write("```python\n")
            f.write("from thalia.training.visualization import TrainingMonitor\n\n")
            f.write("# Monitor training from checkpoints\n")
            f.write("monitor = TrainingMonitor(checkpoint_dir='checkpoints/my_run')\n")
            f.write("monitor.display_all()  # Shows progress, metrics, and growth\n")
            f.write("```\n\n")

            f.write("### Real-Time Diagnostics\n\n")
            f.write("```python\n")
            f.write(
                "from thalia.training.visualization.live_diagnostics import LiveDiagnostics\n\n"
            )
            f.write("# Create live diagnostic dashboard\n")
            f.write("diagnostics = LiveDiagnostics(brain, update_interval=10)\n")
            f.write("diagnostics.start()  # Auto-refreshing dashboard\n")
            f.write("\n")
            f.write("# Training loop\n")
            f.write("for step in range(num_steps):\n")
            f.write("    loss = train_step(brain, batch)\n")
            f.write("    diagnostics.update(step, loss)\n")
            f.write("```\n\n")

            f.write("### Critical Period Analysis\n\n")
            f.write("```python\n")
            f.write("from thalia.training.visualization.critical_period_plots import (\n")
            f.write("    plot_critical_period_windows,\n")
            f.write("    plot_sensory_critical_periods\n")
            f.write(")\n\n")
            f.write("# Visualize developmental windows\n")
            f.write("fig1 = plot_critical_period_windows(max_steps=500000)\n")
            f.write("fig2 = plot_sensory_critical_periods(modality='visual')\n")
            f.write("```\n\n")

            # Cross-references
            f.write("## See Also\n\n")
            f.write(
                "- [DIAGNOSTICS_REFERENCE.md](DIAGNOSTICS_REFERENCE.md) - Diagnostic monitor classes\n"
            )
            f.write("- [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) - More visualization examples\n")
            f.write(
                "- [../MONITORING_GUIDE.md](../MONITORING_GUIDE.md) - Complete monitoring guide\n\n"
            )

        print(f"✅ Generated: {output_file.relative_to(self.docs_dir.parent)}")

    def _generate_api_index(self):
        """Generate comprehensive API index with search functionality."""
        output_file = self.api_dir / "API_INDEX.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with output_file.open("w", encoding="utf-8") as f:
            f.write("# API Index\n\n")
            f.write("> **Auto-generated documentation** - Do not edit manually!\n")
            f.write(f"> Last updated: {timestamp}\n")
            f.write("> Generated from: `scripts/generate_api_docs.py`\n\n")

            f.write("Comprehensive searchable index of all Thalia API components.\n\n")

            # Summary statistics
            f.write("## 📊 API Coverage\n\n")
            f.write("| Category | Count | Documentation |\n")
            f.write("|----------|-------|---------------|\n")
            f.write(
                f"| **Regions** | {len(self.regions)} | [COMPONENT_CATALOG.md](COMPONENT_CATALOG.md) |\n"
            )
            f.write(
                f"| **Pathways** | {len(self.pathways)} | [COMPONENT_CATALOG.md](COMPONENT_CATALOG.md) |\n"
            )
            f.write(
                f"| **Learning Strategies** | {len(self.strategies)} | [LEARNING_STRATEGIES_API.md](LEARNING_STRATEGIES_API.md) |\n"
            )
            f.write(
                f"| **Configurations** | {len(self.configs)} | [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) |\n"
            )
            f.write(
                f"| **Datasets** | {len(self.datasets)} | [DATASETS_REFERENCE.md](DATASETS_REFERENCE.md) |\n"
            )
            f.write(
                f"| **Monitors** | {len(self.monitors)} | [DIAGNOSTICS_REFERENCE.md](DIAGNOSTICS_REFERENCE.md) |\n"
            )
            f.write(
                f"| **Mixins** | {len(self.mixins)} | [MIXINS_REFERENCE.md](MIXINS_REFERENCE.md) |\n"
            )
            f.write(
                f"| **Exceptions** | {len(self.exceptions)} | [EXCEPTIONS_REFERENCE.md](EXCEPTIONS_REFERENCE.md) |\n"
            )
            f.write(
                f"| **Constants** | {len(self.constants)} | [CONSTANTS_REFERENCE.md](CONSTANTS_REFERENCE.md) |\n"
            )
            f.write(
                f"| **Protocols** | {len(self.protocols)} | [PROTOCOLS_REFERENCE.md](PROTOCOLS_REFERENCE.md) |\n"
            )
            f.write(
                f"| **Type Aliases** | {len(self.type_aliases)} | [TYPE_ALIASES.md](TYPE_ALIASES.md) |\n"
            )
            f.write(
                f"| **Enumerations** | {len(self.enumerations)} | [ENUMERATIONS_REFERENCE.md](ENUMERATIONS_REFERENCE.md) |\n"
            )
            f.write(
                f"| **State Classes** | {len(self.state_classes)} | [STATE_CLASSES_REFERENCE.md](STATE_CLASSES_REFERENCE.md) |\n"
            )
            f.write(
                f"| **Neuron Factories** | {len(self.neuron_factories)} | [NEURON_FACTORIES_REFERENCE.md](NEURON_FACTORIES_REFERENCE.md) |\n"
            )
            f.write(
                f"| **Compute Functions** | {len(self.compute_functions)} | [COMPUTE_FUNCTIONS_REFERENCE.md](COMPUTE_FUNCTIONS_REFERENCE.md) |\n"
            )
            f.write(
                f"| **Visualization Functions** | {len(self.visualization_functions)} | [VISUALIZATION_REFERENCE.md](VISUALIZATION_REFERENCE.md) |\n"
            )
            total = (
                len(self.regions)
                + len(self.pathways)
                + len(self.strategies)
                + len(self.configs)
                + len(self.datasets)
                + len(self.monitors)
            )
            f.write(f"| **Total Components** | **{total}** | - |\n\n")

            # Alphabetical index
            f.write("## 🔤 Alphabetical Index\n\n")

            # Collect all items with their categories
            all_items = []
            for region in self.regions:
                all_items.append((region.name, "Region", "COMPONENT_CATALOG.md"))
            for pathway in self.pathways:
                all_items.append((pathway.name, "Pathway", "COMPONENT_CATALOG.md"))
            for strategy in self.strategies:
                all_items.append((strategy.name, "Strategy", "LEARNING_STRATEGIES_API.md"))
            for config in self.configs:
                all_items.append((config.name, "Config", "CONFIGURATION_REFERENCE.md"))
            for dataset in self.datasets:
                all_items.append((dataset.name, "Dataset", "DATASETS_REFERENCE.md"))
            for monitor in self.monitors:
                all_items.append((monitor.name, "Monitor", "DIAGNOSTICS_REFERENCE.md"))
            for mixin in self.mixins:
                all_items.append((mixin.name, "Mixin", "MIXINS_REFERENCE.md"))
            for exception in self.exceptions:
                all_items.append((exception.name, "Exception", "EXCEPTIONS_REFERENCE.md"))

            # Sort alphabetically
            all_items.sort(key=lambda x: x[0].lower())

            # Group by first letter
            current_letter = None
            for name, category, doc_file in all_items:
                first_letter = name[0].upper()
                if first_letter != current_letter:
                    if current_letter is not None:
                        f.write("\n")
                    f.write(f"### {first_letter}\n\n")
                    current_letter = first_letter

                f.write(f"- **{name}** ({category}) → [{doc_file}]({doc_file})\n")

            # Category index
            f.write("\n## 📂 By Category\n\n")

            categories = {
                "Core Components": ["Region", "Pathway"],
                "Learning & Training": ["Strategy", "Dataset", "Monitor"],
                "Configuration": ["Config", "Constant"],
                "Utilities": ["Mixin", "Protocol", "Exception"],
            }

            for cat_name, cat_types in categories.items():
                f.write(f"### {cat_name}\n\n")
                for name, category, doc_file in all_items:
                    if category in cat_types:
                        f.write(f"- `{name}` → [{doc_file}]({doc_file})\n")
                f.write("\n")

            # Quick search guide
            f.write("## 🔍 Search Guide\n\n")
            f.write("### By Task\n\n")
            f.write("- **Building a brain**: See [COMPONENT_CATALOG.md](COMPONENT_CATALOG.md)\n")
            f.write(
                "- **Implementing learning**: See [LEARNING_STRATEGIES_API.md](LEARNING_STRATEGIES_API.md)\n"
            )
            f.write("- **Creating datasets**: See [DATASETS_REFERENCE.md](DATASETS_REFERENCE.md)\n")
            f.write(
                "- **Monitoring training**: See [DIAGNOSTICS_REFERENCE.md](DIAGNOSTICS_REFERENCE.md)\n"
            )
            f.write(
                "- **Visualizing results**: See [VISUALIZATION_REFERENCE.md](VISUALIZATION_REFERENCE.md)\n"
            )
            f.write(
                "- **Computing biological effects**: See [COMPUTE_FUNCTIONS_REFERENCE.md](COMPUTE_FUNCTIONS_REFERENCE.md)\n"
            )
            f.write(
                "- **Handling errors**: See [EXCEPTIONS_REFERENCE.md](EXCEPTIONS_REFERENCE.md)\n\n"
            )

            f.write("### By Biological Region\n\n")
            bio_regions = [
                "cortex",
                "hippocampus",
                "striatum",
                "cerebellum",
                "thalamus",
                "prefrontal",
            ]
            for region in bio_regions:
                matching = [r for r in self.regions if region in r.name.lower()]
                if matching:
                    f.write(f"- **{region.title()}**: ")
                    f.write(", ".join(f"`{r.name}`" for r in matching))
                    f.write("\n")
            f.write("\n")

        print(f"✅ Generated: {output_file.relative_to(self.docs_dir.parent)}")

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

    def _generate_dependency_graph(self):
        """Generate DEPENDENCY_GRAPH.md with module dependencies."""
        output_file = self.api_dir / "DEPENDENCY_GRAPH.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with output_file.open("w", encoding="utf-8") as f:
            f.write("# Dependency Graph\n\n")
            f.write("> **Auto-generated documentation** - Do not edit manually!\n")
            f.write(f"> Last updated: {timestamp}\n")
            f.write("> Generated from: `scripts/generate_api_docs.py`\n\n")

            f.write(
                "This document visualizes the dependency relationships between Thalia modules.\n\n"
            )

            # Component-level dependency diagram
            f.write("## 🔗 Component Dependencies\n\n")
            f.write("```mermaid\n")
            f.write("graph TD\n")
            f.write('    Core["Core (protocols, errors)"]\n')
            f.write('    Components["Components (neurons, synapses)"]\n')
            f.write('    Regions["Regions (cortex, hippocampus, etc.)"]\n')
            f.write('    Pathways["Pathways (axonal projection)"]\n')
            f.write('    Learning["Learning (strategies, registry)"]\n')
            f.write('    Brain["Brain (DynamicBrain)"]\n')
            f.write('    Training["Training (curriculum, monitors)"]\n')
            f.write('    Datasets["Datasets"]\n\n')

            f.write("    Core --> Components\n")
            f.write("    Core --> Learning\n")
            f.write("    Components --> Regions\n")
            f.write("    Components --> Pathways\n")
            f.write("    Learning --> Regions\n")
            f.write("    Regions --> Brain\n")
            f.write("    Pathways --> Brain\n")
            f.write("    Brain --> Training\n")
            f.write("    Datasets --> Training\n")
            f.write("    Learning --> Training\n")
            f.write("```\n\n")

            # Region dependency diagram
            f.write("## 🧠 Region Dependencies\n\n")
            f.write("```mermaid\n")
            f.write("graph LR\n")
            f.write('    NeuralRegion["NeuralRegion (base)"]\n')
            f.write('    Mixins["Mixins"]\n')
            f.write('    Config["*Config"]\n')
            f.write('    Neurons["ConductanceLIF"]\n')
            f.write('    Strategy["LearningStrategy"]\n\n')

            for region in self.regions[:6]:  # Show first 6 regions
                safe_name = region.name.replace("-", "_").replace(" ", "_")
                f.write(f'    {safe_name}["{region.name}"]\n')
                f.write(f"    NeuralRegion --> {safe_name}\n")
                f.write(f"    Mixins --> {safe_name}\n")
                f.write(f"    Config --> {safe_name}\n")
                f.write(f"    Neurons --> {safe_name}\n")
                f.write(f"    Strategy --> {safe_name}\n")

            if len(self.regions) > 6:
                f.write(f'    More["... +{len(self.regions)-6} more regions"]\n')
                f.write("    NeuralRegion --> More\n")
            f.write("```\n\n")

            # Import hierarchy
            f.write("## 📦 Module Import Layers\n\n")
            f.write("```mermaid\n")
            f.write("graph TB\n")
            f.write('    subgraph Layer1["Layer 1: Foundation"]\n')
            f.write('        L1A["core.protocols"]\n')
            f.write('        L1B["core.errors"]\n')
            f.write('        L1C["config"]\n')
            f.write("    end\n\n")

            f.write('    subgraph Layer2["Layer 2: Components"]\n')
            f.write('        L2A["components.neurons"]\n')
            f.write('        L2B["components.synapses"]\n')
            f.write('        L2C["neuromodulation"]\n')
            f.write("    end\n\n")

            f.write('    subgraph Layer3["Layer 3: Learning"]\n')
            f.write('        L3A["learning.rules"]\n')
            f.write('        L3B["learning.strategies"]\n')
            f.write('        L3C["learning.registry"]\n')
            f.write("    end\n\n")

            f.write('    subgraph Layer4["Layer 4: Regions & Pathways"]\n')
            f.write('        L4A["regions.*"]\n')
            f.write('        L4B["pathways.*"]\n')
            f.write('        L4C["mixins.*"]\n')
            f.write("    end\n\n")

            f.write('    subgraph Layer5["Layer 5: Brain"]\n')
            f.write('        L5A["core.dynamic_brain"]\n')
            f.write('        L5B["core.builder"]\n')
            f.write("    end\n\n")

            f.write('    subgraph Layer6["Layer 6: Training & Apps"]\n')
            f.write('        L6A["training.*"]\n')
            f.write('        L6B["datasets.*"]\n')
            f.write('        L6C["diagnostics.*"]\n')
            f.write("    end\n\n")

            f.write("    Layer1 --> Layer2\n")
            f.write("    Layer2 --> Layer3\n")
            f.write("    Layer2 --> Layer4\n")
            f.write("    Layer3 --> Layer4\n")
            f.write("    Layer4 --> Layer5\n")
            f.write("    Layer5 --> Layer6\n")
            f.write("```\n\n")

            # Dependency guidelines
            f.write("## 📋 Dependency Guidelines\n\n")
            f.write("### Import Rules\n\n")
            f.write("1. **Downward dependencies only**: Higher layers import from lower layers\n")
            f.write(
                "2. **No circular imports**: Modules at the same layer should not import each other\n"
            )
            f.write("3. **Core is foundation**: All modules can import from `core`\n")
            f.write(
                "4. **Regions are independent**: Regions should not import from other regions\n\n"
            )

            f.write("### Common Import Patterns\n\n")
            f.write("```python\n")
            f.write("# Layer 1 (Foundation)\n")
            f.write("from thalia.core.protocols import NeuralComponent\n")
            f.write("from thalia.core.errors import ConfigurationError\n")
            f.write("from thalia.config import ThaliaConfig\n\n")

            f.write("# Layer 2 (Components)\n")
            f.write("from thalia.components.neurons import ConductanceLIF\n")
            f.write("from thalia.components.synapses import WeightInitializer\n")
            f.write("from thalia.neuromodulation import NeuromodulatorManager\n\n")

            f.write("# Layer 3 (Learning)\n")
            f.write("from thalia.learning import create_strategy\n")
            f.write("from thalia.learning.rules import STDPRule\n\n")

            f.write("# Layer 4 (Regions)\n")
            f.write("from thalia.regions.cortex import LayeredCortex\n")
            f.write("from thalia.mixins import GrowthMixin\n\n")

            f.write("# Layer 5 (Brain)\n")
            f.write("from thalia.core.dynamic_brain import DynamicBrain\n")
            f.write("from thalia.core.builder import BrainBuilder\n\n")

            f.write("# Layer 6 (Training)\n")
            f.write("from thalia.training import CurriculumTrainer\n")
            f.write("from thalia.datasets import create_stage0_temporal_dataset\n")
            f.write("from thalia.diagnostics import HealthMonitor\n")
            f.write("```\n\n")

            # Circular dependency warnings
            f.write("## ⚠️ Avoiding Circular Dependencies\n\n")
            f.write("### Common Pitfalls\n\n")
            f.write("- **Region importing Brain**: Use dependency injection instead\n")
            f.write("- **Config importing Components**: Keep configs as pure data\n")
            f.write("- **Cross-region imports**: Use protocols/interfaces instead\n")
            f.write("- **Training importing specific regions**: Use registry pattern\n\n")

            f.write("### Solutions\n\n")
            f.write("1. **Protocols**: Define interfaces in `core.protocols`\n")
            f.write("2. **Registry**: Use `ComponentRegistry` for dynamic lookup\n")
            f.write("3. **Dependency Injection**: Pass dependencies through constructors\n")
            f.write("4. **Type hints**: Use string literals for forward references\n\n")

        print(f"✅ Generated: {output_file.relative_to(self.docs_dir.parent)}")

    def _generate_architecture_guide(self):
        """Generate ARCHITECTURE_GUIDE.md with system architecture diagrams."""
        output_file = self.api_dir / "ARCHITECTURE_GUIDE.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with output_file.open("w", encoding="utf-8") as f:
            f.write("# Architecture Guide\n\n")
            f.write("> **Auto-generated documentation** - Do not edit manually!\n")
            f.write(f"> Last updated: {timestamp}\n")
            f.write("> Generated from: `scripts/generate_api_docs.py`\n\n")

            f.write(
                "This guide provides architectural diagrams and design patterns for the Thalia framework.\n\n"
            )

            # System overview
            f.write("## 🏗️ System Architecture Overview\n\n")
            f.write("```mermaid\n")
            f.write("graph TB\n")
            f.write('    subgraph User["User Layer"]\n')
            f.write('        Script["Training Script"]\n')
            f.write("    end\n\n")

            f.write('    subgraph API["High-Level API"]\n')
            f.write('        Builder["BrainBuilder"]\n')
            f.write('        Trainer["CurriculumTrainer"]\n')
            f.write('        Datasets["Dataset Factories"]\n')
            f.write("    end\n\n")

            f.write('    subgraph Brain["Brain Layer"]\n')
            f.write('        DB["DynamicBrain"]\n')
            f.write('        Registry["ComponentRegistry"]\n')
            f.write('        Regions["Neural Regions"]\n')
            f.write('        Pathways["Axonal Pathways"]\n')
            f.write("    end\n\n")

            f.write('    subgraph Components["Component Layer"]\n')
            f.write('        Neurons["Neuron Models"]\n')
            f.write('        Synapses["Synaptic Weights"]\n')
            f.write('        Learning["Learning Strategies"]\n')
            f.write('        Neuromod["Neuromodulators"]\n')
            f.write("    end\n\n")

            f.write('    subgraph Support["Support Systems"]\n')
            f.write('        Config["Configuration"]\n')
            f.write('        Diagnostics["Diagnostics"]\n')
            f.write('        Checkpoints["Checkpointing"]\n')
            f.write("    end\n\n")

            f.write("    Script --> Builder\n")
            f.write("    Script --> Trainer\n")
            f.write("    Script --> Datasets\n")
            f.write("    Builder --> DB\n")
            f.write("    Trainer --> DB\n")
            f.write("    DB --> Registry\n")
            f.write("    DB --> Regions\n")
            f.write("    DB --> Pathways\n")
            f.write("    Regions --> Neurons\n")
            f.write("    Regions --> Synapses\n")
            f.write("    Regions --> Learning\n")
            f.write("    Regions --> Neuromod\n")
            f.write("    Pathways --> Neurons\n")
            f.write("    Trainer --> Diagnostics\n")
            f.write("    DB --> Config\n")
            f.write("    DB --> Checkpoints\n")
            f.write("```\n\n")

            # Data flow diagram
            f.write("## 📊 Data Flow Architecture\n\n")
            f.write("```mermaid\n")
            f.write("graph LR\n")
            f.write('    Input["Input Data"]\n')
            f.write('    Encoding["Spike Encoding"]\n')
            f.write('    Thalamus["Thalamus<br/>(relay)"]  \n')
            f.write('    Cortex["Cortex<br/>(processing)"]\n')
            f.write('    Hippo["Hippocampus<br/>(memory)"]\n')
            f.write('    Striatum["Striatum<br/>(action)"]\n')
            f.write('    Output["Output Spikes"]\n')
            f.write('    Dopamine["Dopamine<br/>(reward)"]  \n\n')

            f.write("    Input --> Encoding\n")
            f.write("    Encoding --> Thalamus\n")
            f.write("    Thalamus --> Cortex\n")
            f.write("    Cortex --> Hippo\n")
            f.write("    Cortex --> Striatum\n")
            f.write("    Hippo --> Cortex\n")
            f.write("    Striatum --> Output\n")
            f.write("    Dopamine -.->|modulates| Striatum\n")
            f.write("    Dopamine -.->|modulates| Cortex\n")
            f.write("```\n\n")

            # Component composition
            f.write("## 🧩 Component Composition Pattern\n\n")
            f.write("```mermaid\n")
            f.write("classDiagram\n")
            f.write("    class NeuralRegion {\n")
            f.write("        +forward()\n")
            f.write("        +reset_state()\n")
            f.write("        +get_state()\n")
            f.write("    }\n\n")

            f.write("    class NeuromodulatorMixin {\n")
            f.write("        +set_neuromodulators()\n")
            f.write("        +decay_neuromodulators()\n")
            f.write("    }\n\n")

            f.write("    class GrowthMixin {\n")
            f.write("        +grow_output()\n")
            f.write("        +grow_input()\n")
            f.write("    }\n\n")

            f.write("    class ResettableMixin {\n")
            f.write("        +reset_state()\n")
            f.write("    }\n\n")

            f.write("    class DiagnosticsMixin {\n")
            f.write("        +collect_diagnostics()\n")
            f.write("    }\n\n")

            f.write("    class ConductanceLIF {\n")
            f.write("        +forward()\n")
            f.write("        -update_voltage()\n")
            f.write("    }\n\n")

            f.write("    class LearningStrategy {\n")
            f.write("        +compute_update()\n")
            f.write("    }\n\n")

            f.write("    NeuralRegion <|-- LayeredCortex\n")
            f.write("    NeuralRegion <|-- Hippocampus\n")
            f.write("    NeuralRegion <|-- Striatum\n")
            f.write("    NeuromodulatorMixin <|-- LayeredCortex\n")
            f.write("    GrowthMixin <|-- LayeredCortex\n")
            f.write("    ResettableMixin <|-- LayeredCortex\n")
            f.write("    DiagnosticsMixin <|-- LayeredCortex\n")
            f.write("    LayeredCortex *-- ConductanceLIF\n")
            f.write("    LayeredCortex *-- LearningStrategy\n")
            f.write("```\n\n")

            # Brain lifecycle
            f.write("## ⏰ Brain Lifecycle\n\n")
            f.write("```mermaid\n")
            f.write("stateDiagram-v2\n")
            f.write("    [*] --> Configuration\n")
            f.write("    Configuration --> Construction: BrainBuilder.build()\n")
            f.write("    Construction --> Initialization: initialize components\n")
            f.write("    Initialization --> Training: start training loop\n")
            f.write("    Training --> Forward: process batch\n")
            f.write("    Forward --> Learning: update weights\n")
            f.write("    Learning --> Diagnostics: check health\n")
            f.write("    Diagnostics --> Training: next batch\n")
            f.write("    Training --> Checkpoint: save progress\n")
            f.write("    Checkpoint --> Training: resume\n")
            f.write("    Training --> Evaluation: epoch end\n")
            f.write("    Evaluation --> Training: continue\n")
            f.write("    Evaluation --> [*]: training complete\n")
            f.write("```\n\n")

            # Spike processing pipeline
            f.write("## ⚡ Spike Processing Pipeline\n\n")
            f.write("```mermaid\n")
            f.write("sequenceDiagram\n")
            f.write("    participant D as Dataset\n")
            f.write("    participant R as Region\n")
            f.write("    participant N as Neurons\n")
            f.write("    participant L as Learning\n")
            f.write("    participant NM as Neuromodulators\n\n")

            f.write("    D->>R: Input spikes (batch)\n")
            f.write("    R->>R: Synaptic integration\n")
            f.write("    R->>N: Synaptic currents\n")
            f.write("    N->>N: Update membrane voltage\n")
            f.write("    N->>N: Check threshold\n")
            f.write("    N-->>R: Output spikes\n")
            f.write("    R->>L: Pre & post spikes\n")
            f.write("    NM-->>L: Modulator levels\n")
            f.write("    L->>L: Compute weight update\n")
            f.write("    L-->>R: Updated weights\n")
            f.write("    R-->>D: Output for next region\n")
            f.write("```\n\n")

            # Learning strategy selection
            f.write("## 🎓 Learning Strategy Selection\n\n")
            f.write("```mermaid\n")
            f.write("graph TD\n")
            f.write("    Start[Choose Learning Strategy]\n")
            f.write("    Cortical{Cortical<br/>Region?}\n")
            f.write("    Reward{Reward-based<br/>Learning?}\n")
            f.write("    Memory{One-shot<br/>Memory?}\n")
            f.write("    Motor{Motor<br/>Learning?}\n\n")

            f.write("    Start --> Cortical\n")
            f.write("    Cortical -->|Yes| STDP_BCM[create_cortex_strategy<br/>STDP + BCM]\n")
            f.write("    Cortical -->|No| Reward\n")
            f.write("    Reward -->|Yes| ThreeFactor[create_striatum_strategy<br/>Three-factor]\n")
            f.write("    Reward -->|No| Memory\n")
            f.write("    Memory -->|Yes| Hippocampal[create_hippocampus_strategy<br/>Fast STDP]\n")
            f.write("    Memory -->|No| Motor\n")
            f.write("    Motor -->|Yes| Error[create_cerebellum_strategy<br/>Error-corrective]\n")
            f.write("    Motor -->|No| Hebbian[Basic Hebbian]\n")
            f.write("```\n\n")

            # Configuration hierarchy
            f.write("## ⚙️ Configuration Hierarchy\n\n")
            f.write("```mermaid\n")
            f.write("graph TD\n")
            f.write("    Global[ThaliaConfig<br/>Global settings]\n")
            f.write("    Brain[BrainConfig<br/>Architecture]\n")
            f.write("    Regional[*RegionConfig<br/>Region-specific]\n")
            f.write("    Builder[BrainBuilder<br/>Size specification]\n\n")

            f.write("    Global --> Brain\n")
            f.write("    Brain --> Regional\n")
            f.write("    Builder --> Regional\n")
            f.write("    Regional --> Cortex[LayeredCortexConfig]\n")
            f.write("    Regional --> Hippo[HippocampusConfig]\n")
            f.write("    Regional --> Stri[StriatumConfig]\n")
            f.write("```\n\n")
            f.write(
                "**Note**: Region sizes are specified directly in BrainBuilder.add_component() calls.\n\n"
            )

            # Best practices
            f.write("## 💡 Architectural Best Practices\n\n")
            f.write("### Design Principles\n\n")
            f.write("1. **Biological Plausibility**: All designs follow neuroscience principles\n")
            f.write("2. **Local Learning**: No global error signals or backpropagation\n")
            f.write("3. **Spike-Based**: Binary spikes, not firing rates\n")
            f.write("4. **Modular Composition**: Regions are independent, composable units\n")
            f.write("5. **Mixins for Cross-Cutting**: Common functionality via mixins\n\n")

            f.write("### Component Guidelines\n\n")
            f.write("- **Regions**: Inherit from `NeuralRegion`, use standard mixins\n")
            f.write("- **Pathways**: Pure spike routing, no learning\n")
            f.write("- **Learning**: Implement `LearningStrategy` protocol\n")
            f.write("- **Configs**: Pure dataclasses, no logic\n")
            f.write("- **Neurons**: Only `ConductanceLIF` model used\n\n")

            f.write("### Growth Strategy\n\n")
            f.write("1. Start with small networks (64-256 neurons per region)\n")
            f.write("2. Train on Stage 0 (temporal sequences)\n")
            f.write("3. Grow network based on curriculum needs\n")
            f.write("4. Add new regions via `ComponentRegistry`\n")
            f.write("5. Use dynamic weight initialization\n\n")

            f.write("## 📚 Related Documentation\n\n")
            f.write(
                "- [COMPONENT_CATALOG.md](COMPONENT_CATALOG.md) - All available regions and pathways\n"
            )
            f.write("- [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md) - Module dependency structure\n")
            f.write(
                "- [LEARNING_STRATEGIES_API.md](LEARNING_STRATEGIES_API.md) - Learning rule selection\n"
            )
            f.write("- [API_INDEX.md](API_INDEX.md) - Complete component index\n\n")

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
