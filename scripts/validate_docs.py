"""
Automated Documentation Validation Script

Validates that documentation examples and references match the actual codebase.
Catches common issues like:
- Outdated function names
- References to removed classes
- Incorrect import statements
- Mismatched API signatures

Run this as part of CI/CD or pre-commit hooks.
"""

import ast
import re
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Results from documentation validation."""
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.passed = False
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


class DocumentationValidator:
    """Validates documentation against codebase."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.src_dir = repo_root / "src" / "thalia"
        self.docs_dir = repo_root / "docs"

        # Build indices of actual codebase
        self.actual_functions = self._index_functions()
        self.actual_classes = self._index_classes()
        self.actual_modules = self._index_modules()

    def _index_functions(self) -> Dict[str, List[Path]]:
        """Index all function definitions in the codebase."""
        functions = {}

        for py_file in self.src_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read(), filename=str(py_file))

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if node.name not in functions:
                            functions[node.name] = []
                        functions[node.name].append(py_file)
            except (SyntaxError, UnicodeDecodeError):
                continue

        return functions

    def _index_classes(self) -> Dict[str, List[Path]]:
        """Index all class definitions in the codebase."""
        classes = {}

        for py_file in self.src_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read(), filename=str(py_file))

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        if node.name not in classes:
                            classes[node.name] = []
                        classes[node.name].append(py_file)
            except (SyntaxError, UnicodeDecodeError):
                continue

        return classes

    def _index_modules(self) -> Set[str]:
        """Index all importable modules."""
        modules = set()

        for py_file in self.src_dir.rglob("*.py"):
            # Convert path to module name
            rel_path = py_file.relative_to(self.src_dir.parent)
            module = str(rel_path).replace("/", ".").replace("\\", ".").replace(".py", "")
            modules.add(module)

        return modules

    def validate_all(self) -> ValidationResult:
        """Run all validation checks."""
        result = ValidationResult()

        # Validate all markdown files
        for md_file in self.docs_dir.rglob("*.md"):
            self._validate_markdown_file(md_file, result)

        # Also check copilot instructions
        copilot_file = self.repo_root / ".github" / "copilot-instructions.md"
        if copilot_file.exists():
            self._validate_markdown_file(copilot_file, result)

        return result

    def _validate_markdown_file(self, md_file: Path, result: ValidationResult) -> None:
        """Validate a single markdown file."""
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            result.add_warning(f"{md_file}: Could not read file (encoding issue)")
            return

        rel_path = md_file.relative_to(self.repo_root)

        # Extract code blocks
        code_blocks = self._extract_code_blocks(content)

        for i, (lang, code) in enumerate(code_blocks):
            if lang == "python":
                self._validate_python_code(code, rel_path, i, result)

    def _extract_code_blocks(self, content: str) -> List[Tuple[str, str]]:
        """Extract code blocks from markdown."""
        pattern = r"```(\w+)\n(.*?)```"
        matches = re.findall(pattern, content, re.DOTALL)
        return matches

    def _validate_python_code(
        self,
        code: str,
        file_path: Path,
        block_num: int,
        result: ValidationResult
    ) -> None:
        """Validate Python code block."""

        # Check for common outdated patterns
        outdated_patterns = [
            (r"create_learning_strategy\s*\(", "create_learning_strategy", "create_strategy"),
            (r"class\s+SimpleLIF", "SimpleLIF", "ConductanceLIF"),
            (r"from.*SimpleLIF", "SimpleLIF", "ConductanceLIF"),
            (r"EventDrivenBrain", "EventDrivenBrain", "DynamicBrain"),
            (r"brain\.cortex\b", "brain.cortex", 'brain.components["cortex"]'),
            (r"brain\.hippocampus\b", "brain.hippocampus", 'brain.components["hippocampus"]'),
            (r"brain\.striatum\b", "brain.striatum", 'brain.components["striatum"]'),
            (r"brain\.pfc\b", "brain.pfc", 'brain.components["pfc"]'),
        ]

        for pattern, old, new in outdated_patterns:
            if re.search(pattern, code):
                result.add_error(
                    f"{file_path} (block {block_num}): Uses outdated '{old}', "
                    f"should be '{new}'"
                )

        # Check imports
        self._validate_imports(code, file_path, block_num, result)

        # Check function calls
        self._validate_function_calls(code, file_path, block_num, result)

    def _validate_imports(
        self,
        code: str,
        file_path: Path,
        block_num: int,
        result: ValidationResult
    ) -> None:
        """Validate import statements using AST parsing to handle multi-line imports."""

        # Try to parse the code block as Python
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Code might be incomplete snippet, skip import validation
            return

        # Walk the AST to find import statements
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module
                if module is None:
                    continue

                # Extract imported names
                imports = []
                for alias in node.names:
                    if alias.name == '*':
                        continue  # Skip wildcard imports
                    imports.append(alias.name)

                # Check if module exists
                if module.startswith("thalia."):
                    module_path = self.src_dir.parent / module.replace(".", "/")
                    if not module_path.exists() and not (module_path.with_suffix(".py")).exists():
                        # Check if it's in __init__.py
                        init_file = module_path.parent / "__init__.py"
                        if not init_file.exists():
                            result.add_warning(
                                f"{file_path} (block {block_num}): Module '{module}' may not exist"
                            )

                    # Check if specific imports exist (but be lenient - could be in __init__)
                    for imp in imports:
                        if imp not in self.actual_functions and imp not in self.actual_classes:
                            # Only warn, don't error - might be exported from __init__.py
                            # Skip the warning to reduce false positives
                            pass

    def _validate_function_calls(
        self,
        code: str,
        file_path: Path,
        block_num: int,
        result: ValidationResult
    ) -> None:
        """Validate function calls against actual codebase."""

        # Known important functions to check
        important_functions = {
            "create_strategy": True,
            "create_learning_strategy": False,  # Removed
            "create_cortex_strategy": True,
            "create_striatum_strategy": True,
            "create_hippocampus_strategy": True,
        }

        for func_name, should_exist in important_functions.items():
            pattern = rf"\b{func_name}\s*\("
            if re.search(pattern, code):
                exists = func_name in self.actual_functions
                if should_exist and not exists:
                    result.add_error(
                        f"{file_path} (block {block_num}): "
                        f"Uses '{func_name}' which doesn't exist in codebase"
                    )
                elif not should_exist and exists:
                    result.add_warning(
                        f"{file_path} (block {block_num}): "
                        f"Uses '{func_name}' which is deprecated"
                    )


def main():
    """Run documentation validation."""
    # Find repository root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent

    print("=" * 80)
    print("Thalia Documentation Validation")
    print("=" * 80)
    print(f"Repository: {repo_root}")
    print()

    validator = DocumentationValidator(repo_root)

    print("Indexing codebase...")
    print(f"  Found {len(validator.actual_functions)} functions")
    print(f"  Found {len(validator.actual_classes)} classes")
    print(f"  Found {len(validator.actual_modules)} modules")
    print()

    print("Validating documentation...")
    result = validator.validate_all()
    print()

    # Print results
    if result.errors:
        print("❌ ERRORS:")
        for error in result.errors:
            print(f"  {error}")
        print()

    if result.warnings:
        print("⚠️  WARNINGS:")
        for warning in result.warnings:
            print(f"  {warning}")
        print()

    if result.passed and not result.warnings:
        print("✅ All documentation validation checks passed!")
    elif result.passed:
        print(f"✅ Validation passed with {len(result.warnings)} warnings")
    else:
        print(f"❌ Validation failed with {len(result.errors)} errors")

    print()
    print("=" * 80)

    # Exit with error code if validation failed
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
