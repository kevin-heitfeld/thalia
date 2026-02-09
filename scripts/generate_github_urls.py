"""Generate a list of GitHub raw URLs for all files in the repository."""

import os
from pathlib import Path

# Configuration
GITHUB_USER = "kevin-heitfeld"
REPO_NAME = "thalia"
BRANCH = "main"
BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/refs/heads/{BRANCH}"

# Directories to skip
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".vscode",
    ".idea",
    "dist",
    "build",
    "*.egg-info",
}

# File extensions or names to skip (optional)
SKIP_FILES = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dll",
    ".dylib",
    ".DS_Store",
    "Thumbs.db",
}


def should_skip(path: Path) -> bool:
    """Check if a path should be skipped."""
    # Check if any parent directory is in SKIP_DIRS
    for part in path.parts:
        if part in SKIP_DIRS:
            return True

    # Check if file extension or name is in SKIP_FILES
    if path.suffix in SKIP_FILES or path.name in SKIP_FILES:
        return True

    return False


def generate_github_urls(repo_root: Path, output_file: Path) -> None:
    """Generate GitHub raw URLs for all files in the repository."""
    urls = []

    # Walk through all files in the repository
    for file_path in sorted(repo_root.rglob("*")):
        # Skip directories
        if file_path.is_dir():
            continue

        # Skip files/directories we don't want
        if should_skip(file_path):
            continue

        # Get relative path from repo root
        try:
            rel_path = file_path.relative_to(repo_root)
        except ValueError:
            continue

        # Convert Windows path separators to forward slashes for URLs
        url_path = str(rel_path).replace("\\", "/")

        # Generate the raw GitHub URL
        github_url = f"{BASE_URL}/{url_path}"
        urls.append(github_url)

    # Write URLs to output file
    with open(output_file, "w", encoding="utf-8") as f:
        for url in urls:
            f.write(f"- {url}\n")

    print(f"Generated {len(urls)} URLs")
    print(f"Output written to: {output_file}")


def main():
    """Main entry point."""
    # Get the repository root (parent of scripts directory)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent

    # Output file
    output_file = repo_root / "temp" / "_github_file_urls.txt"

    print(f"Scanning repository: {repo_root}")
    print("Generating GitHub raw URLs...")

    generate_github_urls(repo_root, output_file)


if __name__ == "__main__":
    main()
