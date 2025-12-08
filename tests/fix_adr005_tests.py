"""
Script to fix ADR-005 batch dimension violations in test files.

Replaces patterns like:
- torch.randn(1, n) → torch.randn(n)
- torch.zeros(1, n) → torch.zeros(n)
- torch.ones(1, n) → torch.ones(n)
- .reshape(1, -1) → (remove)
- [0, :] → (remove indexing)

Run from repository root:
    python tests/fix_adr005_tests.py
"""

import re
from pathlib import Path
from typing import List, Tuple


def find_test_files() -> List[Path]:
    """Find all test Python files."""
    test_dir = Path("tests")
    return list(test_dir.rglob("test_*.py"))


def fix_tensor_creation(content: str) -> Tuple[str, int]:
    """Fix torch tensor creation calls to remove batch dimension."""
    changes = 0
    
    # Pattern 1: torch.randn(1, n) → torch.randn(n)
    pattern1 = r'torch\.randn\(1,\s*(\d+|[a-z_]+)\)'
    new_content, count1 = re.subn(pattern1, r'torch.randn(\1)', content)
    changes += count1
    
    # Pattern 2: torch.zeros(1, n) → torch.zeros(n)
    pattern2 = r'torch\.zeros\(1,\s*(\d+|[a-z_]+)\)'
    new_content, count2 = re.subn(pattern2, r'torch.zeros(\1)', new_content)
    changes += count2
    
    # Pattern 3: torch.ones(1, n) → torch.ones(n)
    pattern3 = r'torch\.ones\(1,\s*(\d+|[a-z_]+)\)'
    new_content, count3 = re.subn(pattern3, r'torch.ones(\1)', new_content)
    changes += count3
    
    # Pattern 4: torch.randn((1, n)) → torch.randn(n) [tuple form]
    pattern4 = r'torch\.randn\(\(1,\s*(\d+|[a-z_]+)\)\)'
    new_content, count4 = re.subn(pattern4, r'torch.randn(\1)', new_content)
    changes += count4
    
    return new_content, changes


def fix_shape_assertions(content: str) -> Tuple[str, int]:
    """Fix shape assertions that expect 2D."""
    changes = 0
    
    # Pattern: assert x.shape == (1, n) → assert x.shape == (n,) [with variable]
    pattern1 = r'assert\s+(\w+)\.shape\s*==\s*\(1,\s*([a-z_\d]+)\)'
    new_content, count1 = re.subn(pattern1, r'assert \1.shape == (\2,)', content)
    changes += count1
    
    # Pattern: assert x.shape == (1, 10) → assert x.shape == (10,) [with number]
    pattern2 = r'assert\s+(\w+)\.shape\s*==\s*\(1,\s*(\d+)\)'
    new_content, count2 = re.subn(pattern2, r'assert \1.shape == (\2,)', new_content)
    changes += count2
    
    return new_content, changes


def remove_squeeze_operations(content: str) -> Tuple[str, int]:
    """Remove unnecessary .squeeze(0) operations."""
    changes = 0
    
    # Pattern: x.squeeze(0) → x [when x is already 1D]
    # This is tricky - only remove if safe
    # For now, we'll be conservative
    
    return content, changes


def fix_indexing(content: str) -> Tuple[str, int]:
    """Fix tensor indexing that assumes batch dimension."""
    changes = 0
    
    # Pattern: x[0, :] → x (when x is 1D)
    # Pattern: x[0, i:j] → x[i:j]
    # These are complex and context-dependent, skip for manual review
    
    return content, changes


def process_file(filepath: Path, dry_run: bool = True) -> Tuple[int, bool]:
    """Process a single test file."""
    try:
        content = filepath.read_text(encoding='utf-8')
        original_content = content
        total_changes = 0
        
        # Apply fixes
        content, c1 = fix_tensor_creation(content)
        total_changes += c1
        
        content, c2 = fix_shape_assertions(content)
        total_changes += c2
        
        content, c3 = remove_squeeze_operations(content)
        total_changes += c3
        
        content, c4 = fix_indexing(content)
        total_changes += c4
        
        # Write back if changes made and not dry run
        if total_changes > 0 and not dry_run:
            filepath.write_text(content, encoding='utf-8')
            return total_changes, True
        
        return total_changes, content != original_content
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return 0, False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix ADR-005 violations in tests')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without applying')
    parser.add_argument('--file', type=str, help='Process single file')
    args = parser.parse_args()
    
    if args.file:
        files = [Path(args.file)]
    else:
        files = find_test_files()
    
    print(f"Found {len(files)} test files")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'APPLY CHANGES'}")
    print()
    
    total_files_changed = 0
    total_changes = 0
    
    for filepath in sorted(files):
        changes, modified = process_file(filepath, dry_run=args.dry_run)
        if changes > 0:
            status = "[DRY RUN]" if args.dry_run else "[MODIFIED]"
            print(f"{status} {filepath.relative_to('tests')}: {changes} changes")
            total_files_changed += 1
            total_changes += changes
    
    print()
    print(f"Summary: {total_changes} changes across {total_files_changed} files")
    if args.dry_run:
        print("Run without --dry-run to apply changes")


if __name__ == '__main__':
    main()
