"""Fix test assertions that check for wrong shapes."""
import re
from pathlib import Path

def fix_test_assertions(content: str) -> tuple[str, int]:
    """Fix assertions that check for (1, n) shapes."""
    changes = 0
    
    # Pattern 1: assert x.shape == (1, n)
    pattern1 = r'assert\s+(\w+)\.shape\s*==\s*\(1,\s*(\d+)\)'
    def repl1(m):
        nonlocal changes
        changes += 1
        var_name = m.group(1)
        size = m.group(2)
        return f'assert {var_name}.shape == ({size},)'
    content = re.sub(pattern1, repl1, content)
    
    # Pattern 2: assert x.shape == (batch_size, n)
    pattern2 = r'assert\s+(\w+)\.shape\s*==\s*\(batch_size,\s*(\d+)\)'
    def repl2(m):
        nonlocal changes
        changes += 1
        var_name = m.group(1)
        size = m.group(2)
        # For ADR-005, we don't have batch_size anymore
        return f'assert {var_name}.shape == ({size},)  # ADR-005: No batch dimension'
    content = re.sub(pattern2, repl2, content)
    
    # Pattern 3: assert output.shape[0] == 1
    pattern3 = r'assert\s+(\w+)\.shape\[0\]\s*==\s*1'
    def repl3(m):
        nonlocal changes
        changes += 1
        var_name = m.group(1)
        # This check is no longer valid in 1D architecture
        return f'# assert {var_name}.shape[0] == 1  # REMOVED: ADR-005 uses 1D tensors'
    content = re.sub(pattern3, repl3, content)
    
    # Pattern 4: assert output.ndim == 2
    pattern4 = r'assert\s+(\w+)\.ndim\s*==\s*2'
    def repl4(m):
        nonlocal changes
        changes += 1
        var_name = m.group(1)
        return f'assert {var_name}.ndim == 1  # ADR-005: 1D tensors'
    content = re.sub(pattern4, repl4, content)
    
    return content, changes

def process_test_files():
    """Process test files to fix assertions."""
    test_dir = Path(__file__).parent
    
    # Files with assertion failures
    test_files = [
        "unit/test_validation.py",
        "unit/test_error_handling.py",
        "unit/test_properties.py",
        "unit/test_robustness.py",
        "unit/test_fixtures.py",
    ]
    
    total_changes = 0
    for file_path in test_files:
        full_path = test_dir / file_path
        if not full_path.exists():
            continue
            
        content = full_path.read_text(encoding='utf-8')
        new_content, changes = fix_test_assertions(content)
        
        if changes > 0:
            full_path.write_text(new_content, encoding='utf-8')
            print(f"[MODIFIED] {file_path}: {changes} changes")
            total_changes += changes
        else:
            print(f"[SKIPPED] {file_path}: No changes")
    
    print(f"\nTotal: {total_changes} assertions fixed")

if __name__ == "__main__":
    process_test_files()
