# GitHub Copilot Available Tools

This document lists all tools available to GitHub Copilot when working in this workspace.

## File Operations

### `create_file`
Create a new file with specified content. Automatically creates directories if needed.

### `read_file`
Read contents of a file between specified line numbers (1-indexed). Always specify start and end lines.

### `replace_string_in_file`
Edit existing file by replacing exact literal text. Requires oldString (with 3+ lines context before/after) and newString.

### `multi_replace_string_in_file`
Apply multiple replace_string_in_file operations in a single call. More efficient for multiple edits.

### `list_dir`
List contents of a directory. Names ending in `/` are folders, others are files.

### `create_directory`
Create directory structure (like mkdir -p). Not needed before create_file.

## Search Operations

### `file_search`
Find files by glob pattern (e.g., `**/*.py`, `src/**`). Returns file paths only.

### `grep_search`
Fast text search in workspace. Supports exact strings or regex. Use includePattern to limit to specific files/folders.

### `semantic_search`
Natural language search for relevant code or documentation. Returns code snippets from workspace.

### `list_code_usages`
Find all usages (references, definitions, implementations) of a function, class, method, or variable.

## Python-Specific Tools

### `configure_python_environment`
**CRITICAL**: Configure Python environment for workspace. ALWAYS call before running Python code, tests, or commands.

### `get_python_environment_details`
Get details of Python environment (type, version, installed packages). Call configure_python_environment first.

### `get_python_executable_details`
Get fully qualified Python executable path/command for terminal usage. Call configure_python_environment first.

### `install_python_packages`
Install Python packages into the workspace environment. Call configure_python_environment first.

### `mcp_pylance_mcp_s_pylanceDocuments`
Search Pylance documentation for Python language server help, configuration, features, and troubleshooting.

### `mcp_pylance_mcp_s_pylanceFileSyntaxErrors`
Check Python file for syntax errors. Returns detailed error list with line numbers and messages.

### `mcp_pylance_mcp_s_pylanceImports`
Analyze imports across workspace files. Returns all top-level module names imported.

### `mcp_pylance_mcp_s_pylanceInstalledTopLevelModules`
Get available top-level modules from installed packages. Shows what can be imported.

### `mcp_pylance_mcp_s_pylanceInvokeRefactoring`
Apply automated code refactoring (remove unused imports, convert import formats, add type annotations, fix all).

### `mcp_pylance_mcp_s_pylancePythonEnvironments`
Get Python environment information: current active environment and all available environments.

### `mcp_pylance_mcp_s_pylanceRunCodeSnippet`
**PREFERRED**: Execute Python code snippets directly in workspace environment. Better than terminal for Python code.

### `mcp_pylance_mcp_s_pylanceSettings`
Get current Python analysis settings and configuration for workspace.

### `mcp_pylance_mcp_s_pylanceSyntaxErrors`
Validate Python code snippets for syntax errors without saving to file.

### `mcp_pylance_mcp_s_pylanceUpdatePythonEnvironment`
Switch active Python environment for workspace to different installation or virtual environment.

### `mcp_pylance_mcp_s_pylanceWorkspaceRoots`
Get workspace root directories for specific file or all workspace roots.

### `mcp_pylance_mcp_s_pylanceWorkspaceUserFiles`
Get list of all user Python files in workspace (excludes libraries). Respects include/exclude settings.

## Terminal & Execution

### `run_in_terminal`
Execute PowerShell commands in persistent terminal session. Context preserved across calls. Set isBackground=true for long-running processes.

### `get_terminal_output`
Get output of a terminal command previously started with run_in_terminal.

### `terminal_last_command`
Get the last command run in the active terminal.

### `terminal_selection`
Get the current selection in the active terminal.

## Testing

### `runTests`
Run unit tests in files. Provides detailed results. Can collect coverage with mode="coverage".

### `test_failure`
Include test failure information in the prompt.

## Tasks

### `create_and_run_task`
Create and run a build, run, or custom task by generating/adding to tasks.json.

### `run_task`
Run a VS Code task by ID. Prefer this over run_in_terminal for build/run tasks.

### `get_task_output`
Get the output of a task by ID.

## Jupyter Notebooks

### `create_new_jupyter_notebook`
Generate new Jupyter Notebook (.ipynb) in VS Code. Only use when user explicitly requests notebook.

### `edit_notebook_file`
Edit existing notebook file. Operations: insert, delete, or edit cells.

### `copilot_getNotebookSummary`
Get summary of notebook: list of cells with IDs, types, languages, execution info, and output mime types.

### `run_notebook_cell`
Run code cell in notebook directly in notebook editor. Returns execution output.

### `read_notebook_cell_output`
Retrieve output for a notebook cell from most recent execution. Higher token limit than run_notebook_cell.

## Diagnostics & Errors

### `get_errors`
Get compile or lint errors in specific file(s) or across all files. Use to see errors user is seeing.

### `get_changed_files`
Get git diffs of current file changes. Filter by staged, unstaged, or merge-conflicts.

## Workspace & Project Setup

### `create_new_workspace`
Get comprehensive setup steps for complete project structures (NEW projects only, not individual files).

### `get_project_setup_info`
Get project setup information for workspace (python-script, python-project, mcp-server, vscode-extension, etc.). Call after create_new_workspace.

## VS Code Integration

### `get_vscode_api`
Get comprehensive VS Code API documentation for extension development (contribution points, proposed APIs, etc.).

### `run_vscode_command`
Run a VS Code command by ID. Use only during workspace creation process.

### `install_extension`
Install VS Code extension by ID. Use only during workspace creation process.

### `vscode_searchExtensions_internal`
Search VS Code Extensions Marketplace by category, keywords, or extension IDs.

### `open_simple_browser`
Preview website or open URL in editor's Simple Browser (http/https only).

## External Resources

### `fetch_webpage`
Fetch main content from webpage. Useful for summarizing or analyzing webpage content.

### `github_repo`
Search GitHub repository for relevant source code snippets. Use only when user explicitly requests code from specific repo.

### `get_search_view_results`
Get results from the search view.

## Task Management

### `manage_todo_list`
Manage structured todo list for complex multi-step work. Operations: write (replace entire list) or read (retrieve current list).

**When to use**: Complex multi-step work, multiple user requests, before starting work (mark in-progress), immediately after completing each todo.

## Subagent

### `runSubagent`
Launch new agent for complex, multi-step tasks autonomously (research, searching, multi-step execution). Agent is stateless and returns single final message.

---

## Usage Notes

- **Python Environment**: ALWAYS call `configure_python_environment` before any Python operations
- **File Edits**: Use `multi_replace_string_in_file` for multiple independent edits (more efficient)
- **Python Execution**: Prefer `mcp_pylance_mcp_s_pylanceRunCodeSnippet` over terminal for Python code
- **Line Numbers**: Always use exact literal text with context when using replace_string_in_file
- **Background Processes**: Set `isBackground=true` for servers, watch tasks, etc.
- **Parallel Operations**: Combine independent read-only operations in parallel batches when appropriate
