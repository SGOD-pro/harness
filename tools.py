import subprocess
from pathlib import Path
from context import note_read
from skills import read_skill as _read_skill
from todos import write_todos as _write_todos, update_todo as _update_todo
from subagent import task

# ── sandbox root — all file operations are confined to this directory ─────────
PLAYGROUND_DIR = (Path(__file__).parent / "playground").resolve()
PLAYGROUND_DIR.mkdir(exist_ok=True)


def _safe_path(path: str) -> tuple[Path, str | None]:
    """
    Resolve path relative to PLAYGROUND_DIR and check it stays inside.
    Returns (resolved_path, None) on success or (None, error_string) if
    the path escapes the playground.
    """
    try:
        # Accept both absolute and relative paths
        p = Path(path)
        if not p.is_absolute():
            p = PLAYGROUND_DIR / p
        resolved = p.resolve()
        # Enforce boundary
        resolved.relative_to(PLAYGROUND_DIR)
        return resolved, None
    except ValueError:
        return None, (
            f"Access denied: '{path}' is outside the playground directory "
            f"({PLAYGROUND_DIR}). All file operations must stay inside playground/."
        )
import sandbox as _sandbox

# ── tool implementations ───────────────────────────────────────────────────────

def bash(command: str) -> str:
    """Run a shell command inside the sandbox and return its output."""
    # Force working directory to playground so relative paths resolve correctly
    full_command = f"cd {PLAYGROUND_DIR} && {command}"
    result = _sandbox.run(full_command, capture_output=True, text=True)
    output = result.stdout + result.stderr
    lines = output.splitlines()
    if len(lines) > 200:
        output = "\n".join(lines[:200]) + f"\n… ({len(lines) - 200} more lines truncated)"
    if len(output) > 8000:
        output = output[:8000] + "\n… (truncated, output too large)"
    return output


def read_file(path: str, line_start: int = None, line_end: int = None) -> str:
    """Read a file from the playground directory."""
    resolved, err = _safe_path(path)
    if err:
        return err
    try:
        if resolved.is_dir():
            entries = sorted(resolved.iterdir())
            names   = "\n".join(
                ("  [dir]  " if e.is_dir() else "  [file] ") + e.name
                for e in entries
            )
            return (
                f"Error: '{path}' is a directory, not a file.\n"
                f"Use list_dir to explore it.\n\nContents:\n{names}"
            )
        note_read(str(resolved))
        lines = resolved.read_text(errors="replace").splitlines(keepends=True)
        if line_start is not None or line_end is not None:
            start = (line_start - 1) if line_start else 0
            end   = line_end if line_end else len(lines)
            lines = lines[start:end]
        return "".join(lines)
    except Exception as e:
        return f"Error reading file: {e}"


def list_dir(path: str = ".") -> str:
    """List the contents of a directory inside the playground."""
    resolved, err = _safe_path(path)
    if err:
        return err
    try:
        if not resolved.exists():
            return f"Error: path '{path}' does not exist."
        if not resolved.is_dir():
            return f"Error: '{path}' is a file, not a directory. Use read_file."
        entries = sorted(resolved.iterdir())
        lines   = []
        for e in entries:
            tag  = "[dir] " if e.is_dir() else "[file]"
            size = f"  {e.stat().st_size:>8} bytes" if e.is_file() else ""
            lines.append(f"  {tag} {e.name}{size}")
        # Show path relative to playground so the model sees "." not "playground/"
        try:
            display = str(resolved.relative_to(PLAYGROUND_DIR))
        except ValueError:
            display = str(resolved)
        header = display if display != "." else "(root)"
        return f"{header}/\n" + "\n".join(lines) if lines else f"{header}/ (empty)"
    except Exception as e:
        return f"Error listing directory: {e}"


def write_file(path: str, content: str) -> str:
    """Write a file inside the playground directory, creating parents if needed."""
    resolved, err = _safe_path(path)
    if err:
        return err
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)
        return f"Successfully wrote to {resolved.relative_to(PLAYGROUND_DIR)}"
    except Exception as e:
        return f"Error writing file: {e}"


def str_replace(path: str, old_str: str, new_str: str, allow_multi_edit: bool = False) -> str:
    """Replace a string in a file inside the playground directory."""
    resolved, err = _safe_path(path)
    if err:
        return err
    try:
        content = resolved.read_text()
        count   = content.count(old_str)
        if count == 0:
            return f"Error: old_str not found in {path}"
        if count > 1 and not allow_multi_edit:
            return f"Error: old_str matches {count} locations — be more specific or set allow_multi_edit=true"
        updated = content.replace(old_str, new_str)
        resolved.write_text(updated)
        replaced = count if allow_multi_edit else 1
        return f"Successfully replaced {replaced} occurrence(s) in {resolved.relative_to(PLAYGROUND_DIR)}"
    except Exception as e:
        return f"Error editing file: {e}"


def read_skill(name: str) -> str:
    """Read the full instructions of a skill by name."""
    return _read_skill(name)


def write_todos(todos: list) -> str:
    """Set the agent's todo list for the current task."""
    return _write_todos(todos)


def update_todo(index: int, status: str) -> str:
    """Update the status of a todo item by index."""
    return _update_todo(index, status)


# ── tool schemas (sent to the model) ──────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk and return its contents. Optionally specify a line range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    },
                    "line_start": {
                        "type": "integer",
                        "description": "1-indexed line to start reading from (optional)",
                    },
                    "line_end": {
                        "type": "integer",
                        "description": "1-indexed line to stop reading at, inclusive (optional)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file on disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write into the file",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace",
            "description": "Replace an exact string in a file. Use this to make targeted edits without rewriting the whole file. Fails if the string is not found or matches multiple times unless allow_multi_edit is set to true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "The exact string to find and replace",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "The string to replace it with",
                    },
                    "allow_multi_edit": {
                        "type": "boolean",
                        "description": "If true, replaces all occurrences instead of failing on duplicates. Defaults to false.",
                    },
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_skill",
            "description": "Read the full instructions of a skill by name. Call this when a skill matches the user's task before proceeding.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the skill to read",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_todos",
            "description": "Set the full todo list for the current task. Call this at the start of any multi-step task to lay out the plan. Each item needs content, activeForm, and status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "List of todo items",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content":    {"type": "string", "description": "Short label for the task"},
                                "activeForm": {"type": "string", "description": "Gerund phrase used while in_progress, e.g. 'Reading the file'"},
                                "status":     {"type": "string", "enum": ["pending", "in_progress", "done"]},
                            },
                            "required": ["content", "activeForm", "status"],
                        },
                    }
                },
                "required": ["todos"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_todo",
            "description": "Update the status of a single todo item by its 0-based index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index":  {"type": "integer", "description": "0-based index of the todo to update"},
                    "status": {"type": "string",  "enum": ["pending", "in_progress", "done"]},
                },
                "required": ["index", "status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": (
                "Spawn a read-only subagent to explore the codebase and answer one question. "
                "Use this when you need to understand how something works — where a feature "
                "lives, how data flows, what calls what — instead of grepping yourself. "
                "The subagent explores in its own context window and hands you back just the "
                "findings, so the search does not fill yours. "
                "It cannot see this conversation, so write the question so it stands alone. "
                "Do all editing yourself; the subagent only reads. "
                "You can call task multiple times in the same response for independent "
                "questions — they run in parallel and each returns only its final answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "The self-contained question or exploration task for the subagent.",
                    }
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List the contents of a directory. Use this instead of read_file when given a directory path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the directory to list (defaults to current directory)",
                    }
                },
                "required": [],
            },
        },
    },
]

# ── dispatch map (name → function) ────────────────────────────────────────────

TOOLS = {
    "task":        task,
    "bash":        bash,
    "read_file":   read_file,
    "list_dir":    list_dir,
    "write_file":  write_file,
    "str_replace": str_replace,
    "read_skill":  read_skill,
    "write_todos": write_todos,
    "update_todo": update_todo,
}
