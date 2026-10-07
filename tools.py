import subprocess
from context import note_read
from skills import read_skill as _read_skill
from todos import write_todos as _write_todos, update_todo as _update_todo
import sandbox as _sandbox

# ── tool implementations ───────────────────────────────────────────────────────

def bash(command: str) -> str:
    """Run a shell command inside the sandbox and return its output."""
    result = _sandbox.run(command, capture_output=True, text=True)
    output = result.stdout + result.stderr
    # Hard cap to prevent giant outputs from blowing the context window
    lines = output.splitlines()
    if len(lines) > 200:
        output = "\n".join(lines[:200]) + f"\n… ({len(lines) - 200} more lines truncated)"
    if len(output) > 8000:
        output = output[:8000] + "\n… (truncated, output too large)"
    return output


def read_file(path: str, line_start: int = None, line_end: int = None) -> str:
    """Read a file and return its contents, optionally sliced by line range."""
    try:
        p = Path(path)
        if p.is_dir():
            # Give a helpful nudge instead of an opaque error
            entries = sorted(p.iterdir())
            names   = "\n".join(
                ("  [dir]  " if e.is_dir() else "  [file] ") + e.name
                for e in entries
            )
            return (
                f"Error: '{path}' is a directory, not a file.\n"
                f"Use list_dir to explore it, or bash to inspect it.\n\n"
                f"Contents of {path}:\n{names}"
            )
        note_read(path)
        with open(p, "r") as f:
            lines = f.readlines()
        if line_start is not None or line_end is not None:
            start = (line_start - 1) if line_start else 0
            end   = line_end if line_end else len(lines)
            lines = lines[start:end]
        return "".join(lines)
    except Exception as e:
        return f"Error reading file: {e}"


def list_dir(path: str = ".") -> str:
    """List the contents of a directory."""
    try:
        p       = Path(path)
        entries = sorted(p.iterdir())
        lines   = []
        for e in entries:
            tag  = "[dir] " if e.is_dir() else "[file]"
            size = f"  {e.stat().st_size:>8} bytes" if e.is_file() else ""
            lines.append(f"  {tag} {e.name}{size}")
        return f"{path}/\n" + "\n".join(lines) if lines else f"{path}/ (empty)"
    except Exception as e:
        return f"Error listing directory: {e}"


def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        with open(path, "w") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def str_replace(path: str, old_str: str, new_str: str, allow_multi_edit: bool = False) -> str:
    """Replace a string in a file. By default fails if old_str matches multiple times unless allow_multi_edit is True."""
    try:
        with open(path, "r") as f:
            content = f.read()
        count = content.count(old_str)
        if count == 0:
            return f"Error: old_str not found in {path}"
        if count > 1 and not allow_multi_edit:
            return f"Error: old_str matches {count} locations in {path} — make it more specific or set allow_multi_edit=true"
        updated = content.replace(old_str, new_str)
        with open(path, "w") as f:
            f.write(updated)
        replaced = count if allow_multi_edit else 1
        return f"Successfully replaced {replaced} occurrence(s) in {path}"
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
]

# ── dispatch map (name → function) ────────────────────────────────────────────

TOOLS = {
    "bash": bash,
    "read_file": read_file,
    "write_file": write_file,
    "str_replace": str_replace,
    "read_skill": read_skill,
    "write_todos": write_todos,
    "update_todo": update_todo,
}
