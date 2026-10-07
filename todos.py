"""
Todo list for the agent to plan and track long multi-step tasks.

Statuses:
  pending     [ ]  — not started
  in_progress [~]  — currently being worked on
  done        [x]  — completed
"""

from typing import Literal

Status = Literal["pending", "in_progress", "done"]

STATUS_ICON = {
    "done":        "[x]",
    "in_progress": "[~]",
    "pending":     "[ ]",
}

# In-memory todo list — reset each session
_todos: list[dict] = []


def write_todos(todos: list[dict]) -> str:
    """Replace the entire todo list with a new set of tasks.

    Each item must have:
      content    : str   — short label shown to the user
      activeForm : str   — gerund phrase used while in_progress ("Reading the file")
      status     : str   — pending | in_progress | done
    """
    global _todos
    _todos = [
        {
            "content":    t["content"],
            "activeForm": t.get("activeForm", t["content"]),
            "status":     t.get("status", "pending"),
        }
        for t in todos
    ]
    return f"Todo list updated with {len(_todos)} task(s)."


def update_todo(index: int, status: Status) -> str:
    """Update the status of a single todo by its 0-based index."""
    if index < 0 or index >= len(_todos):
        return f"Error: index {index} out of range (0–{len(_todos) - 1})"
    _todos[index]["status"] = status
    return f"Task {index} marked as {status}."


def get_todos() -> list[dict]:
    """Return the current todo list."""
    return _todos


def todos_prompt() -> str:
    """Render the todo list as a plain-text block for context injection."""
    if not _todos:
        return ""
    lines = []
    for t in _todos:
        icon = STATUS_ICON.get(t["status"], "[ ]")
        label = t["activeForm"] if t["status"] == "in_progress" else t["content"]
        lines.append(f"{icon} {label}")
    return "\n".join(lines)
