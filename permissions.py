"""
Permission rules for tool calls.

Each rule maps a glob-style pattern to an action:
  "allow"  — run without asking
  "deny"   — block and return an error immediately
  "ask"    — pause and ask the user before running

Last matching rule wins, so put the catch-all first.
"""

import fnmatch
from pathlib import Path

PROJECT = Path.cwd().resolve()

# ── bash command rules ────────────────────────────────────────────────────────
# Pattern matched against the full command string.

BASH_RULES: dict[str, str] = {
    "*": "ask",                 # catch-all: ask by default
    # read-only: let them through silently
    "ls*":         "allow",
    "pwd":         "allow",
    "cd *":        "allow",
    "echo *":      "allow",
    "sort*":       "allow",
    "uniq*":       "allow",
    "cut *":       "allow",
    "basename *":  "allow",
    "dirname *":   "allow",
    "date*":       "allow",
    "env":         "allow",
    "cat *":       "allow",
    "head *":      "allow",
    "tail *":      "allow",
    "wc *":        "allow",
    "file *":      "allow",
    "which *":     "allow",
    "grep *":      "allow",
    "find *":      "allow",
    "tree*":       "allow",
    "git status*": "ask",
    "git log*":    "ask",
    "git diff*":   "ask",
    "git branch*": "ask",
    "git show*":   "ask",
    # writes/installs: always ask
    "rm *":        "ask",
    "rmdir *":     "ask",
    "mv *":        "ask",
    "cp *":        "ask",
    "chmod *":     "ask",
    "chown *":     "ask",
    "pip *":       "ask",
    "uv *":        "ask",
    "npm *":       "ask",
    "curl *":      "ask",
    "wget *":      "ask",
    # always deny
    "sudo *":      "deny",
    "su *":        "deny",
    ":(){:|:&};:": "deny",      # fork bomb
}

# ── file path rules ───────────────────────────────────────────────────────────
# Pattern matched against the resolved absolute path string.

FILE_RULES: dict[str, str] = {
    "*":                        "allow",   # default: allow file reads/writes
    f"{PROJECT}/.git/*":        "deny",    # never touch .git internals
    f"{PROJECT}/.env":          "ask",     # secrets file — always confirm
    f"{PROJECT}/.env*":         "ask",
}


def _match(rules: dict[str, str], subject: str) -> str:
    """Return the action for the last matching pattern (last-wins semantics)."""
    action = "ask"  # fallback if nothing matches
    for pattern, rule in rules.items():
        if fnmatch.fnmatch(subject, pattern):
            action = rule
    return action


def check_bash(command: str) -> str:
    """Return 'allow', 'deny', or 'ask' for a bash command."""
    return _match(BASH_RULES, command.strip())


def check_file(path: str) -> str:
    """Return 'allow', 'deny', or 'ask' for a file path."""
    try:
        resolved = str(Path(path).resolve())
    except Exception:
        resolved = path
    return _match(FILE_RULES, resolved)


def check_tool(name: str, args: dict) -> str:
    """Return 'allow', 'deny', or 'ask' for any tool call."""
    if name == "bash":
        return check_bash(args.get("command", ""))
    if name in ("read_file", "write_file", "str_replace"):
        return check_file(args.get("path", ""))
    # all other tools (write_todos, update_todo, read_skill, etc.) are safe
    return "allow"
