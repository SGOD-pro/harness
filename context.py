"""
Late injection: a small block appended just before we send.

It goes at the END of the message list so the stable prefix in front of it
stays cached.
"""

import os
import subprocess
from datetime import datetime
from todos import todos_prompt

# path -> mtime when the agent last read it
SEEN: dict[str, float] = {}


def note_read(path: str) -> None:
    """Record the mtime of a file when the agent reads it."""
    try:
        SEEN[path] = os.path.getmtime(path)
    except OSError:
        pass


def stale_files() -> list[str]:
    """Return paths the agent has read that have been modified since."""
    stale = []
    for p, mtime in list(SEEN.items()):
        try:
            if os.path.exists(p) and os.path.getmtime(p) != mtime:
                stale.append(p)
        except OSError:
            pass
    return stale


def stale_note() -> str:
    """Warn about files that changed on disk since the agent read them."""
    changed = stale_files()
    if not changed:
        return ""
    return (
        "\n<system-reminder>\n"
        "These files changed on disk since you read them. Read them again "
        "before editing:\n" + "\n".join(changed) + "\n</system-reminder>"
    )


def git_branch() -> str:
    result = subprocess.run(
        "git branch --show-current", shell=True, capture_output=True, text=True
    )
    return result.stdout.strip() or "(detached)"


def git_status() -> str:
    result = subprocess.run(
        ["git", "status", "--short", "--branch"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "not a git repository"
    return result.stdout.strip()


def todos_note() -> str:
    """Render the current todo list as a <todos> block, or empty string if none."""
    plan = todos_prompt()
    return f"\n<todos>\n{plan}\n</todos>" if plan else ""


def build_context() -> dict:
    """The block we append to the messages list on every turn."""
    from tools import PLAYGROUND_DIR
    stale = stale_files()
    stale_section = (
        "files changed since last read:\n" + "\n".join(f"  - {p}" for p in stale)
        if stale
        else "no files changed since last read"
    )

    return {
        "role": "system",
        "content": (
            "<env>\n"
            f"time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"branch: {git_branch()}\n"
            f"working dir: {PLAYGROUND_DIR}\n"
            f"git status:\n{git_status()}\n"
            f"{stale_section}\n"
            "</env>"
            f"{todos_note()}"
            f"{stale_note()}"
        ),
    }
