"""
history.py — session message history + context compression.

Owns the messages list and the active system prompt for the session.
Also manages the /tmp log directory lifecycle:
  - created once at import time
  - tool results AND large write args that exceed CAP are spilled to log files
  - the whole directory is deleted automatically on cleanup()

KV-cache note
─────────────
Tool results are the biggest context killers. By capping them at CAP chars
and spilling the rest to disk, we keep the token count low without losing
the information — the agent can always head/tail/grep the log file.

For write_file / str_replace we go further: the content arg itself is
never stored in history. We log it to disk and store only a one-liner
"wrote N bytes to <path> — full content at <log>" so the model knows
the write succeeded without eating the whole file worth of tokens.
"""

import atexit
import shutil
import tempfile
from pathlib import Path

# ── log-spill directory ───────────────────────────────────────────────────────

_LOG_DIR:     Path = Path(tempfile.mkdtemp(prefix="swyracode-"))
_log_counter: int  = 0

# Tools whose large content args we compress rather than echo into history
_WRITE_TOOLS = {"write_file", "str_replace"}

# Content arg names per tool that hold the bulk payload
_CONTENT_ARGS = {
    "write_file": "content",
    "str_replace": "new_str",
}


def _cleanup() -> None:
    if _LOG_DIR.exists():
        shutil.rmtree(_LOG_DIR, ignore_errors=True)


atexit.register(_cleanup)


def log_dir() -> Path:
    return _LOG_DIR


# Optional callback fired every time a log file is written.
# main.py sets this to a rich console.print so the user sees the spill.
_spill_notify = None

def set_spill_notify(fn) -> None:
    global _spill_notify
    _spill_notify = fn


# ── spill / cap ───────────────────────────────────────────────────────────────

CAP     = 2_000   # chars kept inline; rest spills to a log file
TRIMMED = "[TRIMMED]"


def spill(text: str, label: str = "") -> Path:
    """Write text to a numbered log file and return its path."""
    global _log_counter
    _log_counter += 1
    suffix = f"-{label}" if label else ""
    path   = _LOG_DIR / f"{_log_counter:04d}{suffix}-log.txt"
    path.write_text(text, encoding="utf-8")
    if _spill_notify:
        _spill_notify(path, len(text), label)
    return path


def cap(text: str) -> str:
    """
    Trim a tool *result*, leaving a pointer to the full output.

    Short results pass through unchanged.
    Long results: first CAP chars stay inline, rest goes to a log file.
    """
    if len(text) <= CAP:
        return text
    path = spill(text, "result")
    return (
        text[:CAP]
        + f"\n\n{TRIMMED} {len(text) - CAP} of {len(text)} chars cut. "
        + f"Full output at {path} — page with head/tail/sed/grep. "
        + "Deleted when this session ends."
    )


def cap_args(tool_name: str, args: dict) -> dict:
    """
    For write_file / str_replace, replace the large content arg with a
    log-file reference so it never enters the message history.

    Returns a new args dict safe to store; the original args (with full
    content) are still used for the actual tool call.
    """
    if tool_name not in _WRITE_TOOLS:
        return args

    content_key = _CONTENT_ARGS.get(tool_name)
    if not content_key or content_key not in args:
        return args

    content = args[content_key]
    if len(content) <= CAP:
        return args   # small enough, pass through

    path     = spill(content, tool_name)
    new_args = dict(args)
    new_args[content_key] = (
        f"[content spilled — {len(content)} chars written to {path}; "
        "file already written to disk, not stored in context]"
    )
    return new_args


# ── message history ───────────────────────────────────────────────────────────

class History:
    """Holds the messages list and live system prompt for one session."""

    def __init__(self, system_prompt: str) -> None:
        self._system_prompt = system_prompt
        self._messages: list = [{"role": "system", "content": system_prompt}]

    # ── read ──────────────────────────────────────────────────────────────────

    @property
    def messages(self) -> list:
        return self._messages

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def user_turn_count(self) -> int:
        return sum(1 for m in self._messages if m.get("role") == "user")

    def total(self) -> int:
        return len(self._messages)

    # ── write ─────────────────────────────────────────────────────────────────

    def append(self, message: dict) -> None:
        """
        Append any message. For assistant messages that carry tool_calls,
        compress large content args before storing.
        """
        if message.get("role") == "assistant" and message.get("tool_calls"):
            import json, copy
            compressed = copy.deepcopy(message)
            for tc in compressed.get("tool_calls", []):
                fn   = tc.get("function", {})
                name = fn.get("name", "")
                try:
                    raw_args  = json.loads(fn.get("arguments", "{}"))
                    slim_args = cap_args(name, raw_args)
                    fn["arguments"] = json.dumps(slim_args)
                except Exception:
                    pass
            self._messages.append(compressed)
        else:
            self._messages.append(message)

    def append_tool_result(self, tool_call_id: str, result: str) -> None:
        """Append a tool result, capping it before it enters the history."""
        self._messages.append({
            "role":         "tool",
            "tool_call_id": tool_call_id,
            "content":      cap(result),
        })

    def reset(self) -> None:
        self._messages = [{"role": "system", "content": self._system_prompt}]

    def update_system_prompt(self, new_prompt: str) -> None:
        self._system_prompt = new_prompt
        if self._messages and self._messages[0].get("role") == "system":
            self._messages[0]["content"] = new_prompt
        else:
            self._messages.insert(0, {"role": "system", "content": new_prompt})

    def inject(self, extra: dict) -> list:
        """Return messages + a late-injected message without storing it."""
        return self._messages + [extra]

    # ── compaction bridge ─────────────────────────────────────────────────────

    def replace(self, new_messages: list, new_system_prompt: str) -> None:
        self._messages      = new_messages
        self._system_prompt = new_system_prompt
