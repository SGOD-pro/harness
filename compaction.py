"""
Compaction — summarise old conversation turns into a handoff note.

KV-cache strategy
─────────────────
The summary is injected at the START of the system prompt, so every
token before it is identical every turn → the cache never misses.

  [system: SUMMARY + original prompt]   ← stable prefix, always cached
  [user]
  [assistant + tool_calls]
  [tool results]          ← stripped to one-liners after compaction
  [user] ...

We compact when prompt_tokens > CONTEXT_WINDOW * COMPACT_AT.
We strip tool results down to a single line so they stop eating tokens.
We never throw away the most recent N exchanges — the agent needs live
context to keep working.
"""

from llm import call_llm

# ── thresholds (tune to your model's context window) ─────────────────────────

CONTEXT_WINDOW = 32_000   # tokens the model supports
COMPACT_AT     = 0.75     # compact when we hit 75 % of the window
KEEP_RECENT    = 6        # number of recent message pairs to leave untouched

# ── role labels used inside the transcript sent to the summariser ─────────────

ROLES = {"user": "USER", "assistant": "ASSISTANT", "tool": "TOOL RESULT"}

# ── summariser system prompt ──────────────────────────────────────────────────

_SUMMARISE_PROMPT = """\
You are compacting the transcript of a coding session. The session is out of
context window. Write the handoff note that lets a fresh agent pick the work up
without re-reading anything.

Use these sections, in this order. Skip any that would be empty.

## Goal
What the user asked for. Quote them where the exact wording matters.

## What happened
Decisions taken and the reasoning behind them. Include approaches that were
tried and abandoned, and why — those are the expensive lessons, and an agent
without them will try the same dead end again.

## Files
Every file touched: path, and what changed in it.

## State
What works, what is broken, what was left half-finished.

## Next
The immediate next step.

Rules:
- Be specific. Real paths, function names, error text, exact commands.
- Keep anything the user explicitly asked for, corrected, or rejected.
- Never invent progress. If something was not finished, say it was not.
- No preamble and no sign-off. Start at the first heading.
"""

# ── handoff wrapper injected into the system message ─────────────────────────

_HANDOFF = """\
<summary>
Everything before this point has been compacted out of the context window to
free up room. This is the record of it — treat it as your own memory of the
work so far, not as something the user told you.

{summary}
</summary>"""


# ── helpers ───────────────────────────────────────────────────────────────────

def needed(usage) -> bool:
    """Return True when the last request grew past the compaction threshold."""
    if usage is None:
        return False
    return usage.prompt_tokens > CONTEXT_WINDOW * COMPACT_AT


def _strip_tool_results(messages: list) -> list:
    """
    Reduce tool-result messages to a single summary line.
    Keeps the role/tool_call_id so the thread stays valid.
    """
    out = []
    for m in messages:
        if m.get("role") == "tool":
            content = m.get("content", "")
            lines   = content.splitlines()
            short   = lines[0][:120] if lines else "(empty)"
            if len(lines) > 1:
                short += f"  … ({len(lines)-1} more lines)"
            out.append({**m, "content": short})
        else:
            out.append(m)
    return out


def _transcript(messages: list) -> str:
    """Render messages as a plain transcript for the summariser."""
    parts = []
    for m in messages:
        role  = ROLES.get(m.get("role", ""), m.get("role", "").upper())
        content = m.get("content") or ""
        # include tool_calls as readable text
        if m.get("tool_calls"):
            import json
            for tc in m["tool_calls"]:
                fn   = tc.get("function", {})
                args = fn.get("arguments", "")
                try:
                    args = json.dumps(json.loads(args), ensure_ascii=False)
                except Exception:
                    pass
                content += f"\n[tool_call] {fn.get('name','?')}({args[:200]})"
        if content.strip():
            parts.append(f"### {role}\n{content.strip()}")
    return "\n\n".join(parts)


# ── main entry point ──────────────────────────────────────────────────────────

def compact(messages: list, system_prompt: str) -> tuple[list, str]:
    """
    Summarise old turns, return a rebuilt (messages, new_system_prompt) pair.

    The new system prompt has the summary prepended so it forms a stable
    cached prefix for every subsequent request.

    Strategy:
    1. Split messages into [system] + history.
    2. Keep the most recent KEEP_RECENT user/assistant pairs untouched.
    3. Send everything older to the summariser model.
    4. Strip tool results in the kept tail to save tokens.
    5. Prepend the summary to the system prompt.
    6. Return [new system msg] + stripped tail.
    """
    # Separate system message from the rest
    history = [m for m in messages if m.get("role") != "system"]

    # Find the split point: keep the last KEEP_RECENT *user* turns
    user_indices = [i for i, m in enumerate(history) if m.get("role") == "user"]
    if len(user_indices) <= KEEP_RECENT:
        # Not enough history to compact — nothing to do
        return messages, system_prompt

    split_at  = user_indices[-KEEP_RECENT]
    old_turns = history[:split_at]
    new_turns = history[split_at:]

    # Summarise the old turns
    transcript = _transcript(old_turns)
    summarise_messages = [
        {"role": "system", "content": _SUMMARISE_PROMPT},
        {"role": "user",   "content": transcript},
    ]
    summary_msg, _ = call_llm(summarise_messages)
    summary        = (summary_msg.content or "").strip()

    # Build updated system prompt with summary at the top
    new_system = _HANDOFF.format(summary=summary) + "\n\n" + system_prompt

    # Strip tool results in the kept tail to save tokens
    compact_tail = _strip_tool_results(new_turns)

    new_messages = [
        {"role": "system", "content": new_system},
        *compact_tail,
    ]

    return new_messages, new_system
