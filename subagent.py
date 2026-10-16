"""
subagent.py — lightweight exploration subagent.

Rules (structural, not just prompt-level):
  1. Fresh context every call — the main agent's conversation never goes in.
  2. Read-only toolset — write_file, str_replace, write_todos, task are
     withheld, so a subagent cannot edit files or spawn its own subagents.
  3. Only the final answer comes back — the whole tool call log stays here.
  4. Under 150 words in the final reply — findings only, no preamble.
"""

from llm import call_llm

# ── tools the subagent must NOT have ─────────────────────────────────────────
# task       → prevents recursive subagent spawning
# write_file → read-only
# str_replace→ read-only
# write_todos→ belongs to the main agent's plan, not a guest session

WITHHELD = {"task", "write_file", "str_replace", "write_todos", "update_todo"}


def _toolset() -> list:
    """All TOOL_SCHEMAS minus the withheld ones."""
    # Import here, not at module top — tools imports subagent (via tools.py
    # registering 'task'), which would create a circular import.
    from tools import TOOL_SCHEMAS
    return [
        s for s in TOOL_SCHEMAS
        if s["function"]["name"] not in WITHHELD
    ]


_SYSTEM_PROMPT = f"""\
You are an exploration subagent. You were given one question by a lead agent
and you answer it. That is the whole job.

You cannot see the conversation that spawned you, and the lead agent cannot
see anything you do here. Only your final message crosses back, so it has to
stand on its own.

You are working in the playground directory. Search only inside it.
Never search from / or ~ — that scans the whole machine and will time out.

Do all your work with the tools. Do not edit files — you have no write tools.
When you have found what you need, write your final answer and stop.

Your final message is the entire report, and it is the only thing that costs
the lead agent anything — so keep it short. Aim for under 150 words. Findings
only: file paths with line numbers, names, values. No preamble, no restating
the question, no long code blocks — cite the path and line and let the lead
agent open it. Say plainly what you could not find; a gap is useful, a guess
is not.
"""

MAX_TURNS = 10   # subagents get fewer turns than the main agent


def task(description: str) -> str:
    """
    Run a fresh read-only subagent on one question.
    Returns only its final text answer — nothing else crosses back.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": description},
    ]

    toolset = _toolset()
    report  = None   # newest thing it has said, kept if we run out of turns

    # Spill the full subagent transcript to a log file for debugging
    from history import spill as _spill

    call_history = []
    halt_subagent = False

    for _ in range(MAX_TURNS):
        message, usage = call_llm(messages, tools=toolset)
        messages.append(message.model_dump(exclude_none=True))
        report = message.content or report   # keep last non-empty text

        # No tool calls → subagent has stopped exploring and written its answer
        if not message.tool_calls:
            break

        import json, inspect
        from tools import TOOLS

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            raw_args = tool_call.function.arguments
            call_sig = (name, raw_args.strip())
            call_history.append(call_sig)

            # Check consecutive repetitions
            consecutive = 0
            for s in reversed(call_history):
                if s == call_sig:
                    consecutive += 1
                else:
                    break

            if consecutive >= 3:
                result = f"Error: loop limit reached for '{name}'. Halting subagent exploration."
                halt_subagent = True
            elif consecutive == 2:
                result = f"Error: you already called '{name}' with these arguments. Try a different query or summarize findings."
            elif name in WITHHELD:
                result = f"Tool '{name}' is not available to subagents."
            elif name not in TOOLS:
                result = f"Unknown tool '{name}'."
            else:
                try:
                    args    = json.loads(raw_args)
                    fn      = TOOLS[name]
                    valid   = inspect.signature(fn).parameters
                    cleaned = {k: v for k, v in args.items() if k in valid}
                    result  = fn(**cleaned)
                except Exception as e:
                    result = f"Error: {e}"

            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,
                "content":      str(result)[:4000],   # cap subagent tool results too
            })

            if halt_subagent:
                break

        if halt_subagent:
            break

    # Write full transcript to a task log for debugging
    import json as _json
    transcript = "\n\n".join(
        f"[{m.get('role','?').upper()}]\n{m.get('content') or _json.dumps(m.get('tool_calls',''), ensure_ascii=False)[:500]}"
        for m in messages[1:]   # skip system prompt
    )
    _spill(transcript, "task")

    return report or "Subagent produced no output."
