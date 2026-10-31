import json
import ast
import os
import sys
import re
import time
import threading
import termios
import tty
import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

from llm import call_llm, MODEL
from tools import TOOLS, TOOL_SCHEMAS
from skills import skills_prompt
from context import build_context
from permissions import check_tool
from compaction import compact, needed as compaction_needed
from history import History, cap_args, set_spill_notify
import sandbox as _sandbox

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text
from rich.live import Live

console = Console()

# ── spill notification ────────────────────────────────────────────────────────

def _on_spill(path, size: int, label: str) -> None:
    """Fired every time history.py writes a log file."""
    kind = {
        "result":     "tool output",
        "write_file": "write_file content",
        "str_replace":"str_replace content",
        "task":       "subagent task log",
    }.get(label, label or "content")
    console.print(
        f"  [dim]↳ spilled {size:,} chars ({kind}) → "
        f"[bold]{path}[/bold][/dim]"
    )

set_spill_notify(_on_spill)

# ── config ────────────────────────────────────────────────────────────────────

SHOW_TOOL_RESULT = True

TOOL_ICONS = {
    "task":        "🤖",
    "bash":        "🐚",
    "read_file":   "📄",
    "list_dir":    "📁",
    "write_file":  "✏️",
    "str_replace": "🔍",
    "read_skill":  "🧠",
    "write_todos": "📋",
    "update_todo": "✅",
}

PLAYGROUND = (Path(__file__).parent / "playground").resolve()
PLAYGROUND.mkdir(exist_ok=True)

SYSTEM_PROMPT = f"""
You are a coding agent. Use the available tools to inspect, modify, and verify files to accomplish the user's task.
Answer back to the user once your work is done.

Your working directory is: {PLAYGROUND}
ALL file operations (read, write, list, bash) are confined to this directory.
You cannot see or touch any files outside of it — paths outside will be blocked.
Use relative paths (e.g. "index.html", "src/app.js") — they resolve to {PLAYGROUND}/<path>.
The harness source code lives outside this directory and is invisible to you.

You have skills available. Each one is a set of instructions for a task.
If a skill matches what the user wants, call read_skill first and follow it.

{skills_prompt()}

When you need to understand how something works — where a feature lives, how
data flows, what calls what — send a task subagent instead of grepping your
way there yourself. It explores in its own context window and hands you back
just the findings, so the search does not fill yours. It cannot see this
conversation, so write the question so it stands alone. Do all editing
yourself; the subagent only reads.

For independent exploration tasks, call task multiple times in the same
response — the calls run in parallel. For example, if you need to understand
three separate files or systems at once, issue three task calls together
rather than one after another. Each subagent works in isolation and only its
final answer comes back.

Long tool output is cut short, and the whole thing is written to a temp file
whose path is given at the cut. Page through it with head, tail, sed -n or
grep rather than asking for it again. That file only exists for the current
turn, so read it now or re-run the command later.

RULES:
- When your task is completed and verified, STOP calling tools immediately and output your final response to the user.
- Always prefer to ignore .venv, node_modules and other heavy library folders.
- For multi-step tasks, start by calling write_todos to lay out your plan.
- Update each todo with update_todo as you progress. When all todos are done, deliver your final report and stop.
- NEVER search from / or ~ — always use relative paths from the working directory.
- NEVER repeat a tool call that already returned an error for the same arguments — fix the arguments instead.
- When write_file fails with "No such file or directory", the parent dir is missing — use bash mkdir -p first, or just call write_file again (it now auto-creates parents).
- Do not call the same tool with the same args more than once without changing something.
- Do not loop: if a tool call returns the same result or content you already have, move forward to the next step or conclude.

CONTINUATION RULES (when you hit a turn limit or are asked to continue):
- DO NOT restart the task from the beginning.
- Treat the stopped execution as a checkpoint and continue from exactly where it stopped.
- Break big tasks in smaller todos, build one by one and validate.
- Preserve all existing context, tool results, decisions, files created/modified, errors, and completed work.
- Never repeat work that has already been successfully completed.
- Inspect the current project/filesystem state before making any changes.
- Identify exactly what remains unfinished, then execute only that.
- Do not create a new plan from scratch unless the previous plan is clearly invalid.
- Do not undo or recreate files that were already successfully written.
- If the previous tool call succeeded, treat it as completed even if the agent stopped immediately afterward.
- If the previous tool call failed or its result is uncertain, verify the current state before retrying.
- Continue until the original user request is fully completed, verified, or a genuine blocking error requires user input.
- If another turn limit is reached, emit a concise checkpoint with:
    1. Original goal
    2. Completed work
    3. Files changed
    4. Remaining work
    5. Current errors/blockers
    6. Exact next action
  Then stop and wait. The user will say "continue" and you resume from that checkpoint.
"""

# ── helpers ───────────────────────────────────────────────────────────────────

def parse_args(raw: str) -> dict:
    if not raw or not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        result = ast.literal_eval(raw)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    # Strip streaming artifacts or special tokens like <|...|>
    cleaned = re.sub(r"<\|.*?\|>", "", raw).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    try:
        obj, _ = json.JSONDecoder().raw_decode(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    raise ValueError(f"Could not parse tool arguments: {raw!r}")


# ── 3×3 orbiting dot loader ───────────────────────────────────────────────────

# Clockwise around a 2×2 grid:
#   0  1
#   3  2
_PATH = [0, 1, 2, 3]

# Unicode quadrant characters.
# Each character represents a 2×2 pixel/grid:
#   1 2
#   3 4
_QUADRANTS = {
    0: "▘",  # top-left
    1: "▝",  # top-right
    2: "▗",  # bottom-right
    3: "▖",  # bottom-left
}


def _make_loader(frame: int, label: str) -> Text:
    cur     = frame % len(_PATH)
    spinner = _QUADRANTS[_PATH[cur]]
    text    = Text()
    text.append(spinner, style="white")
    text.append(f"  {label}", style="dim white")
    return text


@contextmanager
def working(label: str = "Thinking…"):
    stop  = threading.Event()
    frame = [0]

    def _tick(live):
        while not stop.is_set():
            time.sleep(0.08)
            frame[0] += 1
            live.update(_make_loader(frame[0], label))

    with Live(
        _make_loader(0, label),
        console=console,
        transient=True,
        refresh_per_second=20,
    ) as live:
        thread = threading.Thread(target=_tick, args=(live,), daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=0.2)


# ── low-level key reading (arrow keys) ───────────────────────────────────────

def _read_key() -> str:
    """Read a single keypress, returning a string token."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            if ch2 == "[":
                if ch3 == "A": return "UP"
                if ch3 == "B": return "DOWN"
            return "ESC"
        if ch in ("\r", "\n"): return "ENTER"
        if ch == "\x03":        return "CTRL_C"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ── input prompt ─────────────────────────────────────────────────────────────

def bottom_input(hint_right: str = "") -> str:
    """Simple styled input line — no raw mode, no Live loop."""
    try:
        console.print(
            f"[bold bright_white]>[/bold bright_white] ",
            end="",
            highlight=False,
        )
        value = input()
        return value.strip()
    except (EOFError, KeyboardInterrupt):
        return ""


# ── arrow-navigable permission menu ──────────────────────────────────────────

def ask_permission(name: str, args: dict) -> tuple[bool, str | None]:
    """
    Arrow-key navigable permission menu.
    Uses ANSI cursor-up to redraw in-place — no Live, no staircase.
    """
    icon         = TOOL_ICONS.get(name, "🔧")
    args_preview = json.dumps(args, ensure_ascii=False)
    display_args = args_preview if len(args_preview) < 120 else args_preview[:120] + "…"

    choices = [
        ("Yes, run",                    True,  None),
        ("Yes, always allow this tool", True,  "__always__"),
        ("No, cancel",                  False, "User denied this tool call."),
        ("Send feedback to agent…",     None,  None),
    ]

    # Number of lines the menu occupies (redrawn in-place)
    MENU_LINES = 1 + len(choices) + 1   # header + choices + hint

    console.print(Panel(
        Text.assemble(
            (f"{icon} {name}  ", "bold yellow"),
            (display_args, "dim white"),
        ),
        title="[bold red]⚠  Permission required[/bold red]",
        border_style="red",
        padding=(0, 1),
    ))

    def _draw(sel: int, first: bool = False) -> None:
        if not first:
            # Move cursor up MENU_LINES and erase to end of screen
            sys.stdout.write(f"\033[{MENU_LINES}A\033[J")
        lines = ["\033[1mRun this tool?\033[0m"]
        for i, (label, _, _) in enumerate(choices):
            if i == sel:
                lines.append(f"\033[92m▶ {i+1}. {label}\033[0m")
            else:
                lines.append(f"\033[2m  {i+1}. {label}\033[0m")
        lines.append("\033[2m↑/↓ Navigate · Enter select · Esc cancel\033[0m")
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

    sel = 0
    _draw(sel, first=True)

    while True:
        key = _read_key()

        if key == "UP":
            sel = (sel - 1) % len(choices)
            _draw(sel)
        elif key == "DOWN":
            sel = (sel + 1) % len(choices)
            _draw(sel)
        elif key == "ENTER":
            # Erase menu before returning
            sys.stdout.write(f"\033[{MENU_LINES}A\033[J")
            sys.stdout.flush()
            _, proceed, extra = choices[sel]
            if proceed is None:
                break   # fall through to feedback input
            if extra == "__always__":
                return True, None
            return proceed, extra
        elif key in ("ESC", "CTRL_C"):
            sys.stdout.write(f"\033[{MENU_LINES}A\033[J")
            sys.stdout.flush()
            return False, "User cancelled."

    # Feedback input
    console.print("[dim]Enter feedback for the agent:[/dim]")
    try:
        fb = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        fb = ""
    return False, fb or "User cancelled."


# ── rendering ─────────────────────────────────────────────────────────────────

def render_tool_call(name: str, args: dict, result: str | None = None) -> None:
    icon    = TOOL_ICONS.get(name, "🔧")
    content = Text()
    content.append(f"{icon} {name} ", style="bold yellow")
    content.append(json.dumps(args, ensure_ascii=False)[:400] + "\n", style="dim white")

    if result is not None and SHOW_TOOL_RESULT:
        content.append("─" * 60 + "\n", style="dim")
        lines   = result.splitlines()
        preview = "\n".join(lines[:30])
        if len(lines) > 30:
            preview += f"\n… ({len(lines) - 30} more lines)"
        content.append(preview or "(empty)", style="white")

    console.print(Panel(content, border_style="dim white", padding=(0, 1)))


_last_injected_content: str | None = None

def render_late_injection(ctx: dict, force: bool = False) -> None:
    global _last_injected_content
    content = ctx.get("content", "")
    if not force and content == _last_injected_content:
        return
    _last_injected_content = content
    console.print(Panel(
        Text(content, style="dim white"),
        title="[dim]late injection[/dim]",
        border_style="dim",
        padding=(0, 1),
    ))


def render_usage(usage) -> None:
    prompt_details     = usage.prompt_tokens_details
    completion_details = usage.completion_tokens_details
    cached    = getattr(prompt_details,     "cached_tokens",    None) if prompt_details    else None
    reasoning = getattr(completion_details, "reasoning_tokens", None) if completion_details else None

    parts = [
        f"[yellow]{usage.prompt_tokens:,}[/yellow] [dim]prompt[/dim]",
        f"[yellow]{usage.completion_tokens:,}[/yellow] [dim]completion[/dim]",
    ]
    if cached:    parts.append(f"[yellow]{cached:,}[/yellow] [dim]cached[/dim]")
    if reasoning: parts.append(f"[yellow]{reasoning:,}[/yellow] [dim]reasoning[/dim]")
    console.print("  " + " [dim]·[/dim] ".join(parts))


def render_agent(content: str) -> None:
    console.print(Panel(
        Markdown(content),
        title="[bold magenta]🤖  Agent[/bold magenta]",
        border_style="magenta",
        padding=(1, 2),
    ))


# ── loop detection ────────────────────────────────────────────────────────────

class LoopDetector:
    """Tracks tool calls across turns to detect and prevent infinite loops."""
    def __init__(self, max_consecutive: int = 3, max_window: int = 8) -> None:
        self.history: list[tuple[str, str]] = []
        self.max_consecutive = max_consecutive
        self.max_window = max_window

    def normalize(self, name: str, args: dict | str) -> str:
        if isinstance(args, str):
            try:
                args = parse_args(args)
            except Exception:
                return args.strip()
        if isinstance(args, dict):
            try:
                return json.dumps(args, sort_keys=True, ensure_ascii=False)
            except Exception:
                return str(args)
        return str(args)

    def check(self, name: str, args: dict | str) -> tuple[str, str | None]:
        """
        Evaluate if this tool call represents a repetitive loop.
        Returns:
            ("ok", None) - proceed normally
            ("warn", msg) - warn the model, skip re-running identical tool
            ("halt", msg) - hard stop the entire agent loop
        """
        sig = (name, self.normalize(name, args))
        self.history.append(sig)

        # 1. Consecutive identical calls (across turns or within same turn)
        consecutive = 0
        for s in reversed(self.history):
            if s == sig:
                consecutive += 1
            else:
                break

        if consecutive >= self.max_consecutive:
            return "halt", (
                f"Loop limit exceeded: tool '{name}' called {consecutive} times consecutively "
                "with identical arguments. Stopping agent loop."
            )

        if consecutive == 2:
            return "warn", (
                f"Error: loop detected. You already called '{name}' with these exact arguments "
                "on the previous call and received that result. Do not retry identical calls. "
                "If your work is done, output your final response to the user. "
                "If an action failed, change your approach."
            )

        # 2. Oscillating / ping-pong loop, e.g. [A, B, A, B, A, B]
        if len(self.history) >= 6:
            recent = self.history[-6:]
            if recent[0] == recent[2] == recent[4] and recent[1] == recent[3] == recent[5]:
                return "halt", (
                    f"Oscillating loop detected between '{recent[0][0]}' and '{recent[1][0]}'. "
                    "Stopping agent loop to prevent infinite cycling."
                )

        # 3. Excessive repetitions in recent window (e.g. 4+ times in last 8 calls)
        window = self.history[-self.max_window:]
        same_count = sum(1 for s in window if s == sig)
        if same_count >= 4:
            return "halt", (
                f"Repetition limit exceeded: tool '{name}' called {same_count} times in the last "
                f"{len(window)} tool calls. Stopping agent loop."
            )

        return "ok", None


# ── agent loop ────────────────────────────────────────────────────────────────

def run_agent(user_input: str, hist: History) -> None:
    """Run the agentic loop against a History object. Mutates hist in place."""
    hist.append({"role": "user", "content": user_input})

    usage     = None
    MAX_TURNS = 20
    loop_detector  = LoopDetector()
    halted_by_loop = False

    global _last_injected_content
    _last_injected_content = None

    for turn in range(MAX_TURNS):
        injected_ctx = build_context()
        injected     = hist.inject(injected_ctx)
        render_late_injection(injected_ctx, force=(turn == 0))

        with working("Thinking…"):
            message, usage = call_llm(injected, tools=TOOL_SCHEMAS)

        hist.append(message.model_dump(exclude_none=True))

        if message.content:
            render_agent(message.content)

        # ── compaction check ──────────────────────────────────────────
        if compaction_needed(usage):
            console.print(Rule("Conversation compacted", style="dim yellow"))
            with working("Compacting…"):
                new_msgs, new_sys = compact(hist.messages, hist.system_prompt)
            hist.replace(new_msgs, new_sys)
            console.print(Rule("Context window compressed — continuing", style="dim yellow"))

        if not message.tool_calls:
            break

        # ── parallel task calls, sequential everything else ───────────────
        task_calls  = [tc for tc in message.tool_calls if tc.function.name == "task"]
        other_calls = [tc for tc in message.tool_calls if tc.function.name != "task"]

        results: dict[str, str] = {}

        # ── Check loop detector for task calls ──
        tasks_to_run = []
        for tc in task_calls:
            action, msg = loop_detector.check("task", tc.function.arguments)
            if action == "halt":
                results[tc.id] = msg
                console.print(f"[bold red]🛑 Halting: {msg}[/bold red]")
                halted_by_loop = True
            elif action == "warn":
                results[tc.id] = msg
            else:
                tasks_to_run.append(tc)

        # Deduplicate task calls by description — identical questions get one answer
        seen_descriptions: dict[str, str] = {}
        unique_tasks = []
        for tc in tasks_to_run:
            try:
                desc = parse_args(tc.function.arguments).get("description", "")
            except ValueError:
                desc = tc.function.arguments
            if desc in seen_descriptions:
                pass  # duplicate within the same batch; will copy result below
            else:
                seen_descriptions[desc] = tc.id
                unique_tasks.append(tc)

        # Run unique task calls in parallel
        if unique_tasks:
            with ThreadPoolExecutor(max_workers=len(unique_tasks)) as pool:
                futures = {}
                for tc in unique_tasks:
                    try:
                        args = parse_args(tc.function.arguments)
                    except ValueError as e:
                        results[tc.id] = str(e)
                        continue
                    futures[pool.submit(TOOLS["task"], **args)] = tc

                for future in as_completed(futures):
                    tc = futures[future]
                    try:
                        results[tc.id] = future.result()
                    except Exception as e:
                        results[tc.id] = f"Subagent error: {e}"

        # Copy results to duplicate task calls in the batch
        for tc in tasks_to_run:
            if tc.id not in results:
                try:
                    desc = parse_args(tc.function.arguments).get("description", "")
                except ValueError:
                    desc = tc.function.arguments
                primary_id = seen_descriptions.get(desc)
                if primary_id and primary_id in results:
                    results[tc.id] = results[primary_id]
                else:
                    results[tc.id] = "Duplicate subagent call."

        # Append task results for ALL task_calls in original order, render each
        for tc in task_calls:
            result = results.get(tc.id, "No result.")
            try:
                args = parse_args(tc.function.arguments)
            except ValueError:
                args = {}
            hist.append_tool_result(tc.id, result)
            render_tool_call(tc.function.name, cap_args(tc.function.name, args),
                             result if SHOW_TOOL_RESULT else None)

        if halted_by_loop:
            break

        # Run all other tool calls sequentially
        for tool_call in other_calls:
            # Sanitize tool name — strip any streaming artifacts like <|channel|>
            name = tool_call.function.name
            name = name.split("<|")[0].split("|>")[0].strip()

            try:
                args = parse_args(tool_call.function.arguments)
            except ValueError as e:
                hist.append_tool_result(tool_call.id, str(e))
                render_tool_call(name, {}, str(e) if SHOW_TOOL_RESULT else None)
                continue

            action = check_tool(name, args)

            if action == "deny":
                result = f"Permission denied: tool '{name}' is not allowed."
                hist.append_tool_result(tool_call.id, result)
                render_tool_call(name, args, result if SHOW_TOOL_RESULT else None)
                continue

            if action == "ask":
                proceed, feedback = ask_permission(name, args)
                if not proceed:
                    result = feedback or "User denied this tool call."
                    hist.append_tool_result(tool_call.id, result)
                    render_tool_call(name, args, result if SHOW_TOOL_RESULT else None)
                    continue

            # Loop detection
            l_action, l_msg = loop_detector.check(name, args)
            if l_action == "halt":
                hist.append_tool_result(tool_call.id, l_msg)
                render_tool_call(name, cap_args(name, args), l_msg if SHOW_TOOL_RESULT else None)
                console.print(f"[bold red]🛑 Halting: {l_msg}[/bold red]")
                halted_by_loop = True
                break

            if l_action == "warn":
                hist.append_tool_result(tool_call.id, l_msg)
                render_tool_call(name, cap_args(name, args), l_msg if SHOW_TOOL_RESULT else None)
                continue

            with working(f"{TOOL_ICONS.get(name, '🔧')} {name}…"):
                if name not in TOOLS:
                    result = f"Error: unknown tool '{name}'"
                else:
                    fn      = TOOLS[name]
                    valid   = inspect.signature(fn).parameters
                    cleaned = {k: v for k, v in args.items() if k in valid}
                    result  = fn(**cleaned)

            hist.append_tool_result(tool_call.id, result)
            # render compressed args so the panel doesn't flood the screen
            render_tool_call(name, cap_args(name, args), result if SHOW_TOOL_RESULT else None)

        if halted_by_loop:
            break

    else:
        if not halted_by_loop:
            console.print(f"[yellow]⚠ Reached {MAX_TURNS}-turn limit.[/yellow]")
            # Inject a system nudge so the model emits a checkpoint on the next user message
            hist.append({
                "role": "user",
                "content": (
                    "You have reached the turn limit. "
                    "Emit a concise checkpoint: original goal, completed work, "
                    "files changed, remaining work, blockers, and exact next action. "
                    "Do not restart — the user will say 'continue' to resume."
                ),
            })
            with working("Checkpointing…"):
                checkpoint_msg, _ = call_llm(hist.inject(build_context()), tools=None)
            hist.append(checkpoint_msg.model_dump(exclude_none=True))
            if checkpoint_msg.content:
                render_agent(checkpoint_msg.content)

    if usage:
        render_usage(usage)


# ── CLI ───────────────────────────────────────────────────────────────────────

LOGO = """\
[bold white]
  ██████╗ ██╗    ██╗██╗   ██╗██████╗  █████╗  ██████╗ ██████╗ ██████╗ ███████╗
 ██╔════╝ ██║    ██║╚██╗ ██╔╝██╔══██╗██╔══██╗██╔════╝██╔═══██╗██╔══██╗██╔════╝
 ╚█████╗  ██║ █╗ ██║ ╚████╔╝ ██████╔╝███████║██║     ██║   ██║██║  ██║█████╗
  ╚═══██╗ ██║███╗██║  ╚██╔╝  ██╔══██╗██╔══██║██║     ██║   ██║██║  ██║██╔══╝
 ██████╔╝ ╚███╔███╔╝   ██║   ██║  ██║██║  ██║╚██████╗╚██████╔╝██████╔╝███████╗
 ╚═════╝   ╚══╝╚══╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝
[/bold white]
[dim white]                   Your AI coding agent — powered by AWS Bedrock[/dim white]"""

HELP_TEXT = """\
[bold]Commands[/bold]
  [green]/exit[/green]       quit swyracode
  [green]/clear[/green]      clear message history (keep system prompt)
  [green]/history[/green]    show number of messages in context
  [green]/help[/green]       show this help

[bold]Permission menu[/bold]
  ↑/↓ navigate · Enter select · Esc cancel
  "Send feedback" opens a text input to redirect the agent.\
"""


def print_banner() -> None:
    console.print()
    console.print(Panel(LOGO, border_style="green", style="on dark_green", padding=(0, 2)))
    console.print(f"  [dim]sandbox: [bold]{_sandbox.name()}[/bold]  |  model: [bold]{MODEL}[/bold][/dim]\n")


if __name__ == "__main__":
    print_banner()

    hist = History(SYSTEM_PROMPT)

    while True:
        try:
            query = bottom_input(hint_right=f"{MODEL} · high")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not query:
            continue

        if query == "/exit":
            console.print("[dim]Goodbye.[/dim]")
            break

        if query == "/clear":
            hist.reset()
            from todos import clear_todos
            clear_todos()
            from context import SEEN
            SEEN.clear()
            console.print("[dim]History cleared.[/dim]")
            continue

        if query == "/history":
            console.print(f"[dim]{hist.total()} messages ({hist.user_turn_count()} user turns)[/dim]")
            continue

        if query == "/help":
            console.print(Panel(HELP_TEXT, border_style="dim", padding=(0, 2)))
            continue

        console.print(Rule(f"[dim]{query[:80]}{'…' if len(query) > 80 else ''}[/dim]", style="dim white"))
        run_agent(query, hist)
