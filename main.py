import json
import ast
import os
import sys
import time
import threading
import termios
import tty
import inspect
from contextlib import contextmanager

from llm import call_llm, MODEL
from tools import TOOLS, TOOL_SCHEMAS
from skills import skills_prompt
from context import build_context
from permissions import check_tool
from compaction import compact, needed as compaction_needed
from history import History, cap_args
import sandbox as _sandbox

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text
from rich.live import Live

console = Console()

# ── config ────────────────────────────────────────────────────────────────────

SHOW_TOOL_RESULT = True

TOOL_ICONS = {
    "bash":        "🐚",
    "read_file":   "📄",
    "write_file":  "✏️",
    "str_replace": "🔍",
    "read_skill":  "🧠",
    "write_todos": "📋",
    "update_todo": "✅",
}

SYSTEM_PROMPT = f"""
You are a coding agent. Your job is to code. Always code.
Use the available tools to inspect and modify files.
Answer back to the user once exploration is done.

Your current working directory is: {os.getcwd()}

You have skills available. Each one is a set of instructions for a task.
If a skill matches what the user wants, call read_skill first and follow it.

{skills_prompt()}

RULES:
- Always prefer to ignore .venv, node_modules and other heavy library folders.
- For multi-step tasks, start by calling write_todos to lay out your plan.
- Update each todo with update_todo as you progress.

CONTINUATION RULES (when you hit a turn limit or are asked to continue):
- DO NOT restart the task from the beginning.
- Treat the stopped execution as a checkpoint and continue from exactly where it stopped.
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
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            result = ast.literal_eval(raw)
            if isinstance(result, dict):
                return result
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


def render_late_injection(ctx: dict) -> None:
    console.print(Panel(
        Text(ctx["content"], style="dim white"),
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


# ── agent loop ────────────────────────────────────────────────────────────────

def run_agent(user_input: str, hist: History) -> None:
    """Run the agentic loop against a History object. Mutates hist in place."""
    hist.append({"role": "user", "content": user_input})

    usage     = None
    MAX_TURNS = 20

    for turn in range(MAX_TURNS):
        injected_ctx = build_context()
        injected     = hist.inject(injected_ctx)
        render_late_injection(injected_ctx)

        with working("Thinking…"):
            message, usage = call_llm(injected, tools=TOOL_SCHEMAS)

        hist.append(message.model_dump(exclude_none=True))

        if message.content:
            render_agent(message.content)

        # ── compaction check ──────────────────────────────────────────
        if compaction_needed(usage):
            console.print("[dim]⚙ Context filling — compacting…[/dim]")
            with working("Compacting…"):
                new_msgs, new_sys = compact(hist.messages, hist.system_prompt)
            hist.replace(new_msgs, new_sys)
            console.print("[dim]✓ Compacted.[/dim]")

        if not message.tool_calls:
            break

        for tool_call in message.tool_calls:
            name = tool_call.function.name

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

    else:
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
            checkpoint_msg, _ = call_llm(hist.inject(build_context()), tools=TOOL_SCHEMAS)
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
