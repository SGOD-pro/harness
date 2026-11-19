from rich.console import Console
from rich.theme import Theme
from rich.rule import Rule
from rich.text import Text
AGENT_THEME = Theme(
    {
        # General
        "info": "cyan",
        "warning": "yellow",
        "error": "bright_red bold",
        "success": "green",
        "dim": "dim",
        "muted": "grey50",
        "border": "grey35",
        "highlight": "bold cyan",
        # Roles
        "user": "bright_blue bold",
        "assistant": "bright_white",
        "system": "dark_orange bold",
        # Tools
        "tool": "bright_magenta bold",
        "tool.read": "cyan",
        "tool.write": "yellow",
        "tool.shell": "magenta",
        "tool.search": "bright_cyan",
        "tool.network": "bright_blue",
        "tool.memory": "green",
        "tool.mcp": "bright_cyan",
        "tool.result": "green",
        "tool.error": "bright_red",
        # Events / lifecycle
        "event.start": "bold green",
        "event.stop": "bold red",
        "event.pending": "yellow",
        "event.done": "bright_green bold",
        # Metadata
        "token": "grey50",
        "model": "bright_yellow",
        "timestamp": "grey42",
        "step": "bold white",
        "code":"white"
    }
)

_console =None

def get_console() -> Console:

    global _console
    if _console is None:
        _console = Console(theme=AGENT_THEME,highlight=False)

    return _console


class TUI:
    def __init__(self,console:Console) -> None:
        self.console=console or get_console()
        self._assistant_stream_open= False
        
    def begin_assist(self) -> None:
        self.console.print()
        self.console.print(Rule(Text("Assistant",style="assistant")))

    def stream_assistant_delta(self, content:str) -> None:
        self.console.print(content,end="",markup=False)

    def end_assist(self) -> None:
        if self._assistant_stream_open:
            self.console.print()
        self._assistant_stream_open=False
        
