"""
OS-level sandbox wrapper for bash commands.

Tries to wrap commands in the tightest available sandbox:
  - Linux + bwrap (bubblewrap): network-isolated, project-dir only write access
  - No sandbox available:       runs the command unwrapped

Returns a list of argv to prepend to the shell invocation, or None if no
sandbox is available on this platform.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT = Path.cwd().resolve()

# macOS seatbelt profile (sandbox-exec)
_SEATBELT_PROFILE = f"""(version 1)
(deny default)
(allow process-exec process-fork signal)
(allow file-read*)
(allow sysctl-read)
(deny network*)
(allow file-write* (subpath "{PROJECT}") (literal "/dev/null"))
(deny file-write* (subpath "{PROJECT}/.git"))
"""


def wrap(command: str) -> list[str] | None:
    """
    Return an argv prefix to sandbox the command, or None if unavailable.

    Usage:
        prefix = wrap(cmd)
        argv = (prefix or []) + ["/bin/sh", "-c", cmd]
        subprocess.run(argv, ...)
    """
    if sys.platform == "darwin":
        profile = Path(tempfile.gettempdir()) / "swyracode.sb"
        profile.write_text(_SEATBELT_PROFILE)
        return ["sandbox-exec", "-f", str(profile), "/bin/sh", "-c", command]

    if sys.platform.startswith("linux") and shutil.which("bwrap"):
        return [
            "bwrap",
            "--ro-bind", "/", "/",
            "--bind",    str(PROJECT), str(PROJECT),
            "--dev",     "/dev",
            "--proc",    "/proc",
            "--unshare-net",
            "--die-with-parent",
            "/bin/sh", "-c", command,
        ]

    return None  # Windows or Linux without bubblewrap — no sandbox


def name() -> str:
    """Return a human-readable name for the active sandbox."""
    if sys.platform == "darwin":
        return "seatbelt"
    if sys.platform.startswith("linux") and shutil.which("bwrap"):
        return "bubblewrap"
    return "none"


def run(command: str, **kwargs) -> subprocess.CompletedProcess:
    """
    Run a shell command inside the sandbox (or bare if no sandbox available).
    Passes **kwargs straight to subprocess.run.
    """
    prefix = wrap(command)
    if prefix:
        argv = prefix  # wrap() already includes /bin/sh -c command
    else:
        argv = ["/bin/sh", "-c", command]

    return subprocess.run(argv, **kwargs)
