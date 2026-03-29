"""
Sandboxed Python execution for evaluating LLM-generated code.

Adapted from nanochat/execution.py and OpenAI's human-eval execution.py.

Protections:
- Runs in a separate process (killable on hang/crash)
- Timeout enforced via SIGALRM
- Memory limit (256 MB, Linux only — skipped on macOS)
- Temporary working directory (deleted afterwards)
- Dangerous builtins and os/shutil/subprocess functions disabled via setattr

Not a true security sandbox: network access and dynamic Python features
(ctypes, etc.) are not blocked.
"""

from __future__ import annotations

import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import tempfile
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any


@dataclass
class ExecutionResult:
    """Result of executing Python code in the sandbox.

    Attributes:
        success: True if the code ran without raising an exception.
        stdout: Captured standard output.
        stderr: Captured standard error.
        error: Exception description if execution failed, else None.
        timeout: True if execution was killed due to a timeout.
        memory_exceeded: True if the memory limit was exceeded.
    """

    success: bool
    stdout: str
    stderr: str
    error: str | None = None
    timeout: bool = False
    memory_exceeded: bool = False


class _TimeoutException(Exception):
    pass


class _WriteOnlyStringIO(io.StringIO):
    """StringIO that raises OSError on any read operation."""

    def read(self, size: int = -1) -> str:  # type: ignore[override]
        raise OSError

    def readline(self, size: int | None = -1) -> str:  # type: ignore[override]
        raise OSError

    def readlines(self, hint: int = -1) -> list[str]:  # type: ignore[override]
        raise OSError

    def readable(self, *args: object, **kwargs: object) -> bool:
        return False


class _RedirectStdin(contextlib._RedirectStream):  # type: ignore[type-arg]
    _stream = "stdin"


@contextlib.contextmanager
def _time_limit(seconds: float) -> Generator[None, None, None]:
    def _handler(signum: int, frame: object) -> None:
        raise _TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, _handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def _capture_io() -> Generator[tuple[io.StringIO, io.StringIO], None, None]:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf):
        with contextlib.redirect_stderr(stderr_buf):
            with _RedirectStdin(_WriteOnlyStringIO()):
                yield stdout_buf, stderr_buf


@contextlib.contextmanager
def _tempdir() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as d:
        old = os.getcwd()
        os.chdir(d)
        try:
            yield d
        finally:
            os.chdir(old)


def _reliability_guard(maximum_memory_bytes: int | None = None) -> None:
    if platform.uname().system != "Darwin" and maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    setattr(builtins, "exit", None)  # noqa: B010
    setattr(builtins, "quit", None)  # noqa: B010

    os.environ["OMP_NUM_THREADS"] = "1"

    for attr in (
        "kill",
        "system",
        "putenv",
        "remove",
        "removedirs",
        "rmdir",
        "fchdir",
        "setuid",
        "fork",
        "forkpty",
        "killpg",
        "rename",
        "renames",
        "truncate",
        "replace",
        "unlink",
        "fchmod",
        "fchown",
        "chmod",
        "chown",
        "chroot",
        "chdir",
        "getcwd",
    ):
        if hasattr(os, attr):
            setattr(os, attr, None)

    import shutil

    for attr in ("rmtree", "move", "chown"):
        setattr(shutil, attr, None)

    import subprocess

    setattr(subprocess, "Popen", None)  # noqa: B010

    import sys

    for mod in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
        sys.modules[mod] = None  # type: ignore[assignment]


def _unsafe_execute(
    code: str,
    timeout: float,
    maximum_memory_bytes: int | None,
    result_dict: dict[str, object],
) -> None:
    """Run code in a subprocess; write results into result_dict."""
    import os as _os
    import shutil as _shutil

    rmtree = _shutil.rmtree
    rmdir = _os.rmdir
    chdir = _os.chdir
    unlink = _os.unlink

    with _tempdir():
        _reliability_guard(maximum_memory_bytes=maximum_memory_bytes)

        result_dict.update(
            {"success": False, "stdout": "", "stderr": "", "timeout": False, "memory_exceeded": False, "error": None}
        )

        try:
            exec_globals: dict[str, object] = {}
            with _capture_io() as (out, err):
                with _time_limit(timeout):
                    exec(code, exec_globals)  # noqa: S102
            result_dict.update({"success": True, "stdout": out.getvalue(), "stderr": err.getvalue()})

        except _TimeoutException:
            result_dict.update({"timeout": True, "error": "Execution timed out"})

        except MemoryError as exc:
            result_dict.update({"memory_exceeded": True, "error": f"Memory limit exceeded: {exc}"})

        except BaseException as exc:  # noqa: BLE001
            result_dict.update({"error": f"{type(exc).__name__}: {exc}"})

        _shutil.rmtree = rmtree
        _os.rmdir = rmdir
        _os.chdir = chdir
        _os.unlink = unlink


def execute_code(
    code: str,
    timeout: float = 5.0,
    maximum_memory_bytes: int | None = 256 * 1024 * 1024,
) -> ExecutionResult:
    """Execute Python code in a sandboxed subprocess.

    Args:
        code: Python source code to execute.
        timeout: Maximum wall-clock seconds before the process is killed.
        maximum_memory_bytes: RSS memory limit in bytes (None = unlimited;
            enforced on Linux only).

    Returns:
        ExecutionResult with success status, captured output, and error info.
    """
    manager = multiprocessing.Manager()
    result_dict: Any = manager.dict()

    p = multiprocessing.Process(target=_unsafe_execute, args=(code, timeout, maximum_memory_bytes, result_dict))
    p.start()
    p.join(timeout=timeout + 1)

    if p.is_alive():
        p.kill()
        return ExecutionResult(success=False, stdout="", stderr="", error="Process timed out", timeout=True)

    if not result_dict:
        return ExecutionResult(success=False, stdout="", stderr="", error="No result returned", timeout=True)

    return ExecutionResult(
        success=bool(result_dict["success"]),
        stdout=str(result_dict["stdout"]),
        stderr=str(result_dict["stderr"]),
        error=str(result_dict["error"]) if result_dict["error"] is not None else None,
        timeout=bool(result_dict["timeout"]),
        memory_exceeded=bool(result_dict["memory_exceeded"]),
    )
