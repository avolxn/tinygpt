"""Execute evaluation programs in disposable Docker containers."""

from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    error: str | None = None
    timeout: bool = False
    memory_exceeded: bool = False


def execute_code(
    code: str,
    timeout: float = 5.0,
    maximum_memory_bytes: int | None = 256 * 1024 * 1024,
) -> ExecutionResult:
    """Run Python without host mounts, network, capabilities, or host credentials."""
    if timeout <= 0 or (maximum_memory_bytes is not None and maximum_memory_bytes <= 0):
        raise ValueError("Execution timeout and memory limit must be positive")
    name = f"tinygpt-eval-{uuid.uuid4().hex}"
    command = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--name",
        name,
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=32",
        "--cpus=1",
        "--user=65534:65534",
        "--tmpfs=/tmp:rw,noexec,nosuid,size=16m",
    ]
    if maximum_memory_bytes is not None:
        command.extend(["--memory", str(maximum_memory_bytes), "--memory-swap", str(maximum_memory_bytes)])
    command.extend(["-i", "python:3.12-slim", "python", "-I", "-B", "-"])
    try:
        result = subprocess.run(command, input=code, capture_output=True, text=True, timeout=timeout)
        if result.returncode in {125, 126, 127}:
            raise RuntimeError(f"Evaluation container failed: {result.stderr.strip()}")
        return ExecutionResult(
            success=result.returncode == 0,
            stdout=result.stdout,
            stderr=result.stderr,
            error=result.stderr.strip().splitlines()[-1] if result.stderr.strip() else None,
            memory_exceeded=result.returncode == 137,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(False, "", "", error="Execution timed out", timeout=True)
    finally:
        subprocess.run(["docker", "rm", "--force", name], capture_output=True, check=False)
