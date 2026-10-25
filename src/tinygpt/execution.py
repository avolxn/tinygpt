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
        "--pull=never",
        "--name",
        name,
        "--network=none",
        "--read-only",
        "--log-driver=none",
        "--ulimit=fsize=1048576:1048576",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=32",
        "--cpus=1",
        "--user=65534:65534",
        "--tmpfs=/tmp:rw,noexec,nosuid,size=16m",
    ]
    if maximum_memory_bytes is not None:
        command.extend(["--memory", str(maximum_memory_bytes), "--memory-swap", str(maximum_memory_bytes)])
    command.extend(
        [
            "--init",
            "-i",
            "python:3.12-slim",
            "sh",
            "-c",
            'timeout --signal=KILL "$1" python -I -B - > /tmp/stdout 2> /tmp/stderr; '
            'status=$?; cat /tmp/stdout; cat /tmp/stderr >&2; exit "$status"',
            "evaluation",
            f"{timeout}s",
        ]
    )
    try:
        result = subprocess.run(command, input=code, capture_output=True, text=True, timeout=timeout + 30)
        if result.returncode in {125, 126, 127}:
            raise RuntimeError(f"Evaluation container failed: {result.stderr.strip()}")
        state = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.OOMKilled}}", name],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        timed_out = result.returncode in {124, 137} and state.stdout.strip() != "true"
        return ExecutionResult(
            success=result.returncode == 0,
            stdout=result.stdout,
            stderr=result.stderr,
            error="Execution timed out"
            if timed_out
            else (result.stderr.strip().splitlines()[-1] if result.stderr.strip() else None),
            timeout=timed_out,
            memory_exceeded=state.stdout.strip() == "true",
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Evaluation container exceeded its startup or execution deadline") from None
    finally:
        subprocess.run(["docker", "rm", "--force", name], capture_output=True, check=False)
