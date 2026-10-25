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
        "--log-driver=local",
        "--log-opt=max-size=1m",
        "--log-opt=max-file=1",
        "--log-opt=compress=false",
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
            "--detach",
            "--init",
            "python:3.12-slim",
            "timeout",
            "--signal=KILL",
            f"{timeout}s",
            "python",
            "-I",
            "-B",
            "-c",
            code,
        ]
    )
    try:
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
        result = subprocess.run(
            ["docker", "wait", name],
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout + 5,
        )
        exit_code = int(result.stdout.strip())
        output = subprocess.run(
            ["docker", "logs", "--tail", "64", name],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        state = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.OOMKilled}}", name],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        timed_out = exit_code in {124, 137} and state.stdout.strip() != "true"
        return ExecutionResult(
            success=exit_code == 0,
            stdout=output.stdout,
            stderr=output.stderr,
            error="Execution timed out"
            if timed_out
            else (output.stderr.strip().splitlines()[-1] if output.stderr.strip() else None),
            timeout=timed_out,
            memory_exceeded=state.stdout.strip() == "true",
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Evaluation container exceeded its startup or execution deadline") from None
    finally:
        subprocess.run(["docker", "rm", "--force", name], capture_output=True, check=False)
