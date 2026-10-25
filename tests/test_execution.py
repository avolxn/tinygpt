import subprocess

import pytest

from tinygpt.execution import execute_code


@pytest.fixture(autouse=True)
def require_docker() -> None:
    try:
        result = subprocess.run(["docker", "image", "inspect", "python:3.12-slim"], capture_output=True)
    except FileNotFoundError:
        pytest.skip("Docker is required for isolated execution tests")
    if result.returncode:
        pytest.skip("Docker daemon and python:3.12-slim are required")


def test_execute_code_captures_stdout() -> None:
    result = execute_code("print(2 + 2)", timeout=1.0)

    assert result.success
    assert result.stdout.strip() == "4"


def test_execute_code_reports_exception() -> None:
    result = execute_code('raise ValueError("x")', timeout=1.0)

    assert not result.success
    assert result.error == "ValueError: x"


def test_execute_code_times_out() -> None:
    result = execute_code("while True: pass", timeout=0.2)

    assert not result.success
    assert result.timeout


def test_execute_code_bounds_output() -> None:
    result = execute_code('import os\nwhile True: os.write(1, b"x" * 65536)')
    assert not result.success
    assert len(result.stdout.encode()) <= 2 * 1048576
