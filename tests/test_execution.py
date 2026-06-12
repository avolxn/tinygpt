from tinygpt.execution import execute_code


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
