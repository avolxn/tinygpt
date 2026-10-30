"""Static checks for the Airflow DAG without adding Airflow to the app environment."""

import ast
import os
import subprocess
from pathlib import Path

DAG_PATH = Path(__file__).parents[1] / "infra" / "airflow" / "dags" / "tinygpt_pipeline.py"
EXPECTED_TASKS = {
    "prepare_teacher",
    "pretrain",
    "evaluate_base",
    "distill",
    "evaluate_chat",
    "resume_pretrain",
    "resume_distill",
}


def parse_dag() -> ast.Module:
    return ast.parse(DAG_PATH.read_text(encoding="utf-8"), filename=str(DAG_PATH))


def test_airflow_dag_is_valid_python() -> None:
    compile(parse_dag(), str(DAG_PATH), "exec")


def test_airflow_dag_has_complete_training_pipeline() -> None:
    tree = parse_dag()
    task_ids = {
        call.args[0].value
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "bash_task"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
    }

    assert task_ids == EXPECTED_TASKS


def test_user_params_are_passed_through_environment() -> None:
    tree = parse_dag()
    bash_commands = [
        call.args[1].value
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "bash_task"
        and len(call.args) > 1
        and isinstance(call.args[1], ast.Constant)
        and isinstance(call.args[1].value, str)
    ]

    assert bash_commands
    assert all("{{ params." not in command for command in bash_commands)


def test_bash_tasks_inherit_secrets_without_xcom() -> None:
    tree = parse_dag()
    operator_call = next(
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "BashOperator"
    )
    keywords = {keyword.arg: keyword.value for keyword in operator_call.keywords}

    assert isinstance(keywords["append_env"], ast.Constant)
    assert keywords["append_env"].value is True
    assert isinstance(keywords["do_xcom_push"], ast.Constant)
    assert keywords["do_xcom_push"].value is False


def test_remote_transport_quotes_inputs_and_propagates_failure(tmp_path: Path) -> None:
    """Run the transmitted script in a clean shell, as a remote SSH server does."""
    ssh = tmp_path / "ssh"
    ssh.write_text('#!/bin/bash\n[[ " $* " == *" -n "* ]] && exit 0\nexec env -i HOME="$HOME" PATH="$PATH" bash -se\n')
    ssh.chmod(0o755)
    tree = parse_dag()
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and node.targets[0].id == "COMMON_ENV")
    env = {ast.literal_eval(key): "" for key in assignment.value.keys}
    literal = "example'; touch INJECTED; # $(touch INJECTED)"
    env.update(
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        HOME=str(tmp_path),
        REMOTE_HOST="root@worker",
        REMOTE_WORKSPACE=str(tmp_path),
        RUN_NAME=literal,
        RESUME_STEPS="2",
        MLFLOW_EXPERIMENT_NAME="tinygpt",
        LOCAL_SECRET="must-not-be-transmitted",
    )
    result = subprocess.run(
        ["bash", str(DAG_PATH.parents[3] / "runs" / "remote.sh")],
        input='printf "%s\\n" "$RUN_NAME" "$RESUME_STEPS" "${LOCAL_SECRET:-absent}"\nexit 17\n',
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 17, result.stderr
    assert result.stdout.splitlines() == [literal, "2", "absent"]
    assert not (tmp_path / "INJECTED").exists()
