"""Static checks for the Airflow DAG without adding Airflow to the app environment."""

import ast
from pathlib import Path

DAG_PATH = Path(__file__).parents[1] / "infra" / "airflow" / "dags" / "tinygpt_pipeline.py"
EXPECTED_TASKS = {
    "verify_mlflow",
    "train_tokenizer",
    "evaluate_tokenizer",
    "pretrain",
    "evaluate_base",
    "download_identity",
    "finetune",
    "evaluate_chat",
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
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "BashOperator"
    )
    keywords = {keyword.arg: keyword.value for keyword in operator_call.keywords}

    assert isinstance(keywords["append_env"], ast.Constant)
    assert keywords["append_env"].value is True
    assert isinstance(keywords["do_xcom_push"], ast.Constant)
    assert keywords["do_xcom_push"].value is False
