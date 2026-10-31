"""Airflow DAG for the teacher-to-student distillation pipeline."""

from __future__ import annotations

from datetime import UTC, datetime

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Param

PYTHON_BIN = "/opt/tinygpt/.venv/bin/python"
TORCHRUN_BIN = "/opt/tinygpt/.venv/bin/torchrun"
WORKSPACE = "/workspace"
TEACHER_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

COMMON_ENV = {
    "PYTHON_BIN": PYTHON_BIN,
    "TORCHRUN_BIN": TORCHRUN_BIN,
    "PYTHONPATH": f"{WORKSPACE}/src:{WORKSPACE}",
    "RUN_NAME": "{{ params.experiment }}-{{ ts_nodash }}",
    "TEACHER_MODEL": "{{ params.teacher_model }}",
    "DATASET": "{{ params.dataset }}",
    "TEXT_FIELD": "{{ params.text_field }}",
    "TXT_INPUT": "{{ params.txt }}",
    "DEVICE_TYPE": "{{ params.device_type }}",
    "NPROC_PER_NODE": "{{ params.nproc_per_node }}",
    "TENSOR_PARALLEL_SIZE": "{{ params.tensor_parallel_size }}",
    "DEPTH": "{{ params.depth }}",
    "FP8": "{{ params.fp8 | lower }}",
    "MAX_SEQ_LEN": "{{ params.max_seq_len }}",
    "DEVICE_BATCH_SIZE": "{{ params.device_batch_size }}",
    "DISTILL_BATCH_SIZE": "{{ params.distill_batch_size }}",
    "TOTAL_BATCH_SIZE": "{{ params.total_batch_size }}",
    "PRETRAIN_ITERATIONS": "{{ params.pretrain_iterations }}",
    "DISTILL_ITERATIONS": "{{ params.distill_iterations }}",
    "DISTILL_TASKS": "{{ params.distill_tasks }}",
    "CHAT_EVAL_TASKS": "{{ params.chat_eval_tasks }}",
    "MAX_EVAL_PROBLEMS": "{{ params.max_eval_problems }}",
    "EVAL_TOKENS": "{{ params.eval_tokens }}",
    "EVAL_EVERY": "{{ params.eval_every }}",
    "CORE_MAX_PER_TASK": "{{ params.core_max_per_task }}",
    "RESUME_STEPS": "{{ params.resume_steps }}",
}


def bash_task(task_id: str, bash_command: str, *, retries: int = 0, trigger_rule: str = "all_success") -> BashOperator:
    """Create a project Bash task with inherited secrets and no XCom payload."""
    return BashOperator(
        task_id=task_id,
        bash_command=(
            "TASK_SCRIPT=$(cat <<'TINYGPT_TASK'\n" + bash_command + "\nTINYGPT_TASK\n)\n"
            'if [[ -n "${REMOTE_HOST:-}" ]]; then\n'
            '  printf "%s\\n" "$TASK_SCRIPT" | bash runs/remote.sh\n'
            'else\n  bash -c "$TASK_SCRIPT"\nfi\n'
        ),
        cwd=WORKSPACE,
        env=COMMON_ENV,
        append_env=True,
        do_xcom_push=False,
        retries=retries,
        trigger_rule=trigger_rule,
    )


with DAG(
    dag_id="tinygpt_training",
    description="Prepare a teacher, pretrain a student, distill, and evaluate",
    schedule=None,
    start_date=datetime(2025, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    params={
        "experiment": Param(
            "tinygpt",
            type="string",
            pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$",
            description="Safe artifact and MLflow run prefix",
        ),
        "teacher_model": Param(
            TEACHER_MODEL,
            type="string",
            minLength=1,
            description="Hugging Face teacher model ID",
        ),
        "text_field": Param("text", type="string", pattern=r"^[A-Za-z_][A-Za-z0-9_]*$"),
        "dataset": Param(
            "karpathy/climbmix-400b-shuffle",
            type="string",
            minLength=1,
            description="Hugging Face dataset used when txt is empty",
        ),
        "txt": Param("", type="string", description="Optional local text file with one document per line"),
        "device_type": Param("cuda", type="string", enum=["cpu", "cuda"]),
        "nproc_per_node": Param(8, type="integer", minimum=1, maximum=64),
        "tensor_parallel_size": Param(1, type="integer", minimum=1, maximum=64),
        "depth": Param(24, type="integer", minimum=1, maximum=128),
        "fp8": Param(True, type="boolean"),
        "max_seq_len": Param(2048, type="integer", minimum=32, maximum=65_536),
        "device_batch_size": Param(16, type="integer", minimum=1),
        "distill_batch_size": Param(4, type="integer", minimum=1),
        "total_batch_size": Param(-1, type="integer", minimum=-1),
        "pretrain_iterations": Param(-1, type="integer", minimum=-1),
        "distill_iterations": Param(-1, type="integer", minimum=-1),
        "distill_tasks": Param(
            "smoltalk,mmlu,gsm8k",
            type="string",
            pattern=r"^(smoltalk|mmlu|gsm8k)(,(smoltalk|mmlu|gsm8k))*$",
        ),
        "chat_eval_tasks": Param(
            "",
            type="string",
            enum=["", "ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval"],
        ),
        "max_eval_problems": Param(-1, type="integer", minimum=-1),
        "eval_tokens": Param(80 * 524288, type="integer", minimum=1),
        "eval_every": Param(250, type="integer", minimum=1),
        "core_max_per_task": Param(-1, type="integer", minimum=-1),
        "resume_steps": Param(0, type="integer", minimum=0, description="Additional steps to verify checkpoint resume"),
    },
    tags=["tinygpt", "distillation"],
) as dag:
    prepare_worker = BashOperator(
        task_id="prepare_worker",
        bash_command='if [[ -n "${REMOTE_HOST:-}" ]]; then bash runs/remote.sh setup; fi',
        cwd=WORKSPACE,
        append_env=True,
        do_xcom_push=False,
    )
    download_teacher = bash_task(
        "download_teacher",
        """set -euo pipefail
if [[ "$DEVICE_TYPE" == "cuda" ]]; then
"$PYTHON_BIN" - <<'PY'
from tinygpt.execution import execute_code, use_calculator
from tinygpt.train import require_mlflow_tracking

require_mlflow_tracking()
result = execute_code("print(2 + 2)")
assert result.success and result.stdout.strip() == "4", result
assert use_calculator("2 + 2") == "4"
PY
fi
ARTIFACT_DIR="data/airflow/$RUN_NAME"
mkdir -p "$ARTIFACT_DIR"
"$PYTHON_BIN" -m scripts.download_teacher \
  --model "$TEACHER_MODEL" \
  --out-dir "$ARTIFACT_DIR/teacher"
""",
        retries=1,
    )

    pretrain = bash_task(
        "pretrain",
        """set -euo pipefail
ARTIFACT_DIR="data/airflow/$RUN_NAME"
ARGS=(
  scripts.pretrain
  --teacher-model "$ARTIFACT_DIR/teacher"
  --dataset "$DATASET"
  --text-field "$TEXT_FIELD"
  --device-type "$DEVICE_TYPE"
  --depth "$DEPTH"
  --max-seq-len "$MAX_SEQ_LEN"
  --device-batch-size "$DEVICE_BATCH_SIZE"
  --total-batch-size "$TOTAL_BATCH_SIZE"
  --num-iterations "$PRETRAIN_ITERATIONS"
  --eval-every "$EVAL_EVERY"
  --eval-tokens "$EVAL_TOKENS"
  --out-dir "$ARTIFACT_DIR"
  --run-name "$RUN_NAME"
  --run "$RUN_NAME-pretrain"
)
if [[ -n "$TXT_INPUT" ]]; then
  ARGS+=(--txt "$TXT_INPUT" --eval-every -1)
fi
if [[ "$FP8" == "true" ]]; then
  ARGS+=(--fp8)
fi
if [[ "$NPROC_PER_NODE" -eq 1 ]]; then
  "$PYTHON_BIN" -m "${ARGS[@]}"
else
  "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" -m -- "${ARGS[@]}"
fi
""",
    )

    resume_pretrain = bash_task(
        "resume_pretrain",
        """set -euo pipefail
if [[ "$RESUME_STEPS" -eq 0 ]]; then exit 0; fi
if [[ "$PRETRAIN_ITERATIONS" -le 0 ]]; then
  echo "Resume verification requires a positive pretrain_iterations" >&2
  exit 1
fi
"$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" -m -- scripts.pretrain \
  --resume-from "data/airflow/$RUN_NAME/pretrain_checkpoints/$RUN_NAME" \
  --num-iterations "$((PRETRAIN_ITERATIONS + RESUME_STEPS))" \
  --run "$RUN_NAME-pretrain-resume"
""",
    )

    evaluate_base = bash_task(
        "evaluate_base",
        """set -euo pipefail
ARTIFACT_DIR="data/airflow/$RUN_NAME"
ARGS=(
  scripts.evaluate_model
  --checkpoint "$ARTIFACT_DIR/pretrain_checkpoints/$RUN_NAME"
  --eval core,bpb
  --device-type "$DEVICE_TYPE"
  --dataset "$DATASET"
  --text-field "$TEXT_FIELD"
  --device-batch-size "$DEVICE_BATCH_SIZE"
  --max-per-task "$CORE_MAX_PER_TASK"
  --split-tokens "$EVAL_TOKENS"
)
"$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" -m -- "${ARGS[@]}"
""",
        retries=1,
    )

    distill = bash_task(
        "distill",
        """set -euo pipefail
ARTIFACT_DIR="data/airflow/$RUN_NAME"
ARGS=(
  scripts.distill
  --checkpoint "$ARTIFACT_DIR/pretrain_checkpoints/$RUN_NAME"
  --teacher-model "$ARTIFACT_DIR/teacher"
  --tasks "$DISTILL_TASKS"
  --device-type "$DEVICE_TYPE"
  --max-seq-len "$MAX_SEQ_LEN"
  --device-batch-size "$DISTILL_BATCH_SIZE"
  --num-iterations "$DISTILL_ITERATIONS"
  --eval-every "$EVAL_EVERY"
  --eval-tokens "$EVAL_TOKENS"
  --out-dir "$ARTIFACT_DIR"
  --run-name "$RUN_NAME"
  --run "$RUN_NAME-distill"
)
if [[ "$NPROC_PER_NODE" -eq 1 ]]; then
  "$PYTHON_BIN" -m "${ARGS[@]}"
else
  "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" -m -- "${ARGS[@]}"
fi
""",
        trigger_rule="none_failed",
    )

    resume_distill = bash_task(
        "resume_distill",
        """set -euo pipefail
if [[ "$RESUME_STEPS" -eq 0 ]]; then exit 0; fi
if [[ "$DISTILL_ITERATIONS" -le 0 ]]; then
  echo "Resume verification requires a positive distill_iterations" >&2
  exit 1
fi
"$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" -m -- scripts.distill \
  --resume-from "data/airflow/$RUN_NAME/distill_checkpoints/$RUN_NAME" \
  --num-iterations "$((DISTILL_ITERATIONS + RESUME_STEPS))" \
  --run "$RUN_NAME-distill-resume"
""",
    )

    evaluate_chat = bash_task(
        "evaluate_chat",
        """set -euo pipefail
if [[ "$DEVICE_TYPE" != "cuda" ]]; then
  echo "Skipping vLLM chat evaluation on non-CUDA workers"
  exit 99
fi
ARTIFACT_DIR="data/airflow/$RUN_NAME"
ARGS=(
  scripts.evaluate_model
  --checkpoint "$ARTIFACT_DIR/distill_checkpoints/$RUN_NAME"
  --device-type "$DEVICE_TYPE"
  --eval chat
  --vllm-model "$ARTIFACT_DIR/distill_checkpoints/$RUN_NAME"
  --tasks "$CHAT_EVAL_TASKS"
)
if [[ "$MAX_EVAL_PROBLEMS" -gt 0 ]]; then
  ARGS+=(--max-problems "$MAX_EVAL_PROBLEMS")
fi
"$PYTHON_BIN" -m "${ARGS[@]}" --vllm-tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
""",
        retries=1,
    )

    prepare_worker >> download_teacher >> pretrain >> resume_pretrain >> evaluate_base
    evaluate_base >> distill >> resume_distill >> evaluate_chat
