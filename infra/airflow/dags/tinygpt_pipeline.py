"""Airflow DAG for the teacher-to-student distillation pipeline."""

from __future__ import annotations

from datetime import UTC, datetime

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Param

PYTHON_BIN = "/opt/tinygpt/.venv/bin/python"
TORCHRUN_BIN = "/opt/tinygpt/.venv/bin/torchrun"
WORKSPACE = "/workspace"

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
    "DEPTH": "{{ params.depth }}",
    "MAX_SEQ_LEN": "{{ params.max_seq_len }}",
    "DEVICE_BATCH_SIZE": "{{ params.device_batch_size }}",
    "TOTAL_BATCH_SIZE": "{{ params.total_batch_size }}",
    "PRETRAIN_ITERATIONS": "{{ params.pretrain_iterations }}",
    "DISTILL_ITERATIONS": "{{ params.distill_iterations }}",
    "DISTILL_TASKS": "{{ params.distill_tasks }}",
    "CHAT_EVAL_TASKS": "{{ params.chat_eval_tasks }}",
    "MAX_EVAL_PROBLEMS": "{{ params.max_eval_problems }}",
}


def bash_task(task_id: str, bash_command: str, *, retries: int = 0) -> BashOperator:
    """Create a project Bash task with inherited secrets and no XCom payload."""
    return BashOperator(
        task_id=task_id,
        bash_command=bash_command,
        cwd=WORKSPACE,
        env=COMMON_ENV,
        append_env=True,
        do_xcom_push=False,
        retries=retries,
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
            "airflow-smoke",
            type="string",
            pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$",
            description="Safe artifact and MLflow run prefix",
        ),
        "teacher_model": Param(
            "hf-internal-testing/tiny-random-LlamaForCausalLM",
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
        "device_type": Param("cpu", type="string", enum=["cpu", "cuda"]),
        "nproc_per_node": Param(1, type="integer", minimum=1, maximum=64),
        "depth": Param(2, type="integer", minimum=1, maximum=128),
        "max_seq_len": Param(128, type="integer", minimum=32, maximum=65_536),
        "device_batch_size": Param(1, type="integer", minimum=1),
        "total_batch_size": Param(128, type="integer", minimum=1),
        "pretrain_iterations": Param(2, type="integer", minimum=1),
        "distill_iterations": Param(2, type="integer", minimum=1),
        "distill_tasks": Param(
            "identity",
            type="string",
            pattern=r"^(smoltalk|mmlu|gsm8k|identity|spelling)(,(smoltalk|mmlu|gsm8k|identity|spelling))*$",
        ),
        "chat_eval_tasks": Param(
            "MMLU",
            type="string",
            enum=["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval"],
        ),
        "max_eval_problems": Param(4, type="integer", minimum=1),
    },
    tags=["tinygpt", "distillation"],
) as dag:
    prepare_teacher = bash_task(
        "prepare_teacher",
        """set -euo pipefail
ARTIFACT_DIR="data/airflow/$RUN_NAME"
mkdir -p "$ARTIFACT_DIR"
"$PYTHON_BIN" -m scripts.prepare_teacher \
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
  -m scripts.pretrain
  --teacher-model "$ARTIFACT_DIR/teacher"
  --dataset "$DATASET"
  --text-field "$TEXT_FIELD"
  --device-type "$DEVICE_TYPE"
  --depth "$DEPTH"
  --max-seq-len "$MAX_SEQ_LEN"
  --device-batch-size "$DEVICE_BATCH_SIZE"
  --total-batch-size "$TOTAL_BATCH_SIZE"
  --num-iterations "$PRETRAIN_ITERATIONS"
  --eval-every -1
  --out-dir "$ARTIFACT_DIR"
  --run-name "$RUN_NAME"
  --run "$RUN_NAME-pretrain"
)
if [[ -n "$TXT_INPUT" ]]; then
  ARGS+=(--txt "$TXT_INPUT")
fi
if [[ "$NPROC_PER_NODE" -eq 1 ]]; then
  "$PYTHON_BIN" "${ARGS[@]}"
else
  "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" "${ARGS[@]}"
fi
""",
    )

    evaluate_base = bash_task(
        "evaluate_base",
        """set -euo pipefail
if [[ "$DEVICE_TYPE" != "cuda" ]]; then
  echo "Skipping base vLLM evaluation on non-CUDA workers"
  exit 0
fi
ARTIFACT_DIR="data/airflow/$RUN_NAME"
ARGS=(
  -m scripts.evaluate_model
  --checkpoint "$ARTIFACT_DIR/pretrain_checkpoints/$RUN_NAME"
  --eval sample
  --vllm-model "$ARTIFACT_DIR/pretrain_checkpoints/$RUN_NAME"
)
if [[ "$NPROC_PER_NODE" -eq 1 ]]; then
  "$PYTHON_BIN" "${ARGS[@]}"
else
  "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" "${ARGS[@]}"
fi
""",
        retries=1,
    )

    download_identity = bash_task(
        "download_identity",
        """set -euo pipefail
IDENTITY_PATH="data/airflow/$RUN_NAME/identity_conversations.jsonl"
mkdir -p "$(dirname "$IDENTITY_PATH")"
curl -fsSL --retry 3 \
  -o "$IDENTITY_PATH" \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
""",
        retries=1,
    )

    distill = bash_task(
        "distill",
        """set -euo pipefail
ARTIFACT_DIR="data/airflow/$RUN_NAME"
ARGS=(
  -m scripts.distill
  --checkpoint "$ARTIFACT_DIR/pretrain_checkpoints/$RUN_NAME"
  --teacher-model "$ARTIFACT_DIR/teacher"
  --identity-conversations "$ARTIFACT_DIR/identity_conversations.jsonl"
  --tasks "$DISTILL_TASKS"
  --device-type "$DEVICE_TYPE"
  --max-seq-len "$MAX_SEQ_LEN"
  --device-batch-size "$DEVICE_BATCH_SIZE"
  --total-batch-size "$TOTAL_BATCH_SIZE"
  --num-iterations "$DISTILL_ITERATIONS"
  --eval-every -1
  --out-dir "$ARTIFACT_DIR"
  --run-name "$RUN_NAME"
  --run "$RUN_NAME-distill"
)
if [[ "$NPROC_PER_NODE" -eq 1 ]]; then
  "$PYTHON_BIN" "${ARGS[@]}"
else
  "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" "${ARGS[@]}"
fi
""",
    )

    evaluate_chat = bash_task(
        "evaluate_chat",
        """set -euo pipefail
if [[ "$DEVICE_TYPE" != "cuda" ]]; then
  echo "Skipping vLLM chat evaluation on non-CUDA workers"
  exit 0
fi
ARTIFACT_DIR="data/airflow/$RUN_NAME"
ARGS=(
  -m scripts.evaluate_model
  --checkpoint "$ARTIFACT_DIR/distill_checkpoints/$RUN_NAME"
  --device-type "$DEVICE_TYPE"
  --eval chat
  --vllm-model "$ARTIFACT_DIR/distill_checkpoints/$RUN_NAME"
  --tasks "$CHAT_EVAL_TASKS"
  --max-problems "$MAX_EVAL_PROBLEMS"
)
if [[ "$NPROC_PER_NODE" -eq 1 ]]; then
  "$PYTHON_BIN" "${ARGS[@]}"
else
  "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" "${ARGS[@]}"
fi
""",
        retries=1,
    )

    prepare_teacher >> pretrain >> evaluate_base >> distill >> evaluate_chat
    prepare_teacher >> download_identity >> distill
