# tinygpt

Distributed language model pretraining and online teacher-student distillation with
Transformers, PyTorch FSDP, vLLM inference, Airflow orchestration, and MLflow tracking.

## Pipeline

1. Download and prepare `karpathy/nanochat-d32` from Hugging Face. The preparation
   script selects its converted Transformers revision and saves the model and tokenizer.
2. Pretrain a 20-layer Llama student using the teacher's tokenizer and streaming data.
3. Distill teacher logits into the student, then evaluate with vLLM.

The teacher and student must have identical token-to-ID mappings. Model directories
use the Hugging Face format. Training checkpoints also include Trainer state for resume.
AdamW is the default optimizer; MuonAdamW requires `--sharding-strategy NO_SHARD`.

## Training

Requires Python 3.12+, uv, and a supported CUDA environment for vLLM.

Run from the repository root:

```bash
NPROC_PER_NODE=8 bash runs/distill.sh
```

The script installs locked GPU dependencies, prepares the teacher, pretrains,
distills, and evaluates. Set `TEACHER_MODEL`, `TEACHER_DIR`, `TEACHER_DEVICE`,
`STUDENT_RUN`, or `DISTILL_RUN` to override its defaults.

For a short pipeline run:

```bash
DEVICE_TYPE=cuda bash runs/smoke.sh
```

CPU execution is available with `DEVICE_TYPE=cpu`; vLLM evaluation requires CUDA.
Both modes use the actual NanoChat teacher and download its full weights.

MLflow tracking is required. Set `MLFLOW_TRACKING_URI` and
`MLFLOW_EXPERIMENT_NAME` for a tracking server; direct runs default to
`sqlite:///mlflow.db` and the `tinygpt` experiment.

Generated model and dataset artifacts are stored under `data/` and ignored by Git.

## Inference and evaluation

```bash
uv run --no-sync python -m scripts.chat --vllm-model data/distill_checkpoints/distill
uv run --no-sync python -m scripts.evaluate_model --eval chat --vllm-model data/distill_checkpoints/distill
```

Use `--help` on `scripts.prepare_teacher`, `scripts.pretrain`, `scripts.distill`,
`scripts.evaluate_model`, or `scripts.chat` for individual stage options.
Pretraining with `--txt` requires a separate `--val-txt` when validation is enabled.
`--resume-from` requires a complete Trainer checkpoint.

## Airflow and MLflow

The Compose stack runs separate Airflow API, scheduler, and DAG processor services,
a MLflow server, and PostgreSQL databases for both services.

```bash
bash runs/airflow.sh
```

Settings have built-in defaults; no secrets file or manual configuration is required.
Override them through environment variables or an optional `infra/airflow/.env`.

Airflow is available at `http://localhost:8081`; MLflow at `http://localhost:5000`.
Trigger the `tinygpt_training` DAG to configure a pipeline run. Airflow's Simple Auth
Manager stores generated passwords in
`/opt/airflow/simple_auth_manager_passwords.json.generated`.

For GPU training, install the NVIDIA Container Toolkit and start with
`AIRFLOW_GPU=1 bash runs/airflow.sh`. GPUs are assigned to the scheduler that executes
training tasks. CPU runs skip vLLM evaluation without skipping distillation.
HumanEval executes generated code in isolated Docker containers and requires
a Docker daemon with the `python:3.12-slim` image available to the evaluator.

## Development

```bash
uv sync --frozen --extra cpu --group dev
uv run --no-sync pytest
```

CI runs lint, formatting, type checking, tests, and a live Airflow/MLflow startup check.
The source is organized into `src/tinygpt/` for training components, `tasks/` for
datasets and benchmarks, `scripts/` for entry points, and `infra/airflow/` for orchestration.
