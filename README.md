# TinyGPT

> **Training curves — MLflow screenshot to be added after the full run.**
> Show pretraining and distillation loss and validation curves against training
> steps, with run names and key parameters visible.

A distributed language model training pipeline covering teacher preparation,
student pretraining, online distillation, evaluation, and inference. The project
uses Hugging Face Transformers, PyTorch FSDP, vLLM, Airflow, and MLflow.

## Architecture

The default teacher is `karpathy/nanochat-d32`. Teacher preparation downloads its
converted Transformers revision from Hugging Face and saves a model directory
with its tokenizer and chat template. A 20-layer Llama student is pretrained with
the same tokenizer, then distilled using a weighted combination of supervised
cross-entropy and temperature-scaled KL divergence over assistant tokens.

Training uses Transformers Trainer with FSDP, gradient accumulation, streaming
datasets, and sequence packing. AdamW is the default optimizer; optional MuonAdamW
combines PyTorch Muon for transformer matrices with AdamW for other parameters
and requires `--sharding-strategy NO_SHARD`.

Checkpoints use the Hugging Face format and include tokenizer assets and run
metadata. Trainer checkpoints retain optimizer, scheduler, and RNG state for
resuming training. vLLM handles generation and tensor-parallel inference.

Evaluation supports bits per byte, ARC, MMLU, GSM8K, and HumanEval. HumanEval runs
generated Python in restricted Docker containers with resource and execution
limits. MLflow records training parameters and metrics.

## Training

Requires Python 3.12+, uv, and Linux with a compatible NVIDIA CUDA environment
for vLLM. Run commands from the repository root.

```bash
NPROC_PER_NODE=8 bash runs/distill.sh
```

The script installs locked dependencies, prepares the teacher, pretrains the
student, runs distillation, and evaluates the resulting checkpoint.
`TEACHER_MODEL`, `TEACHER_DIR`, `TEACHER_DEVICE`, `STUDENT_RUN`, and `DISTILL_RUN`
override the corresponding defaults.

A short end-to-end run is available through:

```bash
DEVICE_TYPE=cuda bash runs/smoke.sh
```

CPU execution is supported with `DEVICE_TYPE=cpu`; vLLM evaluation is skipped.
Both modes download the full teacher model. Generated artifacts are stored under
`data/`, which is excluded from version control.

Individual stages are exposed as `scripts.prepare_teacher`, `scripts.pretrain`,
`scripts.distill`, and `scripts.evaluate_model`. Each module provides `--help`.
Local text validation requires a separate `--val-txt`; `--resume-from` requires a
complete Trainer checkpoint.

## Orchestration and tracking

The stack starts separate Airflow API, scheduler, and DAG processor services,
MLflow, and PostgreSQL with built-in defaults. No environment file is required.

> **Pipeline execution — Airflow screenshot to be added after the full run.**
> Show the completed GPU DAG with teacher preparation, pretraining, distillation,
> and evaluation tasks visible, including their final statuses and durations.

With Docker Compose installed:

```bash
bash runs/airflow.sh
```

Airflow is available at `http://localhost:8081` and MLflow at
`http://localhost:5000`. Trigger the `tinygpt_training` DAG to run the pipeline.
The generated Airflow login password is stored inside the container at
`/opt/airflow/simple_auth_manager_passwords.json.generated`.

GPU workers require the NVIDIA Container Toolkit and
`AIRFLOW_GPU=1 bash runs/airflow.sh`. HumanEval additionally requires a Docker
daemon and the `python:3.12-slim` image available to its evaluator.

Direct training runs use local SQLite tracking by default. Set
`MLFLOW_TRACKING_URI` to connect to the MLflow server and
`MLFLOW_EXPERIMENT_NAME` to select an experiment. Compose settings can be
overridden through environment variables or an optional `infra/airflow/.env`.

## Inference

```bash
uv run --no-sync python -m scripts.chat --vllm-model data/distill_checkpoints/distill
uv run --no-sync python -m scripts.evaluate_model --eval chat --vllm-model data/distill_checkpoints/distill
```
