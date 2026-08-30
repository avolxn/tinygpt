# tinygpt

`tinygpt` is a compact training and evaluation stack for small GPT-style models.
The repository is organized around five opinionated workflows:

1. Full training from scratch: tokenizer training, pretraining, SFT, and evaluation.
2. Pretraining with a converted teacher tokenizer.
3. Online distillation from `karpathy/nanochat-d32` into a student trained with the same tokenizer.
4. A smoke test for CPU or a small GPU.
5. An Airflow DAG that orchestrates Spark preparation, training, evaluation, and mandatory W&B tracking.

## Repository Workflows

| Workflow | Script | What it does |
| --- | --- | --- |
| From scratch | `runs/from_scratch.sh` | Trains a tokenizer, pretrains a base model, runs SFT, then evaluates the result. |
| Teacher tokenizer pretrain | `runs/pretrain_reference_d32.sh` | Converts the teacher tokenizer and runs pretraining only. |
| Distillation | `runs/distill_reference_d32.sh` | Distills from `karpathy/nanochat-d32` into a student checkpoint produced by `pretrain_reference_d32.sh`. |
| Smoke test | `runs/smoke.sh` | Runs a minimal end-to-end validation path on CPU or a small GPU. |
| Airflow | `runs/airflow.sh` | Starts the local Airflow UI and the parameterized `tinygpt_training` DAG. |

## Recommended Usage

Run commands from the `tinygpt` root:

```bash
bash runs/from_scratch.sh
bash runs/pretrain_reference_d32.sh
bash runs/distill_reference_d32.sh
bash runs/smoke.sh
bash runs/airflow.sh
```

## Storage Layout

All generated artifacts and support files are stored under `data/`.

Typical outputs:

- `data/tokenizer_from_scratch`
- `data/tokenizer_reference_d32`
- `data/teacher_reference_d32`
- `data/tokenizer_smoke`
- `data/pretrain_checkpoints/from_scratch`
- `data/pretrain_checkpoints/pretrain_reference_d32`
- `data/distill_checkpoints/distill_reference_d32`
- `data/sft_checkpoints/from_scratch`
- `data/sft_checkpoints/smoke`
- `data/airflow/<experiment>-<timestamp>`
- `data/identity_conversations.jsonl`

## Runtime Overrides

The run scripts are intentionally simple. Only a small number of environment overrides are supported:

- `WANDB_RUN`: Weights & Biases run name. Each run script provides a stage-specific default.
- `WANDB_PROJECT`: Required tracking project. Defaults to `tinygpt`.
- `WANDB_ENTITY`: Optional W&B team or user entity.
- `WANDB_BASE_URL`: Optional W&B Self-Managed server URL; public cloud is used by default.
- `NPROC_PER_NODE`: Number of `torchrun` processes per node for GPU workflows.
- `DEVICE_TYPE`: Runtime override for `runs/smoke.sh`, typically `cpu`, `cuda`, or `mps`.
- `TEACHER_DEVICE`: Teacher placement override for `runs/distill_reference_d32.sh`.
- `REFERENCE_MODEL`: Optional override for the teacher repository/path. Defaults to `karpathy/nanochat-d32`.
- `AIRFLOW_GPU`: Set to `1` to build the CUDA image and request all NVIDIA GPUs from Docker Compose.

Examples:

```bash
WANDB_RUN=from_scratch_exp bash runs/from_scratch.sh
WANDB_RUN=student_d32 bash runs/pretrain_reference_d32.sh
WANDB_RUN=distill_d32 TEACHER_DEVICE=cpu bash runs/distill_reference_d32.sh
DEVICE_TYPE=cpu bash runs/smoke.sh
```

Online W&B tracking is mandatory for pretraining, SFT, and distillation. Run
`uv run wandb login --verify` before invoking a Python entry point directly.
The shell workflows run the same verified login preflight automatically and
stop before allocating training resources when authentication fails.

## Local Airflow Orchestration

The local stack packages Airflow 3.1, Java 17, PySpark, and the tinygpt runtime
in one reproducible image. Airflow owns orchestration and observability; model
metrics and artifacts are logged to W&B; Spark is used only for offline data
preparation.

Create a local secrets file and add a valid W&B API key:

```bash
cp infra/airflow/.env.example infra/airflow/.env
$EDITOR infra/airflow/.env
bash runs/airflow.sh
```

Alternatively, export `WANDB_API_KEY` instead of creating the file. The Compose
configuration refuses to start when the key is missing or empty, and the first
DAG task verifies it online before any data or training work begins.

Open `http://localhost:8081`, sign in as `admin`, enable the
`tinygpt_training` DAG, and trigger it with the generated parameter form. The
standalone server writes its generated development password to:

```text
/opt/airflow/simple_auth_manager_passwords.json.generated
```

The default parameters run two small CPU training steps. Set `raw_input` to a
text, JSON, or Parquet path when Spark should build deterministic local shards;
leave it empty to use the configured Hugging Face dataset directly. Artifacts
for each run are isolated under `data/airflow/<experiment>-<timestamp>`.

On a Linux host with the NVIDIA Container Toolkit, request the GPU image with:

```bash
AIRFLOW_GPU=1 bash runs/airflow.sh
```

This Compose stack intentionally uses `airflow standalone` and the Simple Auth
Manager, so it is a local development environment, not a production Airflow
deployment. For production, keep the DAG but run it in managed Airflow or the
official Kubernetes Helm deployment, inject `WANDB_API_KEY` through a secrets
backend, move durable artifacts to object storage, and submit Spark/training
jobs to dedicated compute instead of running them inside the Airflow service.

## Important Constraint

Online KL distillation in this codebase requires tokenizer compatibility between teacher and student.
In practice, the distillation workflow assumes:

- the teacher is `karpathy/nanochat-d32`, unless `REFERENCE_MODEL` overrides it
- the student was pretrained with `runs/pretrain_reference_d32.sh`

If the student uses a different tokenizer or token ID mapping, distillation will fail by design.

## Python Entry Points

Primary modules:

- `python -m scripts.train_tokenizer`
- `python -m scripts.pretrain`
- `python -m scripts.finetune`
- `python -m scripts.distill`
- `python -m scripts.evaluate_tokenizer`
- `python -m scripts.evaluate_model`
- `python -m scripts.chat`

Defaults are aligned with the `data/` directory layout used by the run scripts.

## Environment

Expected baseline:

- Python 3.12+
- `uv` for environment setup
- PyTorch-compatible CPU, CUDA, or MPS runtime

The run scripts create or reuse `.venv` and install dependencies via `uv sync`.

## Optional Spark Data Preparation

PySpark is isolated in the `spark` extra and is used only for offline ETL. It
normalizes line endings, removes short and duplicate documents, creates a
deterministic validation split, and writes versionable Parquet shards.

Java 17 or newer is required outside Docker:

```bash
bash runs/prepare_data.sh \
  --input 'data/raw/*.txt' \
  --input-format text \
  --output data/processed/corpus \
  --master 'local[*]'
```

The resulting directory can be passed directly to tokenizer training and
pretraining:

```bash
python -m scripts.train_tokenizer --dataset data/processed/corpus
python -m scripts.pretrain --dataset data/processed/corpus
```
