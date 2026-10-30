# TinyGPT

> **Training curves — MLflow screenshot to be added after the full run.**
> Show pretraining and distillation loss and validation curves against training
> steps, with run names and key parameters visible.

A distributed training pipeline based on [nanochat](https://github.com/karpathy/nanochat),
using Transformers, PyTorch FSDP2, vLLM, Airflow, and MLflow. It follows nanochat's
architecture and default training recipe, replacing SFT with online distillation
and taking the tokenizer from the teacher.

## Architecture

The reference implementation is pinned to nanochat commit `92d63d4`. The student
has 20 Transformer blocks with ReLU² MLPs, rotary QK-normalized attention,
`SSSL` short/full attention windows, value embeddings, residual scalars,
input smear, and backout. Transformers provides the model format and supported
components; the remaining architecture is implemented in the project.

The default teacher is `karpathy/nanochat-d32`, downloaded from a pinned converted
Transformers revision. Its tokenizer and token IDs are preserved. Pretraining
uses ClimbMix with nanochat's row-group partitioning and BOS-aligned best-fit
packing. Distillation uses SmolTalk, MMLU, and GSM8K with the reference sampling
ratios and assistant-token masks. Its loss combines supervised cross-entropy
and temperature-scaled teacher KL divergence.

Training uses Transformers Trainer, FSDP2, gradient accumulation, and mixed
precision. Muon follows nanochat's Polar Express update, variance reduction,
cautious decay, parameter groups, and schedules; other parameters use PyTorch
AdamW. Complete matrices are gathered for Muon updates under FSDP2. Optional
FP8 pretraining uses TorchAO on H100 or newer GPUs.

Checkpoints use the Hugging Face format and include tokenizer assets and run
metadata. Trainer checkpoints retain optimizer, scheduler, and RNG state for
resuming training. vLLM handles generation and tensor-parallel inference.

Evaluation uses nanochat's 22-task DCLM CORE bundle, bits per byte, and ChatCORE
tasks: ARC, MMLU, GSM8K, and HumanEval. Calculator calls and HumanEval programs
execute in restricted Docker containers. MLflow records parameters, metrics,
and evaluation results.

## Training

Requires Python 3.12+, uv, Linux with compatible NVIDIA CUDA drivers, and Docker
for calculator and HumanEval evaluation. Run commands from the repository root.

```bash
NPROC_PER_NODE=8 bash runs/distill.sh
```

The script installs locked dependencies, prepares the teacher, pretrains the
student, runs distillation, and evaluates the resulting checkpoint.
`TEACHER_MODEL`, `TEACHER_DIR`, `TEACHER_DEVICE`, `STUDENT_RUN`, and `DISTILL_RUN`
override the corresponding defaults.
`DEVICE_BATCH_SIZE` and `DISTILL_BATCH_SIZE` control per-GPU batches while
preserving the global token batch through accumulation. FP8 pretraining is enabled by default; set `FP8=0` to disable it.

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
Resume restores the saved recipe and Trainer state; `--num-iterations` can extend a run.

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
On macOS, the launch script uses port `5001` for MLflow to avoid AirPlay Receiver.
The generated Airflow login password is stored inside the container at
`/opt/airflow/simple_auth_manager_passwords.json.generated`.

GPU workers require the NVIDIA Container Toolkit and
`AIRFLOW_GPU=1 bash runs/airflow.sh`. The GPU scheduler mounts the host Docker
socket for isolated evaluation jobs. The DAG defaults to the full d24 speedrun recipe;
its parameters can be reduced for shorter runs.

To keep Airflow, MLflow, and the evaluation sandbox on a local machine while
training on a Runpod GPU worker, register the local SSH public key with Runpod
and use its direct TCP SSH address:

```bash
REMOTE_HOST=root@GPU_HOST REMOTE_PORT=SSH_PORT bash runs/airflow.sh
```

The scheduler copies the project to `/workspace/tinygpt` on the worker and
installs the locked GPU dependencies. OpenSSH reverse forwarding connects the
worker to local MLflow and the local Docker sandbox; neither service requires a
public port. The local machine must remain awake and connected throughout the
run. Checkpoints remain on the worker's persistent `/workspace` volume.

For an end-to-end validation on two H100 GPUs, set `depth=24` and
`max_seq_len=2048`, set `nproc_per_node=2`, `tensor_parallel_size=2`,
`device_batch_size=2`, `distill_batch_size=1`, `total_batch_size=16384`,
`pretrain_iterations=20`, `distill_iterations=20`, `resume_steps=2`,
`eval_every=10`, `eval_tokens=16384`, `core_max_per_task=2`, and
`max_eval_problems=2`. Leave `chat_eval_tasks` empty to cover every benchmark.
This executes every stage with the real teacher and student, including both
checkpoint resumes; the reduced training horizon does not measure final model
quality. The previous GPU validation used BF16; FP8 requires a separate GPU validation.

Direct training runs use local SQLite tracking by default. Set
`MLFLOW_TRACKING_URI` to connect to the MLflow server and
`MLFLOW_EXPERIMENT_NAME` to select an experiment. Compose settings can be
overridden through environment variables or an optional `infra/airflow/.env`.

## Inference

```bash
uv run --no-sync python -m scripts.chat --vllm-model data/distill_checkpoints/distill
uv run --no-sync python -m scripts.evaluate_model --eval chat --vllm-model data/distill_checkpoints/distill
```

Importing `tinygpt` registers its architecture with Transformers. The installed
package also registers it with vLLM, including the per-request smear cache.
Standard serving is also available:

```bash
uv run --no-sync vllm serve data/distill_checkpoints/distill/checkpoint-STEP --dtype bfloat16
```

Training ranks and inference tensor parallelism are independent. Set
`VLLM_TENSOR_PARALLEL_SIZE` for the pipeline, or `--vllm-tensor-parallel-size`
for individual scripts. The default is one GPU; the d24 model has twelve attention
heads, so eight-way tensor parallelism is not valid.
