"""HuggingFace Trainer integration for tinygpt.

TinyGPTTrainer subclasses transformers.Trainer to plug in:
- nanochat's MuonAdamW parameter groups and update schedules
- tinygpt's warmup + linear LR schedule (get_lr_multiplier)
- Pre-batched infinite iterators as data sources (no re-batching)
- tinygpt's Hugging Face style model directory format
- Pluggable eval_fn for bpb / distillation-loss evaluation
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Iterator
from typing import Any

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from tinygpt.checkpoint import METADATA_NAME, resolve_model_directory, save_checkpoint_assets
from tinygpt.distillation import masked_distillation_loss
from tinygpt.distributed import print0
from tinygpt.optimizer import get_lr_multiplier, make_optimizer, schedule_muon


class MlflowTrackingError(RuntimeError):
    """Raised when the configured MLflow tracking backend is unavailable."""


def parse_training_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Restore the saved recipe on resume, allowing explicit CLI overrides.

    Args:
        parser: Training entry point's parser, including ``--resume-from``.

    Returns:
        Arguments inheriting the original run's settings when resuming.
    """
    args = parser.parse_args()
    if args.resume_from:
        directory = resolve_model_directory(args.resume_from)
        with open(os.path.join(directory, METADATA_NAME), encoding="utf-8") as stream:
            parser.set_defaults(**json.load(stream)["user_config"])
        args = parser.parse_args()
    return args


def require_mlflow_tracking() -> str:
    """Configure and verify the MLflow tracking backend before training."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "tinygpt")
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    except Exception as exc:
        raise MlflowTrackingError(f"MLflow tracking is required but unavailable at {tracking_uri!r}: {exc}") from exc
    return tracking_uri


def model_logits(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Run a Transformers causal LM and return its logits tensor."""
    output = model(input_ids=input_ids, use_cache=False)
    logits = getattr(output, "logits", output)
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"Model output does not contain logits: {type(output)!r}")
    return logits


def causal_lm_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute loss against labels that are already shifted by the dataloader."""
    logits = model_logits(model, input_ids)
    if reduction == "mean" and not bool(labels.ne(-1).any()):
        return logits.sum() * 0.0
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-1,
        reduction=reduction,
    )
    return loss.reshape(labels.shape) if reduction == "none" else loss


def resolve_training_value(
    name: str,
    requested: int | float | None,
    fallback: int | float,
    *candidates: int | float | None,
) -> int | float:
    """Resolve an optional training value from CLI, checkpoint metadata, or a fallback."""
    if requested is not None:
        return requested
    for candidate in candidates:
        if candidate is not None:
            print0(f"Inherited {name}={candidate} from pretrain checkpoint")
            return candidate
    print0(f"Using fallback {name}={fallback}")
    return fallback


def build_training_arguments(
    *,
    output_dir: str,
    max_steps: int,
    device_batch_size: int,
    grad_accum_steps: int,
    warmup_steps: int,
    weight_decay: float,
    grad_clip: float,
    logging_steps: int,
    eval_every: int,
    save_steps: int,
    run_name: str,
    seed: int,
    report_to: list[str],
    device_type: str,
    compute_dtype: torch.dtype,
    disable_tqdm: bool,
    fsdp_sharding_strategy: str | None = None,
) -> TrainingArguments:
    """Build the shared HuggingFace Trainer arguments for all training phases."""
    evaluation_enabled = eval_every > 0
    fsdp = False
    fsdp_config: dict[str, Any] | None = None
    if fsdp_sharding_strategy in {"FULL_SHARD", "SHARD_GRAD_OP"}:
        fsdp = True
        fsdp_config = {
            "version": 2,
            "transformer_layer_cls_to_wrap": ["Block"],
            "reshard_after_forward": fsdp_sharding_strategy == "FULL_SHARD",
        }
    return TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=device_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        max_grad_norm=grad_clip,
        logging_steps=logging_steps,
        eval_strategy="steps" if evaluation_enabled else "no",
        eval_steps=eval_every if evaluation_enabled else None,
        save_strategy="steps",
        save_steps=save_steps,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to=report_to,
        run_name=run_name,
        seed=seed,
        label_names=["labels"],
        use_cpu=device_type == "cpu",
        bf16=compute_dtype == torch.bfloat16 and device_type == "cuda",
        fp16=False,
        prediction_loss_only=True,
        disable_tqdm=disable_tqdm,
        fsdp=fsdp,  # type: ignore[arg-type]
        fsdp_config=fsdp_config,
        ddp_find_unused_parameters=False,
        torch_compile=device_type == "cuda",
    )


class PreBatchedIterableDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Wraps an infinite (inputs, targets) generator into an IterableDataset.

    Items are already full batches of shape (B, T). The DataLoader must be
    created with batch_size=None so they are passed through unchanged.

    Args:
        loader: Infinite iterator yielding (inputs, targets) tensor pairs.
    """

    def __init__(self, loader: Iterator[tuple[torch.Tensor, torch.Tensor]]) -> None:
        self._loader = loader

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for inputs, targets in self._loader:
            yield {
                "input_ids": inputs.clone(),
                "labels": targets.clone(),
            }


class TinyGPTTrainer(Trainer):
    """Trainer subclass tailored for tinygpt.

    Args:
        matrix_lr: Learning rate for transformer block weight matrices (Muon).
        embedding_lr: Learning rate for embedding parameters (AdamW).
        scalar_lr: Learning rate for scalar/1-D parameters (AdamW).
        warmdown_ratio: Fraction of total steps used for linear learning-rate decay.
        final_lr_frac: LR at the end as a fraction of peak LR.
        init_lr_frac: Initial multiplier for every parameter family, including smear.
        train_loader: Infinite iterator yielding (inputs, targets) batches.
        eval_fn: Optional callable (model, step) -> dict[str, float] for
        custom evaluation metrics (e.g. bpb or distillation loss).
    """

    def __init__(
        self,
        *args: Any,
        matrix_lr: float,
        embedding_lr: float,
        scalar_lr: float,
        lm_head_lr: float = 0.008,
        warmdown_ratio: float = 0.65,
        final_lr_frac: float = 0.05,
        init_lr_frac: float = 1.0,
        train_loader: Iterator[tuple[torch.Tensor, torch.Tensor]],
        eval_fn: Callable[[nn.Module, int], dict[str, float]] | None = None,
        teacher_model: nn.Module | None = None,
        distill_alpha: float = 0.0,
        distill_temperature: float = 1.0,
        checkpoint_metadata: dict[str, Any] | None = None,
        tokenizer_dir: str | None = None,
        schedule_pretrain: bool = False,
        optimizer_checkpoint: str | None = None,
        conversation_progress: list[float] | None = None,
        warmup_ratio: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if teacher_model is not None and not 0.0 < distill_alpha <= 1.0:
            raise ValueError("Teacher-student training requires a positive distillation weight")
        self.model_accepts_loss_kwargs = False
        self._matrix_lr = matrix_lr
        self._embedding_lr = embedding_lr
        self._scalar_lr = scalar_lr
        self._lm_head_lr = lm_head_lr
        self._warmdown_ratio = warmdown_ratio
        self._final_lr_frac = final_lr_frac
        self._init_lr_frac = init_lr_frac
        self._train_loader = train_loader
        self._eval_fn = eval_fn
        self._teacher_model = teacher_model
        self._distill_alpha = distill_alpha
        self._distill_temperature = distill_temperature
        self._checkpoint_metadata = checkpoint_metadata or {}
        self._tokenizer_dir = tokenizer_dir
        self._optimizer_checkpoint = optimizer_checkpoint
        if schedule_pretrain:
            self.add_callback(MuonScheduleCallback(warmdown_ratio))
        if conversation_progress is not None:
            self.add_callback(
                ConversationScheduleCallback(conversation_progress, warmup_ratio, warmdown_ratio, final_lr_frac)
            )

    def get_train_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        """Return a DataLoader that passes pre-batched items through unchanged.

        Returns:
            DataLoader wrapping our infinite bestfit-packed iterator.
        """
        dataset = PreBatchedIterableDataset(self._train_loader)
        return DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=False)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute loss against the dataloader's already-shifted labels.

        Args:
            model: A Transformers causal language model.
            inputs: Dict with 'input_ids' and 'labels' keys.
            return_outputs: If True, also return a dict with the loss.
            num_items_in_batch: Ignored; present for API compatibility.

        Returns:
            Loss scalar, or (loss, {"loss": loss}) if return_outputs is True.
        """
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]

        if self._teacher_model is None:
            loss = causal_lm_loss(model, input_ids, labels)
            return (loss, {"loss": loss}) if return_outputs else loss

        student_logits = model_logits(model, input_ids)
        if bool(labels.ne(-1).any()):
            ce_loss = torch.nn.functional.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1),
                ignore_index=-1,
            )
        else:
            ce_loss = student_logits.sum() * 0.0

        teacher_device = next(self._teacher_model.parameters()).device
        teacher_input_ids = input_ids.to(teacher_device)
        with torch.no_grad():
            teacher_logits = model_logits(self._teacher_model, teacher_input_ids)
        teacher_logits = teacher_logits.to(student_logits.device)

        distill_loss = masked_distillation_loss(
            student_logits,
            teacher_logits,
            labels,
            temperature=self._distill_temperature,
        )
        loss = (1.0 - self._distill_alpha) * ce_loss + self._distill_alpha * distill_loss
        outputs = {
            "loss": loss,
            "ce_loss": ce_loss.detach(),
            "distill_loss": distill_loss.detach(),
        }
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self, model: Any = None) -> torch.optim.Optimizer:
        """Create the selected optimizer with project parameter groups.

        Args:
            model: Ignored; uses self.model.

        Returns:
            AdamW or MuonAdamW with the configured parameter groups.
        """
        assert self.model is not None
        self.optimizer = make_optimizer(
            self.model,
            matrix_lr=self._matrix_lr,
            embedding_lr=self._embedding_lr,
            scalar_lr=self._scalar_lr,
            lm_head_lr=self._lm_head_lr,
            weight_decay=self.args.weight_decay,
        )
        for group in self.optimizer.param_groups:
            group["lr"] *= self._init_lr_frac
            group["initial_lr"] = group["lr"]
        return self.optimizer

    def _load_optimizer_and_scheduler(self, checkpoint: str | None) -> None:
        """Resume training or warm-start distillation with native Trainer/FSDP loading."""
        if checkpoint is not None or self._optimizer_checkpoint is None:
            super()._load_optimizer_and_scheduler(checkpoint)
            return
        assert self.optimizer is not None and self.lr_scheduler is not None
        groups = [{k: group[k] for k in ("lr", "initial_lr")} for group in self.optimizer.param_groups]
        scheduler_state = self.lr_scheduler.state_dict()
        super()._load_optimizer_and_scheduler(self._optimizer_checkpoint)

        for group, settings in zip(self.optimizer.param_groups, groups, strict=True):
            group.update(settings)
        self.lr_scheduler.load_state_dict(scheduler_state)

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Create a LambdaLR scheduler using tinygpt's warmup + linear decay.

        Args:
            num_training_steps: Total number of training steps.
            optimizer: Optimizer to attach the scheduler to.

        Returns:
            LambdaLR scheduler wrapping get_lr_multiplier.
        """
        if optimizer is None:
            optimizer = self.optimizer
        assert optimizer is not None
        warmup_steps = int(self.args.warmup_steps)
        warmdown_ratio = self._warmdown_ratio
        final_lr_frac = self._final_lr_frac

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: get_lr_multiplier(
                step, num_training_steps, warmup_steps, warmdown_ratio, final_lr_frac
            ),
        )
        return self.lr_scheduler

    def evaluate(
        self,
        eval_dataset: Any = None,
        ignore_keys: Any = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Run evaluation using the pluggable eval_fn.

        Args:
            eval_dataset: Ignored; evaluation uses eval_fn.
            ignore_keys: Ignored.
            metric_key_prefix: Prefix for metric keys in the returned dict.

        Returns:
            Dict of evaluation metrics prefixed with metric_key_prefix.
        """
        if self._eval_fn is None:
            return {}
        assert self.model is not None
        model = self.model_wrapped
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                metrics = self._eval_fn(model, self.state.global_step)
        finally:
            model.train(was_training)
        prefixed = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}
        self.log(dict(prefixed))
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, prefixed)
        return prefixed

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        """Save model weights and config in a Hugging Face style directory.

        Args:
            output_dir: Directory to save the checkpoint. Defaults to
                self.args.output_dir.
            _internal_call: Forwarded to Trainer to distinguish internal checkpoint saves.
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        assert output_dir is not None
        assert self.model is not None
        super().save_model(output_dir, _internal_call=_internal_call)
        if self.args.should_save:
            save_checkpoint_assets(output_dir, self._checkpoint_metadata, self._tokenizer_dir)


class MlflowMetadataCallback(TrainerCallback):
    """Publish checkpoint metadata to the active MLflow run."""

    def __init__(self, metadata: dict[str, Any]) -> None:
        self._metadata = metadata

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        if not state.is_world_process_zero:
            return
        if mlflow.active_run() is not None:
            mlflow.log_dict(self._metadata, "tinygpt_metadata.json")


class MuonScheduleCallback(TrainerCallback):
    """Update momentum and cautious decay before each pretraining optimizer step.

    Args:
        warmdown_ratio: Fraction of training used for the reference warmdown.
    """

    def __init__(self, warmdown_ratio: float) -> None:
        self.warmdown_ratio = warmdown_ratio

    def on_step_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        """Keep optimizer scheduling aligned with Trainer's restored global step."""
        schedule_muon(kwargs["optimizer"], state.global_step, state.max_steps, self.warmdown_ratio)


class ConversationScheduleCallback(TrainerCallback):
    """Schedule and stop distillation by consumed conversations, as in nanochat.

    Args:
        progress: Consumed-epoch fraction updated by the packing iterator.
        warmup: Warmup fraction of the conversational epoch.
        warmdown: Warmdown fraction of the conversational epoch.
        final_lr: Final learning-rate fraction.
    """

    def __init__(self, progress: list[float], warmup: float, warmdown: float, final_lr: float) -> None:
        self.progress = progress
        self.warmup = warmup
        self.warmdown = warmdown
        self.final_lr = final_lr

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
        """Synchronize consumed progress and apply the next optimizer step's rates."""
        progress = torch.tensor(self.progress[0], device=args.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(progress, op=torch.distributed.ReduceOp.MAX)
        value = float(progress.item())
        if value < self.warmup:
            multiplier = (value + 1e-8) / self.warmup
        elif value <= 1 - self.warmdown or not self.warmdown:
            multiplier = 1.0
        else:
            fraction = (value - (1 - self.warmdown)) / self.warmdown
            multiplier = (1 - fraction) + fraction * self.final_lr
        for group in kwargs["optimizer"].param_groups:
            group["lr"] = group["initial_lr"] * multiplier
        if value >= 1:
            control.should_training_stop = True
            control.should_save = True
            control.should_evaluate = args.eval_strategy != "no"
