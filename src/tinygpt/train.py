"""
HuggingFace Trainer integration for tinygpt.

TinyGPTTrainer subclasses transformers.Trainer to plug in:
- tinygpt's multi-group AdamW optimizer (make_optimizer)
- tinygpt's warmup + cosine LR schedule (get_lr_multiplier)
- Pre-batched infinite iterators as data sources (no re-batching)
- tinygpt's Hugging Face style model directory format
- Pluggable eval_fn for bpb / SFT-loss evaluation
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from tinygpt.checkpoint import save_model_checkpoint
from tinygpt.distillation import masked_distillation_loss
from tinygpt.inference import Engine
from tinygpt.optimizer import make_optimizer
from tinygpt.scheduler import get_lr_multiplier
from tinygpt.utils import get_model_device, print0


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
                "input_ids": inputs.cpu(),
                "labels": targets.cpu(),
            }


class TinyGPTTrainer(Trainer):
    """Trainer subclass tailored for tinygpt.

    Args:
        matrix_lr: Learning rate for transformer block weight matrices.
        embedding_lr: Learning rate for embedding parameters.
        scalar_lr: Learning rate for scalar/1-D parameters.
        warmdown_ratio: Fraction of total steps used for cosine decay.
        final_lr_frac: LR at the end as a fraction of peak LR.
        train_loader: Infinite iterator yielding (inputs, targets) batches.
        eval_fn: Optional callable (model, step) -> dict[str, float] for
            custom evaluation metrics (e.g. bpb or SFT loss).
    """

    def __init__(
        self,
        *args: Any,
        matrix_lr: float,
        embedding_lr: float,
        scalar_lr: float,
        warmdown_ratio: float = 0.65,
        final_lr_frac: float = 0.05,
        train_loader: Iterator[tuple[torch.Tensor, torch.Tensor]],
        eval_fn: Callable[[nn.Module, int], dict[str, float]] | None = None,
        teacher_model: nn.Module | None = None,
        distill_alpha: float = 0.0,
        distill_temperature: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._matrix_lr = matrix_lr
        self._embedding_lr = embedding_lr
        self._scalar_lr = scalar_lr
        self._warmdown_ratio = warmdown_ratio
        self._final_lr_frac = final_lr_frac
        self._train_loader = train_loader
        self._eval_fn = eval_fn
        self._teacher_model = teacher_model
        self._distill_alpha = distill_alpha
        self._distill_temperature = distill_temperature

    def get_train_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        """Return a DataLoader that passes pre-batched items through unchanged.

        Returns:
            DataLoader wrapping our infinite bestfit-packed iterator.
        """
        dataset = PreBatchedIterableDataset(self._train_loader)
        return DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=False)

    def get_eval_dataloader(self, eval_dataset: Any = None) -> DataLoader[dict[str, torch.Tensor]]:
        """Return a no-op DataLoader; evaluation is handled in evaluate().

        Returns:
            Empty DataLoader.
        """
        empty: IterableDataset[dict[str, torch.Tensor]] = PreBatchedIterableDataset(iter([]))
        return DataLoader(empty, batch_size=None, num_workers=0)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute loss by calling model(input_ids, labels) directly.

        Args:
            model: The GPT model.
            inputs: Dict with 'input_ids' and 'labels' keys.
            return_outputs: If True, also return a dict with the loss.
            num_items_in_batch: Ignored; present for API compatibility.

        Returns:
            Loss scalar, or (loss, {"loss": loss}) if return_outputs is True.
        """
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]

        if self._teacher_model is None or self._distill_alpha <= 0:
            loss = model(input_ids, labels)
            return (loss, {"loss": loss}) if return_outputs else loss

        student_logits = model(input_ids)
        ce_loss = torch.nn.functional.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-1,
        )

        teacher_device = get_model_device(self._teacher_model)
        teacher_input_ids = input_ids.to(teacher_device)
        with torch.inference_mode():
            teacher_logits = self._teacher_model(teacher_input_ids)
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
        """Create the multi-group AdamW optimizer via make_optimizer.

        Args:
            model: Ignored; uses self.model.

        Returns:
            Configured AdamW optimizer with per-group learning rates.
        """
        assert self.model is not None
        self.optimizer = make_optimizer(
            self.model,
            matrix_lr=self._matrix_lr,
            embedding_lr=self._embedding_lr,
            scalar_lr=self._scalar_lr,
            weight_decay=self.args.weight_decay,
        )
        return self.optimizer

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Create a LambdaLR scheduler using tinygpt's warmup + cosine decay.

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
        self.model.eval()
        with torch.no_grad():
            metrics = self._eval_fn(self.model, self.state.global_step)
        self.model.train()
        prefixed = {f"{metric_key_prefix}/{k}": v for k, v in metrics.items()}
        self.log(prefixed)
        return prefixed

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        """Save model weights and config in a Hugging Face style directory.

        Args:
            output_dir: Directory to save the checkpoint. Defaults to
                self.args.output_dir.
            _internal_call: Ignored; present for API compatibility.
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        assert output_dir is not None
        assert self.model is not None
        save_model_checkpoint(output_dir, self.model)


class SamplerCallback(TrainerCallback):
    """Generate sample text from the model every sample_every steps.

    Args:
        tokenizer: HuggingFaceTokenizer for encoding/decoding.
        device: Device the model is on.
        sample_every: Generate samples every this many steps. 0 = disabled.
        master_process: Only generate on rank 0.
        prompts: List of prompt strings to sample from.
    """

    def __init__(
        self,
        tokenizer: Any,
        device: torch.device,
        sample_every: int,
        master_process: bool,
        prompts: list[str] | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._device = device
        self._sample_every = sample_every
        self._master_process = master_process
        self._prompts = prompts or ["The capital of France is", "The chemical symbol of gold is"]

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        """Generate samples at the configured interval.

        Args:
            args: Training arguments.
            state: Trainer state with global_step.
            control: Trainer control object.
            model: The current model.
        """
        if model is None:
            return
        if self._sample_every <= 0 or not self._master_process:
            return
        if state.global_step % self._sample_every != 0:
            return
        model.eval()
        engine = Engine(model, self._tokenizer)
        for prompt in self._prompts:
            tokens = self._tokenizer(prompt, prepend="<|bos|>")
            sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=32, temperature=0.0)
            print0(f"  [{prompt!r}] → {self._tokenizer.decode(sample[0])}")
        model.train()
