"""tinygpt — a trainable Transformers/vLLM language-model pipeline."""

from tinygpt.model import TinyGPTForCausalLM as TinyGPTForCausalLM


def register_vllm() -> None:
    """Register the student architecture in every vLLM worker process."""
    from vllm import ModelRegistry

    ModelRegistry.register_model("TinyGPTForCausalLM", "tinygpt.vllm:TinyGPTForCausalLM")
