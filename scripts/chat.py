"""Interactive CLI chat with a vLLM model.

Usage:
    python -m scripts.chat --vllm-model data/distill_checkpoints/student
"""

import argparse

from tinygpt.checkpoint import resolve_model_directory
from tinygpt.inference import VLLM
from tinygpt.tokenizer import HuggingFaceTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with tinygpt")
    parser.add_argument("--prompt", type=str, default="", help="Single-turn prompt (interactive mode if empty)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--vllm-model", type=str, required=True, help="Prepared model directory or Trainer checkpoint")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")

    args = parser.parse_args()

    model_dir = resolve_model_directory(args.vllm_model)
    tokenizer = HuggingFaceTokenizer.from_directory(model_dir)
    vllm = VLLM(
        model_dir,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
    )

    print("\ntinygpt Interactive Chat")
    print("-" * 50)
    print("Type 'quit' or 'exit' to end  |  'clear' to reset conversation")
    print("-" * 50)

    messages: list[dict[str, str]] = []

    def run_turn(user_input: str) -> str:
        messages.append({"role": "user", "content": user_input})
        prompt = list(tokenizer.tokenizer.apply_chat_template(messages, add_generation_prompt=True))
        response = vllm.generate_conversation(
            prompt,
            tokenizer,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(response, flush=True)
        messages.append({"role": "assistant", "content": response})
        return response

    if args.prompt:
        run_turn(args.prompt)
    else:
        while True:
            try:
                user_input = input("\nUser: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break
            if user_input.lower() == "clear":
                messages.clear()
                print("[Conversation cleared]")
                continue

            print("Assistant: ", end="", flush=True)
            run_turn(user_input)
