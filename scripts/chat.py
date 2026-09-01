"""
Interactive CLI chat with a vLLM model.

Usage:
    python -m scripts.chat --vllm-model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse

from tinygpt.inference import VLLM
from tinygpt.tokenizer import HuggingFaceTokenizer

parser = argparse.ArgumentParser(description="Chat with tinygpt")
parser.add_argument("--tokenizer-dir", type=str, default="data/tokenizer")
parser.add_argument("--prompt", type=str, default="", help="Single-turn prompt (interactive mode if empty)")
parser.add_argument("--temperature", type=float, default=0.6)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--max-tokens", type=int, default=512)
parser.add_argument("--vllm-model", type=str, required=True, help="Model path or Hugging Face ID")
parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
parser.add_argument("--trust-remote-code", action="store_true")


args = parser.parse_args()

tokenizer = HuggingFaceTokenizer.from_directory(args.tokenizer_dir)
vllm = VLLM(
    args.vllm_model,
    tensor_parallel_size=args.vllm_tensor_parallel_size,
    trust_remote_code=args.trust_remote_code,
)

def required_special(name: str) -> int:
    token_id = tokenizer.encode_special(name)
    if token_id is None:
        raise RuntimeError(f"Tokenizer missing required special token: {name}")
    return token_id


bos = tokenizer.get_bos_token_id()
user_start = required_special("<|user_start|>")
user_end = required_special("<|user_end|>")
assistant_start = required_special("<|assistant_start|>")
assistant_end = required_special("<|assistant_end|>")

print("\ntinygpt Interactive Chat")
print("-" * 50)
print("Type 'quit' or 'exit' to end  |  'clear' to reset conversation")
print("-" * 50)

conversation_tokens = [bos]


def run_turn(user_input: str) -> str:
    global conversation_tokens
    user_ids = tokenizer.encode(user_input)
    prompt = conversation_tokens + [user_start] + user_ids + [user_end] + [assistant_start]
    response = vllm.generate(
        tokenizer.decode(prompt),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        stop=["<|assistant_end|>", "<|bos|>"],
    )
    response = response.split("<|assistant_end|>", 1)[0].split("<|bos|>", 1)[0]
    response_tokens = tokenizer.encode(response)
    print(response, end="", flush=True)
    print()
    conversation_tokens = (
        conversation_tokens
        + [user_start]
        + user_ids
        + [user_end]
        + [assistant_start]
        + response_tokens
        + [assistant_end]
    )
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
            conversation_tokens = [bos]
            print("[Conversation cleared]")
            continue

        print("Assistant: ", end="", flush=True)
        run_turn(user_input)
