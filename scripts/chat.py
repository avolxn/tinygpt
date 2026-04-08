"""
Interactive CLI chat with a fine-tuned tinygpt model.

Usage:
    python -m scripts.chat --checkpoint out/checkpoints/sft
    python -m scripts.chat --checkpoint out/checkpoints/sft --prompt "What is 2+2?"
"""

import argparse

from tinygpt.checkpoint import build_model_from_checkpoint
from tinygpt.inference import Engine
from tinygpt.tokenizer import HuggingFaceTokenizer
from tinygpt.utils import autodetect_device_type, compute_init

parser = argparse.ArgumentParser(description="Chat with tinygpt")
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to a model directory or Trainer output directory",
)
parser.add_argument("--tokenizer-dir", type=str, default="out/tokenizer")
parser.add_argument("--prompt", type=str, default="", help="Single-turn prompt (interactive mode if empty)")
parser.add_argument("--temperature", type=float, default=0.6)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--max-tokens", type=int, default=512)
parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps", ""])
args = parser.parse_args()

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
_, _, _, _, device = compute_init(device_type)

model, _ = build_model_from_checkpoint(args.checkpoint, device, phase="eval")
tokenizer = HuggingFaceTokenizer.from_directory(args.tokenizer_dir)
engine = Engine(model, tokenizer)

bos = tokenizer.get_bos_token_id()
user_start = tokenizer.encode_special("<|user_start|>")
user_end = tokenizer.encode_special("<|user_end|>")
assistant_start = tokenizer.encode_special("<|assistant_start|>")
assistant_end = tokenizer.encode_special("<|assistant_end|>")

print("\ntinygpt Interactive Chat")
print("-" * 50)
print("Type 'quit' or 'exit' to end  |  'clear' to reset conversation")
print("-" * 50)

conversation_tokens = [bos]


def run_turn(user_input: str) -> str:
    global conversation_tokens
    user_ids = tokenizer.encode(user_input)
    prompt = conversation_tokens + [user_start] + user_ids + [user_end] + [assistant_start]
    response_tokens: list[int] = []
    for token_column, _ in engine.generate(
        prompt,
        num_samples=1,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    ):
        token = token_column[0]
        if token == assistant_end or token == bos:
            break
        response_tokens.append(token)
        print(tokenizer.decode([token]), end="", flush=True)
    print()
    response = tokenizer.decode(response_tokens)
    # Append to conversation history
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
