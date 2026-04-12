"""
Evaluate compression ratio of the trained tokenizer.

Compares our tokenizer against GPT-2 and GPT-4 baselines on several
text types (news, code, math, Korean, science).

Usage:
    python -m scripts.evaluate_tokenizer
    python -m scripts.evaluate_tokenizer --tokenizer-dir data/tokenizer
"""

import argparse
import os
from typing import TypedDict

from tinygpt.tokenizer import HuggingFaceTokenizer


class CompressionMetrics(TypedDict):
    bytes: int
    tokens: int
    ratio: float

parser = argparse.ArgumentParser(description="Evaluate tokenizer compression")
parser.add_argument("--tokenizer-dir", type=str, default="data/tokenizer", help="Directory containing tokenizer.json")
args = parser.parse_args()

news_text = """
(Washington, D.C.)- Mexico's National Service of Agro-Alimentary Health, Safety, and Quality
reported a new case of New World Screwworm in Ixhuatlan de Madero, Veracruz in Mexico,
approximately 160 miles northward of the current sterile fly dispersal grid.
""".strip()

korean_text = """
정직한 사실 위에, 공정한 시선을 더하다
헤럴드코리아타임즈는 정치, 경제, 사회, 문화 등 한국 사회 전반의 주요 이슈를 심도 있게 다루는 종합 온라인 신문사입니다.
""".strip()

code_text = r"""
class BasicTokenizer:
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
""".strip()

math_text = r"""
\begin{theorem}
For every integer $n \ge 1$,
\[ \sum_{k=1}^{n} k^{3} = \left(\frac{n(n+1)}{2}\right)^{2}. \]
\end{theorem}
\begin{proof}[Proof (Induction)]
Let $S(n) = \sum_{k=1}^{n} k^3$. For $n=1$, $S(1)=1=(1\cdot 2/2)^2$.
Assume the formula holds for $n$; then $S(n+1)=S(n)+(n+1)^3 = \left(\frac{(n+1)(n+2)}{2}\right)^2$.
\end{proof}
""".strip()

science_text = """
Photosynthesis is a photochemical energy transduction process in which light-harvesting
pigment-protein complexes within the thylakoid membranes of oxygenic phototrophs absorb
photons and initiate charge separation at the reaction center, driving the linear electron
transport chain from water to NADP+ via photosystem II, the cytochrome b6f complex, and
photosystem I, concomitantly generating a trans-thylakoid proton motive force utilized by
chloroplastic ATP synthase.
""".strip()

all_text = [
    ("news", news_text),
    ("korean", korean_text),
    ("code", code_text),
    ("math", math_text),
    ("science", science_text),
]

tokenizers: dict[str, HuggingFaceTokenizer] = {}

tokenizers["gpt2"] = HuggingFaceTokenizer.from_pretrained("gpt2")
tokenizers["gpt4"] = HuggingFaceTokenizer.from_pretrained("cl100k_base")

if os.path.exists(os.path.join(args.tokenizer_dir, "tokenizer.json")):
    tokenizers["ours"] = HuggingFaceTokenizer.from_directory(args.tokenizer_dir)
    print(f"Loaded our tokenizer from {args.tokenizer_dir}")
else:
    print(f"WARNING: tokenizer not found at {args.tokenizer_dir} — skipping 'ours'")

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

results: dict[str, dict[str, CompressionMetrics]] = {}
vocab_sizes: dict[str, int] = {}

for name, tok in tokenizers.items():
    vocab_sizes[name] = tok.get_vocab_size()
    results[name] = {}
    for text_name, text in all_text:
        encoded = tok.encode(text)
        decoded = tok.decode(encoded)
        if decoded != text:
            raise ValueError(f"Round-trip failed for {name}/{text_name}")
        nb = len(text.encode("utf-8"))
        results[name][text_name] = {"bytes": nb, "tokens": len(encoded), "ratio": nb / len(encoded)}

print("\nVocab sizes:")
for name, size in vocab_sizes.items():
    print(f"  {name:<8}: {size:,}")

if "ours" not in tokenizers:
    print("\nNo 'ours' tokenizer found — skipping comparison.")
else:
    for baseline_name in ("gpt2", "gpt4"):
        print(f"\nComparison with {baseline_name.upper()}:")
        print("=" * 90)
        print(f"{'Text':<10} {'Bytes':<8} {baseline_name:<14} {'Ours':<14} {'Relative':<10} Better")
        print(f"{'':10} {'':8} {'Tokens':<7} {'Ratio':<7} {'Tokens':<7} {'Ratio':<7}")
        print("-" * 90)
        for text_name, _ in all_text:
            b = results[baseline_name][text_name]
            o = results["ours"][text_name]
            rel = (b["tokens"] - o["tokens"]) / b["tokens"] * 100
            b_color = RED if b["ratio"] < o["ratio"] else GREEN
            o_color = GREEN if o["ratio"] > b["ratio"] else RED
            better = "Ours" if o["ratio"] > b["ratio"] else baseline_name.upper()
            diff_color = GREEN if rel > 0 else RED
            print(
                f"{text_name:<10} {b['bytes']:<8} "
                f"{b_color}{b['tokens']:<7}{RESET} {b_color}{b['ratio']:<7.2f}{RESET} "
                f"{o_color}{o['tokens']:<7}{RESET} {o_color}{o['ratio']:<7.2f}{RESET} "
                f"{diff_color}{rel:+7.1f}%{RESET}  {better}"
            )
