"""
SpellingBee and SimpleSpelling tasks.

SpellingBee: count occurrences of a letter in a word, solved with manual
enumeration + Python tool call verification.

SimpleSpelling: spell a word character by character.

Both tasks are procedurally generated from an English word list.
https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt
"""

from __future__ import annotations

import random
import re
from typing import Any

from tasks.base import Task
from tinygpt.utils import download_file_with_lock

LETTERS = "abcdefghijklmnopqrstuvwxyz"
WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"
TEST_RANDOM_SEED_OFFSET = 10_000_000

ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

USER_MSG_TEMPLATES = [
    "How many {letter} are in the word {word}",
    "How many {letter} are in {word}",
    "Count the number of {letter} in {word}",
    "How many times does {letter} appear in {word}",
    "What's the count of {letter} in {word}",
    "In the word {word}, how many {letter} are there",
    "How many letter {letter} are in the word {word}",
    "Count how many {letter} appear in {word}",
    "Tell me the number of {letter} in {word}",
    "How many occurrences of {letter} are in {word}",
    "Find the count of {letter} in {word}",
    "Can you count the {letter} letters in {word}",
    "What is the frequency of {letter} in {word}",
    "How many {letter}s are in {word}",
    "How many {letter}'s are in {word}",
    "Count all the {letter} in {word}",
    "How many times is {letter} in {word}",
    "Number of {letter} in {word}",
    "Total count of {letter} in {word}",
    "How many {letter} does {word} have",
    "How many {letter} does {word} contain",
    "What's the number of {letter} in {word}",
    "{word} has how many {letter}",
    "In {word}, count the {letter}",
    "How many {letter} appear in {word}",
    "Count the {letter} in {word}",
    "Give me the count of {letter} in {word}",
    "How many instances of {letter} in {word}",
    "Show me how many {letter} are in {word}",
    "Calculate the number of {letter} in {word}",
    # Spanish
    "¿Cuántas {letter} hay en {word}?",
    "¿Cuántas veces aparece {letter} en {word}?",
    "Cuenta las {letter} en {word}",
    "¿Cuántas letras {letter} tiene {word}?",
    # Chinese
    "{word}中有多少个{letter}",
    "{word}里有几个{letter}",
    "数一下{word}中的{letter}",
    "{word}这个词里有多少{letter}",
    # Korean
    "{word}에 {letter}가 몇 개 있나요",
    "{word}에서 {letter}의 개수는",
    "{word}에 {letter}가 몇 번 나오나요",
    "{word}라는 단어에 {letter}가 몇 개",
    # French
    "Combien de {letter} dans {word}",
    "Combien de fois {letter} apparaît dans {word}",
    "Compte les {letter} dans {word}",
    # German
    "Wie viele {letter} sind in {word}",
    "Wie oft kommt {letter} in {word} vor",
    "Zähle die {letter} in {word}",
    # Japanese
    "{word}に{letter}は何個ありますか",
    "{word}の中に{letter}がいくつ",
    "{word}に{letter}が何回出てくる",
]


def _load_word_list() -> list[str]:
    filename = WORD_LIST_URL.split("/")[-1]
    path = download_file_with_lock(WORD_LIST_URL, filename)
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _extract_answer(text: str) -> str | None:
    match = ANSWER_RE.search(text)
    if match:
        return match.group(1).strip().replace(",", "")
    return None


class SpellingBee(Task):
    """Count occurrences of a letter in a word, verified with Python.

    Args:
        size: Number of procedurally generated examples.
        split: Either 'train' or 'test' (different random seeds).
        **kwargs: Forwarded to Task.__init__ (start, stop, step).
    """

    def __init__(self, size: int = 1000, split: str = "train", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert split in ("train", "test"), "split must be train|test"
        self.size = size
        self.split = split
        self.words = _load_word_list()

    @property
    def eval_type(self) -> str:
        """Return 'generative' — answer extracted from free-form completion."""
        return "generative"

    def num_examples(self) -> int:
        """Return the configured number of examples.

        Returns:
            The size parameter passed at construction.
        """
        return self.size

    def get_example(self, index: int) -> dict[str, Any]:
        """Generate a SpellingBee problem deterministically from index.

        Args:
            index: Logical example index (seed for deterministic generation).

        Returns:
            Conversation dict with a multi-part assistant response.
        """
        seed = index if self.split == "train" else TEST_RANDOM_SEED_OFFSET + index
        rng = random.Random(seed)

        word = rng.choice(self.words)
        letter = rng.choice(word) if rng.random() < 0.9 else rng.choice(LETTERS)
        count = word.count(letter)

        template = rng.choice(USER_MSG_TEMPLATES)
        if rng.random() < 0.3:
            template = template.lower()
        q = rng.choice(["", "'", '"'])
        wq = rng.choice(["", "'", '"'])
        user_msg = template.format(letter=f"{q}{letter}{q}", word=f"{wq}{word}{wq}")
        if rng.random() < 0.5:
            user_msg += "?"

        word_letters = ",".join(list(word))
        manual = (
            f"We are asked to find the number '{letter}' in the word '{word}'. "
            f"Let me try a manual approach first.\n\n"
            f"First spell the word out:\n{word}:{word_letters}\n\n"
            f"Then count the occurrences of '{letter}':\n"
        )
        running = 0
        for i, char in enumerate(word, 1):
            if char == letter:
                running += 1
                manual += f"{i}:{char} hit! count={running}\n"
            else:
                manual += f"{i}:{char}\n"
        manual += f"\nThis gives us {running}."

        assistant_parts: list[dict[str, str]] = [
            {"type": "text", "text": manual},
            {"type": "text", "text": "\n\nLet me double check this using Python:\n\n"},
            {"type": "python", "text": f"'{word}'.count('{letter}')"},
            {"type": "python_output", "text": str(count)},
            {"type": "text", "text": f"\n\nPython gives us {count}.\n\nMy final answer is:\n\n#### {count}"},
        ]
        return {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_parts},
            ],
        }

    def evaluate(self, problem: dict[str, Any], completion: str) -> bool:
        """Check whether the predicted count matches the ground-truth answer.

        Args:
            problem: Conversation dict from get_example.
            completion: Model's free-form response string.

        Returns:
            True if the extracted answer matches the ground-truth count.
        """
        assert isinstance(completion, str)
        last_part = problem["messages"][-1]["content"][-1]["text"]
        ref = _extract_answer(last_part)
        pred = _extract_answer(completion)
        return bool(pred is not None and pred == ref)

    def reward(self, problem: dict[str, Any], completion: str) -> float:
        """Return 1.0 if correct, 0.0 otherwise.

        Args:
            problem: Conversation dict from get_example.
            completion: Model's free-form response string.

        Returns:
            Float reward signal.
        """
        return float(self.evaluate(problem, completion))


class SimpleSpelling(Task):
    """Spell a word character by character.

    Simpler companion to SpellingBee for practising token-to-character mapping.

    Args:
        size: Number of procedurally generated examples.
        split: Either 'train' or 'test'.
        **kwargs: Forwarded to Task.__init__ (start, stop, step).
    """

    def __init__(self, size: int = 1000, split: str = "train", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert split in ("train", "test"), "split must be train|test"
        self.size = size
        self.split = split
        words = _load_word_list()
        rng = random.Random(42)
        rng.shuffle(words)
        self.words = words

    @property
    def eval_type(self) -> str:
        """Return 'generative'."""
        return "generative"

    def num_examples(self) -> int:
        """Return the configured number of examples.

        Returns:
            The size parameter passed at construction.
        """
        return self.size

    def get_example(self, index: int) -> dict[str, Any]:
        """Generate a SimpleSpelling example deterministically from index.

        Args:
            index: Logical example index.

        Returns:
            Conversation dict with a comma-separated letter spelling.
        """
        seed = index if self.split == "train" else TEST_RANDOM_SEED_OFFSET + index
        rng = random.Random(seed)
        word = rng.choice(self.words)
        word_letters = ",".join(list(word))
        return {
            "messages": [
                {"role": "user", "content": f"Spell the word: {word}"},
                {"role": "assistant", "content": f"{word}:{word_letters}"},
            ],
        }
