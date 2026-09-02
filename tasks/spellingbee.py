"""
Spelling/counting tasks used to strengthen small-model distillation.

1. SimpleSpelling: spell a word character-by-character
2. SpellingBee: count occurrences of a letter in a word, including Python verification
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


def extract_answer(completion: str) -> str | None:
    """Extract the numerical answer after the #### marker."""
    match = ANSWER_RE.search(completion)
    if match:
        return match.group(1).strip().replace(",", "")
    return None


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
    "¿Cuántas {letter} hay en {word}?",
    "¿Cuántas veces aparece {letter} en {word}?",
    "Cuenta las {letter} en {word}",
    "¿Cuántas letras {letter} tiene {word}?",
    "{word}中有多少个{letter}",
    "{word}里有几个{letter}",
    "数一下{word}中的{letter}",
    "{word}这个词里有多少{letter}",
    "{word}에 {letter}가 몇 개 있나요",
    "{word}에서 {letter}의 개수는",
    "{word}에 {letter}가 몇 번 나오나요",
    "{word}라는 단어에 {letter}가 몇 개",
    "Combien de {letter} dans {word}",
    "Combien de fois {letter} apparaît dans {word}",
    "Compte les {letter} dans {word}",
    "Wie viele {letter} sind in {word}",
    "Wie oft kommt {letter} in {word} vor",
    "Zähle die {letter} in {word}",
    "{word}に{letter}は何個ありますか",
    "{word}の中に{letter}がいくつ",
    "{word}に{letter}が何回出てくる",
]


def _load_words() -> list[str]:
    filename = WORD_LIST_URL.rsplit("/", 1)[-1]
    word_list_path = download_file_with_lock(WORD_LIST_URL, filename)
    with open(word_list_path, encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    if not words:
        raise ValueError(f"Word list is empty: {word_list_path}")
    return words


class SpellingBee(Task):
    """Count occurrences of a letter in a word."""

    def __init__(self, size: int = 1000, split: str = "train", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if size <= 0:
            raise ValueError(f"SpellingBee size must be positive, got {size}")
        if split not in ("train", "test"):
            raise ValueError("SpellingBee split must be train|test")
        self.size = size
        self.split = split
        self.words = _load_words()

    @property
    def eval_type(self) -> str:
        return "generative"

    def num_examples(self) -> int:
        return self.size

    def get_example(self, index: int) -> dict[str, Any]:
        seed = index if self.split == "train" else TEST_RANDOM_SEED_OFFSET + index
        rng = random.Random(seed)

        word = rng.choice(self.words)
        letter = rng.choice(word) if rng.random() < 0.9 else rng.choice(LETTERS)
        count = word.count(letter)

        template = rng.choice(USER_MSG_TEMPLATES)
        if rng.random() < 0.3:
            template = template.lower()
        quote_options = ["", "'", '"']
        letter_quote = rng.choice(quote_options)
        word_quote = rng.choice(quote_options)
        user_msg = template.format(
            letter=f"{letter_quote}{letter}{letter_quote}",
            word=f"{word_quote}{word}{word_quote}",
        )
        if rng.random() < 0.5:
            user_msg += "?"

        word_letters = ",".join(list(word))
        manual_text = (
            f"We are asked to find the number '{letter}' in the word '{word}'. "
            "Let me try a manual approach first.\n\n"
            f"First spell the word out:\n{word}:{word_letters}\n\n"
            f"Then count the occurrences of '{letter}':\n"
        )
        running_count = 0
        for i, char in enumerate(word, 1):
            if char == letter:
                running_count += 1
                manual_text += f"{i}:{char} hit! count={running_count}\n"
            else:
                manual_text += f"{i}:{char}\n"
        manual_text += f"\nThis gives us {running_count}."

        assistant_parts = [
            {"type": "text", "text": manual_text},
            {"type": "text", "text": "\n\nLet me double check this using Python:\n\n"},
            {"type": "python", "text": f"'{word}'.count('{letter}')"},
            {"type": "python_output", "text": str(count)},
            {"type": "text", "text": f"\n\nPython gives us {count}.\n\nMy final answer is:\n\n#### {count}"},
        ]
        return {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_parts},
            ]
        }

    def evaluate(self, conversation: dict[str, Any], assistant_response: str) -> bool:
        assistant_message = conversation["messages"][-1]
        if assistant_message["role"] != "assistant" or not isinstance(assistant_message["content"], list):
            return False
        last_text_part = assistant_message["content"][-1]["text"]
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        return pred_num == ref_num

    def reward(self, conversation: dict[str, Any], assistant_response: str) -> float:
        return float(self.evaluate(conversation, assistant_response))


class SimpleSpelling(Task):
    """Spell words character-by-character."""

    def __init__(self, size: int = 1000, split: str = "train", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if size <= 0:
            raise ValueError(f"SimpleSpelling size must be positive, got {size}")
        if split not in ("train", "test"):
            raise ValueError("SimpleSpelling split must be train|test")
        self.size = size
        self.split = split
        self.words = _load_words()
        rng = random.Random(42)
        rng.shuffle(self.words)

    @property
    def eval_type(self) -> str:
        return "generative"

    def num_examples(self) -> int:
        return self.size

    def get_example(self, index: int) -> dict[str, Any]:
        seed = index if self.split == "train" else TEST_RANDOM_SEED_OFFSET + index
        rng = random.Random(seed)
        word = rng.choice(self.words)
        word_letters = ",".join(list(word))
        return {
            "messages": [
                {"role": "user", "content": f"Spell the word: {word}"},
                {"role": "assistant", "content": f"{word}:{word_letters}"},
            ]
        }
