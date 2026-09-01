from collections.abc import Iterable

from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from tinygpt.tokenizer import SPECIAL_TOKENS, HuggingFaceTokenizer


def make_test_tokenizer(texts: Iterable[str], vocab_size: int = 512) -> HuggingFaceTokenizer:
    tokenizer = Tokenizer(BPE(byte_fallback=True, unk_token=None, fuse_unk=False))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        texts,
        BpeTrainer(
            vocab_size=vocab_size,
            show_progress=False,
            min_frequency=0,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        ),
    )
    return HuggingFaceTokenizer(tokenizer)
