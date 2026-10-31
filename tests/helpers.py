from collections.abc import Iterable

from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from tinygpt.tokenizer import HuggingFaceTokenizer

SPECIAL_TOKENS = ["<bos>", "<eos>", "<user>", "<assistant>", "<system>"]
CHAT_TEMPLATE = "{{ bos_token }}{% for message in messages %}{{ '<' + message['role'] + '>' + message['content'] }}{% if message['role'] == 'assistant' %}{{ eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}"


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
    return HuggingFaceTokenizer(
        PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<bos>",
            eos_token="<eos>",
            additional_special_tokens=SPECIAL_TOKENS,
            chat_template=CHAT_TEMPLATE,
        )
    )
