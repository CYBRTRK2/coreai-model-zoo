# Community port — NOT an Apple model.
"""Fun-ASR-Nano prompt contract: token ids around the audio slots, and the stop set.

funasr 1.4.16 (``FunASRNano.get_prompt`` / ``generate_chatml`` / ``data_load_speech``, language None,
itn True, no hotwords) builds the ChatML turn below and puts ``N = fake_token_len(L)`` placeholder
ids (0) where the ``<|startofspeech|>...<|endofspeech|>`` span was; ``inference_prepare`` then
overwrites those rows of ``inputs_embeds`` with the adaptor output ``[:N]``. No marker token
surrounds the audio. The port writes the placeholders as ``V + slot`` (``V`` = vocab size, slot
0..N-1) so the decoder graph can ``index_select`` row ``slot`` of its ``audio_embeds`` input:

    ids = enc(PREFIX_TEXT) + [V + 0, ..., V + N - 1] + enc(SUFFIX_TEXT)      (18 + N + 5 tokens)

``enc`` is the official ``Qwen3-0.6B/`` tokenizer (Qwen2Tokenizer; it adds no BOS, so
``add_special_tokens`` makes no difference — checked in ``parity_decoder.py``).
Generation stops on either id in ``EOS_IDS``.

``get_prompt``'s options change only the user text before the audio (``user_text``): hotwords put a
context preamble and the list first, ``language`` turns ``语音转写`` into ``语音转写成{language}``,
``itn=False`` appends ``，不进行文本规整``, and the text always ends with ``：``. funasr tokenizes the
whole text before the audio in one ``encode`` call, so the variants keep the ``enc(prefix)`` shape.
"""
from __future__ import annotations

from collections.abc import Sequence

V = 151936
SYSTEM_TURN = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
HOTWORD_PREAMBLE = "请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n"
SUFFIX_TEXT = "<|im_end|>\n<|im_start|>assistant\n"
EOS_IDS = (151645, 151643)   # <|im_end|>, <|endoftext|>
MAX_NEW_TOKENS = 512


def user_text(hotwords: Sequence[str] = (), language: str | None = None, itn: bool = True) -> str:
    """``FunASRNano.get_prompt``: the user text that precedes the audio."""
    text = HOTWORD_PREAMBLE + f"热词列表：[{', '.join(hotwords)}]\n" if hotwords else ""
    text += "语音转写" if language is None else f"语音转写成{language}"
    if not itn:
        text += "，不进行文本规整"
    return text + "："


def prefix_text(hotwords: Sequence[str] = (), language: str | None = None, itn: bool = True) -> str:
    return SYSTEM_TURN + user_text(hotwords, language, itn)


PREFIX_TEXT = prefix_text()
assert PREFIX_TEXT == "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n语音转写："


def prompt_segments(tokenizer, hotwords: Sequence[str] = (), language: str | None = None,
                    itn: bool = True) -> tuple[list[int], list[int]]:
    """Token ids of the text before and after the audio slots."""
    prefix = tokenizer(prefix_text(hotwords, language, itn), add_special_tokens=False).input_ids
    suffix = tokenizer(SUFFIX_TEXT, add_special_tokens=False).input_ids
    return list(prefix), list(suffix)


def build_prompt_ids(tokenizer, n_audio: int, hotwords: Sequence[str] = (), language: str | None = None,
                     itn: bool = True) -> tuple[list[int], int]:
    """``(ids, prefix_len)``: the full prompt with the audio slots as ``V + slot``."""
    prefix, suffix = prompt_segments(tokenizer, hotwords, language, itn)
    return prefix + [V + slot for slot in range(n_audio)] + suffix, len(prefix)
