#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""The Audio8-TTS prompt, built from the tokenizer alone — the host spec, asserted against the oracle's prompts.

`processing_arktts.py` builds the prompt from segments, each encoded with `add_special_tokens=False`:

  no reference:   "<|im_start|>system\\n" "convert the provided text to speech" "<|im_end|>\\n"
                  "<|im_start|>user\\n" <text> "<|im_end|>\\n" "<|im_start|>assistant\\n<|voice|>"
  voice clone:    prefix = "<|im_start|>system\\n" "convert the provided text to speech reference to the following:\\n\\nText:\\n"
                           <"<|speaker:0|>" + reference text> "\\n\\nSpeech:\\n"
                  then the reference's codebook-0 codes as semantic ids (151678 + code), then
                  suffix = "<|im_end|>\\n" "<|im_start|>user\\n" <text> "<|im_end|>\\n" "<|im_start|>assistant\\n<|voice|>"

Text is whitespace-normalised (`" ".join(text.split())`). The packed prompt is `[11, P]` int: row 0 the ids above,
rows 1..10 zero except under the reference codes, where they carry the reference's 10 codebooks (row 1 = codebook 0
= the same code the semantic id encodes). Segments are encoded one at a time — the BPE never sees across a boundary,
which is what makes `<|im_start|>system\\n` deterministic — and the Swift host has to encode the same segments.

    python prompt.py            # asserts the 18 oracle prompts are reproduced id for id
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
from _paths import hf_snapshot, work_path  # noqa: E402

HF_ID = "Edge0/Audio8-TTS-Preview-0.6b"
REVISION = "f07040f3d151f1ba0253bfb92cb2f5dd38b44594"
SEMANTIC_BEGIN = 151678
NUM_CODEBOOKS = 10

SYSTEM_PLAIN = "convert the provided text to speech"
SYSTEM_REF = "convert the provided text to speech reference to the following:\n\nText:\n"
SPEECH_TAG = "\n\nSpeech:\n"


def clean(text: str) -> str:
    return " ".join(str(text).strip().split())


class PromptBuilder:
    def __init__(self, tokenizer_json: Path):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(tokenizer_json))

    def enc(self, s: str) -> list[int]:
        return list(self.tok.encode(s, add_special_tokens=False).ids)

    def segments(self, text: str, reference_text: str | None):
        target = clean(text)
        if not target:
            raise ValueError("empty text")
        if reference_text is None:
            full = ["<|im_start|>system\n", SYSTEM_PLAIN, "<|im_end|>\n", "<|im_start|>user\n", target, "<|im_end|>\n",
                    "<|im_start|>assistant\n<|voice|>"]
            return full, []
        ref = clean(reference_text)
        if "<|speaker:" not in ref:
            ref = "<|speaker:0|>" + ref
        prefix = ["<|im_start|>system\n", SYSTEM_REF, ref, SPEECH_TAG]
        suffix = ["<|im_end|>\n", "<|im_start|>user\n", target, "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>"]
        return prefix, suffix

    def build(self, text: str, reference_text: str | None = None, reference_codes: np.ndarray | None = None) -> np.ndarray:
        """-> [11, P] int64."""
        prefix, suffix = self.segments(text, reference_text)
        pre = [t for s in prefix for t in self.enc(s)]
        suf = [t for s in suffix for t in self.enc(s)]
        if reference_codes is None:
            row0 = pre + suf
            out = np.zeros((NUM_CODEBOOKS + 1, len(row0)), np.int64)
            out[0] = row0
            return out
        codes = np.asarray(reference_codes, np.int64)
        assert codes.ndim == 2 and codes.shape[0] == NUM_CODEBOOKS and codes.shape[1] > 0
        sem = (codes[0] + SEMANTIC_BEGIN).tolist()
        row0 = pre + sem + suf
        out = np.zeros((NUM_CODEBOOKS + 1, len(row0)), np.int64)
        out[0] = row0
        out[1:, len(pre): len(pre) + codes.shape[1]] = codes
        return out


def main():
    snap = Path(hf_snapshot(HF_ID, revision=REVISION))
    pb = PromptBuilder(snap / "tokenizer.json")
    work = work_path("_audio8_tts")
    fx = json.loads((HERE / "fixtures.json").read_text())["fixtures"]
    meta = json.loads((work / "oracle" / "oracle.json").read_text())["fixtures"]
    ok = 0
    for f in fx:
        orc = np.load(work / "oracle" / f"{f['name']}.npz")
        info = meta[f["name"]]
        if info.get("reference"):
            got = pb.build(f["text"], info["reference"]["reference_text"], orc["ref_codes"])
        else:
            got = pb.build(f["text"])
        same = got.shape == orc["prompt"].shape and bool(np.array_equal(got, orc["prompt"]))
        ok += same
        print(f"[{f['name']}] P={got.shape[1]} {'== oracle' if same else '!= ORACLE'}")
        if not same:
            print("  got ", got[0, :60].tolist())
            print("  want", orc["prompt"][0, :60].tolist())
    print(f"prompt parity {ok}/{len(fx)}")
    sys.exit(0 if ok == len(fx) else 1)


if __name__ == "__main__":
    main()
