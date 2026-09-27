#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""ASR round trip for the Audio8-TTS gate: transcribe wavs with the fp32 Fun-ASR-Nano oracle, score WER / CER.

Tensor cosine passes on unintelligible audio (pocket-tts-port.md), so the TTS gate ends in a transcript: every
generated wav — the oracle's and the port's — is transcribed by the same fp32 ASR (the publisher's `funasr` 1.4.16
running FunAudioLLM/Fun-ASR-Nano-2512 on the CPU, the oracle of the zoo's Fun-ASR port), and scored against the
fixture text with the zoo's one normalizer (`conversion/funasr_nano/metrics.py`: NFKC, lowercase, punctuation to
space; en = WER over words, ja / zh = CER over characters).

Runs in the Fun-ASR oracle venv (`$ZOO_WORK_ROOT/_funasr_nano/venv-oracle`), which the gate calls as a subprocess:

    <venv-oracle>/bin/python asr_judge.py --manifest gate/asr_manifest.json --out gate/asr.json

manifest = [{"id": ..., "wav": path, "lang": "ja|en|zh", "text": reference}] ; output adds "hyp" per row and the
per-language corpus WER / CER.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE.parents[0] / "funasr_nano"))
from _paths import work_path  # noqa: E402
from metrics import score  # noqa: E402

OFFICIAL = work_path("_funasr_nano", "official")


def load_16k(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != 16000:
        from torchaudio.functional import resample

        audio = resample(torch.from_numpy(audio), sr, 16000).numpy()
    return audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ncpu", type=int, default=8)
    args = ap.parse_args()
    rows = json.loads(Path(args.manifest).read_text())

    from funasr import AutoModel

    torch.set_num_threads(args.ncpu)
    am = AutoModel(model=str(OFFICIAL), device="cpu", disable_update=True, hub="hf", disable_pbar=True,
                   disable_log=True, ncpu=args.ncpu, frontend_conf={"dither": 0.0})
    import tempfile

    lang_name = {"ja": "日文", "en": "英文", "zh": "中文"}
    tmp = Path(tempfile.mkdtemp(prefix="audio8_asr_"))
    for i, r in enumerate(rows):
        audio = load_16k(r["wav"])
        r["seconds"] = round(audio.size / 16000.0, 2)
        if audio.size < 1600:
            r["hyp"] = ""
            continue
        wav16 = tmp / f"{i:03d}.wav"
        sf.write(wav16, audio, 16000, subtype="PCM_16")
        # the zoo's Fun-ASR oracle call (conversion/funasr_nano/make_oracle.py RUNTIME) + the language hint
        res = am.generate(input=[str(wav16)], cache={}, batch_size=1, language=lang_name.get(r["lang"]), itn=True,
                          hotwords=[], llm_kwargs={"do_sample": False, "num_beams": 1})
        r["hyp"] = res[0]["text"] if res else ""
    groups: dict[str, dict[str, list]] = {}
    for r in rows:
        groups.setdefault(r["lang"], {}).setdefault(r.get("arm", "all"), []).append((r["lang"], r["text"], r["hyp"]))
    summary = {lang: {arm: score(pairs) for arm, pairs in arms.items()} for lang, arms in groups.items()}
    Path(args.out).write_text(json.dumps({"asr": "FunAudioLLM/Fun-ASR-Nano-2512 fp32 cpu (funasr 1.4.16), dither 0, itn on, language forced",
                                          "rows": rows, "summary": summary}, ensure_ascii=False, indent=1))
    for lang, arms in summary.items():
        for arm, s in arms.items():
            print(f"[asr] {lang} {arm}: n {s['n']} rate {s['rate']:.4f} exact {s['normalized_exact']}/{s['n']}")


if __name__ == "__main__":
    main()
