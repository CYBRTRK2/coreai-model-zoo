#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""E5: transcript metrics for the Fun-ASR-Nano gates — one normalizer, WER (en) / CER (zh, ja, ko, yue).

``normalize(text)``: NFKC -> lowercase -> every character whose Unicode category is P* (punctuation;
after NFKC this covers full-width and CJK punctuation such as ``，。、「」``) or S* (symbols) becomes a
space -> runs of whitespace collapse to one space -> strip. Punctuation becomes a space rather than
nothing so that hyphenated and comma-joined words split the same way on both sides
(FLEURS writes ``well rounded`` where the model writes ``well-rounded``).

Units: en = whitespace-separated words (WER); zh / ja / ko / yue = characters with every
whitespace removed (CER). Rates are corpus-level: total Levenshtein edits / total reference units.
No number or spelling normalization (``2`` vs ``二``, ``10,000`` vs ``ten thousand`` count as errors).

Pure Python, no dependencies. As a script it scores the ``gate_e2e.py`` arms:
    python conversion/funasr_nano/metrics.py --arms enc16_int8hu enc16_fp16 enc32_int8hu enc32_fp16
-> ``logs/r2_metrics.json`` with (a) port vs oracle, (b) oracle vs FLEURS ``transcription``,
(c) port vs FLEURS ``transcription``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

NORMALIZER = ("NFKC -> lowercase -> Unicode category P* and S* characters (full-width/CJK punctuation "
              "included, being P* after NFKC) replaced by a space -> whitespace runs collapsed to one "
              "space -> strip. en: WER over whitespace words. zh/ja/ko/yue: CER over characters with all "
              "whitespace removed. Corpus rate = sum(Levenshtein edits) / sum(reference units). No number, "
              "spelling or script normalization.")
WORD_LANGS = {"en"}


def lang_of(name: str) -> str:
    """Fixture clip name -> language code (``en_us_1660`` -> en, ``cmn_hans_cn_*`` -> zh, examples as named)."""
    for prefix, lang in (("en_us_", "en"), ("cmn_hans_cn_", "zh"), ("ja_jp_", "ja")):
        if name.startswith(prefix):
            return lang
    return name   # the model-repo examples are named zh / en / ja / ko / yue


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    return re.sub(r"\s+", " ", text).strip()


def units(text: str, lang: str) -> list[str]:
    t = normalize(text)
    return t.split() if lang in WORD_LANGS else list(re.sub(r"\s+", "", t))


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def score(pairs: list[tuple[str, str, str]]) -> dict:
    """``pairs`` = [(lang, reference, hypothesis)] of one language -> corpus rate and exact matches."""
    edits = ref_units = exact = 0
    for lang, ref, hyp in pairs:
        r, h = units(ref, lang), units(hyp, lang)
        edits += edit_distance(r, h)
        ref_units += len(r)
        exact += r == h
    return {"n": len(pairs), "edits": edits, "ref_units": ref_units,
            "rate": edits / ref_units if ref_units else None, "normalized_exact": exact}


def by_language(rows: list[tuple[str, str, str]]) -> dict:
    """[(clip name, reference, hypothesis)] -> {lang: score}; metric = WER for en, CER otherwise."""
    groups: dict[str, list] = {}
    for name, ref, hyp in rows:
        lang = lang_of(name)
        groups.setdefault(lang, []).append((lang, ref, hyp))
    return {lang: {"metric": "WER" if lang in WORD_LANGS else "CER", **score(p)} for lang, p in sorted(groups.items())}


def _selftest() -> None:
    assert normalize("Hello, World!  It's  ＡＢＣ。") == "hello world it s abc"
    assert units("浪漫主义，具有 很强的。", "zh") == list("浪漫主义具有很强的")
    assert edit_distance(list("kitten"), list("sitting")) == 3
    assert score([("en", "a b c d", "a x c")])["edits"] == 2
    print("metrics selftest ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _paths import work_path

    work = work_path("_funasr_nano")
    meta = {c["name"]: c for c in json.loads((work / "fixtures" / "meta.json").read_text())["clips"]}
    oracle = {c["name"]: c["text"] for c in json.loads((work / "oracle" / "oracle.json").read_text())["clips"]}
    fleurs = {n: c["reference_text"]["transcription"] for n, c in meta.items() if c.get("reference_text")}

    samples = {}
    for prefix in ("en_us_", "cmn_hans_cn_", "ja_jp_"):
        names = sorted(n for n in fleurs if n.startswith(prefix))[:3]
        samples[prefix.rstrip("_")] = [{"name": n, "transcription": fleurs[n],
                                        "raw_transcription": meta[n]["reference_text"]["raw_transcription"],
                                        "oracle": oracle[n]} for n in names]
    out = {"normalizer": NORMALIZER, "fleurs_transcription_samples": samples,
           "oracle_vs_fleurs": by_language([(n, fleurs[n], oracle[n]) for n in sorted(fleurs)]),
           "arms": {}}
    for arm in args.arms:
        res = json.loads((work / "logs" / f"r2_gate_e2e_{arm}.json").read_text())
        port = {c["name"]: c["text"] for c in res["clips"]}
        exact_tokens = sum(c["exact"] for c in res["clips"])
        out["arms"][arm] = {
            "clips": len(port), "token_exact": exact_tokens,
            "text_exact": sum(port[n] == oracle[n] for n in port),
            "port_vs_oracle": by_language([(n, oracle[n], port[n]) for n in sorted(port)]),
            "port_vs_fleurs": by_language([(n, fleurs[n], port[n]) for n in sorted(port) if n in fleurs]),
        }
    path = Path(args.out) if args.out else work / "logs" / "r2_metrics.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "fleurs_transcription_samples"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
