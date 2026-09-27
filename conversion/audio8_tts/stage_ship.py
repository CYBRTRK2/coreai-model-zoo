#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""Lay out the Hugging Face repository for the Audio8-TTS Core AI port in `$ZOO_WORK_ROOT/_audio8_tts/ship/` (no upload).

    ship/
      audio8_dualar_int8_cl2048_w32.aimodel/       prefill + frame + first_frame: the slow AR (int8 linears, fp16 tables,
                                                   KV 2048 slots), the sampler and the fast AR (fp16) in one asset
      audio8_codec_decoder_fp16_t160.aimodel/      codec decoder, 160-frame bucket (7.4 s), fp16
      audio8_codec_encoder_fp16_t216.aimodel/      codec encoder for voice registration, 216-frame bucket (10 s), fp16 (optional)
      tokenizer/                                   tokenizer.json, tokenizer_config.json (class retagged Qwen2Tokenizer), special_tokens_map.json
      metadata.json                                graph contracts, sampling defaults, source revision, sha256 of every model file
      config.json                                  the checkpoint's config.json, verbatim
      LICENSE  NOTICE                              Apache-2.0, the upstream repository's files
      README.md                                    the zoo card (models/audio8-tts/README.md)

The JIT `.aimodel`s serve macOS and iOS alike (the iPhone 18 Pro specializes them on first load).
    ~/code/coreai/coreai-models/.venv/bin/python conversion/audio8_tts/stage_ship.py [--with-encoder]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
from _paths import hf_snapshot, repo_root, work_path  # noqa: E402

HF_ID = "Edge0/Audio8-TTS-Preview-0.6b"
REVISION = "f07040f3d151f1ba0253bfb92cb2f5dd38b44594"
WORK = work_path("_audio8_tts")
DUALAR = "audio8_dualar_int8_cl2048_w32"
CODEC = "audio8_codec_decoder_fp16_t160"
ENCODER = "audio8_codec_encoder_fp16_t216"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-encoder", action="store_true")
    ap.add_argument("--out", default=str(WORK / "ship"))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    snap = Path(hf_snapshot(HF_ID, revision=REVISION))
    exports = WORK / "exports"
    bundles = [DUALAR, CODEC] + ([ENCODER] if args.with_encoder else [])
    files = {}
    for b in bundles:
        src = exports / f"{b}.aimodel"
        assert src.is_dir(), src
        shutil.copytree(src, out / f"{b}.aimodel")
        for f in sorted((out / f"{b}.aimodel").rglob("*")):
            if f.is_file():
                files[str(f.relative_to(out))] = {"sha256": sha256(f), "bytes": f.stat().st_size}
    tok = out / "tokenizer"
    tok.mkdir()
    shutil.copyfile(snap / "tokenizer.json", tok / "tokenizer.json")
    shutil.copyfile(snap / "special_tokens_map.json", tok / "special_tokens_map.json")
    tc = json.loads((snap / "tokenizer_config.json").read_text())
    tc["tokenizer_class"] = "Qwen2Tokenizer"          # swift-transformers rejects PreTrainedTokenizerFast; tokenizer.json drives encode/decode
    (tok / "tokenizer_config.json").write_text(json.dumps(tc, ensure_ascii=False, indent=1))
    shutil.copyfile(snap / "config.json", out / "config.json")
    for name in ("LICENSE", "NOTICE"):
        shutil.copyfile(WORK / "ship_src" / name, out / name)
    card = repo_root() / "models" / "audio8-tts" / "README.md"
    if card.exists():
        shutil.copyfile(card, out / "README.md")
    meta = {
        "metadata_version": "0.2",
        "kind": "tts",
        "name": "Audio8-TTS-Preview-0.6b-CoreAI",
        "source": {"model_definition": "torch", "hf_model_id": HF_ID, "hf_revision": REVISION,
                   "model_safetensors_sha256": sha256(snap / "model.safetensors"), "codec_pth_sha256": sha256(snap / "codec.pth"),
                   "license": "apache-2.0"},
        "audio": {"sample_rate": 44100, "frame_samples": 2048, "frames_per_second": 44100 / 2048},
        "graphs": {
            "dualar": {"asset": f"{DUALAR}.aimodel", "functions": {
                "prefill": {"inputs": {"codes": "[1, 11, 32] int32 (row 0 ids, rows 1-10 codebooks; pad rows: 151643 / 0)", "pos": "[1] int32 first position"},
                            "outputs": {"logits": "[32, 4097] float16 (semantic 0..4095 then eos)", "hidden": "[32, 896] float16"}},
                "frame": {"inputs": {"codes": "[1, 11, 1] int32 (the previous frame: semantic id, 10 codebooks)", "pos": "[1] int32 (its position)",
                                     "noise_slow": "[2, 4097] float32 uniform draws (normal branch, RAS-high branch)", "window": "[10] int32 RAS window (-1 = none)",
                                     "noise_fast": "[9, 4096] float32 uniform draws", "forced": "[11] int32 (teacher forcing)", "use_forced": "[1] float32 0/1"},
                          "outputs": {"semantic": "[1] int32 token id (151645 = eos)", "codes": "[10] int32 codebooks 0..9", "logits": "[4097] float16",
                                      "hidden": "[896] float16", "fast_logits": "[9, 4096] float16", "sampled_semantic": "[1] int32", "sampled_codes": "[10] int32"}},
                "first_frame": {"inputs": {"logits": "[4097] float16 (the prefill's last row)", "hidden": "[896] float16", "noise_slow": "…", "window": "…",
                                           "noise_fast": "…", "forced": "…", "use_forced": "…"},
                                "outputs": "as frame"}},
                "state": {"k_cache": "[24, 1, 2, 2048, 64] float16", "v_cache": "[24, 1, 2, 2048, 64] float16 (prefill and frame)"},
                "compression": "slow AR: int8 weight-only, symmetric with clipping, per-block-32 (input axis) on the 24 layers' linears; embeddings, codebook embeddings, the 4,097-row head, norms, the fast AR (4 layers) float16"},
            "codec_decoder": {"asset": f"{CODEC}.aimodel", "functions": {
                "main": {"inputs": {"codes": "[1, 10, 160] int32 (right-pad with 0; codebook 0 < 4096, codebooks 1-9 < 1024, clamped in-graph)"},
                         "outputs": {"wav": "[1, 327680] float16, 44.1 kHz"}}},
                "note": "every op is causal: a frame's samples depend on at most the 127 frames before it (the codec transformer's window) — decode a stream in 160-frame windows and keep the last 32", "compression": None},
        },
        "sampling": {"top_k": 50, "top_p": 0.9, "temperature": 0.7, "max_new_tokens": 512,
                     "ras": {"window": 10, "top_p": 0.9, "temperature": 1.0},
                     "rule": "argmax(softmax(processed) / -log(u)) with u ~ U(0,1); processed = top-k/top-p mask then / temperature; a semantic id repeated within the 10-frame window is replaced by the (0.9, 1.0) draw",
                     "eos": 151645, "semantic_begin": 151678, "semantic_end": 155773, "pad": 151643},
        "prompt": {"spec": "conversion/audio8_tts/prompt.py (segments encoded one at a time with the tokenizer, no special tokens added)",
                   "tokenizer": {"files": "tokenizer/", "class_retag": "PreTrainedTokenizerFast -> Qwen2Tokenizer for swift-transformers"}},
        "languages": ["yue", "zh", "nl", "en", "fr", "de", "it", "ja", "ko", "pl", "es"],
        "host": {"swift": "CoreAIKit Audio8TTS (Sources/CoreAIKit/Audio8TTS)", "python_gate": "conversion/audio8_tts/gate_audio8.py"},
        "files": files,
        "compilation": {"date": datetime.now(timezone.utc).isoformat(), "targets": ["macos", "ios (jit)"]},
    }
    if args.with_encoder:
        meta["graphs"]["codec_encoder"] = {"asset": f"{ENCODER}.aimodel", "functions": {
            "main": {"inputs": {"audio": "[1, 1, 442368] float32 mono 44.1 kHz, right-pad with 0 (216 frames = 10.03 s)"},
                     "outputs": {"codes": "[1, 10, 216] int32; keep the first ceil(samples / 2048) frames"}}},
            "note": "voice registration (zero-shot cloning): codes + the exact transcript make an Audio8Voice", "compression": None}
    (out / "metadata.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    total = tree_bytes(out)
    print(f"[ship] {out}: {len(files)} model files, {total / 1e6:.0f} MB total")
    for b in bundles:
        print(f"  {b}.aimodel  {tree_bytes(out / (b + '.aimodel')) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
