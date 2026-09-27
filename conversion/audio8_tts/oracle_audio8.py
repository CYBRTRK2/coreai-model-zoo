#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""Audio8-TTS-Preview-0.6b fp32 oracle: the publisher's own remote code, run on the CPU, every step recorded.

The oracle is the publisher's `modeling_arktts.py` (HF `trust_remote_code`) loaded in float32 on the CPU
with its `codec.pth`. The generation loop below re-implements `ArkttsModel.generate` for batch 1 call for
call — the same `_slow_step` / `_fast_step`, the same top-k/top-p/temperature processor, the same Gumbel
draw (`argmax(softmax(scores) / -log(u))`), the same RAS repetition rule — so that every quantity the
port has to reproduce is on disk:

  per fixture `oracle/<name>.npz`
    prompt            [11, P] int64   the packed prompt (row 0 text/semantic ids, rows 1..10 codebooks)
    prefill_logits    [4097]  f32     slow logits at the last prompt position, layout semantic(4096) + eos
    prefill_hidden    [896]   f32     the slow hidden handed to the fast AR (post final RMSNorm)
    slow_logits       [T, 4097] f32   per generated frame (the frame whose semantic was sampled from it)
    slow_hidden       [T, 896] f32
    slow_argmax_full  [T] int64       argmax over the full 155,776 vocab (is it always inside the allowed set?)
    fast_logits       [T, 9, 4096] f32  fast-AR logits for codebooks 1..9
    noise_slow        [T, 2, 4097] f32  the uniform draws at the allowed ids: normal branch, RAS-high branch
    noise_fast        [T, 9, 4096] f32  the uniform draws of the nine codebook samples
    semantic          [T] int64       the sampled semantic id (eos = 151645 ends the run, no frame emitted)
    codes             [10, T_emit] int64  emitted frames (codebook 0 = semantic - 151678)
    wav               [N] f32         codec decode of `codes`, 44.1 kHz
    ref_codes         [10, Tr] int64  (voice-clone fixtures) codec encode of the reference clip
    ref_wav           [Nr] f32        (voice-clone fixtures) the reference at 44.1 kHz as the processor made it

  `oracle/oracle.json`: config digest, per-fixture summary (frames, seconds, eos step, codebook ranges),
  fp16 headroom = abs-max of every transformer block's residual output and of the codec decoder blocks.

`--check-generate` runs the publisher's `model.generate` with the same seed on the first fixture and asserts
the codes are identical: the proof that the re-implemented loop *is* the publisher's loop.

Run with the shared venv (transformers 4.57.x is what the remote code targets):
    ~/code/coreai/coreai-models/.venv/bin/python conversion/audio8_tts/oracle_audio8.py --check-generate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
from _paths import hf_snapshot, work_path  # noqa: E402

HF_ID = "Edge0/Audio8-TTS-Preview-0.6b"
REVISION = "f07040f3d151f1ba0253bfb92cb2f5dd38b44594"
WORK = work_path("_audio8_tts")
FUNASR_FIXTURES = work_path("_funasr_nano", "fixtures")   # FLEURS clips (CC BY 4.0) reused as reference voices


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(snap: str):
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(snap, trust_remote_code=True)
    model = AutoModel.from_pretrained(snap, trust_remote_code=True, dtype=torch.float32).eval()
    return processor, model


# --------------------------------------------------------------------------- sampling (mirrors modeling_arktts)
def legacy_top_k_top_p(scores: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
    cumulative = torch.cumsum(torch.softmax(sorted_scores, dim=-1), dim=-1)
    positions = torch.arange(sorted_scores.shape[-1], device=scores.device)
    threshold = torch.tensor(top_p, dtype=cumulative.dtype, device=cumulative.device)
    remove_sorted = (cumulative > threshold) | (positions >= top_k)
    remove_sorted[..., 0] = False
    remove = torch.zeros_like(remove_sorted).scatter(1, sorted_indices, remove_sorted)
    return scores.masked_fill(remove, float("-inf"))


def processed_scores(scores: torch.Tensor, top_k: int, top_p: float, temperature: float) -> torch.Tensor:
    scores = legacy_top_k_top_p(scores, top_k, top_p)
    temperature_value = torch.tensor(temperature, dtype=scores.dtype, device=scores.device).clamp_min(1e-5)
    return scores / temperature_value


def gumbel_sample(scores: torch.Tensor, generator: torch.Generator):
    """`ArkttsModel._sample`, returning the uniform draw too."""
    probabilities = torch.softmax(scores, dim=-1)
    random = torch.rand(probabilities.shape, dtype=probabilities.dtype, device=probabilities.device,
                        generator=generator)
    noise = -torch.log(random)
    return torch.argmax(probabilities / noise, dim=-1), random


class Recorder:
    """Forward hooks: abs-max of block outputs (the fp16 headroom evidence)."""

    def __init__(self):
        self.absmax: dict[str, float] = {}

    def hook(self, name: str):
        def fn(_module, _inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            v = float(t.detach().abs().max())
            if v > self.absmax.get(name, 0.0):
                self.absmax[name] = v
        return fn

    def attach(self, model, codec):
        handles = []
        for i, layer in enumerate(model.layers):
            handles.append(layer.register_forward_hook(self.hook(f"slow.block{i}")))
            handles.append(layer.attention.wqkv.register_forward_hook(self.hook(f"slow.wqkv{i}")))
            handles.append(layer.feed_forward.w2.register_forward_hook(self.hook(f"slow.w2_{i}")))
        for i, layer in enumerate(model.fast_layers):
            handles.append(layer.register_forward_hook(self.hook(f"fast.block{i}")))
            handles.append(layer.attention.wqkv.register_forward_hook(self.hook(f"fast.wqkv{i}")))
        handles.append(model.norm.register_forward_hook(self.hook("slow.norm_out")))
        handles.append(model.fast_norm.register_forward_hook(self.hook("fast.norm_out")))
        q = codec.quantizer
        handles.append(q.post_module.register_forward_hook(self.hook("codec.post_module")))
        for i, layer in enumerate(q.post_module.layers):
            handles.append(layer.register_forward_hook(self.hook(f"codec.post.block{i}")))
        for i, m in enumerate(q.upsample):
            handles.append(m.register_forward_hook(self.hook(f"codec.upsample{i}")))
        for i, m in enumerate(codec.decoder.model):
            handles.append(m.register_forward_hook(self.hook(f"codec.decoder.{i}.{type(m).__name__}")))
        return handles


@torch.inference_mode()
def generate_recorded(model, prompt: torch.Tensor, prompt_mask: torch.Tensor, gen_cfg: dict, seed: int):
    """Batch-1 re-implementation of `ArkttsModel.generate` that records every tensor the port must match."""
    cfg = model.config
    begin, end, eos = cfg.semantic_begin_id, cfg.semantic_end_id, cfg.eos_token_id
    allowed = torch.cat([torch.arange(begin, end + 1), torch.tensor([eos])])
    top_k, top_p, temperature = int(gen_cfg["top_k"]), float(gen_cfg["top_p"]), float(gen_cfg["temperature"])
    max_new = int(gen_cfg["max_new_tokens"])
    generator = torch.Generator(device="cpu").manual_seed(seed)

    _, _, P = prompt.shape
    assert P < cfg.max_seq_len
    max_new = min(max_new, cfg.max_seq_len - P)
    model._setup_generation_caches(1, P + max_new, torch.float32)

    cache_position = torch.arange(P, dtype=torch.long)
    position_ids = prompt_mask.cumsum(-1).sub(1).clamp_min(0)
    logits, slow_hidden = model._slow_step(prompt, cache_position, position_ids, prompt_mask)
    rec = {
        "prefill_logits": logits[0, allowed].clone().numpy(),
        "prefill_hidden": slow_hidden[0, -1].clone().numpy(),
        "slow_logits": [], "slow_hidden": [], "slow_argmax_full": [], "fast_logits": [],
        "noise_slow": [], "noise_fast": [], "semantic": [], "frames": [],
    }
    previous = None
    prompt_len = int(prompt_mask.sum())
    filtered_template = torch.full_like(logits, float("-inf"))

    for step in range(max_new):
        # ---- semantic token (ArkttsModel._sample_semantic) ------------------------------------
        filtered = filtered_template.clone()
        filtered[:, begin:end + 1] = logits[:, begin:end + 1]
        filtered[:, eos] = logits[:, eos]
        regular = processed_scores(filtered, top_k, top_p, temperature)
        normal, u_normal = gumbel_sample(regular, generator)
        high_scores = processed_scores(filtered, top_k, cfg.ras_top_p, cfg.ras_temperature)
        high, u_high = gumbel_sample(high_scores, generator)
        if previous is None:
            semantic = normal
        else:
            repeated = (previous == normal[:, None]).any(dim=1)
            is_sem = (normal >= begin) & (normal <= end)
            semantic = torch.where(repeated & is_sem, high, normal)
        rec["slow_logits"].append(logits[0, allowed].clone().numpy())
        rec["slow_hidden"].append(slow_hidden[0, -1].clone().numpy())
        rec["slow_argmax_full"].append(int(logits[0].argmax()))
        rec["noise_slow"].append(np.stack([u_normal[0, allowed].numpy(), u_high[0, allowed].numpy()]))
        rec["semantic"].append(int(semantic))

        # ---- codebooks (ArkttsModel._generate_codebooks) ---------------------------------------
        hidden = model.fast_project_in(slow_hidden)
        model._fast_step(hidden, 0)
        current = (semantic - begin).clamp(0, cfg.codebook_size - 1)
        codebooks = [current]
        hidden = model.fast_embeddings(current)[:, None]
        fl, nf = [], []
        for position in range(1, cfg.num_codebooks):
            scores = model._fast_step(hidden, position)
            fl.append(scores[0].clone().numpy().astype(np.float32))
            scores = processed_scores(scores, top_k, top_p, temperature)
            current, u = gumbel_sample(scores, generator)
            nf.append(u[0].numpy())
            codebooks.append(current)
            hidden = model.fast_embeddings(current)[:, None]
        codebooks = torch.stack(codebooks, dim=1)            # [1, 10]
        rec["fast_logits"].append(np.stack(fl))
        rec["noise_fast"].append(np.stack(nf))

        if int(semantic) == eos:
            break
        rec["frames"].append(codebooks[0].clone().numpy())
        if previous is None:
            previous = torch.zeros((1, cfg.ras_window_size), dtype=torch.long)
        else:
            previous = previous.roll(-1, dims=1)
            previous[:, -1] = semantic
        if step + 1 >= max_new:
            break
        next_column = torch.cat((semantic[:, None], codebooks), dim=1).unsqueeze(-1)   # [1, 11, 1]
        prompt_mask = torch.cat((prompt_mask, torch.ones((1, 1), dtype=torch.long)), dim=1)
        physical_position = torch.tensor([P + step])
        token_position = torch.tensor([[prompt_len + step]])
        with sdpa_kernel(SDPBackend.MATH):
            logits, slow_hidden = model._slow_step(next_column, physical_position, token_position, prompt_mask)

    out = {
        "prompt": prompt[0].numpy(),
        "prefill_logits": rec["prefill_logits"],
        "prefill_hidden": rec["prefill_hidden"],
        "slow_logits": np.stack(rec["slow_logits"]),
        "slow_hidden": np.stack(rec["slow_hidden"]),
        "slow_argmax_full": np.asarray(rec["slow_argmax_full"], dtype=np.int64),
        "fast_logits": np.stack(rec["fast_logits"]),
        "noise_slow": np.stack(rec["noise_slow"]),
        "noise_fast": np.stack(rec["noise_fast"]),
        "semantic": np.asarray(rec["semantic"], dtype=np.int64),
        "codes": (np.stack(rec["frames"], axis=1) if rec["frames"] else np.zeros((cfg.num_codebooks, 0), np.int64)),
    }
    return out


def pick_reference(pick: str, index: int) -> tuple[Path, str, dict]:
    meta = json.loads((FUNASR_FIXTURES / "meta.json").read_text())
    clips = [c for c in meta["clips"] if c["path"].startswith(f"fleurs/{pick}/")
             and 5.0 <= c["duration_s"] <= 10.0 and c.get("reference_text")]
    c = clips[index]
    return FUNASR_FIXTURES / c["path"], c["reference_text"]["raw_transcription"], c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default=str(HERE / "fixtures.json"))
    ap.add_argument("--out", default=str(WORK / "oracle"))
    ap.add_argument("--only", nargs="*", default=None, help="fixture names to run (default: all)")
    ap.add_argument("--check-generate", action="store_true",
                    help="also run the publisher's model.generate with the same seed on the first fixture")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    snap = hf_snapshot(HF_ID, revision=REVISION)
    fx = json.loads(Path(args.fixtures).read_text())
    gen_cfg = fx["generation"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    processor, model = load_model(snap)
    codec = model.load_codec(device="cpu", dtype=torch.float32)
    print(f"[load] {HF_ID}@{REVISION[:8]} fp32 cpu in {time.time() - t0:.1f}s; "
          f"params {sum(p.numel() for p in model.parameters()):,} + codec {sum(p.numel() for p in codec.parameters()):,}",
          flush=True)
    recorder = Recorder()
    recorder.attach(model, codec)

    summary = {
        "hf_id": HF_ID, "revision": REVISION,
        "model_safetensors_sha256": sha256(Path(snap) / "model.safetensors"),
        "codec_pth_sha256": sha256(Path(snap) / "codec.pth"),
        "config": json.loads((Path(snap) / "config.json").read_text()),
        "generation": gen_cfg, "dtype": "float32", "device": "cpu",
        "torch": torch.__version__, "fixtures": {},
    }
    names = [f["name"] for f in fx["fixtures"]] if args.only is None else args.only
    first = True
    for f in fx["fixtures"]:
        if f["name"] not in names:
            continue
        t1 = time.time()
        kwargs = {"text": [f["text"]]}
        ref_info = None
        if f.get("reference"):
            wav_path, ref_text, clip = pick_reference(f["reference"]["pick"], f["reference"]["index"])
            kwargs.update(reference_audio=[str(wav_path)], reference_text=[ref_text])
            ref_info = {"clip": clip["name"], "path": str(wav_path), "license": clip["license"],
                        "duration_s": clip["duration_s"], "reference_text": ref_text,
                        "source": clip["source"]}
        inputs = processor(**kwargs, return_tensors="pt")
        extra = {}
        if "reference_audio_values" in inputs:
            ref_codes, ref_lens = model.encode_audio(inputs["reference_audio_values"], inputs["reference_audio_lengths"])
            extra["ref_codes"] = ref_codes[0, :, : int(ref_lens[0])].numpy()
            extra["ref_wav"] = inputs["reference_audio_values"][0, 0, : int(inputs["reference_audio_lengths"][0])].numpy()
            inputs = {k: v for k, v in inputs.items() if not k.startswith("reference_audio")}
            inputs["reference_codes"], inputs["reference_code_lengths"] = ref_codes, ref_lens
        prompt, prompt_mask = model._prepare_prompt(**{k: inputs.get(k) for k in (
            "prefix_input_ids", "prefix_attention_mask", "suffix_input_ids", "suffix_attention_mask",
            "reference_codes", "reference_code_lengths")})
        rec = generate_recorded(model, prompt, prompt_mask, gen_cfg, f["seed"])
        rec.update(extra)
        codes = torch.from_numpy(rec["codes"])
        if codes.shape[1]:
            wav, lengths = model.decode_audio(codes[None])
            wav = wav[0, : int(lengths[0])].numpy()
        else:
            wav = np.zeros(0, np.float32)
        rec["wav"] = wav
        if args.check_generate and first:
            g = torch.Generator(device="cpu").manual_seed(f["seed"])
            ref = model.generate(input_ids=prompt, attention_mask=prompt_mask, max_new_tokens=gen_cfg["max_new_tokens"],
                                 temperature=gen_cfg["temperature"], top_p=gen_cfg["top_p"], top_k=gen_cfg["top_k"],
                                 do_sample=True, generator=g)
            same = ref.shape == codes[None].shape and bool((ref == codes[None]).all())
            print(f"[check-generate] {f['name']}: publisher generate == recorded loop -> {same} "
                  f"({tuple(ref.shape)} vs {tuple(codes[None].shape)})", flush=True)
            assert same, "recorded loop diverges from model.generate"
            summary["check_generate"] = {"fixture": f["name"], "identical": same}
        first = False
        np.savez(out_dir / f"{f['name']}.npz", **rec)
        sf.write(out_dir / f"{f['name']}.wav", wav, model.config.codec_sample_rate, subtype="FLOAT")
        T = int(codes.shape[1])
        info = {
            "text": f["text"], "lang": f["lang"], "seed": f["seed"], "prompt_len": int(prompt.shape[2]),
            "frames": T, "seconds": T * model.config.codec_frame_size / model.config.codec_sample_rate,
            "steps": int(rec["semantic"].shape[0]),
            "ended_with_eos": int(rec["semantic"][-1]) == model.config.eos_token_id,
            "slow_argmax_outside_allowed": int(sum(1 for a in rec["slow_argmax_full"]
                                                  if not (model.config.semantic_begin_id <= a <= model.config.semantic_end_id
                                                          or a == model.config.eos_token_id))),
            "codebook_max": rec["codes"].max(axis=1).tolist() if T else None,
            "codebooks_1_9_above_1023": int((rec["codes"][1:] > 1023).sum()) if T else 0,
            "reference": ref_info, "wall_s": round(time.time() - t1, 1),
        }
        summary["fixtures"][f["name"]] = info
        print(f"[{f['name']}] prompt {info['prompt_len']} -> {T} frames ({info['seconds']:.2f} s), eos {info['ended_with_eos']}, "
              f"cb1-9>1023: {info['codebooks_1_9_above_1023']}, argmax outside allowed: {info['slow_argmax_outside_allowed']}, "
              f"{info['wall_s']} s", flush=True)
        summary["absmax"] = recorder.absmax
        (out_dir / "oracle.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"[done] {len(summary['fixtures'])} fixtures in {time.time() - t0:.0f}s -> {out_dir}")
    am = recorder.absmax
    slow = max(v for k, v in am.items() if k.startswith("slow.block"))
    fast = max(v for k, v in am.items() if k.startswith("fast.block"))
    codec_max = max(v for k, v in am.items() if k.startswith("codec."))
    print(f"[absmax] slow residual {slow:.1f}  fast residual {fast:.1f}  codec {codec_max:.1f}  (float16 max 65504)")


if __name__ == "__main__":
    main()
