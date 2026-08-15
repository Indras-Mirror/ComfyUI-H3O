"""Standalone tests for H3RefBoostV (h3_refboost_v.py).

Validates the two pure-logic pieces against the REAL PackedLayout, without a
GPU: (1) ref block -> packed row-range mapping, and (2) the attention V-scale
math (source damped, char boosted, everything else untouched).

Usage:
    cd /media/mal/Crucible/AI-ART/ComfyUI
    ./venv/bin/python custom_nodes/ComfyUI-H3O/test_h3_refboost_v.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.realpath(__file__))
_COMFY_ROOT = os.path.dirname(sys.prefix)

sys.path.insert(0, _COMFY_ROOT)

import torch

from comfy.ldm.minimax.model import PackedLayout

sys.path.insert(0, _HERE)
import h3_refboost_v as rv

PASS = 0


def check(name, cond, detail=""):
    global PASS
    status = "ok " if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        sys.exit(f"TEST FAILED: {name} {detail}")
    PASS += 1


TEXT_LEN = 8
LATENT_T = 4
LAT_H, LAT_W = 48, 84
AUDIO_T = 8


def build_payload():
    """refs: image0 (scene), image1 (char), video0 (source, with audio), audio0."""
    lat_h, lat_w = LAT_H, LAT_W
    refs = [
        {"kind": "image", "latent_h": lat_h, "latent_w": lat_w, "latent": torch.randn(1, 24, 1, lat_h, lat_w)},
        {"kind": "image", "latent_h": lat_h, "latent_w": lat_w, "latent": torch.randn(1, 24, 1, lat_h, lat_w)},
        {"kind": "video_audio", "latent_t": 2, "latent_h": lat_h, "latent_w": lat_w,
         "ref_audio_t": 4, "latent": torch.randn(1, 24, 2, lat_h, lat_w),
         "audio_latent": torch.randn(1, 32, 2, 4)},
        {"kind": "audio", "ref_audio_t": 4, "audio_latent": torch.randn(1, 32, 2, 4)},
    ]
    layout = PackedLayout(TEXT_LEN, LATENT_T, lat_h, lat_w, AUDIO_T, refs=refs, frame_count=None)
    payload = {"refs": refs, "layout": layout, "seed": 0, "frame_count": None}
    return payload


# ── 1. row-range mapping ────────────────────────────────────────────────────
print("row-range mapping (real PackedLayout)")
payload = build_payload()
refs = payload["refs"]
ranges = rv._ref_block_ranges(refs, payload["layout"])

check("4 ranges returned (one per ref block)", len(ranges) == 4)
check("image0 range is a contiguous span", ranges[0] is not None and ranges[0][0] < ranges[0][1])
check("image1 range starts where image0 ends", ranges[1][0] == ranges[0][1])
check("video_audio range follows image1 (its ref_audio rows come first)",
      ranges[2] is not None and ranges[2][0] >= ranges[1][1])
check("audio block maps to None (no visual rows)", ranges[3] is None)

# verify image ranges are exactly the packed rows of the layout's ref_img
# segments for the two image blocks, in order.
img_segs = [s for s in payload["layout"].segments if s[2] == "ref_img"]
check("image0 range matches first ref_img segment", ranges[0] == (img_segs[0][0], img_segs[0][1]))
check("image1 range matches second ref_img segment", ranges[1] == (img_segs[1][0], img_segs[1][1]))
check("video range matches third ref_img segment", ranges[2] == (img_segs[2][0], img_segs[2][1]))

# ── 2. state: char vs source split ─────────────────────────────────────────
print("\nstate split (char vs source)")
x = (torch.randn(1, 1, LATENT_T, LAT_H, LAT_W), torch.zeros(1, 1, 1, LATENT_T * 8, 8, 8))
ctx = torch.randn(1, TEXT_LEN, 128)
cfg = {"slots": {1}, "ref_v_boost": 1.5, "source_v_scale": 0.7,
       "schedule": "cosine", "step_threshold": 0.5, "debug": False}
state = rv._compute_state(payload, x, ctx, cfg)
check("state built", state is not None)
check("one char range (slot 1)", state["char_ranges"] == [ranges[1]])
check("one source range (video)", state["src_ranges"] == [ranges[2]])

# ── 3. class-level patch: Sol-Attn bypass + V-scaling ───────────────────────
print("\nclass-level patch (Sol-Attn compose proof + V-scaling)")
import comfy.ops
import comfy.quant_ops
import comfy.ldm.modules.attention as attn_mod
from comfy.ldm.minimax.model import Attention

captured = {}
def _fake_rope_(q, k, freqs, qw, kw, **extra):
    return None
comfy.quant_ops.ck.rms_rope_split_half_ = _fake_rope_
def _fake_attn(q, k, v, heads, **extra):
    captured["v"] = v.clone()
    b, h, s, hd = v.shape
    return torch.zeros(b, s, h * hd)
attn_mod.optimized_attention = _fake_attn

hidden, heads, hd = 64, 4, 16
attn = Attention(hidden, heads, hd, eps=1e-5, dtype=torch.float32,
                 device="cpu", operations=comfy.ops.disable_weight_init)
rv._patch_attention_class(debug=False)

# Sol-Attn's _compose_module_patch only wraps INSTANCE attrs: fwd =
# attn.__dict__.get("forward"). With a class-level patch this is None, so
# Sol-Attn never composes and cannot bypass our V-scaling.
check("instance has NO forward attr (Sol-Attn compose skips us)",
      "forward" not in attn.__dict__)
check("Sol-Attn compose hook sees None", attn.__dict__.get("forward") is None)

s = 32
x = torch.randn(s, hidden)
rope_freqs = torch.randn(s, 3, 2, 16)
cfg_v = {"src_ranges": [(4, 6)], "char_ranges": [(10, 12)],
         "ref_v_eff": 1.5, "src_v_eff": 0.7}
attn(x, rope_freqs=rope_freqs, transformer_options={rv._TO_KEY: cfg_v})
v = captured["v"]                     # (1, heads, s, hd)
captured.clear()
neutral = {"src_ranges": [], "char_ranges": [], "ref_v_eff": 1.0, "src_v_eff": 1.0}
attn(x, rope_freqs=rope_freqs, transformer_options={rv._TO_KEY: neutral})
v0 = captured["v"].clone()

check("source rows damped x0.7 (through class patch)",
      torch.allclose(v[:, :, 4:6], v0[:, :, 4:6] * 0.7, atol=1e-5))
check("char rows boosted x1.5 (through class patch)",
      torch.allclose(v[:, :, 10:12], v0[:, :, 10:12] * 1.5, atol=1e-5))
check("other rows untouched",
      torch.allclose(v[:, :, :4], v0[:, :, :4], atol=1e-6)
      and torch.allclose(v[:, :, 12:], v0[:, :, 12:], atol=1e-6))
check("no-cfg call delegates to original (returns [s, hidden])",
      attn(x, rope_freqs=rope_freqs, transformer_options={}).shape == (s, hidden))

# ── 4. wrapper injects config into transformer_options ──────────────────────
print("\nwrapper transformer_options injection")
seen = {}

def fake_executor(x_, t_, c_, to_, **kw):
    seen["to"] = to_
    return "done"

wrapped = rv.make_wrapper(cfg)
t_low = torch.tensor([1.0])  # sigma*1000 tiny -> eff ~1
out = wrapped(fake_executor, x, t_low, ctx, {}, minimax_payload=payload)
check("executor ran", out == "done")
check("config injected into transformer_options", rv._TO_KEY in seen["to"])
injected = seen["to"][rv._TO_KEY]
eff = rv._ramp(rv._sigma_of(t_low), cfg["schedule"], cfg["step_threshold"])
check("ref_v_eff ramped", abs(injected["ref_v_eff"] - (1.0 + (1.5 - 1.0) * eff)) < 1e-6)
check("src_v_eff ramped", abs(injected["src_v_eff"] - (1.0 + (0.7 - 1.0) * eff)) < 1e-6)
check("ranges carried", injected["char_ranges"] == [ranges[1]] and injected["src_ranges"] == [ranges[2]])

# high sigma -> eff 0 -> scales at neutral 1.0
seen.clear()
t_hi = torch.tensor([999.9])
rv.make_wrapper(cfg)(fake_executor, x, t_hi, ctx, {}, minimax_payload=payload)
injected_hi = seen["to"][rv._TO_KEY]
check("high sigma: ref_v_eff neutral", abs(injected_hi["ref_v_eff"] - 1.0) < 1e-6)
check("high sigma: src_v_eff neutral", abs(injected_hi["src_v_eff"] - 1.0) < 1e-6)

print(f"\nALL {PASS} CHECKS PASSED")
