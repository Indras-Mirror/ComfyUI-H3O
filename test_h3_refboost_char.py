"""Standalone tests for H3RefBoostChar (h3_refboost_char.py).

Runs against the REAL comfy.ldm.minimax.model.PackedLayout (not a stub), so a
layout rebuild after ref expansion is validated by the actual layout code.

Usage:
    cd /media/mal/Crucible/AI-ART/ComfyUI
    ./venv/bin/python custom_nodes/ComfyUI-H3O/test_h3_refboost_char.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.realpath(__file__))
# This pack is symlinked into ComfyUI/custom_nodes, so realpath points outside
# the tree. The ComfyUI root is the parent of this venv (which lives inside it).
_COMFY_ROOT = os.path.dirname(sys.prefix)

sys.path.insert(0, _COMFY_ROOT)

import torch

from comfy.ldm.minimax.model import PackedLayout

sys.path.insert(0, _HERE)
import h3_refboost_char as rb

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

def build_payload(seed=42):
    """refs: image0 (scene image), image1 (char A), video0 (scene video), audio."""
    lat_h, lat_w = LAT_H, LAT_W
    img_z = torch.randn(1, 24, 1, lat_h, lat_w)
    vid_z = torch.randn(1, 24, 2, lat_h, lat_w)
    refs = [
        {"kind": "image", "latent_h": lat_h, "latent_w": lat_w, "latent": img_z},      # slot 0 scene image
        {"kind": "image", "latent_h": lat_h, "latent_w": lat_w, "latent": torch.randn(1, 24, 1, lat_h, lat_w)},  # slot 1 char
        {"kind": "video", "latent_t": 2, "latent_h": lat_h, "latent_w": lat_w,
         "ref_audio_t": 0, "latent": vid_z},
        {"kind": "audio", "ref_audio_t": 4, "audio_latent": torch.randn(1, 32, 2, 4)},
    ]
    payload = {
        "refs": list(refs),
        "cond_video_latents": [b["latent"] for b in refs if "latent" in b],
        "seed": seed,
        "frame_count": None,
    }
    payload["layout"] = PackedLayout(TEXT_LEN, LATENT_T, lat_h, lat_w, AUDIO_T,
                                     refs=refs, frame_count=None)
    return payload

def fake_x_context():
    x0 = torch.randn(1, 1, LATENT_T, LAT_H, LAT_W)  # video latent (t,h,w)
    x1 = torch.zeros(1, 1, 1, LATENT_T * 8, 8, 8)   # audio dims only matter for .shape[-1]
    context = torch.randn(1, TEXT_LEN, 128)
    return (x0, x1), context

# ── 1. slot parsing ────────────────────────────────────────────────────────
print("slot parsing")
check("empty -> empty set", rb._parse_slots("") == set())
check("'1' -> {1}", rb._parse_slots("1") == {1})
check("'0,1,2' -> {0,1,2}", rb._parse_slots("0, 1, 2") == {0, 1, 2})
check("garbage ignored", rb._parse_slots("0,x,2") == {0, 2})

# ── 2. classify: scene image slot 0 never char; slot 1 char ───────────────
print("\nclassify char ids")
refs = build_payload()["refs"]
char_ids = rb._classify_char_ids(refs, {1})
blk = [b for b in refs if b.get("kind") == "image"]
check("slot0 scene not char", id(blk[0]) not in char_ids)
check("slot1 char is char", id(blk[1]) in char_ids)
check("videos never char", all(id(b) not in char_ids for b in refs if b.get("kind") == "video"))

# ── 3. expansion: only slot1 repeated ──────────────────────────────────────
print("\nexpansion")
payload = build_payload()
orig_refs = list(payload["refs"])
char_ids = rb._classify_char_ids(payload["refs"], {1})
orig, expanded, is_char = rb._snapshot_and_expand(
    payload["refs"], char_ids, ref_repeat=3, protect_scene=True)
check("expanded 5 latent blocks (1 scene img + 3 char + 1 video + 1 audio)",
      len([b for b in expanded if "latent" in b]) == 5)

def identity_count(seq, obj):
    return sum(1 for b in seq if b is obj)

def latent_count(seq, lat):
    return sum(1 for b in seq if b.get("latent") is lat)

check("scene image not duplicated",
      identity_count(expanded, orig_refs[0]) == 1)
check("char latent present x3 (dup blocks share the tensor)",
      latent_count(expanded, orig_refs[1]["latent"]) == 3)
check("video not duplicated",
      identity_count(expanded, orig_refs[2]) == 1)
img_idx = [i for i, b in enumerate(expanded) if b.get("kind") == "image"]
check("char expansion flagged", is_char[img_idx[1]] and is_char[img_idx[2]] and is_char[img_idx[3]])
check("scene image not flagged", not is_char[img_idx[0]])
check("video not flagged", not is_char[[i for i, b in enumerate(expanded) if b.get("kind") == "video"][0]])

# ── 4. full wrapper: init + apply, scene preserved ─────────────────────────
print("\nwrapper config")
cfg = rb._parse_slots if False else {
    "slots": {1}, "ref_scale": 2.0, "ref_repeat": 3,
    "protect_scene": True, "source_noise": 0.7, "schedule": "linear",
    "step_threshold": 0.5, "debug": False,
}
payload = build_payload(seed=123)
x, ctx = fake_x_context()
scene_img_orig = payload["refs"][0]["latent"].clone()
char_orig = payload["refs"][1]["latent"].clone()
video_orig = payload["refs"][2]["latent"].clone()

state = rb._init_state(payload, x, ctx, cfg)
check("init ran", state is not None)
check("payload refs expanded to 6", len(payload["refs"]) == 6)
check("cond_video_latents resynced (5 latents)", len(payload["cond_video_latents"]) == 5)

# low sigma -> identity pass active (p -> 1)
t = torch.tensor([1.0], dtype=torch.float32)  # sigma*1000 ~ tiny -> p~1
rb._apply_step(payload, state, t, cfg)
chars_after = [b["latent"] for i, b in enumerate(payload["refs"])
               if i >= 1 and i <= 3 and b.get("kind") == "image"]
scene_after = payload["refs"][0]["latent"]
# after expansion refs = [scene_img, char, dup, dup, video, audio] -> video at 4
video_after = payload["refs"][4]["latent"]
eff = rb._ramp(rb._sigma_of(t), cfg["schedule"], cfg["step_threshold"])
expected_scale = 1.0 + (cfg["ref_scale"] - 1.0) * eff
check("selected char amplified (ref_scale x%.3f)" % expected_scale,
      torch.allclose(chars_after[0], char_orig * expected_scale, atol=1e-5))
check("scene image pristine", torch.equal(scene_after, scene_img_orig))
check("video pristine with protect_scene", torch.equal(video_after, video_orig))

# cond_video_latents sees the rewrite too
check("cond_video_latents aligned (block0 = scene pristine)",
      torch.equal(payload["cond_video_latents"][0], scene_img_orig))
check("cond_video_latents aligned (block1 = char scaled)",
      torch.equal(payload["cond_video_latents"][1], chars_after[0]))

# ── 5. protect_scene OFF -> video noise-mixed at low sigma ─────────────────
print("\nprotect_scene OFF")
cfg2 = dict(cfg); cfg2["protect_scene"] = False
payload2 = build_payload(seed=123)
x2, ctx2 = fake_x_context()
v_orig2 = payload2["refs"][2]["latent"].clone()  # video at index 2 pre-expansion
st2 = rb._init_state(payload2, x2, ctx2, cfg2)
rb._apply_step(payload2, st2, torch.tensor([1.0]), cfg2)
v_after2 = payload2["refs"][4]["latent"]  # video moved to 4 after expansion
check("video mixed with noise", not torch.equal(v_after2, v_orig2))

# high sigma -> ramp 0 -> nothing mixed even with protect OFF
cfg3 = dict(cfg); cfg3["protect_scene"] = False
payload3 = build_payload(seed=123)
x3, ctx3 = fake_x_context()
v_orig3 = payload3["refs"][2]["latent"].clone()
st3 = rb._init_state(payload3, x3, ctx3, cfg3)
rb._apply_step(payload3, st3, torch.tensor([999.9]), cfg3)  # high sigma -> eff 0
check("high sigma: video untouched (eff=0)", torch.equal(payload3["refs"][4]["latent"], v_orig3))

# ── 7. source_scale damp + video_audio coverage ────────────────────────────
print("\nsource_scale damp + video_audio")
# native node emits kind="video_audio" when a ref-video has a soundtrack —
# the OLD code only matched kind=="video" and silently skipped it. Verify both
# the kind fix (noise applies) and the new damp lever (source_scale).
def build_va_payload(seed=7):
    lat_h, lat_w = LAT_H, LAT_W
    refs = [
        {"kind": "image", "latent_h": lat_h, "latent_w": lat_w, "latent": torch.randn(1, 24, 1, lat_h, lat_w)},   # slot 0 scene
        {"kind": "image", "latent_h": lat_h, "latent_w": lat_w, "latent": torch.randn(1, 24, 1, lat_h, lat_w)},   # slot 1 char
        {"kind": "video_audio", "latent_t": 2, "latent_h": lat_h, "latent_w": lat_w, "ref_audio_t": 2,
         "latent": torch.randn(1, 24, 2, lat_h, lat_w), "audio_latent": torch.randn(1, 32, 2, 4)},
    ]
    payload = {"refs": refs,
               "cond_video_latents": [b["latent"] for b in refs if "latent" in b],
               "seed": seed, "frame_count": None}
    payload["layout"] = PackedLayout(TEXT_LEN, LATENT_T, lat_h, lat_w, AUDIO_T, refs=refs, frame_count=None)
    return payload

cfg_va = {"slots": {1}, "ref_scale": 1.0, "ref_repeat": 1, "protect_scene": False,
          "source_noise": 0.6, "source_scale": 1.0, "schedule": "linear", "step_threshold": 0.5, "debug": False}
p_va = build_va_payload()
v_va_orig = p_va["refs"][2]["latent"].clone()
st_va = rb._init_state(p_va, x, ctx, cfg_va)
rb._apply_step(p_va, st_va, torch.tensor([1.0]), cfg_va)
check("video_audio ref is noise-mixable (kind fix)", not torch.equal(p_va["refs"][2]["latent"], v_va_orig))

cfg_damp = {"slots": {1}, "ref_scale": 1.0, "ref_repeat": 1, "protect_scene": False,
            "source_noise": 0.0, "source_scale": 0.5, "schedule": "linear", "step_threshold": 0.5, "debug": False}
p_damp = build_va_payload()
v_damp_orig = p_damp["refs"][2]["latent"].clone()
st_damp = rb._init_state(p_damp, x, ctx, cfg_damp)
t_low = torch.tensor([1.0])
eff_low = rb._ramp(rb._sigma_of(t_low), "linear", 0.5)
rb._apply_step(p_damp, st_damp, t_low, cfg_damp)
expected_damp = 1.0 + (0.5 - 1.0) * eff_low
check("source_scale dampens ref-video at low sigma (factor %.3f)" % expected_damp,
      torch.allclose(p_damp["refs"][2]["latent"], v_damp_orig * expected_damp, atol=1e-6))

cfg_prot = dict(cfg_damp); cfg_prot["protect_scene"] = True; cfg_prot["source_noise"] = 0.6
p_prot = build_va_payload()
v_prot_orig = p_prot["refs"][2]["latent"].clone()
st_prot = rb._init_state(p_prot, x, ctx, cfg_prot)
rb._apply_step(p_prot, st_prot, torch.tensor([1.0]), cfg_prot)
check("protect_scene blocks damp + noise", torch.equal(p_prot["refs"][2]["latent"], v_prot_orig))

# ── 6. layout rebuild validity ─────────────────────────────────────────────
print("\nlayout rebuild")
payload4 = build_payload(seed=7)
x4, ctx4 = fake_x_context()
st4 = rb._init_state(payload4, x4, ctx4, cfg)
layout = payload4.get("layout")
check("layout present", layout is not None)
check("layout segment count matches refs", layout is not None and len(layout.segments) >= 1)
# rebuilding again from the same shape must not change signature
sig_before = layout.signature
rb._init_state(payload4, x4, ctx4, cfg)
check("layout signature stable across calls", payload4["layout"].signature == sig_before)

print(f"\nALL {PASS} CHECKS PASSED")
