"""End-to-end harness: H3ReferenceToVideo batch vs separate refs, same conditioning?

Path A: official MiniMaxH3ReferenceToVideo with 3 separate ref_image_N inputs.
Path B: MiniMaxH3ReferenceToVideoBatch with 1 images_batch (real H3BatchImages
output — the actual workflow composition).

Both paths run through the REAL official execute() internals with a recording
fake VAE + fake CLIP. ref_blocks / presentation / layout are validated against
the REAL comfy PackedLayout.

H3BatchImages is now WAN-style AUTOMATIC: black bars always stripped, one exact
megapixels box (dominant aspect x MP, 32px aligned), pad+fit (edge replicate —
never black) or crop+fit (zero pad). It also emits the `refs` LIST output.

Usage:
    cd /media/mal/Crucible/AI-ART/ComfyUI
    ./venv/bin/python custom_nodes/ComfyUI-H3O/test_h3_batch_e2e.py
"""

import hashlib
import math
import os
import sys

_HERE = os.path.dirname(os.path.realpath(__file__))
_COMFY_ROOT = os.path.dirname(sys.prefix)
sys.path.insert(0, _COMFY_ROOT)

import torch

from comfy.ldm.minimax.model import PackedLayout
from comfy_extras.nodes_minimax_h3 import (
    MiniMaxH3ReferenceToVideo,
    _resize,
    CANVAS_MULTIPLE,
    temporal_shape,
)

sys.path.insert(0, _HERE)
from h3_ref_to_video_batch import MiniMaxH3ReferenceToVideoBatch
from h3_batch_images import H3BatchImages

import nodes

W, HT, L = 1344, 768, 124
TEXT_LEN = 16
_frame_count, LATENT_T, AUDIO_T = temporal_shape(L)
LAT_H, LAT_W = HT // 16, W // 16

RESULTS = []

def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    status = "ok " if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

def tensor_hash(x):
    return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()[:16]

def match_scale_dims(w, h, width=W, height=HT):
    """Replicates the official node's 'match' ref sizing math (nodes_minimax_h3.py:222-228)."""
    scale = min(1.0, math.sqrt((width * height) / (w * h)))
    tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return tw, th

class RecordingVAE:
    """Fake VAE: records every per-ref encode input; fixed-shape latent [1,24,1,h/16,w/16]."""

    def __init__(self):
        self.calls = []

    def encode(self, x):
        self.calls.append({
            "shape": tuple(x.shape), "dtype": str(x.dtype),
            "hash": tensor_hash(x), "input": x.detach().clone(),
        })
        lh, lw = x.shape[2] // 16, x.shape[3] // 16
        return torch.full((1, 24, 1, lh, lw), int(self.calls[-1]["hash"][:8], 16) / 1e10)

class RecordingClip:
    def __init__(self):
        self.ref_items = None

    def tokenize(self, prompt, minimax_ref_items=None, **kw):
        self.ref_items = list(minimax_ref_items) if minimax_ref_items else []
        return {"prompt": prompt, "n_items": len(self.ref_items)}

    def encode_from_tokens_scheduled(self, tokens):
        return [[tokens, {}]]

def run(node_cls, **kw):
    vae, clip = RecordingVAE(), RecordingClip()
    out = node_cls.execute(clip=clip, vae=vae, audio_vae=None, prompt="test",
                           width=W, height=HT, length=L, **kw)
    refs = out[0][0][1]["minimax_refs"]
    return vae, clip, refs

def run_region(node_cls, batch, regions, ref_images=None):
    """Run with the content-region side-channel wired (region-aware batch)."""
    return run(node_cls, images_batch=batch, images_batch_regions=regions,
               ref_images=ref_images or {})

def build_layout(refs):
    return PackedLayout(TEXT_LEN, LATENT_T, LAT_H, LAT_W, AUDIO_T,
                        refs=refs, frame_count=None)

def picture_labels(ref_items):
    """Replicates comfy/text_encoders/minimax.py:153-175 label loop."""
    counters = {"image": 0, "audio": 0, "video": 0}
    labels = []
    for item in ref_items:
        kind = item["type"]
        counters[kind] += 1
        if kind == "image":
            labels.append(f"<Picture {counters[kind]}>")
        elif kind == "video":
            labels.append(f"<Video {counters[kind]}>")
        else:
            labels.append(f"<Audio {counters[kind]}>")
    return labels

def synthetic_frame(colors):
    x = torch.zeros(512, 512, 3, dtype=torch.float32)
    for i, c in enumerate(colors):
        x[..., i] = c
    return x.unsqueeze(0)

def _textured(h, w):
    """Textured content (mean~0.5, std~0.14) — never detected as a bar."""
    return (torch.rand(1, h, w, 3) * 0.5 + 0.25).float()

# ── 1. controlled: same-size synthetic frames (padding-free) ───────────────
print("== 1. controlled: same-size synthetic frames (padding-free) ==")
frames = [synthetic_frame((1, 0, 0)), synthetic_frame((0, 1, 0)), synthetic_frame((0, 0, 1))]
refs_src = {"ref_image_0": frames[0], "ref_image_1": frames[1], "ref_image_2": frames[2]}

vae_a, clip_a, refs_a = run(MiniMaxH3ReferenceToVideo, ref_images=dict(refs_src))
vae_b, clip_b, refs_b = run(MiniMaxH3ReferenceToVideoBatch, images_batch=torch.cat(frames, dim=0))

check("ref count A==B==3", len(refs_a) == 3 and len(refs_b) == 3)
check("VAE encode calls 3 per path", len(vae_a.calls) == 3 and len(vae_b.calls) == 3)
check("per-position encode input SIZE identical (A vs B)",
      all(vae_a.calls[i]["shape"] == vae_b.calls[i]["shape"] for i in range(3)),
      str([c["shape"] for c in vae_b.calls]))
check("per-position encode input CONTENT identical (order proof)",
      all(vae_a.calls[i]["hash"] == vae_b.calls[i]["hash"] for i in range(3)),
      "/".join(f"pos{i}:{vae_b.calls[i]['hash'][:8]}" for i in range(3)))
check("ref_blocks latent_h/w equal per position",
      all(refs_a[i]["latent_h"] == refs_b[i]["latent_h"] and
          refs_a[i]["latent_w"] == refs_b[i]["latent_w"] for i in range(3)))
check("Picture labels identical order",
      picture_labels(clip_a.ref_items) == picture_labels(clip_b.ref_items),
      str(picture_labels(clip_b.ref_items)))
la, lb = build_layout(refs_a), build_layout(refs_b)
check("PackedLayout segments + seq_len identical",
      la.segments == lb.segments and la.seq_len == lb.seq_len,
      f"seq A={la.seq_len} B={lb.seq_len}")

# ── 1b. batch subclass passthrough: images_batch=None must equal official ───
print("\n== 1b. batch subclass passthrough (images_batch=None) ==")
vae_p, _, refs_p = run(MiniMaxH3ReferenceToVideoBatch, ref_images=dict(refs_src))
check("passthrough ref_blocks identical to official path",
      len(refs_p) == 3 and all(
          refs_p[i]["latent_h"] == refs_a[i]["latent_h"] and
          torch.equal(refs_p[i]["latent"], refs_a[i]["latent"]) for i in range(3)))

# ── 1c. INPUT_IS_LIST wrapped dict values (executor-style) must not crash ──
print("\n== 1c. INPUT_IS_LIST wrapped dict values (ref_video_0 etc.) ==")
# Regression: with INPUT_IS_LIST the executor wraps each autogrow dict value in
# a 1-element list — {"ref_video_0": [tensor], "ref_image_0": [tensor]}. The
# batch sibling must unwrap those or the official node crashes on
# video_frames.shape (the user's ri2v source-video -> ref_video_0 path).
_video5 = torch.cat([frames[0]] * 5, dim=0)          # 5-frame static ref video
vae_w2, clip_w2, refs_w2 = run(
    MiniMaxH3ReferenceToVideoBatch,
    ref_images={"ref_image_0": [frames[1]]},
    ref_videos={"ref_video_0": [_video5]})
check("WRAPPED: image + video refs both accepted (no crash)",
      len(refs_w2) == 2, f"refs={len(refs_w2)}")
_w2_types = [r["kind"] for r in refs_w2]
check("WRAPPED: ref order image then video",
      _w2_types == ["image", "video"], str(_w2_types))
check("WRAPPED: video latent uses the 5-frame ref",
      refs_w2[1]["latent_h"] == 512 // 16
      and refs_w2[1]["latent_w"] == 512 // 16,
      str((refs_w2[1]["latent_h"], refs_w2[1]["latent_w"])))

# ── 2. real aliclo PNGs (mixed dims, real H3BatchImages composition) ───────
print("\n== 2. real aliclo PNGs (mixed dims, real H3BatchImages composition) ==")
loader = nodes.LoadImage()
sources = [loader.load_image(f)[0] for f in ("aliclo2.png", "aliclo4.png", "Aliclo3.png")]
names = ["aliclo2", "aliclo4", "Aliclo3"]
print(f"  source dims (H,W): {list(zip(names, [(s.shape[1], s.shape[2]) for s in sources]))}")
refsA_src = {"ref_image_0": sources[0], "ref_image_1": sources[1], "ref_image_2": sources[2]}
batch, regions, refsA_list = H3BatchImages.execute(images=dict(refsA_src))
print(f"  H3BatchImages output: {tuple(batch.shape)}  (auto: bars stripped, megapixels=1.0 box, pad+fit replicate)")

vae_a, clip_a, refs_a = run(MiniMaxH3ReferenceToVideo, ref_images=dict(refsA_src))
vae_b, clip_b, refs_b = run(MiniMaxH3ReferenceToVideoBatch, images_batch=batch)

check("ref count A==B==3 (real images)", len(refs_a) == 3 and len(refs_b) == 3)

# Path A ordering: per-position encode input must equal _resize(source_i, exp_i)
exp_a = [match_scale_dims(s.shape[2], s.shape[1]) for s in sources]  # (tw, th)
exp_input_a = [_resize(s, tw, th, "disabled") for s, (tw, th) in zip(sources, exp_a)]
a_order = all(vae_a.calls[i]["hash"] == tensor_hash(exp_input_a[i]) for i in range(3))
a_sizes = [c["shape"] for c in vae_a.calls]
check("Path A per-position content == source order (aliclo2,aliclo4,Aliclo3)",
      a_order, str(a_sizes))

# Batch path: every frame at the SAME exact megapixels box (uniform geometry)
b_sizes = [c["shape"] for c in vae_b.calls]
check("Path B: all batch refs at the same exact box (uniform)",
      len({s for s in b_sizes}) == 1, str(b_sizes))
check("Path B: encode box == H3BatchImages output box",
      (b_sizes[0][1], b_sizes[0][2]) == (batch.shape[1], batch.shape[2]),
      f"encode {b_sizes[0]} batch {tuple(batch.shape)}")

labels_a, labels_b = picture_labels(clip_a.ref_items), picture_labels(clip_b.ref_items)
check("Picture labels identical order (real)", labels_a == labels_b, str(labels_b))
check("Prompt enhancer contract: image0 -> <Picture 1>", labels_a[0] == "<Picture 1>")

# ── 2b. region-aware path: crop the pad back off -> native fitted geometry ──
print("\n== 2b. region-aware batch (content regions wired) ==")
vae_c, clip_c, refs_c = run_region(
    MiniMaxH3ReferenceToVideoBatch, batch, regions)

sizes_c = [c["shape"] for c in vae_c.calls]
# the region path crops each padded frame back to its content extent, so each
# ref keeps its OWN fitted geometry (per-frame, not the uniform box) — the
# encode input must equal the match-scale of the region-cropped content.
fitted_c = []
for _i in range(3):
    _y0, _y1, _x0, _x1 = regions[_i]
    fitted_c.append(batch[_i:_i + 1][:, _y0:_y1, _x0:_x1, :])
exp_c = [_resize(fitted_c[_i],
                 *match_scale_dims(fitted_c[_i].shape[2], fitted_c[_i].shape[1]),
                 "disabled") for _i in range(3)]
check("REGION: per-ref encode == match-scale of region-cropped content",
      all(vae_c.calls[i]["hash"] == tensor_hash(exp_c[i]) for i in range(3)),
      str(sizes_c))
check("REGION: per-frame geometry restored (sizes differ per ref, not uniform)",
      len({s for s in sizes_c}) == 3, str(sizes_c))
labels_c = picture_labels(clip_c.ref_items)
check("REGION: Picture labels identical order",
      labels_c == labels_a, str(labels_c))

# ── 3. split key ordering with a pre-wired ref ─────────────────────────────
print("\n== 3. key-ordering probe: wired ref_image_5 + 2-frame batch ==")
batch2 = torch.cat([frames[1], frames[2]], dim=0)
vae_w, clip_w, refs_w = run(
    MiniMaxH3ReferenceToVideoBatch,
    ref_images={"ref_image_5": frames[0]}, images_batch=batch2)
exp_w = [_resize(frames[0], *match_scale_dims(512, 512), "disabled"),
         _resize(frames[1], *match_scale_dims(512, 512), "disabled"),
         _resize(frames[2], *match_scale_dims(512, 512), "disabled")]
check("wired ref first, batch frames appended after (ref_image_5,6,7)",
      len(refs_w) == 3 and all(vae_w.calls[i]["hash"] == tensor_hash(exp_w[i]) for i in range(3)),
      str(picture_labels(clip_w.ref_items)))

# ── 4. automatic black-bar removal (always on, WAN defaults) ───────────────
print("\n== 4. auto bar removal: always on, WAN defaults ==")
from h3_batch_images import detect_and_crop_black_bars as _crop_bars

def _checkered(h, w):
    """8px checkerboard, high-contrast textured content (std ~0.45)."""
    x = torch.zeros(h, w, 3, dtype=torch.float32)
    yy = torch.arange(h)[:, None] // 8
    xx = torch.arange(w)[None, :] // 8
    x[((yy + xx) % 2) == 0] = 0.9
    return x

def _bar_frame(bar_val, bar_w, content=None, h=128, w=128):
    c = _checkered(h, w) if content is None else content
    x = c.clone()
    x[:, :bar_w] = bar_val
    x[:, w - bar_w:] = bar_val
    return x.unsqueeze(0)

def _dark_textured_edge(bar_w=8, h=128, w=128):
    """Low-mean (0.03) higher-std (0.07) two-tone noise edge — texture, not a bar."""
    x = _checkered(h, w).clone()
    noise = (torch.rand(h, bar_w, 3) < 0.5).float() * 0.14   # std ~0.07 > 0.06
    x[:, :bar_w] = noise
    x[:, w - bar_w:] = noise
    return x.unsqueeze(0)

def _edges_nonuniform(frame, tol=1e-3):
    f = frame if frame.dim() == 3 else frame[0]
    return (f.std(dim=(1, 2))[0] > tol and f.std(dim=(1, 2))[-1] > tol and
            f.std(dim=(0, 2))[0] > tol and f.std(dim=(0, 2))[-1] > tol)

# black pillarbox cropped (default threshold 0.15)
out0 = _crop_bars(_bar_frame(0.0, 8))
check("black pillarbox cropped (8px, mean 0)", out0.shape == (1, 128, 112, 3), str(tuple(out0.shape)))
check("black-bar crop edges non-uniform", _edges_nonuniform(out0))

# gray bars 0.08 — above the old threshold, stripped by the node's always-on auto
out_gray = _crop_bars(_bar_frame(0.08, 8))
check("gray bars 0.08 stripped at default", out_gray.shape == (1, 128, 112, 3), str(tuple(out_gray.shape)))

# variance guard: dark-textured edge (std > 0.06) NOT cropped
out4 = _crop_bars(_dark_textured_edge(8))
check("dark-textured edge NOT cropped (variance guard)", out4.shape == (1, 128, 128, 3), str(tuple(out4.shape)))

# full node: bars gone end-to-end (bar detector no-op on output)
mp_out, mp_regs, mp_refs = H3BatchImages.execute(
    images={"image_0": _bar_frame(0.0, 8), "image_1": _bar_frame(0.08, 8)})
check("NODE: bars stripped before batching (bar detector no-op on output)",
      all(_crop_bars(mp_out[i:i + 1]).shape == mp_out[i:i + 1].shape for i in range(2)),
      str(tuple(mp_out.shape)))

# ── 5. exact box math + empty-frame guard ──────────────────────────────────
print("\n== 5. exact WAN box + empty-frame guard ==")
big = _textured(1000, 3000)
big_out, big_regs, _ = H3BatchImages.execute(images={"image_0": big}, megapixels=1.0)
_exp_tw = max(32, round(math.sqrt(1e6 * 3.0) / 32) * 32)   # dominant aspect 3:1
_exp_th = max(32, round(math.sqrt(1e6 / 3.0) / 32) * 32)
check("exact box from dominant aspect x 1.0MP (3:1)",
      (big_out.shape[1], big_out.shape[2]) == (_exp_th, _exp_tw),
      f"got {tuple(big_out.shape[1:3])} want {(_exp_th, _exp_tw)}")
check("exact box region == full frame (pad+fit of dominant frame)", big_regs[0] == (0, _exp_th, 0, _exp_tw), str(big_regs[0]))
black = torch.zeros(1, 1000, 3000, 3)
blk, _, _ = H3BatchImages.execute(images={"image_0": black})
check("all-black frame kept (non-empty guard)",
      blk.shape[1] > 0 and blk.shape[2] > 0, str(tuple(blk.shape)))

# ── 6. megapixels resize: uniform exact box, pad+fit vs crop+fit ───────────
print("\n== 6. megapixels: uniform exact box, pad+fit never black, crop+fit zero pad ==")
mp_src = {
    "image_0": _textured(400, 300),    # portrait (300x400)
    "image_1": _textured(512, 512),    # square
    "image_2": _textured(450, 800),    # landscape 16:9 — largest by area = dominant
}
_largest = max(mp_src.values(), key=lambda f: f.shape[1] * f.shape[2])
_lh, _lw = _largest.shape[1], _largest.shape[2]
_exp_tw = max(32, round(math.sqrt(1e6 * (_lw / _lh)) / 32) * 32)
_exp_th = max(32, round(math.sqrt(1e6 / (_lw / _lh)) / 32) * 32)

mp_out, mp_regs, mp_refs = H3BatchImages.execute(images=dict(mp_src), megapixels=1.0)
check("megapixels: all frames identical HxW",
      len({tuple(f.shape[1:3]) for f in mp_out}) == 1, str(tuple(mp_out.shape)))
check("megapixels: exact target box (dominant aspect x 1.0MP, aligned 32)",
      (mp_out.shape[1], mp_out.shape[2]) == (_exp_th, _exp_tw),
      f"got {tuple(mp_out.shape[1:3])} want {(_exp_th, _exp_tw)}")
check("pad+fit: no leftover bars (bar detector no-op on output)",
      all(_crop_bars(mp_out[i:i + 1]).shape == (1, _exp_th, _exp_tw, 3) for i in range(3)))
check("pad+fit: content inside box, not cropped (regions)",
      all(r[0] >= 0 and r[1] <= _exp_th and r[2] >= 0 and r[3] <= _exp_tw
          and r[1] > r[0] and r[3] > r[2] for r in mp_regs), str(mp_regs))
check("pad+fit: dominant frame pad is minimal (<= one 32px cell)",
      mp_regs[2][0] < 32 and mp_regs[2][3] > _exp_tw - 32
      and mp_regs[2][1] == _exp_th, str(mp_regs[2]))
# pad+fit pads with EDGE REPLICATE — the padded band is non-black (content edge)
check("pad+fit: pad band is NOT black (edge replicate, no visible bars)",
      all(mp_out[i][0].mean() > 0.05 and mp_out[i][-1].mean() > 0.05
          and mp_out[i][:, 0].mean() > 0.05 and mp_out[i][:, -1].mean() > 0.05
          for i in range(3)), str([float(mp_out[i][0].mean()) for i in range(3)]))

mp_crop, _, _ = H3BatchImages.execute(images=dict(mp_src), megapixels=1.0,
                                      fit_mode="crop+fit")
check("crop+fit: exact box",
      (mp_crop.shape[1], mp_crop.shape[2]) == (_exp_th, _exp_tw),
      str(tuple(mp_crop.shape)))
check("crop+fit: no bars at all (zero pad)",
      all(_crop_bars(mp_crop[i:i + 1]).shape == (1, _exp_th, _exp_tw, 3)
          for i in range(3)))

# ── Refs (list) output: individual [1,H,W,C] frames, same processed pixels ──
check("refs list: is a python list with one entry per frame",
      isinstance(mp_refs, list) and len(mp_refs) == len(mp_out), str(type(mp_refs)))
check("refs list: each frame is [1,H,W,C] at the exact box",
      all(f.shape == (1, _exp_th, _exp_tw, 3) for f in mp_refs),
      [tuple(f.shape) for f in mp_refs])
check("refs list: pixels identical to the batch frames (same processing)",
      all(torch.equal(f, mp_out[i:i + 1]) for i, f in enumerate(mp_refs)))
check("refs list: also emitted from the real-PNG default path",
      isinstance(refsA_list, list) and len(refsA_list) == len(sources)
      and all(f.shape[0] == 1 for f in refsA_list))

# ── 7. the stretched-ref trap (regions from one tensor applied to another) ─
print("\n== 7. stretched-ref trap (the old ri2v wiring) ==")
# images_batch from H3PromptEnhancerPlus.ref_images_out (bilinear-stretched to
# the first ref's dims, aspect destroyed) while images_batch_regions comes from
# H3BatchImages (geometry of the batch box). Regions from one tensor applied to
# a stretched copy = wrong crops + distortion. Must NOT be confused with the
# correct same-node region path (2b).
_stretch_h, _stretch_w = sources[0].shape[1], sources[0].shape[2]
_matched = []
for _img in sources:
    if _img.shape[1] != _stretch_h or _img.shape[2] != _stretch_w:
        _img = torch.nn.functional.interpolate(
            _img.permute(0, 3, 1, 2), size=(_stretch_h, _stretch_w),
            mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    _matched.append(_img)
stretched = torch.cat(_matched, dim=0)
vae_t, _, _ = run_region(MiniMaxH3ReferenceToVideoBatch, stretched, regions)
t_sizes = [c["shape"] for c in vae_t.calls]
check("TRAP: stretched batch NOT identical to the correct region path",
      t_sizes != sizes_c or any(
          vae_t.calls[i]["hash"] != vae_c.calls[i]["hash"] for i in range(3)),
      f"C={sizes_c} T={t_sizes}")
t_ident = all(vae_a.calls[i]["hash"] == vae_t.calls[i]["hash"]
              for i in range(3))
check("TRAP: stretched batch NOT bit-identical to individual refs",
      not t_ident,
      "wrong-crop/stretch produced identical encodes — trap wiring not pinned")

# ── 8. internal split: megapixels batch -> batch-sibling split ─────────────
print("\n== 8. internal split: megapixels batch -> H3ReferenceToVideo ==")
# The real workflow path: H3BatchImages (auto, megapixels) -> images_batch,
# split inside execute() — per-frame refs, no regions needed.
mp_batch, _, _ = H3BatchImages.execute(images=dict(refsA_src), megapixels=0.5)
vae_mp, clip_mp, refs_mp = run(MiniMaxH3ReferenceToVideoBatch,
                               images_batch=mp_batch)
mp_sizes = [c["shape"] for c in vae_mp.calls]
check("INTERNAL: 3 refs from 3-frame batch",
      len(refs_mp) == 3 and len(vae_mp.calls) == 3, str(mp_sizes))
check("INTERNAL: every ref encodes as [1,H,W,C]",
      all(len(s) == 4 and s[0] == 1 for s in mp_sizes), str(mp_sizes))
check("INTERNAL: uniform geometry (one exact box for every ref)",
      len({s for s in mp_sizes}) == 1, str(mp_sizes))
check("INTERNAL: encode box == megapixels batch box",
      (mp_sizes[0][1], mp_sizes[0][2]) == (mp_batch.shape[1], mp_batch.shape[2]),
      f"encode {mp_sizes[0]} batch {tuple(mp_batch.shape)}")
exp_mp = match_scale_dims(mp_batch.shape[2], mp_batch.shape[1])
exp_mp_in = [_resize(mp_batch[i:i + 1], *exp_mp, "disabled") for i in range(3)]
check("INTERNAL: order preserved (frame i == <Picture i+1>)",
      all(vae_mp.calls[i]["hash"] == tensor_hash(exp_mp_in[i]) for i in range(3)),
      str(picture_labels(clip_mp.ref_items)))
check("INTERNAL: labels <Picture 1..3> in batch order",
      picture_labels(clip_mp.ref_items)
      == ["<Picture 1>", "<Picture 2>", "<Picture 3>"],
      str(picture_labels(clip_mp.ref_items)))

# ── 8b. LIST path: H3BatchImages 'Refs (list)' -> video images_batch ───────
print("\n== 8b. list refs -> H3ReferenceToVideo ==")
# The user flow: H3BatchImages 'Refs (list)' output (individual [1,H,W,C]
# frames, internal megapixels processing) wired into the video node's
# images_batch — each frame handled as its OWN reference, never a collated
# image. Must be identical to the batch path (8) since the frames are the
# same processed pixels.
refs_list = H3BatchImages.execute(images=dict(refsA_src), megapixels=0.5)[2]
vae_l, clip_l, refs_l = run(MiniMaxH3ReferenceToVideoBatch,
                            images_batch=refs_list)
l_sizes = [c["shape"] for c in vae_l.calls]
check("LIST: 3 refs from 3-frame list",
      len(refs_l) == 3 and len(vae_l.calls) == 3, str(l_sizes))
check("LIST: every ref encodes as [1,H,W,C]",
      all(len(s) == 4 and s[0] == 1 for s in l_sizes), str(l_sizes))
check("LIST: geometry identical to the batch path",
      l_sizes == mp_sizes, str(l_sizes))
check("LIST: order preserved (frame i == <Picture i+1>)",
      all(vae_l.calls[i]["hash"] == tensor_hash(exp_mp_in[i])
          for i in range(3)),
      str(picture_labels(clip_l.ref_items)))
check("LIST: labels <Picture 1..3> in order",
      picture_labels(clip_l.ref_items)
      == ["<Picture 1>", "<Picture 2>", "<Picture 3>"],
      str(picture_labels(clip_l.ref_items)))

# ── summary ────────────────────────────────────────────────────────────────
fails = [r for r in RESULTS if not r[1]]
print(f"\nSUMMARY: {len(RESULTS) - len(fails)}/{len(RESULTS)} checks passed")
for name, _, detail in fails:
    print(f"  FAILED: {name} — {detail}")
sys.exit(1 if fails else 0)
