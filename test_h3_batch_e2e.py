"""End-to-end harness: H3ReferenceToVideo batch vs separate refs, same conditioning?

Path A: official MiniMaxH3ReferenceToVideo with 3 separate ref_image_N inputs.
Path B: MiniMaxH3ReferenceToVideoBatch with 1 images_batch (real H3BatchImages
output — the actual workflow composition).

Both paths run through the REAL official execute() internals with a recording
fake VAE + fake CLIP. ref_blocks / presentation / layout are validated against
the REAL comfy PackedLayout.

Two data sets:
  1. synthetic same-size frames (controlled, padding-free)
  2. the 3 real aliclo PNGs, mixed dims -> H3BatchImages pads them
     (this is where the batch path can diverge from separate refs)

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
    """Replicates comfy/text_encoders/minimax.py:153-175 label loop.

    Image refs are labeled <Picture N>, video <Video N>, audio <Audio N>.
    """
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

# ── 2. batch subclass passthrough: images_batch=None must equal official ───
print("\n== 1b. batch subclass passthrough (images_batch=None) ==")
vae_p, _, refs_p = run(MiniMaxH3ReferenceToVideoBatch, ref_images=dict(refs_src))
check("passthrough ref_blocks identical to official path",
      len(refs_p) == 3 and all(
          refs_p[i]["latent_h"] == refs_a[i]["latent_h"] and
          torch.equal(refs_p[i]["latent"], refs_a[i]["latent"]) for i in range(3)))

# ── 3. real aliclo PNGs (mixed dims, real H3BatchImages composition) ───────
print("\n== 2. real aliclo PNGs (mixed dims, real H3BatchImages composition) ==")
loader = nodes.LoadImage()
sources = [loader.load_image(f)[0] for f in ("aliclo2.png", "aliclo4.png", "Aliclo3.png")]
names = ["aliclo2", "aliclo4", "Aliclo3"]
print(f"  source dims (H,W): {list(zip(names, [(s.shape[1], s.shape[2]) for s in sources]))}")
refsA_src = {"ref_image_0": sources[0], "ref_image_1": sources[1], "ref_image_2": sources[2]}
batch = H3BatchImages.execute(images=dict(refsA_src))[0]
print(f"  H3BatchImages output: {tuple(batch.shape)}  (fit_mode=max, pad=replicate, black-bar auto)")

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

# Path B ordering: per-position encode input must equal _resize(padded frame i, exp_b)
exp_b = match_scale_dims(batch.shape[2], batch.shape[1])  # all padded frames identical
exp_input_b = [_resize(batch[i:i + 1], *exp_b, "disabled") for i in range(3)]
b_order = all(vae_b.calls[i]["hash"] == tensor_hash(exp_input_b[i]) for i in range(3))
b_sizes = [c["shape"] for c in vae_b.calls]
check("Path B per-position content == batch frame order (frame 0,1,2)",
      b_order, str(b_sizes))

# CRITICAL INVARIANT (OLD, deprecated plain-batch path)
# NOTE: the plain batch path (B, no content regions) CANNOT preserve per-frame
# geometry — the stack forces one common box. These divergences were the bug;
# the region-aware path (section 2b) is the canonical fix and passes all of them.
print("  [note] plain batch path WITHOUT regions remains non-equivalent (old"
      f" behavior): A={a_sizes} B={b_sizes} (fixed by region path, see 2b)")

labels_a, labels_b = picture_labels(clip_a.ref_items), picture_labels(clip_b.ref_items)
check("Picture labels identical order (real)", labels_a == labels_b, str(labels_b))
check("Prompt enhancer contract: image0 -> <Picture 1>", labels_a[0] == "<Picture 1>")

print("  [note] plain batch path PackedLayout diverges (old behavior): "
      f"A.seq={build_layout(refs_a).seq_len} "
      f"B.seq={build_layout(refs_b).seq_len} — fixed by region path, see 2b")
# ── 2b. region-aware path: crop padding back off -> native geometry restored ─
print("\n== 2b. region-aware batch (content regions wired) ==")
batch, regions = H3BatchImages.execute(images=dict(refsA_src))
vae_c, clip_c, refs_c = run_region(
    MiniMaxH3ReferenceToVideoBatch, batch, regions)

sizes_c = [c["shape"] for c in vae_c.calls]
inv_c = [a_sizes[i] == sizes_c[i] for i in range(3)]
check("REGION: per-ref encode SIZE identical to individual (invariant)",
      all(inv_c), f"A={a_sizes} C={sizes_c}")
content_c = all(vae_a.calls[i]["hash"] == vae_c.calls[i]["hash"] for i in range(3))
check("REGION: per-ref encode CONTENT bit-identical to individual",
      content_c,
      "content differs — padding not fully cropped")
blk_hw_a = [(b["latent_h"], b["latent_w"]) for b in refs_a]
blk_hw_c = [(b["latent_h"], b["latent_w"]) for b in refs_c]
check("REGION: ref_blocks latent_h/w equal to individual",
      blk_hw_c == blk_hw_a, f"A={blk_hw_a} C={blk_hw_c}")
labels_c = picture_labels(clip_c.ref_items)
check("REGION: Picture labels identical order",
      labels_c == labels_a, str(labels_c))
lc = build_layout(refs_c)
la_real = build_layout(refs_a)
check("REGION: PackedLayout segments + seq_len identical",
      lc.segments == la_real.segments and lc.seq_len == la_real.seq_len,
      f"A.seq={la_real.seq_len} C.seq={lc.seq_len}")

# ── 4. split key ordering with a pre-wired ref ─────────────────────────────
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

# ── 4. black/gray bar removal (mean + variance guard) ─────────────────────
print("\n== 4. bar removal: mean+std guard, per-frame crop, min-crop guard ==")
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
    """Low-mean (0.03) high-std (0.03) two-tone noise edge — texture, not a bar."""
    x = _checkered(h, w).clone()
    noise = (torch.rand(h, bar_w, 3) < 0.5).float() * 0.06
    x[:, :bar_w] = noise
    x[:, w - bar_w:] = noise
    return x.unsqueeze(0)

def _edges_nonuniform(frame, tol=1e-3):
    f = frame if frame.dim() == 3 else frame[0]
    return (f.std(dim=(1, 2))[0] > tol and f.std(dim=(1, 2))[-1] > tol and
            f.std(dim=(0, 2))[0] > tol and f.std(dim=(0, 2))[-1] > tol)

# AFTER_AUTO: pure black pillarbox cropped
fb0 = _bar_frame(0.0, 8)
out0 = _crop_bars(fb0, threshold=0.10, var_threshold=0.02)
check("black pillarbox cropped (8px, mean 0)", out0.shape == (1, 128, 112, 3), str(tuple(out0.shape)))
check("black-bar crop edges non-uniform", _edges_nonuniform(out0))

# AFTER_AUTO: gray bars ABOVE the old 0.05 threshold, caught by new default 0.10
out1 = _crop_bars(_bar_frame(0.08, 8), threshold=0.10, var_threshold=0.02)
check("gray bars 0.08 cropped at default threshold 0.10", out1.shape == (1, 128, 112, 3), str(tuple(out1.shape)))
out_def = H3BatchImages.execute(
    images={"image_0": _bar_frame(0.08, 8)}, remove_black_bars="auto")[0]
check("node default threshold crops typical gray bars (0.08)",
      out_def.shape == (1, 128, 112, 3), str(tuple(out_def.shape)))

# AFTER_AUTO: gray bars ~0.12 cropped when threshold raised to 0.15
out2 = _crop_bars(_bar_frame(0.12, 8), threshold=0.15, var_threshold=0.02)
check("gray bars 0.12 cropped at threshold 0.15", out2.shape == (1, 128, 112, 3), str(tuple(out2.shape)))

# min_crop_pct guard: narrow gray band is NOT a crop target
out3 = _crop_bars(_bar_frame(0.08, 4), threshold=0.10, var_threshold=0.02)
check("4px gray bars NOT cropped (min_crop_pct guard)", out3.shape == (1, 128, 128, 3), str(tuple(out3.shape)))

# variance guard: dark-textured edge is NOT cropped
out4 = _crop_bars(_dark_textured_edge(8), threshold=0.10, var_threshold=0.02)
check("dark-textured edge NOT cropped (variance guard)", out4.shape == (1, 128, 128, 3), str(tuple(out4.shape)))

# per-frame cropping: frame 0 clean, frame 1 gray-barred -> own content box each
mixed_clean = _checkered(128, 128).unsqueeze(0)
mixed_barred = _bar_frame(0.08, 8)
check("mixed batch: clean frame untouched",
      _crop_bars(mixed_clean, threshold=0.10, var_threshold=0.02).shape == (1, 128, 128, 3))
check("mixed batch: bar frame cropped to own content",
      _crop_bars(mixed_barred, threshold=0.10, var_threshold=0.02).shape == (1, 128, 112, 3))
out_b, regs = H3BatchImages.execute(
    images={"image_0": mixed_clean, "image_1": mixed_barred},
    remove_black_bars="auto", black_bar_threshold=0.10, bar_variance_threshold=0.02)
check("mixed batch regions: clean frame full frame", regs[0] == (0, 128, 0, 128), str(regs[0]))
check("mixed batch regions: bar frame cropped region", regs[1] == (0, 128, 8, 120), str(regs[1]))
check("mixed batch output edges non-uniform",
      all(_edges_nonuniform(out_b[i:i + 1]) for i in range(2)))

# BEFORE_AUTO: 'none' keeps bars; very low threshold disables gray stripping
out_none = H3BatchImages.execute(
    images={"image_0": _bar_frame(0.08, 8)}, remove_black_bars="none")[0]
check("remove_black_bars=none leaves gray bars intact", out_none.shape == (1, 128, 128, 3), str(tuple(out_none.shape)))
out_low = H3BatchImages.execute(
    images={"image_0": _bar_frame(0.08, 8)}, remove_black_bars="auto",
    black_bar_threshold=0.01, bar_variance_threshold=0.02)[0]
check("black_bar_threshold=0.01 disables gray-bar stripping", out_low.shape == (1, 128, 128, 3), str(tuple(out_low.shape)))
out_black_low = H3BatchImages.execute(
    images={"image_0": _bar_frame(0.0, 8)}, remove_black_bars="auto",
    black_bar_threshold=0.01, bar_variance_threshold=0.02)[0]
check("black bars still stripped at threshold=0.01", out_black_low.shape == (1, 128, 112, 3), str(tuple(out_black_low.shape)))

# ── summary ────────────────────────────────────────────────────────────────
fails = [r for r in RESULTS if not r[1]]
print(f"\nSUMMARY: {len(RESULTS) - len(fails)}/{len(RESULTS)} checks passed")
for name, _, detail in fails:
    print(f"  FAILED: {name} — {detail}")
sys.exit(1 if fails else 0)
