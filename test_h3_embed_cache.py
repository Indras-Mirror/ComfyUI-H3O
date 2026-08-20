"""Standalone tests for h3_embed_cache.py (V1 prompt-embedding cache).

Covers: hit/miss semantics, key sensitivity (prompt / ref pixels / video
frames), model-fingerprint namespacing, LRU eviction under a tiny byte cap,
proxy passthrough, and recursive-hasher determinism. The last section drives
the REAL MiniMaxH3ReferenceToVideoBatch node with a fake VAE + fake CLIP to
prove the integration claim: same prompt + same refs, different length => the
second run's encode is a cache hit (encode skipped entirely).

Usage:
    cd /media/mal/Crucible/AI-ART/ComfyUI
    ./venv/bin/python custom_nodes/ComfyUI-H3O/test_h3_embed_cache.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.realpath(__file__))
_COMFY_ROOT = os.path.dirname(sys.prefix)
sys.path.insert(0, _COMFY_ROOT)

import torch  # noqa: E402

sys.path.insert(0, _HERE)
import h3_embed_cache as h3c  # noqa: E402
from h3_embed_cache import (  # noqa: E402
    H3EmbedCachingClip,
    structure_hash,
    stats,
    reset_cache,
    set_cache_max_bytes,
)
from h3_ref_to_video_batch import MiniMaxH3ReferenceToVideoBatch  # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    status = "ok " if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


def make_tokens(prompt="a cat", ref_tensors=None, video=False):
    """Mirror of comfy/text_encoders/minimax.py:141-186 token structure:
    top-level {"qwen3vl_32b": [entries]} with (int|vision-dict, weight)
    tuples; vision entries carry the ref pixels in "data"."""
    entries = []
    for t in (ref_tensors or []):
        entry = {"type": "image", "data": t, "original_type": "image"}
        if video:
            entry["minimax_video_block"] = True
        entries += [(151652, 1.0), (entry, 1.0), (151653, 1.0)]
    entries += [(ord(c), 1.0) for c in prompt]
    return {"qwen3vl_32b": [entries]}


class FakeClip:
    """Deterministic fake CLIP: encode output derives from tokens + seed.

    ``_h3o_cache_fingerprint`` mimics the model-identity hook; when None the
    proxy falls back to the class-based "no-model" fingerprint.
    """

    def __init__(self, seed=0, fingerprint="modelA"):
        self.seed = seed
        self._h3o_cache_fingerprint = fingerprint
        self.encode_calls = 0
        self.tokenize_calls = 0
        self.attr_marker = "passthrough-ok"

    def tokenize(self, prompt, minimax_ref_items=None, **kw):
        self.tokenize_calls += 1
        entries = []
        counters = {"image": 0, "audio": 0, "video": 0}
        for item in (minimax_ref_items or []):
            kind = item["type"]
            counters[kind] += 1
            if kind == "image":
                entry = {"type": "image", "data": item["data"],
                         "original_type": "image"}
                entries += [(151652, 1.0), (entry, 1.0), (151653, 1.0)]
            elif kind == "video":
                entry = {"type": "image", "data": item["data"],
                         "original_type": "image", "minimax_video_block": True}
                entries += [(151652, 1.0), (entry, 1.0), (151653, 1.0)]
            # audio: label text only — skipped (matches minimax.py)
        entries += [(ord(c), 1.0) for c in prompt]
        return {"qwen3vl_32b": [entries]}

    def encode_from_tokens_scheduled(self, tokens, *args, **kwargs):
        self.encode_calls += 1
        seq = len(tokens["qwen3vl_32b"][0])
        cond = torch.full((1, seq, 4), float(seq + self.seed))
        pooled = torch.full((1, 4), float(self.seed))
        tags = torch.arange(seq, dtype=torch.long)
        return [[cond, {"pooled_output": pooled, "minimax_token_tags": tags}]]


class FakeVAE:
    def __init__(self):
        self.calls = 0

    def encode(self, x):
        self.calls += 1
        lh, lw = x.shape[2] // 16, x.shape[3] // 16
        return torch.full((1, 24, 1, lh, lw), 0.25)


def ref_img(seed=0, h=64, w=64):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, h, w, 3, generator=g)


def encode_payload(proxy, tokens):
    """Encode once and extract (cond, pooled) — same shape as the H3 node sees."""
    return proxy.encode_from_tokens_scheduled(tokens)


# ── 1. hit / miss semantics ────────────────────────────────────────────────
print("== 1. hit / miss semantics ==")
reset_cache()
clip = FakeClip(seed=3)
proxy = H3EmbedCachingClip(clip)
t1 = make_tokens("a cat")
p1 = encode_payload(proxy, t1)
p2 = encode_payload(proxy, t1)
check("hit: second encode served from cache (clip.encode called once)",
      clip.encode_calls == 1, f"calls={clip.encode_calls}")
hits, misses, cbytes, entries = stats()
check("hit: counters (hits=1, misses=1)",
      hits == 1 and misses == 1, f"hits={hits} misses={misses}")
check("hit: returned cond bit-equal to the first encode",
      torch.equal(p1[0][0], p2[0][0])
      and torch.equal(p1[0][1]["pooled_output"], p2[0][1]["pooled_output"]))
check("hit: payload byte accounting is positive",
      cbytes > 0 and entries == 1, f"bytes={cbytes} entries={entries}")

t2 = make_tokens("a dog")  # same length, different text ids
encode_payload(proxy, t2)
hits, misses, _, _ = stats()
check("miss on prompt change (same length)",
      clip.encode_calls == 2 and misses == 2, f"calls={clip.encode_calls}")

t3 = make_tokens("a cat", ref_tensors=[ref_img(0)])
encode_payload(proxy, t3)
t4 = make_tokens("a cat", ref_tensors=[ref_img(1)])
encode_payload(proxy, t4)
check("miss on ref pixel change (refs are part of the key)",
      clip.encode_calls == 4, f"calls={clip.encode_calls}")

encode_payload(proxy, t3)
check("hit on identical tokens after refs (still cached)",
      clip.encode_calls == 4, f"calls={clip.encode_calls}")

t5 = make_tokens("a cat", ref_tensors=[torch.rand(2, 32, 32, 3)], video=True)
t6 = make_tokens("a cat", ref_tensors=[torch.rand(2, 32, 32, 3)], video=True)
encode_payload(proxy, t5)
encode_payload(proxy, t6)
check("miss on ref VIDEO frame change (video block in key)",
      clip.encode_calls == 6, f"calls={clip.encode_calls}")

# ── 2. model fingerprint namespacing ───────────────────────────────────────
print("\n== 2. model fingerprint namespacing ==")
reset_cache()
clip_a = FakeClip(seed=1, fingerprint="modelA")
clip_b = FakeClip(seed=1, fingerprint="modelB")
t = make_tokens("same prompt")
encode_payload(H3EmbedCachingClip(clip_a), t)
encode_payload(H3EmbedCachingClip(clip_b), t)
hits, misses, _, _ = stats()
check("different fingerprint -> miss (separate namespace)",
      clip_a.encode_calls == 1 and clip_b.encode_calls == 1 and misses == 2,
      f"misses={misses}")
clip_c = FakeClip(seed=1, fingerprint="modelA")
encode_payload(H3EmbedCachingClip(clip_c), t)
hits, _, _, _ = stats()
check("same fingerprint across instances -> hit, encode skipped entirely",
      clip_c.encode_calls == 0 and hits == 1, f"calls={clip_c.encode_calls} hits={hits}")

clip_f1 = FakeClip(seed=2, fingerprint=None)
clip_f2 = FakeClip(seed=2, fingerprint=None)
t_f = make_tokens("fallback")
encode_payload(H3EmbedCachingClip(clip_f1), t_f)
encode_payload(H3EmbedCachingClip(clip_f2), t_f)
check("no-model fallback fingerprint is class-based (no cond_stage_model)",
      clip_f2.encode_calls == 0, f"calls={clip_f2.encode_calls}")

# ── 3. proxy passthrough ───────────────────────────────────────────────────
print("\n== 3. proxy passthrough ==")
reset_cache()
clip_p = FakeClip()
proxy_p = H3EmbedCachingClip(clip_p)
check("non-encode attributes reachable",
      proxy_p.attr_marker == "passthrough-ok", str(proxy_p.attr_marker))
proxy_p.tokenize("hi")
check("tokenize forwards to the wrapped clip",
      clip_p.tokenize_calls == 1, f"calls={clip_p.tokenize_calls}")
encode_payload(proxy_p, make_tokens("x"))
proxy_p.encode_from_tokens_scheduled(make_tokens("y"), show_pbar=False)
proxy_p.encode_from_tokens_scheduled(make_tokens("y"), show_pbar=False)
hits, misses, _, entries = stats()
check("non-default encode options bypass the cache (no stale risk)",
      clip_p.encode_calls == 3 and entries == 1 and hits == 0 and misses == 1,
      f"calls={clip_p.encode_calls} entries={entries}")
encode_payload(proxy_p, make_tokens("z"))
encode_payload(proxy_p, make_tokens("z"))
hits, misses, _, entries = stats()
check("default call caches again after bypass",
      clip_p.encode_calls == 4 and hits == 1 and misses == 2 and entries == 2,
      f"calls={clip_p.encode_calls} hits={hits} misses={misses} entries={entries}")

# ── 4. LRU eviction under a tiny byte cap ──────────────────────────────────
print("\n== 4. LRU eviction ==")
reset_cache()
clip_l = FakeClip(seed=5)
proxy_l = H3EmbedCachingClip(clip_l)
# Cap derived from a MEASURED entry size so the assertions hold regardless of
# the exact byte accounting: cap = 2 entries + slack -> exactly 2 fit, 3rd evicts.
# The probe must be the SAME payload size as the keys below (equal seq).
encode_payload(proxy_l, make_tokens("pppppp"))  # same seq as keys -> same size
_, _, one_size, _ = stats()
cap = one_size * 2 + 10
set_cache_max_bytes(cap)
keys = [make_tokens(c * 6) for c in "abcde"]  # equal seq -> equal payload size
for k in keys:
    encode_payload(proxy_l, k)
hits, misses, cbytes, entries = stats()
check("byte cap enforced (bytes <= cap, entries bounded)",
      cbytes <= cap and entries <= 2, f"bytes={cbytes} cap={cap} entries={entries}")
encode_payload(proxy_l, keys[0])
check("evicted oldest key re-encodes (miss)",
      clip_l.encode_calls == 7, f"calls={clip_l.encode_calls}")
encode_payload(proxy_l, keys[4])
check("most-recent key still served from cache",
      clip_l.encode_calls == 7, f"calls={clip_l.encode_calls}")
reset_cache()
check("reset restores default cap + zero state",
      stats() == (0, 0, 0, 0) and h3c.DEFAULT_MAX_BYTES == h3c._MAX_BYTES,
      str(stats()))

# ── 5. recursive hasher determinism ────────────────────────────────────────
print("\n== 5. recursive hasher determinism ==")
tA = make_tokens("alpha", ref_tensors=[ref_img(7)])
check("same input -> same hash across calls",
      structure_hash(tA) == structure_hash(tA),
      structure_hash(tA)[:12])
check("dict key order does not change the hash",
      structure_hash({"a": 1, "b": [2, {"c": 3.0}]})
      == structure_hash({"b": [2, {"c": 3.0}], "a": 1}))
check("list vs tuple with same content hash the same (same token stream)",
      structure_hash([1, "x", (2, 3)]) == structure_hash((1, "x", [2, 3])))
check("different tensor values -> different hash",
      structure_hash({"t": torch.zeros(2, 3)})
      != structure_hash({"t": torch.ones(2, 3)}))
check("different tensor shapes -> different hash",
      structure_hash({"t": torch.zeros(2, 3)})
      != structure_hash({"t": torch.zeros(3, 2)}))
check("different tensor dtypes -> different hash",
      structure_hash({"t": torch.zeros(2, dtype=torch.float32)})
      != structure_hash({"t": torch.zeros(2, dtype=torch.float64)}))
check("nested mixed structure deterministic",
      structure_hash({"qwen3vl_32b": [[(1, 1.0), ({"data": ref_img(3)}, 1.0)]]})
      == structure_hash({"qwen3vl_32b": [[(1, 1.0), ({"data": ref_img(3)}, 1.0)]]}))
check("scalar discrimination: None/False/0/0.0/1/1.0 all distinct",
      len({structure_hash(x) for x in (None, False, 0, 0.0, 1, 1.0)}) == 6)
check("empty containers deterministic + distinct",
      structure_hash([]) == structure_hash(tuple())
      and structure_hash({}) != structure_hash([]))
# non-contiguous tensor: the hasher must canonicalize (contiguous cpu bytes)
_t_view = torch.zeros(4, 4)[::2]
check("non-contiguous tensor hashes like its contiguous copy",
      structure_hash({"t": _t_view})
      == structure_hash({"t": _t_view.contiguous()}))

# ── 6. real node integration: length varies, encode skipped ────────────────
print("\n== 6. node integration: length varies, encode skipped ==")
reset_cache()
img = ref_img(0, 128, 128)
W, HT = 1344, 768


def node_run(length, prompt="test", refs=None, fps_seed=0):
    clip_n = FakeClip(seed=0, fingerprint="nodeModel")
    vae_n = FakeVAE()
    out = MiniMaxH3ReferenceToVideoBatch.execute(
        clip=clip_n, vae=vae_n, audio_vae=None, prompt=prompt,
        width=W, height=HT, length=length,
        ref_images=refs or {"ref_image_0": img})
    return clip_n, vae_n, out


clip_n1, vae_n1, _ = node_run(length=124)
clip_n2, vae_n2, _ = node_run(length=362)
# On a cache hit the wrapped clip's encode is NEVER invoked — encode skipped.
check("same prompt+refs, length 124 then 362 -> encode runs ONCE",
      clip_n1.encode_calls == 1 and clip_n2.encode_calls == 0,
      f"calls1={clip_n1.encode_calls} calls2={clip_n2.encode_calls}")
hits, misses, _, _ = stats()
check("node: second run was a cache hit",
      hits == 1 and misses == 1, f"hits={hits} misses={misses}")
check("node: VAE still encodes refs per run (only the CLIP encode is cached)",
      vae_n1.calls == 1 and vae_n2.calls == 1)

clip_n3, vae_n3, _ = node_run(length=124, refs={"ref_image_0": ref_img(9, 128, 128)})
hits, misses, _, _ = stats()
check("node: ref pixel change -> re-encode (refs in the key)",
      clip_n3.encode_calls == 1 and misses == 2, f"misses={misses}")

clip_n4, vae_n4, _ = node_run(length=124, prompt="a different prompt")
hits, misses, _, _ = stats()
check("node: prompt change -> re-encode",
      clip_n4.encode_calls == 1 and misses == 3, f"misses={misses}")

# ── summary ────────────────────────────────────────────────────────────────
reset_cache()
fails = [r for r in RESULTS if not r[1]]
print(f"\nSUMMARY: {len(RESULTS) - len(fails)}/{len(RESULTS)} checks passed")
for name, _, detail in fails:
    print(f"  FAILED: {name} — {detail}")
sys.exit(1 if fails else 0)
