"""MiniMax H3 prompt-embedding cache — V1, in-memory CPU LRU.

The H3 text encode is a 50-layer Qwen3-VL forward over the *spliced* token
stream (refs are inserted INTO the tokens, ``comfy/text_encoders/minimax.py:141-
186``), costing ~15-25 s text-only up to minutes with ref videos. ComfyUI's
node-output cache already covers seed-only changes; this module covers the
blind spot: batches with FIXED prompt + FIXED refs where ``length`` / ``width`` /
sampler vary — those re-encode today because the node signature changes even
though the encode output does not.

Correctness facts the design is built around:

- Refs are part of the encode input, so the cache key MUST include the ref
  pixels (via the tokenized structure). A text-only key would return stale
  conditioning. (Verified: ``minimax.py:141-186`` splices vision entries;
  ``qwen3vl.py:94-113`` runs one bidirectional forward over text+vision.)
- ``encode_from_tokens_scheduled`` returns ``[[cond, pooled_dict]]`` (list of
  ``[tensor, dict]`` pairs; ``comfy/sd.py:330-336``) and the H3 node mutates
  only the pooled dict, through ``conditioning_set_values`` which COPIES it
  (``comfy/hooks.py:692-700`` — ``n = [t[0], t[1].copy()]``). Returning the
  cached object across runs is therefore safe — the same semantics ComfyUI's
  own node cache relies on.
- The encode output is already on the intermediate (CPU) device
  (``comfy/sd1_clip.py:68``, ``comfy/model_management.py:1227-1228``). Nothing
  here ever touches GPU.

Key: ``xxh3(CACHE_VERSION || model_fingerprint || H(tokens))`` — the model
fingerprint separates model-file/quant namespaces; ``H(tokens)`` covers prompt
text, ref pixels, ref video frames and timestamps in one shot. ``length``,
``width``, sampler params and seed are deliberately NOT in the key (they never
reach this proxy).

Caveats (documented, matching ComfyUI node-cache semantics): clip-level hooks /
clip-schedule options are not part of the key — bump ``CACHE_VERSION`` if H3
semantics change. Non-default ``encode_from_tokens_scheduled`` options
(``unprojected`` / ``add_dict``) bypass the cache entirely rather than risk a
stale hit.
"""

import logging
import sys
import time
from collections import OrderedDict

import torch
import xxhash

from comfy_api.latest import io

CACHE_VERSION = 1
DEFAULT_MAX_BYTES = 4 * 1024 ** 3  # 4 GiB of CPU payloads

# --------------------------------------------------------------------------
# Module-level cache (survives queue items, dies on restart / node reload —
# the same lifecycle as ComfyUI's own RAM cache).
# --------------------------------------------------------------------------

_CACHE = OrderedDict()        # key (hex) -> (payload, byte_size)
_TOTAL_BYTES = 0
_HITS = 0
_MISSES = 0
_MAX_BYTES = DEFAULT_MAX_BYTES

_log = logging.getLogger("h3_embed_cache")


def set_cache_max_bytes(n):
    """Set the module-wide byte cap (0 restores the default)."""
    global _MAX_BYTES
    _MAX_BYTES = n if n and n > 0 else DEFAULT_MAX_BYTES


def reset_cache():
    """Drop all entries, counters, and restore the default byte cap."""
    global _TOTAL_BYTES, _HITS, _MISSES, _MAX_BYTES
    _CACHE.clear()
    _TOTAL_BYTES = 0
    _HITS = 0
    _MISSES = 0
    _MAX_BYTES = DEFAULT_MAX_BYTES


def stats():
    """(hits, misses, bytes, entries) — instrumentation for hit-rate study."""
    return _HITS, _MISSES, _TOTAL_BYTES, len(_CACHE)


# --------------------------------------------------------------------------
# Recursive structure hash — H(tokens)
# --------------------------------------------------------------------------

def structure_hash(obj):
    """Deterministic xxh3 over a nested structure of scalars/containers/tensors.

    Handles None, bool, int, float, str, bytes, list, tuple, dict, and
    torch.Tensor (hashed as shape + dtype + raw bytes on CPU). Sequences and
    tuples share a tag (same contents = same token stream = same encode);
    dict keys are visited in sorted order so insertion order never matters.
    Any other object type is hashed via its repr (e.g. numpy arrays).
    """
    h = xxhash.xxh3_64()
    _update(h, obj)
    return h.hexdigest()


def _update(h, obj):
    if obj is None:
        h.update(b"N")
    elif torch.is_tensor(obj):
        # Hash the full payload bytes + shape + dtype. Refs at ~1 MP cost
        # ~10-30 ms to hash vs a 20-200 s encode — acceptable.
        h.update(b"T")
        h.update(repr(tuple(obj.shape)).encode())
        h.update(str(obj.dtype).encode())
        h.update(obj.detach().cpu().contiguous().numpy().tobytes())
    elif isinstance(obj, bool):
        h.update(b"B1" if obj else b"B0")  # bool before int (bool is int subclass)
    elif isinstance(obj, str):
        h.update(b"R")
        h.update(obj.encode("utf-8", "replace"))
    elif isinstance(obj, bytes):
        h.update(b"Y")
        h.update(obj)
    elif isinstance(obj, (list, tuple)):
        h.update(b"S")
        h.update(repr(len(obj)).encode())
        for item in obj:
            _update(h, item)
    elif isinstance(obj, dict):
        h.update(b"D")
        h.update(repr(len(obj)).encode())
        # sorted by repr(key): insertion order must not change the hash
        for k in sorted(obj, key=repr):
            _update(h, k)
            _update(h, obj[k])
    elif isinstance(obj, (int, float)):
        # repr is deterministic for int/float (incl. NaN/Inf spellings)
        h.update(b"V")
        h.update(repr(obj).encode())
    else:
        # anything else (numpy arrays, custom objects, ...) — repr fallback
        h.update(b"O")
        h.update(repr(obj).encode("utf-8", "replace"))


# --------------------------------------------------------------------------
# Model fingerprint — computed lazily ONCE per wrapped CLIP
# --------------------------------------------------------------------------

_SAMPLE_PARAMS = 64   # parameters whose first bytes are sampled
_SAMPLE_BYTES = 2048  # bytes sampled from each of those parameters


def _resolve_transformer(clip):
    """Locate the Qwen3-VL transformer module inside a ComfyUI CLIP, or None.

    ``CLIP.cond_stage_model`` is a MiniMaxH3TEModel (SD1ClipModel) whose inner
    clip model lives under an attribute named after the clip ("qwen3vl_32b",
    sd1_clip.py:717-729); ``SDClipModel.transformer`` is the Qwen3VL module.
    Resolved defensively so test doubles / future refactors degrade to the
    type-name fallback instead of crashing.
    """
    cond_stage = getattr(clip, "cond_stage_model", None)
    if cond_stage is None:
        return None
    for attr in ("qwen3vl_32b", "clip_model", "cond_stage_model"):
        inner = getattr(cond_stage, attr, None)
        if inner is not None and hasattr(inner, "transformer"):
            return inner.transformer
    for _name, mod in cond_stage.named_modules():
        if "Qwen3VL" in type(mod).__name__:
            return mod
    return None


def _model_fingerprint(clip):
    """Stable id for the loaded weights: config repr + sampled weight bytes.

    Must change when the model FILE or its quant changes (int8_convrot vs
    nvfp4_awq vs bf16). Choice: hash of the Llama2_ config dataclass repr
    (covers architecture: layers, hidden size, rope, ...) + the first
    ``_SAMPLE_BYTES`` bytes of the first ``_SAMPLE_PARAMS`` parameters (covers
    the actual weight bits) + the total parameter count (guards sample aliasing
    across differently-sized models). Cheap: ~128 KB of byte reads, once per
    wrapped CLIP (memoized on the proxy).
    """
    override = getattr(clip, "_h3o_cache_fingerprint", None)
    if override is not None:
        return "override|%s" % override
    transformer = _resolve_transformer(clip)
    if transformer is None:
        return "no-model|%s.%s" % (type(clip).__module__, type(clip).__qualname__)
    h = xxhash.xxh3_64()
    config = getattr(getattr(transformer, "model", None), "config", None)
    h.update(("config:" + repr(config)).encode("utf-8", "replace"))
    total = 0
    for i, param in enumerate(transformer.parameters()):
        total += param.numel()
        if i < _SAMPLE_PARAMS:
            h.update(str(param.dtype).encode())
            flat = param.detach().reshape(-1)
            h.update(flat[:_SAMPLE_BYTES].cpu().contiguous().numpy().tobytes())
    h.update(("params:%d" % total).encode())
    return h.hexdigest()


# --------------------------------------------------------------------------
# Payload byte accounting (CPU tensors only — GPU is never touched)
# --------------------------------------------------------------------------

def _payload_bytes(obj):
    """Approximate byte size of a cached payload (for the LRU byte cap)."""
    if torch.is_tensor(obj):
        return max(1, obj.numel() * obj.element_size())
    if isinstance(obj, (list, tuple)):
        return sum(_payload_bytes(x) for x in obj)
    if isinstance(obj, dict):
        return sum(_payload_bytes(k) + _payload_bytes(v) for k, v in obj.items())
    if isinstance(obj, (str, bytes)):
        return max(1, len(obj))
    if obj is None:
        return 1
    if isinstance(obj, (bool, int, float)):
        return 8
    return max(1, sys.getsizeof(obj))


def _put(key, payload, nbytes):
    global _TOTAL_BYTES
    old = _CACHE.get(key)
    if old is not None:
        _TOTAL_BYTES -= old[1]
    _CACHE[key] = (payload, nbytes)
    _TOTAL_BYTES += nbytes
    while _TOTAL_BYTES > _MAX_BYTES and _CACHE:
        _CACHE.popitem(last=False)
        # recompute lazily below (popitem returns the entry; keep accounting exact)
        _TOTAL_BYTES = sum(b for _, b in _CACHE.values())
    if key not in _CACHE:
        # the entry alone exceeded the cap and was evicted immediately
        _TOTAL_BYTES = sum(b for _, b in _CACHE.values())


# --------------------------------------------------------------------------
# The proxy
# --------------------------------------------------------------------------

class H3EmbedCachingClip:
    """Passthrough CLIP proxy that memoizes ``encode_from_tokens_scheduled``.

    ``__getattr__`` forwards everything else (tokenize, vae, ...) to the
    wrapped clip. The cache is module-level, so every wrapped CLIP shares it
    (same model fingerprint => same namespace, across loader instances).
    """

    def __init__(self, clip, cache_dir=None):
        self._clip = clip
        self._cache_dir = cache_dir  # V1 is memory-only; API stays disk-ready for V2
        self._fingerprint = None

    def __getattr__(self, name):
        return getattr(self._clip, name)

    def encode_from_tokens_scheduled(self, tokens, *args, **kwargs):
        # Non-default options (unprojected / add_dict / show_pbar) change the
        # output but are not part of the token key — route around the cache
        # rather than risk a stale hit. The H3 nodes call with no extras.
        if args or kwargs:
            return self._clip.encode_from_tokens_scheduled(tokens, *args, **kwargs)

        if self._fingerprint is None:
            self._fingerprint = _model_fingerprint(self._clip)
        key = _make_key(self._fingerprint, tokens)

        entry = _CACHE.get(key)
        if entry is not None:
            global _HITS
            _HITS += 1
            _CACHE.move_to_end(key)
            return entry[0]

        t0 = time.perf_counter()
        out = self._clip.encode_from_tokens_scheduled(tokens)
        dt = time.perf_counter() - t0
        _put(key, out, _payload_bytes(out))
        global _MISSES
        _MISSES += 1
        _log.info("[H3EmbedCache] miss key=%s encode=%.1fs hits=%d misses=%d",
                  key[:12], dt, _HITS, _MISSES)
        return out


def _make_key(fingerprint, tokens):
    h = xxhash.xxh3_64()
    h.update(CACHE_VERSION.to_bytes(4, "little"))
    h.update(fingerprint.encode())
    h.update(structure_hash(tokens).encode())
    return h.hexdigest()


# --------------------------------------------------------------------------
# Standalone node: CLIP in -> CLIP out (t2va / no-ref / non-batch workflows)
# --------------------------------------------------------------------------

class H3EmbedCache(io.ComfyNode):
    """Wrap a CLIP so prompt-embedding results are cached across runs.

    A cache hit skips the ~20 s+ Qwen3-VL text encode when the prompt and any
    reference images/videos are unchanged (the key covers the refs' pixels).
    Node params that don't affect the encode — length, width, sampler, seed —
    intentionally do NOT invalidate the cache.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3EmbedCache",
            description="Memoize the MiniMax H3 prompt text-encode per "
                        "(model, prompt, refs). CLIP passthrough; cache hits "
                        "skip the Qwen3-VL forward entirely. Keyed on the "
                        "tokenized structure, so ref pixels and timestamps "
                        "are always part of the key. In-memory CPU LRU, "
                        "capped at 4 GiB (see h3_embed_cache.set_cache_max_bytes).",
            display_name="H3 Embed Cache",
            category="model/clip/minimax",
            inputs=[
                io.Clip.Input("clip"),
                io.Int.Input("max_bytes", default=0, min=0, max=128 * 1024 ** 3,
                             tooltip="Byte cap for the module-level LRU; "
                                     "0 = default (4 GiB)."),
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(cls, clip, max_bytes=0):
        if max_bytes and max_bytes > 0:
            set_cache_max_bytes(max_bytes)
        return io.NodeOutput(H3EmbedCachingClip(clip))
