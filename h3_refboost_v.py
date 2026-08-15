"""H3RefBoostV — attention-level V-boost / V-damp for MiniMax H3 ref2va.

Why this exists
---------------
H3's normalization stack cancels pure amplitude scaling: ref_scale (and the
source_scale damp) multiply the *input latent*, which then passes through
video_patch_proj (a Linear with bias) and the DiT's RMSNorm. RMSNorm is
scale-invariant, so only a subtle "bias-mediated" residual survives — that is
why ref_scale is a *weak* dial. Bernini RefBoost V2/V3 moved off amplitude
scaling for exactly this reason.

The one place the model's norms CANNOT cancel a scale is the VALUE projection
inside self-attention:

    out_i = sum_j( attn_ij * v_j )

Scaling reference rows of V makes reference content proportionally louder in
every token update, independent of LayerNorm/RMSNorm on Q and K (Q and K are
RMSNorm'd, V is not). Symmetrically, scaling SOURCE rows of V *down* is the
"anti-copy" dial: it weakens the source person so reference characters win.

This is the H3 port of Bernini's ``ref_v_boost`` / ``source_v_scale``.

How it works (no core edit)
---------------------------
A DIFFUSION_MODEL wrapper runs at the top of every forward. On first call it
reads ``payload["layout"].segments`` and maps each ref block to its absolute
row range in the packed sequence. Every call it computes the sigma-ramped
scales and writes them into ``transformer_options`` under a private key.

Each DiT block's ``attn.forward`` is wrapped (instance level, applied at
``apply()`` time). The wrapper reads that key; when absent it delegates to the
original forward unchanged (zero overhead for any other H3 workflow). When
present it scales the source rows of ``v`` down and the selected character
image rows of ``v`` up, between ``qkv_proj`` and ``optimized_attention``.

    packed sequence:  [text | ref blocks ... | audio | video]

Row indices in the packed ``h`` tensor are 1:1 with the attention ``v`` rows
(the block runs attention on the full packed sequence, no reordering), so a
``layout.segments`` ``(start, stop)`` range is directly a ``v[start:stop]``
slice.

Notes / limits
--------------
- Does NOT touch latents — the conditioning latents stay pristine; only the
  attention VALUE of the source/char rows is modulated.
- Composes with latent-level nodes (H3RefBoostChar) but NOT tested against
  attention-level patches that replace ``attn.forward`` themselves (e.g.
  Sol-Attn / Patch-Sage). Test V-boost WITHOUT those in the chain first, then
  add them back to check composition.
"""

import logging

import torch

from comfy.patcher_extension import WrappersMP

try:
    from .h3_refboost_char import (
        _ramp,
        _sigma_of,
        _parse_slots,
        _classify_char_ids,
    )
except ImportError:  # standalone import (tests) — module has no package
    from h3_refboost_char import (  # noqa: F401
        _ramp,
        _sigma_of,
        _parse_slots,
        _classify_char_ids,
    )

logger = logging.getLogger("h3o.refboost_v")

_WRAPPER_KEY = "h3_refboost_v"
_STATE_KEY = "__h3_refboost_v_state"
_TO_KEY = "h3_refboost_v_config"  # transformer_options key read by attn wrapper


def _ref_block_ranges(refs, layout):
    """Map each ref block -> (start, stop) row range in the packed sequence.

    Mirrors PackedLayout.__init__'s segment emission order: text, then
    (keyframes/cond, absent in ref2va), then refs in order, then target
    audio/video. Returns a list aligned to ``refs``; ``None`` for ref blocks
    that occupy no visual rows (pure audio).
    """
    seg = list(layout.segments) if layout is not None else []  # (start, stop, kind)
    i = 0
    while i < len(seg) and seg[i][2] in ("text", "cond"):
        i += 1
    ranges = []
    for blk in refs:
        kind = blk.get("kind")
        if kind == "image":
            ranges.append((seg[i][0], seg[i][1]))
            i += 1
        elif kind == "audio":
            if blk.get("ref_audio_t", 0) > 0:
                i += 1  # one ref_audio segment
            ranges.append(None)
        elif kind in ("video", "video_audio"):
            if blk.get("ref_audio_t", 0) > 0:
                i += 1  # ref_audio segment precedes the video's image rows
            ranges.append((seg[i][0], seg[i][1]))
            i += 1
        else:
            ranges.append(None)
    return ranges


def _compute_state(payload, x, context, cfg):
    """Resolve char/source row ranges once per run. Returns state or None."""
    refs = payload.get("refs") or []
    if not refs:
        return None
    char_ids = _classify_char_ids(refs, cfg["slots"])
    layout = payload.get("layout")
    if layout is None:
        try:
            from .h3_refboost_char import _rebuild_layout
        except ImportError:
            from h3_refboost_char import _rebuild_layout
        payload["layout"] = _rebuild_layout(payload, x, context)
        layout = payload["layout"]
    ranges = _ref_block_ranges(refs, layout)
    char_ranges = []
    src_ranges = []
    for blk, rng in zip(refs, ranges):
        if rng is None:
            continue
        kind = blk.get("kind")
        if kind == "image" and id(blk) in char_ids:
            char_ranges.append(rng)
        elif kind in ("video", "video_audio"):
            src_ranges.append(rng)
    if not char_ranges and not src_ranges:
        return None
    if cfg["debug"]:
        logger.info(
            "H3RefBoostV: %d char row range(s) %s | %d source row range(s) %s",
            len(char_ranges), char_ranges, len(src_ranges), src_ranges)
    return {"char_ranges": char_ranges, "src_ranges": src_ranges}


def make_wrapper(cfg):
    def wrapper(executor, x, timestep, context, transformer_options={},
                minimax_payload=None, **kwargs):
        payload = minimax_payload
        if payload and payload.get("refs"):
            state = payload.get(_STATE_KEY)
            if state is None:
                state = _compute_state(payload, x, context, cfg)
                if state is not None:
                    payload[_STATE_KEY] = state
            if state is not None:
                eff = _ramp(_sigma_of(timestep), cfg["schedule"], cfg["step_threshold"])
                ref_v_eff = 1.0 + (cfg["ref_v_boost"] - 1.0) * eff
                src_v_eff = 1.0 + (cfg["source_v_scale"] - 1.0) * eff
                to = dict(transformer_options)
                to[_TO_KEY] = {
                    "char_ranges": state["char_ranges"],
                    "src_ranges": state["src_ranges"],
                    "ref_v_eff": ref_v_eff,
                    "src_v_eff": src_v_eff,
                }
                transformer_options = to
        return executor(x, timestep, context, transformer_options,
                        minimax_payload=minimax_payload, **kwargs)
    return wrapper


def _make_class_attn_forward(orig):
    """Return a class-level replacement for Attention.forward.

    Class-level (NOT instance-level) is required so Sol-Attn cannot bypass it:
    Sol-Attn's compose hook only wraps *instance* attrs (`attn.__dict__.get(
    "forward")`); with the patch on the class there is no instance attr, so it
    never composes, our V-scaling runs every call, and the scaled q/k/v still
    reach ``optimized_attention`` where Sol-Attn's override dispatches eligible
    calls to its kernel. Both compose.
    """
    def forward(self, x, rope_freqs=None, transformer_options={}):
        cfg = transformer_options.get(_TO_KEY) if transformer_options else None
        if cfg is None or rope_freqs is None:
            return orig(self, x, rope_freqs=rope_freqs,
                        transformer_options=transformer_options)

        import comfy.model_management
        import comfy.quant_ops
        from comfy.ldm.modules.attention import optimized_attention

        s = x.shape[0]
        q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
        v = v.view(s, self.heads, self.head_dim)
        # V-scaling (the only norm-cancellation-proof lever). Out-of-place
        # mask multiply: `v` is a view of the split() output, so in-place
        # slice assignment (`v[a:b] = ...`) is rejected by autograd.
        scale = torch.ones(s, 1, 1, device=v.device, dtype=v.dtype)
        for a, b in cfg["src_ranges"]:
            scale[a:b] = cfg["src_v_eff"]
        for a, b in cfg["char_ranges"]:
            scale[a:b] = cfg["ref_v_eff"]
        v = v * scale

        q = q.view(1, s, self.heads, self.head_dim)
        k = k.view(1, s, self.heads, self.head_dim)
        qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
        rot = rope_freqs.shape[-3] * 2
        if comfy.model_management.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(
                q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        q = q[0]
        k = k[0]
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        out = optimized_attention(q, k, v, self.heads, mask=None,
                                  skip_reshape=True,
                                  transformer_options=transformer_options)
        return self.out_proj(out.squeeze(0))
    return forward


_PATCHED = False


def _patch_attention_class(debug=False):
    """Patch comfy.ldm.minimax.model.Attention.forward ONCE (class level).

    A runtime monkeypatch (like comfyui-rh-bernini does for Wan), NOT a core
    file edit. Inert for every call without the ``_TO_KEY`` in
    transformer_options, so other H3 workflows are unaffected. Returns the
    number of block attention modules the patch covers (for the debug log).
    """
    global _PATCHED
    if _PATCHED:
        return None
    try:
        from comfy.ldm.minimax.model import Attention
    except ImportError:
        return None
    orig = Attention.forward
    Attention.forward = _make_class_attn_forward(orig)
    _PATCHED = True
    if debug:
        logger.info("H3RefBoostV: patched class Attention.forward (composes "
                    "with Sol-Attn via optimized_attention override)")
    return 1


class H3RefBoostV:
    """Attention-level reference V-boost + source V-damp for MiniMax H3.

    Place between the model loader and the sampler (like H3RefBoostChar), and
    keep it BEFORE Spectrum in the chain so the expanded/observed layout is
    consistent. Only affects H3 ref2va runs (payload has "refs"); everything
    else passes through untouched.

    ref_slots selects which IMAGE ref slots are the character(s) to amplify
    ("0", "0,1", or "all"). Ref VIDEOS are the source and get damped.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "ref_slots": ("STRING", {
                    "default": "0",
                    "multiline": False,
                    "tooltip": "Image-ref slots to boost, 0-based, comma list "
                               "('0,1') or 'all'. Slots number the ref_images in "
                               "input order. Unselected image refs are left alone; "
                               "ref videos are the source and get damped.",
                }),
                "ref_v_boost": ("FLOAT", {
                    "default": 1.3, "min": 1.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Amplifies the SELECTED character refs inside "
                               "self-attention VALUE (the one place H3's norms "
                               "can't cancel it). 1.0 = off, 1.15-1.3 = "
                               "noticeable, 1.4-1.6 = strong, >1.8 risks "
                               "oversaturation.",
                }),
                "source_v_scale": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.5, "step": 0.05,
                    "tooltip": "Dampens the SOURCE ref-video inside attention "
                               "VALUE. <1.0 weakens the source person so refs "
                               "win; 0.7-0.9 typical. 1.0 = off.",
                }),
                "schedule": (["cosine", "linear", "sqrt", "power2", "step"], {
                    "default": "cosine",
                }),
                "step_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "For the 'step' schedule only: identity pass "
                               "switches on at this sigma fraction.",
                }),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "MiniMax"

    def apply(self, model, enabled=True, ref_slots="0", ref_v_boost=1.3,
              source_v_scale=0.8, schedule="cosine", step_threshold=0.5,
              debug=False):
        if not enabled:
            return (model,)
        diffusion_model = getattr(model.model, "diffusion_model", None)
        if diffusion_model is None or type(diffusion_model).__name__ != "MiniMaxH3Model":
            logger.warning(
                "H3RefBoostV: model is not MiniMax H3 (%s) — pass-through",
                type(diffusion_model).__name__ if diffusion_model is not None else "None")
            return (model,)
        if model.get_wrappers(WrappersMP.DIFFUSION_MODEL, _WRAPPER_KEY):
            logger.info("H3RefBoostV: wrapper already installed — pass-through")
            return (model,)

        slots = _parse_slots(ref_slots)
        cfg = {
            "slots": slots,
            "ref_v_boost": float(ref_v_boost),
            "source_v_scale": float(source_v_scale),
            "schedule": schedule,
            "step_threshold": float(step_threshold),
            "debug": bool(debug),
        }
        _patch_attention_class(debug=debug)
        patched = model.clone()
        patched.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, _WRAPPER_KEY,
                                     make_wrapper(cfg))
        logger.info(
            "H3RefBoostV installed: slots=%s ref_v_boost=%.2f source_v_scale=%.2f "
            "schedule=%s",
            "all" if slots is None else sorted(slots),
            cfg["ref_v_boost"], cfg["source_v_scale"], cfg["schedule"])
        return (patched,)


NODE_CLASS_MAPPINGS = {"H3RefBoostV": H3RefBoostV}
NODE_DISPLAY_NAME_MAPPINGS = {"H3RefBoostV": "H3 Ref Boost V (attention)"}
