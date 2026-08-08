"""H3 Region Attention Mask — real regional control for MiniMax H3.

Why this exists
---------------
The popular "ComfyUI-MiniMaxH3-AttentionMask" approach uses
``set_model_attn1_patch``. On MiniMax H3 that is a silent no-op: the model's
own Attention class (comfy/ldm/minimax/model.py) calls ``optimized_attention``
directly and never dispatches ``attn1_patch``. Even when it fires, the naive
``q * mask.reshape(-1)`` scales *every* packed token (text, reference images,
audio) with a vector that does not match H3's t-major frame-major video token
layout.

This node does it correctly:

1. A DIFFUSION_MODEL wrapper reads the per-run ``PackedLayout`` (geometry
   available in ``minimax_payload``) and builds a factor vector aligned to the
   exact video token span: one spatial factor per 2x2 latent patch, replicated
   across every latent frame (static spatial mask).
2. An ``optimized_attention_override`` (the same hook SolAttnPatch uses) damps
   Q or K by that factor — composed with whatever override was already
   installed on the model (SolAttn's).

Modes
-----
- ``preserve_foreground``   : background Q damped -> background tokens attend
  weakly, freeing them to change (background replacement drifts, foreground
  stays clean and locked).
- ``suppress_background``   : background K damped -> no query can read the
  background columns (foreground cannot bleed from background; the "hold
  character, suppress background" intent from Ref-V2V masking).

``strength`` fades the effect 0..1. ``sigma_start``/``sigma_end`` gate by the
current sampling sigma (1.0/0.0 = always on; set 0.8/0.2 for a warm-up style
dense schedule).

The mask is optional: disconnected = pass-through (factor all ones).
"""

import logging

import torch
import torch.nn.functional as F

from comfy.patcher_extension import WrappersMP

logger = logging.getLogger("h3o.region_mask")

_WRAPPER_KEY = "h3_region_attn_mask"
_CFG_KEY = "h3_attn_mask_cfg"


# ── factor construction ─────────────────────────────────────────────────────

def _build_factor(cfg, mask, h_grid, w_grid, seq_len, vstart, n_video,
                  frame_rows):
    """Spatial K/Q factor per video token, t-major frame-major.

    mask: [B, H, W] float 0-1 (foreground = 1.0). Resized to the 2x2 latent
    patch grid by bilinear interpolation. Returns None for a degenerate
    (all-foreground) mask so the override passes through.
    """
    if mask is None:
        return None
    m = mask.detach().float()
    if m.dim() == 2:
        m = m.unsqueeze(0)
    if m.dim() == 3:
        m = m.unsqueeze(1)                      # [1, 1, H, W]
    m = m.clamp(0.0, 1.0)
    if m.shape[-1] != w_grid or m.shape[-2] != h_grid:
        m = F.interpolate(m, size=(h_grid, w_grid),
                          mode="bilinear", align_corners=False)
    strength = float(cfg.get("strength", 0.8))
    grid = m[0, 0].reshape(-1)                  # [h_grid * w_grid]
    if grid.numel() < 1:
        return None
    per_frame = 1.0 - (1.0 - grid) * strength   # fg -> 1.0, bg -> 1-strength
    if float(per_frame.min()) >= 0.999:
        return None                             # all foreground -> no-op
    if n_video % frame_rows != 0:
        return None                             # geometry mismatch -> safe off
    latent_t = n_video // frame_rows
    factor = torch.ones(seq_len, dtype=torch.float32)
    # video tokens are t-major: token = t * frame_rows + fr, so repeat the
    # per-frame factor for every latent frame in order.
    factor[vstart:vstart + n_video] = per_frame.repeat(latent_t)
    return factor


# ── wrapper: capture H3 geometry + build factor once per run ───────────────

def make_wrapper(cfg, mask):
    def wrapper(executor, x, timestep, context, transformer_options={},
                minimax_payload=None, **kwargs):
        to = transformer_options
        c = to.get(_CFG_KEY)
        if c is not None and c.get("factor") is None and not c.get("pass"):
            payload = minimax_payload or {}
            layout = payload.get("layout")
            if layout is not None and layout.segments:
                a, b, kind = layout.segments[-1]      # target video always last
                if kind == "video":
                    try:
                        lat_h = int(x[0].shape[3])
                        lat_w = int(x[0].shape[4])
                        lat_h = (lat_h + 1) // 2 * 2  # 2x2 patch padding
                        lat_w = (lat_w + 1) // 2 * 2
                        h_grid, w_grid = lat_h // 2, lat_w // 2
                        frame_rows = h_grid * w_grid
                        n_video = b - a
                        factor = _build_factor(cfg, mask, h_grid, w_grid,
                                               layout.seq_len, a, n_video,
                                               frame_rows)
                        if factor is None:
                            c["pass"] = True
                        else:
                            c["factor"] = factor
                            logger.info(
                                "[h3o] region mask active: mode=%s strength=%.2f "
                                "grid=%dx%d frames=%d tokens=%d",
                                cfg["mode"], cfg["strength"],
                                h_grid, w_grid, n_video // frame_rows,
                                layout.seq_len)
                    except Exception as e:
                        logger.warning("[h3o] region mask init failed: %s", e)
                        c["pass"] = True
        return executor(x, timestep, context, transformer_options,
                        minimax_payload=minimax_payload, **kwargs)
    return wrapper


# ── optimized_attention_override: damp Q or K per video token ──────────────

def make_override(previous):
    from functools import partial

    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        def dense(qq, kk, vv):
            target = func if previous is None else partial(previous, func)
            return target(qq, kk, vv, heads, mask=mask,
                          attn_precision=attn_precision,
                          skip_reshape=skip_reshape,
                          skip_output_reshape=skip_output_reshape, **kwargs)

        to = kwargs.get("transformer_options") or {}
        c = to.get(_CFG_KEY)
        factor = c.get("factor") if c else None
        if factor is None:
            return dense(q, k, v)

        # token axis: H3's Attention passes [1, N, heads, hd] (skip_reshape)
        n = q.shape[1] if q.dim() == 4 else (q.shape[2] if q.dim() == 3 else 0)
        if n != factor.shape[0]:
            return dense(q, k, v)

        # sigma gate
        sigmas = to.get("sample_sigmas")
        if sigmas is not None and len(sigmas):
            s = float(sigmas[0])
            ss = float(c.get("sigma_start", 1.0))
            se = float(c.get("sigma_end", 0.0))
            if s > ss or s < se:
                return dense(q, k, v)

        g = factor.to(device=q.device, dtype=q.dtype)
        if q.dim() == 4:
            fq = g.view(1, n, 1, 1)      # [1, N, heads, hd] -> scale tokens
            fk = fq
        else:
            fq = g.view(1, 1, n, 1)      # [B, H, N, D] -> scale token axis
            fk = fq

        if c.get("mode", "preserve_foreground") == "suppress_background":
            return dense(q, k * fk, v)
        return dense(q * fq, k, v)

    return override


# ── ComfyUI node ────────────────────────────────────────────────────────────

class H3RegionAttentionMask:
    """Regional attention control for MiniMax H3 DiT self-attention.

    Damp Q (preserve_foreground) or K (suppress_background) per latent-patch
    factor from a MASK, geometrically aligned to H3's packed video tokens.
    Composes with SolAttnPatch / Spectrum / FirstBlockCache via the standard
    optimized_attention_override chain.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["preserve_foreground", "suppress_background"], {
                    "default": "preserve_foreground",
                    "tooltip": "preserve_foreground: damp background QUERY tokens "
                               "(background attends weakly, frees it to change). "
                               "suppress_background: damp background KV columns "
                               "(no query can read background — holds character "
                               "while the background is suppressed).",
                }),
                "strength": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0 = off, 1 = full damping.",
                }),
                "sigma_start": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Dense schedule gate: effect active for sigma "
                               "<= this. 1.0 = always on.",
                }),
                "sigma_end": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Effect active for sigma >= this. 0.0 = always "
                               "on.",
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Foreground = 1.0, background = 0.0. Any spatial "
                               "size — resized to the latent patch grid. Leave "
                               "unconnected for pass-through.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "h3/attention"

    def apply(self, model, mode="preserve_foreground", strength=0.8,
              sigma_start=1.0, sigma_end=0.0, mask=None):
        if not (0.0 < strength <= 1.0):
            return (model,)
        m = model.clone()
        to = m.model_options["transformer_options"]
        cfg = {
            "mode": mode,
            "strength": float(strength),
            "sigma_start": float(sigma_start),
            "sigma_end": float(sigma_end),
        }
        # chain onto any existing override (SolAttn etc.) — ours wraps it
        prev = to.get("optimized_attention_override")
        to[_CFG_KEY] = cfg
        to["optimized_attention_override"] = make_override(prev)
        m.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, _WRAPPER_KEY,
                               make_wrapper(cfg, mask))
        return (m,)


NODE_CLASS_MAPPINGS = {
    "H3RegionAttentionMask": H3RegionAttentionMask,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RegionAttentionMask": "H3 Region Attention Mask",
}
