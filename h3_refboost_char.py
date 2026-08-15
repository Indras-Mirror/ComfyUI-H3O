"""MiniMax H3 RefBoost Char — boost only the character ref-image slot(s).

Why this exists
---------------
MiniMaxH3RefBoost applies ref_repeat / ref_scale to EVERY image reference
block. In a scene+character setup (the H3 rv2v pattern: a ref VIDEO or ref
IMAGE carries the scene, ref IMAGES define the replacement characters), that
means the scene image gets duplicated and amplified right alongside the
characters, competing with them for attention mass.

This node adds slot selection: only the ref-image slots listed in `ref_slots`
get repeated and amplified. Everything else is left alone:
  - unselected scene IMAGE refs: never scaled, never repeated
  - VIDEO refs: never scaled, never repeated; with `protect_scene` ON they are
    also never source_noise-mixed (the scene survives the identity pass)

Slot numbering
--------------
Ordered as MiniMaxH3ReferenceToVideo sends them (nodes_minimax_h3.py:218-267):
reference images first (slot 0, 1, 2, ... in the order of the ref_image_N
inputs), then reference videos. So for the rv2v pattern:

  ref_image_0  = scene image          -> ref_slots: "1,2"  (boost chars only)
  ref_image_1  = character A          -> ref_slots: "1,2"
  ref_image_2  = character B          -> ref_slots: "1,2"

or with a scene VIDEO:

  ref_video_0  = scene video (5 frames) -> never an image slot; protect_scene
  ref_image_0  = character             -> ref_slots: "0"
  ref_image_1  = second character      -> ref_slots: "0,1"

or a full character batch (images_batch input — slot count varies):

  ref_video_0  = scene video           -> protect_scene
  images_batch = 10 character frames   -> ref_slots: "all"

Everything original MiniMaxH3RefBoost does still works — expansion rebuilds
the PackedLayout and per-step rewriting keeps cond_video_latents in lockstep.
"""

import logging
import math

import torch

from comfy.patcher_extension import WrappersMP

logger = logging.getLogger("h3o.refboost_char")

_WRAPPER_KEY = "h3_refboost_char"
_STATE_KEY = "__h3_refboost_char_state"

def _ramp(p, schedule, step_threshold):
    """0 at high sigma -> 1 at low sigma, per schedule.

    p < 1e-4 is floored to 0 so the identity pass is a genuine no-op at very
    high sigma (sigma clamps to 1-1e-6, which would otherwise leave a sub-eps
    mix that still rewrites video tensors every step).
    """
    p = min(1.0, max(0.0, float(p)))
    if p < 1e-4:
        return 0.0
    if schedule == "linear":
        return p
    if schedule == "sqrt":
        return math.sqrt(p)
    if schedule == "power2":
        return p * p
    if schedule == "step":
        return 1.0 if p >= step_threshold else 0.0
    # cosine
    return (1.0 - math.cos(p * math.pi)) / 2.0

def _sigma_of(timestep):
    """model_base passes sigma*1000; flow sigma runs 1 -> 0 during sampling."""
    if torch.is_tensor(timestep):
        t = float(timestep.flatten()[0])
    else:
        t = float(timestep)
    sigma = max(t / 1000.0, 1e-6)
    return min(1.0, max(0.0, 1.0 - sigma))  # p: 0 at start, ~1 at end

def _parse_slots(slots_str):
    """'0,1,2' -> {0, 1, 2}; 'all' -> None (every image slot).

    Empty string selects no image slots (pass-through). 'all' boosts every
    image ref — useful with the images_batch input, where slot count varies.
    """
    if str(slots_str or "").strip().lower() in ("all", "*"):
        return None
    wanted = set()
    for part in str(slots_str or "").split(","):
        part = part.strip()
        if part:
            try:
                wanted.add(int(part))
            except ValueError:
                pass
    return wanted

def _classify_char_ids(refs, wanted):
    """Set of block ids (identity) whose IMAGE slot is selected for boosting.

    Image blocks are numbered 0..N in arrival order (which is also the
    ref_image_N input order). Video blocks and audio blocks never qualify.
    wanted=None means "all image slots".
    """
    char_ids = set()
    img_slot = 0
    for blk in refs:
        kind = blk.get("kind")
        if kind == "image":
            if wanted is None or img_slot in wanted:
                char_ids.add(id(blk))
            img_slot += 1
    return char_ids

def _snapshot_and_expand(refs, char_ids, ref_repeat, protect_scene):
    """Pristine latents aligned to the EXPANDED blocks + expanded refs.

    Only blocks in char_ids are marked for per-step boosting and duplicated.
    Every other latent-bearing block (scene image, video) is left structurally
    alone and, with protect_scene, never rewritten at all.
    """
    orig = []        # pristine latent per expanded latent-bearing block (dups share)
    expanded = []    # payload blocks (dups are shallow copies)
    is_char = []     # True for expanded blocks that should be boosted
    for blk in refs:
        if "latent" not in blk:
            expanded.append(blk)
            continue
        char = id(blk) in char_ids
        orig.append(blk["latent"])
        expanded.append(blk)
        is_char.append(char)
        if char and ref_repeat > 1:
            for _ in range(ref_repeat - 1):
                dup = dict(blk)  # shallow copy: same latent tensor + dims
                expanded.append(dup)
                orig.append(blk["latent"])
                is_char.append(True)
    return orig, expanded, is_char

def _rebuild_layout(payload, x, context):
    """Mirror _forward's dims (including 2x2 patch padding) for the expanded refs."""
    from comfy.ldm.minimax.model import PackedLayout

    video_x, audio_x = x[0], x[1]
    latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    lat_h = (lat_h + 1) // 2 * 2
    lat_w = (lat_w + 1) // 2 * 2
    audio_t = audio_x.shape[-1]
    text_len = context.shape[1]
    return PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                        keyframes=payload.get("keyframes"),
                        refs=payload.get("refs"),
                        frame_count=payload.get("frame_count"))

def _init_state(payload, x, context, cfg):
    """Snapshot pristine latents, expand selected image refs, rebuild layout."""
    refs = payload.get("refs") or []
    if not refs:
        return None
    char_ids = _classify_char_ids(refs, cfg["slots"])
    if not char_ids:
        return None
    orig, expanded, is_char = _snapshot_and_expand(
        refs, char_ids, cfg["ref_repeat"], cfg["protect_scene"])
    if not orig:
        return None
    changed = len(expanded) != len(refs)
    if changed:
        payload["refs"] = expanded
        payload["cond_video_latents"] = [b["latent"] for b in expanded if "latent" in b]
        if payload.get("layout") is not None:
            payload["layout"] = _rebuild_layout(payload, x, context)
    if cfg["debug"]:
        n_char = sum(1 for b in expanded if b.get("kind") == "image")
        logger.info("MiniMaxH3RefBoostChar: %d ref blocks (%d image, repeat x%d) slots=%s — %s",
                    len(expanded), n_char, cfg["ref_repeat"],
                    "all" if cfg["slots"] is None else sorted(cfg["slots"]),
                    "layout rebuilt" if changed else "layout untouched")
    return {"orig": orig, "is_char": is_char}

def _apply_step(payload, state, timestep, cfg):
    """Rewrite ONLY the selected character ref latents for this sampling step."""
    refs = payload.get("refs") or []
    orig = state["orig"]
    is_char = state["is_char"]
    p = _sigma_of(timestep)
    eff = _ramp(p, cfg["schedule"], cfg["step_threshold"])
    ref_scale = 1.0 + (cfg["ref_scale"] - 1.0) * eff
    # source damp ramps 1.0 (high sigma, fully present) -> source_scale (low
    # sigma, identity pass). source_scale factorizes so it composes with
    # source_noise: z = src * ((1-mix)*z0 + mix*noise).
    src_scale = 1.0 + (cfg.get("source_scale", 1.0) - 1.0) * eff
    mix = cfg["source_noise"] * eff
    seed = int(payload.get("seed", 0))
    cl = payload.get("cond_video_latents")
    idx = 0
    for i, blk in enumerate(refs):
        if "latent" not in blk:
            continue
        z0 = orig[idx]
        char = is_char[idx]
        kind = blk.get("kind")
        is_video = kind in ("video", "video_audio")
        if char and kind == "image":
            z = z0 if ref_scale == 1.0 else z0 * ref_scale
        elif not cfg["protect_scene"] and is_video:
            z = z0
            if mix > 0.0:
                gen = torch.Generator("cpu").manual_seed(seed)
                noise = torch.randn(z0.shape, generator=gen, dtype=torch.float32)
                noise = noise.to(device=z0.device, dtype=z0.dtype)
                z = (1.0 - mix) * z0 + mix * noise
            if src_scale != 1.0:
                z = z * src_scale
        else:
            # scene image or protected video: pristine, untouched
            z = z0
        blk["latent"] = z
        if cl is not None and idx < len(cl):
            cl[idx] = z
        idx += 1

def make_wrapper(cfg):
    def wrapper(executor, x, timestep, context, transformer_options={},
                minimax_payload=None, **kwargs):
        payload = minimax_payload
        if payload and payload.get("refs"):
            state = payload.get(_STATE_KEY)
            if state is None:
                state = _init_state(payload, x, context, cfg)
                if state is not None:
                    payload[_STATE_KEY] = state
            if state is not None:
                _apply_step(payload, state, timestep, cfg)
        return executor(x, timestep, context, transformer_options,
                        minimax_payload=minimax_payload, **kwargs)
    return wrapper

class H3RefBoostChar:
    """Boost selected reference-IMAGE slots; leave the scene ref alone.

    Place between the model loader and the sampler like MiniMaxH3RefBoost.
    Only affects H3 ref2va runs (payload has "refs"); t2va / fl2va / keyframe
    runs pass through untouched.

    rv2v pattern: wire the scene (static ref-video or scene ref-image) and the
    replacement-character ref images as normal. Set ref_slots to the character
    image slot numbers ("0", "0,1", ...). Those get ref_repeat + ref_scale;
    every other ref — including a scene ref-IMAGE in an unselected slot — is
    never scaled, never repeated, and (with protect_scene) never noise-mixed.
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
                               "('0,1') or 'all' for every image ref. Slots "
                               "number the ref_images in input order. Unselected "
                               "ref images and ALL ref videos are left untouched.",
                }),
                "ref_scale": ("FLOAT", {
                    "default": 1.5, "min": 1.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Amplitude of the SELECTED character image refs at "
                               "the identity pass (low sigma). Weak dial — RMSNorm "
                               "cancels pure scaling; pair with ref_repeat.",
                }),
                "ref_repeat": ("INT", {
                    "default": 2, "min": 1, "max": 6, "step": 1,
                    "tooltip": "Duplicate each SELECTED character image ref N times "
                               "for extra attention mass. Unselected refs are never "
                               "duplicated. 2-3 good start.",
                }),
                "protect_scene": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When ON, ref VIDEO latents are NEVER noise-mixed — "
                               "the scene video stays pristine through the identity "
                               "pass. Use ONLY when the scene video is a pure "
                               "backdrop with NO character to replace. When OFF "
                               "(default), source_noise applies to videos at low "
                               "sigma — breaking the identity of any character in "
                               "the scene while the already-committed structure "
                               "(room, lighting, framing) survives. Scene videos "
                               "that contain the person being replaced MUST be OFF.",
                }),
                "source_noise": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Only applies when protect_scene is OFF: mixes "
                               "ref-video latents toward noise at low sigma. With "
                               "protect_scene ON (rv2v scene pattern), this is "
                               "ignored.",
                }),
                "schedule": (["cosine", "linear", "sqrt", "power2", "step"], {
                    "default": "cosine",
                }),
                "step_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "For the 'step' schedule: identity pass kicks in "
                               "below this sigma fraction.",
                }),
                "debug": ("BOOLEAN", {"default": False}),
                # NOTE: appended LAST so existing saved workflows keep their
                # widgets_values alignment (trailing entries default gracefully).
                "source_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05,
                    "tooltip": "SOURCE DAMPENING (Bernini source_scale port). "
                               "Scales ref-video latents from 1.0 at high sigma "
                               "(structure pass, source fully present) down to "
                               "this value at low sigma (identity pass). <1.0 "
                               "weakens the SOURCE person so image refs win the "
                               "identity pass — the 'dampen source, amplify refs "
                               "in relation to it' lever. 0.6-0.8 typical. 1.0 = "
                               "off. Unlike source_noise, this attenuates the "
                               "source without replacing it with noise. Ignored "
                               "when protect_scene is ON."
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "MiniMax"

    def apply(self, model, enabled=True, ref_slots="0", ref_scale=1.5,
              ref_repeat=2, protect_scene=False, source_noise=0.0,
              schedule="cosine", step_threshold=0.5, debug=False,
              source_scale=1.0):
        if not enabled:
            return (model,)
        diffusion_model = getattr(model.model, "diffusion_model", None)
        if diffusion_model is None or type(diffusion_model).__name__ != "MiniMaxH3Model":
            logger.warning(
                "H3RefBoostChar: model is not MiniMax H3 (%s) — pass-through",
                type(diffusion_model).__name__ if diffusion_model is not None else "None")
            return (model,)
        if model.get_wrappers(WrappersMP.DIFFUSION_MODEL, _WRAPPER_KEY):
            logger.info("H3RefBoostChar: wrapper already installed — pass-through")
            return (model,)

        slots = _parse_slots(ref_slots)
        cfg = {
            "slots": slots,
            "ref_scale": float(ref_scale),
            "ref_repeat": int(ref_repeat),
            "protect_scene": bool(protect_scene),
            "source_noise": float(source_noise),
            "source_scale": float(source_scale),
            "schedule": schedule,
            "step_threshold": float(step_threshold),
            "debug": bool(debug),
        }
        patched = model.clone()
        patched.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, _WRAPPER_KEY,
                                     make_wrapper(cfg))
        logger.info(
            "H3RefBoostChar installed: slots=%s ref_scale=%.2f ref_repeat=%d "
            "protect_scene=%s schedule=%s",
            "all" if slots is None else sorted(slots),
            cfg["ref_scale"], cfg["ref_repeat"],
            cfg["protect_scene"], cfg["schedule"])
        return (patched,)

NODE_CLASS_MAPPINGS = {"H3RefBoostChar": H3RefBoostChar}
NODE_DISPLAY_NAME_MAPPINGS = {"H3RefBoostChar": "H3 Ref Boost Char"}
