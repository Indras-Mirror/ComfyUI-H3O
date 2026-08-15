"""H3 TAE Preview — live preview decoder for MiniMax H3 via madebyollin's TAE.

Why this exists
---------------
ComfyUI's built-in previewer only auto-loads a TAE when the latent format sets
``taesd_decoder_name`` — and ``MiniMaxH3Video`` (comfy/latent_formats.py:570)
inherits ``None``, so H3 runs always fall back to Latent2RGB (ugly, no detail).
The core ``TAEHV`` class also hardcodes a 3-channel image head, while the H3
TAE (taeh3.safetensors) uses a 12-channel head + 24 latent channels with H3's
17-frame chunking (madebyollin's H3-specific encode/decode paths).

This node installs an OUTER_SAMPLE wrapper (same mechanism Kijai's
ModelPreviewOverride uses) that decodes each sampling step's x0 with the H3
TAE and feeds the image into ComfyUI's standard progress-bar preview channel —
no custom JS, no core edits. Place it in the model chain before the sampler:

  UNETLoader -> ... -> H3TAEPreview -> BasicGuider -> sampler

Model file: models/vae_approx/taeh3.safetensors (auto-detected, 22.7MB).

Requires tqdm (ships with ComfyUI's venv).
"""

import io
import logging
import os

import torch
from PIL import Image

from comfy.patcher_extension import WrappersMP

from .h3_tae_preview_taehv import TAEHV

logger = logging.getLogger("h3o.tae_preview")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def register_node(identifier: str, display_name: str):
    def decorator(cls):
        NODE_CLASS_MAPPINGS[identifier] = cls
        NODE_DISPLAY_NAME_MAPPINGS[identifier] = display_name
        return cls
    return decorator

_WRAPPER_KEY = "h3_tae_preview"

def _find_tae():
    import folder_paths
    for fn in folder_paths.get_filename_list("vae_approx"):
        if fn.startswith("taeh3"):
            return folder_paths.get_full_path("vae_approx", fn)
    return None

def _decode_to_preview(tae, x0, max_side=1024):
    """x0: [B, C, T, H, W] H3 video latent -> PIL RGB image (first frame)."""
    # TAEHV.decode_video expects NTCHW; x0 is [B, C, T, H, W].
    lat = x0[:1]                       # [1, C, T, H, W]
    lat = lat.movedim(1, 2)            # [1, T, C, H, W] (NTCHW)
    with torch.no_grad():
        frames = tae.decode_video(lat, parallel=False, show_progress_bar=False)
    img = frames[0, 0]                 # [C, H, W] first frame
    img = img.clamp_(0, 1).movedim(0, -1)  # [H, W, C]
    pil = Image.fromarray((img * 255).to(torch.uint8).cpu().numpy())
    if max_side and max(pil.size) > max_side:
        pil.thumbnail((max_side, max_side), Image.LANCZOS)
    return pil

class _H3TAEPreviewWrapper:
    def __init__(self, tae, max_side=1024):
        self.tae = tae
        self.max_side = max_side

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask,
                 callback, disable_pbar, seed, latent_shapes):
        pbar = None
        try:
            from comfy.utils import ProgressBar
            n_steps = max(0, len(sigmas) - 1) if sigmas is not None else 0
            pbar = ProgressBar(n_steps)
        except Exception as e:
            logger.warning(f"[H3TAEPreview] ProgressBar init failed: {e}")

        def new_callback(step, x0, x, total_steps):
            if pbar is not None:
                preview_bytes = None
                try:
                    if x0.is_nested:
                        x0 = x0.tensors[0]
                    pil = _decode_to_preview(self.tae, x0, self.max_side)
                    buf = io.BytesIO()
                    pil.save(buf, format="JPEG", quality=85)
                    preview_bytes = buf.getvalue()
                except Exception as e:
                    logger.warning(f"[H3TAEPreview] decode failed (falling back): {e}")
                pbar.update_absolute(step + 1, total_steps, preview_bytes)
            if callback is not None:
                callback(step, x0, x, total_steps)

        return executor(noise, latent_image, sampler, sigmas, denoise_mask,
                        new_callback, disable_pbar, seed, latent_shapes)

@register_node("H3TAEPreview", "H3 TAE Preview")
class H3TAEPreview:
    """Live H3 preview via taeh3.safetensors (madebyollin's TAE).

    Decodes each sampling step's latent to a JPEG preview through ComfyUI's
    standard progress-bar channel. Place in the model chain before the sampler.
    """

    CATEGORY = "MiniMax"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "max_preview_side": ("INT", {
                    "default": 1024, "min": 128, "max": 2048, "step": 64,
                    "tooltip": "Max preview side in pixels (downscaled for transport).",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    OUTPUT_NODE = False

    def apply(self, model, max_preview_side=1024):
        path = _find_tae()
        if path is None:
            raise RuntimeError(
                "H3TAEPreview: models/vae_approx/taeh3.safetensors not found. "
                "Download it from "
                "https://github.com/madebyollin/taehv/blob/main/safetensors/taeh3.safetensors")
        tae = TAEHV(checkpoint_path=path).eval()
        sd = torch.load(path, map_location="cpu")
        tae.load_state_dict(sd, strict=False)
        logger.info("H3TAEPreview: loaded %s", os.path.basename(path))

        if model.get_wrappers(WrappersMP.DIFFUSION_MODEL, _WRAPPER_KEY):
            logger.info("H3TAEPreview: wrapper already installed — pass-through")
            return (model,)
        patched = model.clone()
        patched.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, _WRAPPER_KEY,
                                     _H3TAEPreviewWrapper(tae, max_preview_side))
        return (patched,)
