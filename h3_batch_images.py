"""H3 Batch Images — WAN-style automatic reference batching.

One knob (megapixels) + one fit choice. Everything else is automatic:

  1. black bars are STRIPPED from every input first (WAN-style detector —
     dark + near-uniform rows/cols cropped, never a false-positive crop on
     dark-but-textured subject edges)
  2. ONE exact target box is computed for the whole batch: the dominant
     (largest-area, post-bar-strip) input's aspect at `megapixels` MP,
     aligned to 32px (WAN Aspect Ratio convention)
  3. every frame is LANCZOS-resized to that exact box:
       pad+fit  -> aspect-preserving fit inside the box, short dimension
                   padded with EDGE REPLICATE (never black — no visible bars)
       crop+fit -> cover-scale to fill the box + center-crop the overhang
                   (zero pad bands, every output pixel is subject)
  4. frames are emitted in input order as a [B, H, W, C] batch, plus the
     per-frame content regions and the `refs` LIST of individual [1,H,W,C]
     frames (same processed pixels) for the list-aware H3 ref-to-video path.

Frame ordering matches the prompt enhancer's images_batch numbering (frame 0 ->
image0 -> <Picture 1>), so labels stay aligned.
"""

import math

import torch
import torch.nn.functional as F

from comfy_api.latest import io

_NODE_ID = "H3BatchImages"
_MP_ALIGN = 32  # WAN-style alignment multiple for the megapixels target box
_BAR_THRESHOLD = 0.15   # WAN-tuned: catches dark-gray bars
_BAR_VAR = 0.06         # WAN-tuned: tolerates codec noise, guards dark texture
_MIN_CROP_PCT = 0.05    # false-positive guard: only crop bars >= 5% of the side

def detect_and_crop_black_bars(image, threshold=_BAR_THRESHOLD,
                               min_crop_pct=_MIN_CROP_PCT,
                               var_threshold=_BAR_VAR):
    """Detect and remove letterbox/pillarbox bars from a [1,H,W,C] frame.

    A row/column is a bar only if BOTH its mean brightness is below
    `threshold` AND its std is below `var_threshold` (near-uniform). The
    variance guard separates true bars (flat black/dark-gray padding) from
    dark-but-textured subject edges (shadows, vignettes), which have high
    std and are never cropped. Scanning stops at the first non-bar row/col
    from each edge. Only crops if the bar band is at least `min_crop_pct`
    of the dimension (no false-positive crops on tiny dark regions).
    Evolved from WANAspectRatioResizer.detect_and_crop_black_bars.
    """
    _, h, w, _ = image.shape
    ref = image[0]  # (H, W, C)
    row_means = ref.mean(dim=(1, 2))  # (H,)
    row_stds = ref.std(dim=(1, 2))    # (H,)
    col_means = ref.mean(dim=(0, 2))  # (W,)
    col_stds = ref.std(dim=(0, 2))    # (W,)

    row_bar = (row_means < threshold) & (row_stds < var_threshold)
    col_bar = (col_means < threshold) & (col_stds < var_threshold)

    top = 0
    while top < h and row_bar[top]:
        top += 1
    bottom = h
    while bottom > top and row_bar[bottom - 1]:
        bottom -= 1
    left = 0
    while left < w and col_bar[left]:
        left += 1
    right = w
    while right > left and col_bar[right - 1]:
        right -= 1

    min_v = int(h * min_crop_pct)
    min_h = int(w * min_crop_pct)
    if top < min_v: top = 0
    if (h - bottom) < min_v: bottom = h
    if left < min_h: left = 0
    if (w - right) < min_h: right = w

    if top == 0 and bottom == h and left == 0 and right == w:
        return image
    # Safety: never emit an empty frame (all-black input crops to nothing).
    if bottom <= top or right <= left:
        return image
    return image[:, top:bottom, left:right, :]

def _mp_target_box(frames, megapixels, align=_MP_ALIGN):
    """Exact target box: the batch's dominant aspect (largest input by area,
    AFTER bar stripping) at `megapixels` MP, aligned to `align` px.
    Matches WAN Aspect Ratio conventions (max_area + 32px)."""
    largest = max(frames, key=lambda f: f.shape[1] * f.shape[2])
    _, h, w, _ = largest.shape
    area = max(1.0, float(megapixels) * 1e6)
    aspect = w / h
    tw = max(align, int(round(math.sqrt(area * aspect) / align) * align))
    th = max(align, int(round(math.sqrt(area / aspect) / align) * align))
    return tw, th

def _resize_exact(image, target_w, target_h):
    """LANCZOS resize to the exact target box (upscale allowed).
    image: [1,H,W,C] -> [1,target_h,target_w,C]"""
    _, h, w, _ = image.shape
    if w == target_w and h == target_h:
        return image
    x = image.movedim(-1, 1)                      # [1,C,H,W]
    x = F.interpolate(x, size=(target_h, target_w),
                      mode="lanczos", antialias=True)
    return x.movedim(1, -1)                       # [1,target_h,target_w,C]

def _fit_exact(image, target_w, target_h):
    """Aspect-preserving fit inside the exact box (upscale allowed), no pad yet.
    image: [1,H,W,C] -> [1,th,tw,C]"""
    _, h, w, _ = image.shape
    if w == target_w and h == target_h:
        return image
    scale = min(target_w / w, target_h / h)
    tw, th = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    if (tw, th) == (w, h):
        return image
    return _resize_exact(image, tw, th)

def _fit_pad_replicate(image, target_w, target_h):
    """Pad a fitted image to the target box with EDGE REPLICATE (never black).
    image: [1,H,W,C] -> [1,target_h,target_w,C]"""
    _, h, w, _ = image.shape
    if w == target_w and h == target_h:
        return image
    pad_r = target_w - w
    pad_b = target_h - h
    # pad order for F.pad (channels-first): (left, right, top, bottom)
    x = image.movedim(-1, 1)                      # [1,C,H,W]
    left, top = pad_r // 2, pad_b // 2
    right, bottom = pad_r - left, pad_b - top
    x = F.pad(x, (left, right, top, bottom), mode="replicate")
    return x.movedim(1, -1)                       # [1,target_h,target_w,C]

def _cover_exact(image, target_w, target_h):
    """Aspect-preserving cover to the exact box (upscale allowed) + center crop.

    Every output pixel comes from the subject; the scaled overhang outside the
    box is center-cropped, so no pad band survives. scale = max(target_w/w,
    target_h/h). image: [1,H,W,C] -> [1,target_h,target_w,C]
    """
    _, h, w, _ = image.shape
    if w == target_w and h == target_h:
        return image
    scale = max(target_w / w, target_h / h)
    tw, th = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    if (tw, th) == (target_w, target_h):
        return _resize_exact(image, target_w, target_h)
    x = _resize_exact(image, tw, th)
    top = (th - target_h) // 2
    left = (tw - target_w) // 2
    return x[:, top:top + target_h, left:left + target_w, :]

class H3BatchImages(io.ComfyNode):
    """Reference batching, WAN-style: strip bars, one exact megapixels box,
    pad+fit (replicate edges, never black) or crop+fit (zero pad)."""

    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image"), prefix="image", min=1, max=50)
        return io.Schema(
            node_id=_NODE_ID,
            display_name="H3 Batch Images (auto)",
            description="Strip black bars, resize to one exact megapixels box "
                        "(WAN-style), pad+fit or crop+fit. Automatic.",
            category="image/batch",
            search_aliases=["batch", "image batch", "wan batch", "megapixels",
                            "stack images", "no bars"],
            inputs=[
                io.Autogrow.Input("images", template=autogrow_template),
                io.Combo.Input(
                    "fit_mode", options=["pad+fit", "crop+fit"],
                    default="pad+fit",
                    tooltip="pad+fit = aspect-preserving fit inside the exact "
                            "box, short side padded with EDGE REPLICATE (never "
                            "black — no visible bars). crop+fit = cover-scale "
                            "to fill the box + center-crop the overhang "
                            "(zero pad bands, every pixel is subject)."),
                io.Float.Input(
                    "megapixels", default=1.0, min=0.1, max=16.0, step=0.1,
                    tooltip="WAN-style target resolution in megapixels "
                            "(e.g. 0.3 / 0.4 / 1.0 / 2.0). The WHOLE batch is "
                            "resized to one exact box of that resolution: "
                            "aspect from the largest input, dimensions aligned "
                            "to 32px, every frame identical HxW. LANCZOS."),
            ],
            outputs=[
                io.Image.Output(),
                io.Array.Output(
                    "content_regions",
                    display_name="Content regions",
                    tooltip="Per-frame (y0, y1, x0, x1) subject extent INSIDE "
                            "the exact box. Optional — with megapixels the box "
                            "is uniform, so regions are only needed for the "
                            "legacy padded-batch crop path."),
                io.Image.Output(
                    "refs",
                    display_name="Refs (list)",
                    is_output_list=True,
                    tooltip="LIST of the processed frames as individual "
                            "[1,H,W,C] images (same pixels as the batch output "
                            "— bars stripped, megapixels box, padded/cropped). "
                            "Wire into H3ReferenceToVideo (Batch) "
                            "'images_batch' (or the enhancer's images_batch) so "
                            "each frame is handled as its own reference, never "
                            "a collated image."),
            ],
        )

    @classmethod
    def execute(cls, images, fit_mode="pad+fit", megapixels=1.0):
        frames = list(images.values())
        if not frames:
            return io.NodeOutput(None, None, None)

        # 1. strip source pad lines ALWAYS (WAN-style auto)
        frames = [detect_and_crop_black_bars(f) for f in frames]

        # 2. ONE exact WAN box for the whole batch
        target_w, target_h = _mp_target_box(frames, megapixels)

        # 3. every frame to the exact box: pad+fit (replicate, never black) or
        #    crop+fit (zero pad bands)
        out = []
        regions = []
        for f in frames:
            if fit_mode == "crop+fit":
                f = _cover_exact(f, target_w, target_h)
                regions.append((0, f.shape[1], 0, f.shape[2]))
            else:
                f = _fit_exact(f, target_w, target_h)
                _, h_res, w_res, _ = f.shape
                f = _fit_pad_replicate(f, target_w, target_h)
                pad_l = (target_w - w_res) // 2
                pad_t = (target_h - h_res) // 2
                regions.append((pad_t, pad_t + h_res, pad_l, pad_l + w_res))
            out.append(f)
        return io.NodeOutput(torch.cat(out, dim=0), regions, out)


NODE_CLASS_MAPPINGS = {_NODE_ID: H3BatchImages}
NODE_DISPLAY_NAME_MAPPINGS = {_NODE_ID: "H3 Batch Images (auto)"}
