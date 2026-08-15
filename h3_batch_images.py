"""H3 Batch Images — stack reference images WITHOUT cropping.

Why this exists
---------------
The core ``BatchImagesNode`` (comfy_extras/nodes_post_processing.py:514) stacks
images by resizing every frame to the FIRST image's dimensions with bilinear +
"center" crop. For mixed-aspect reference shots (a face closeup, a full-body
shot, a tattoo detail) that CROPS content out before the H3 reference pipeline
encodes it — body detail and framing silently lost. Individual ref_image_N
inputs don't have this problem because the H3 node match-scales each ref
independently.

This node stacks with ZERO content loss:
  1. every image is scaled to FIT inside the target box (aspect preserved,
     never upscaled — matching the H3 node's "down only" ref sizing)
  2. the short dimension is padded to the target box (replicate-edge or
     constant), so no pixel of the original is discarded
  3. frames are emitted in input order as one [B, H, W, C] batch

The H3ReferenceToVideo node then splits the batch back into per-frame refs and
match-scales each independently — the padding is neutral canvas that the VAE
encode absorbs; the subject content is intact.

Frame ordering matches the prompt enhancer's images_batch numbering (frame 0 ->
image0 -> <Picture 1>), so labels stay aligned.
"""

import torch
import torch.nn.functional as F

from comfy_api.latest import io

_NODE_ID = "H3BatchImages"

def detect_and_crop_black_bars(image, threshold=0.15, min_crop_pct=0.05,
                               var_threshold=0.06):
    """Detect and remove letterbox/pillarbox bars from a [1,H,W,C] frame.

    A row/column is a bar only if BOTH its mean brightness is below
    `threshold` AND its std is below `var_threshold` (near-uniform). The
    variance guard separates true bars (flat black/dark-gray padding) from
    dark-but-textured subject edges (shadows, vignettes), which have high
    std and are never cropped. Scanning stops at the first non-bar row/col
    from each edge. Only crops if the bar band is at least `min_crop_pct`
    of the dimension (no false-positive crops on tiny dark regions).
    Defaults tuned for real-world compressed bars: threshold 0.15 catches
    dark-gray bars, var_threshold 0.06 tolerates codec noise, min_crop_pct
    0.05 keeps the false-positive guard on thin dark edges.
    Evolved from WANAspectRatioResizer.detect_and_crop_black_bars so batched
    reference images are stripped of their own pad lines BEFORE fit+pad —
    the subject fills the frame instead of carrying dead black/gray bars
    into the VAE encode.
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
    return image[:, top:bottom, left:right, :]

def _fit_resize(image, target_w, target_h):
    """Aspect-preserving fit (never upscale). image: [1,H,W,C] -> [1,h,w,C]."""
    _, h, w, _ = image.shape
    if w == target_w and h == target_h:
        return image
    scale = min(target_w / w, target_h / h, 1.0)  # down only
    tw, th = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    if (tw, th) == (w, h):
        return image
    # channels-first for interpolate
    x = image.movedim(-1, 1)                      # [1,C,H,W]
    x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
    return x.movedim(1, -1)                       # [1,th,tw,C]

def _cover_crop(image, target_w, target_h):
    """Aspect-preserving cover (upscale allowed) + center crop to the box.

    Every output pixel comes from the subject; the ~2-6% of scaled pixels that
    fell outside the box (short-axis overhang) are cropped, so no pad band
    survives to the H3 encode. scale = max(target_w/w, target_h/h, 1.0).
    image: [1,H,W,C] -> [1,target_h,target_w,C]
    """
    _, h, w, _ = image.shape
    if w == target_w and h == target_h:
        return image
    scale = max(target_w / w, target_h / h, 1.0)
    tw, th = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    if (tw, th) == (target_w, target_h):
        if (tw, th) == (w, h):
            return image
        return F.interpolate(
            image.movedim(-1, 1), size=(th, tw),
            mode="bilinear", align_corners=False).movedim(1, -1)
    x = image.movedim(-1, 1)                      # [1,C,H,W]
    x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
    top = (th - target_h) // 2
    left = (tw - target_w) // 2
    x = x[:, :, top:top + target_h, left:left + target_w]
    return x.movedim(1, -1)                       # [1,target_h,target_w,C]

def _fit_pad(image, target_w, target_h, pad_mode):
    """Pad a fitted image to the target box. pad_mode: 'replicate'|'black'|'white'."""
    _, h, w, _ = image.shape
    if w == target_w and h == target_h:
        return image
    pad_r = target_w - w
    pad_b = target_h - h
    # pad order for F.pad (channels-first): (left, right, top, bottom)
    x = image.movedim(-1, 1)                      # [1,C,H,W]
    left, top = pad_r // 2, pad_b // 2
    right, bottom = pad_r - left, pad_b - top
    if pad_mode == "replicate":
        x = F.pad(x, (left, right, top, bottom), mode="replicate")
    else:
        value = 0.0 if pad_mode == "black" else 1.0
        x = F.pad(x, (left, right, top, bottom), mode="constant", value=value)
    return x.movedim(1, -1)                       # [1,target_h,target_w,C]

class H3BatchImages(io.ComfyNode):
    """Stack IMAGE refs into one batch without cropping (fit + pad).

    Unlike the core Batch Images node (which bilinear-resizes everything to the
    first image's box and center-crops), this scales each frame to fit inside a
    common box and pads the rest — every pixel of every reference survives.
    """

    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image"), prefix="image", min=1, max=50)
        return io.Schema(
            node_id=_NODE_ID,
            display_name="H3 Batch Images (no crop)",
            description="Stack reference images without cropping: fit + pad each frame.",
            category="image/batch",
            search_aliases=["batch", "image batch", "no crop batch", "stack images"],
            inputs=[
                io.Autogrow.Input("images", template=autogrow_template),
                io.Combo.Input(
                    "fit_mode", options=["max", "first", "fill"], default="max",
                    tooltip="Target box: 'max' = largest input dims (default, best "
                            "fidelity); 'first' = first image's dims (core-node "
                            "behavior but with pad instead of crop); 'fill' = "
                            "cover-scale each frame to fill the box, center-crop "
                            "the ~2-6% overhang, ZERO pad bands (max subject "
                            "area per encoded frame; content loss is the crop-only "
                            "overhang — for the reference portrait set the subject "
                            "encodes at ~92-95% of its native-match area vs "
                            "~88-94% for 'max', and with the subject's native "
                            "aspect instead of a padded one)."),
                io.Combo.Input(
                    "pad_mode", options=["replicate", "black", "white"],
                    default="replicate",
                    tooltip="How to fill the padding band: 'replicate' mirrors the "
                            "edge pixels (no new colors, recommended); "
                            "'black'/'white' use a constant fill."),
                io.Combo.Input(
                    "remove_black_bars", options=["none", "auto"], default="auto",
                    tooltip="'auto' strips letterbox/pillarbox bars from each input "
                            "frame (row/col mean brightness below the threshold) "
                            "BEFORE fit+pad, so source images with their own pad "
                            "lines contribute clean subject content to the encode."),
                io.Float.Input(
                    "black_bar_threshold", default=0.15, min=0.01, max=0.3,
                    step=0.01,
                    tooltip="Brightness threshold (0-1) for bar detection. "
                            "Rows/columns with mean brightness below this AND "
                            "near-uniform (std below bar_variance_threshold) "
                            "are treated as bars. 0.15 catches dark-gray "
                            "letterbox; raise toward 0.2-0.25 if brighter "
                            "bars persist."),
                io.Float.Input(
                    "bar_variance_threshold", default=0.06, min=0.0, max=0.2,
                    step=0.001,
                    tooltip="Max per-channel std (0-1) for a row/column to "
                            "count as a bar. A bar is near-uniform; dark but "
                            "textured subject edges (shadows, vignettes) have "
                            "high std and are never cropped. 0.06 tolerates "
                            "codec noise on real bars."),
            ],
            outputs=[
                io.Image.Output(),
                io.Array.Output(
                    "content_regions",
                    display_name="Content regions",
                    tooltip="Per-frame (y0, y1, x0, x1) subject extent INSIDE the "
                            "padded common box. Wire to H3ReferenceToVideo (Batch) "
                            "'images_batch_regions' to crop padding back off so each "
                            "ref keeps its native geometry for H3's per-ref scaling "
                            "(batch == individually-wired refs)."),
            ],
        )

    @classmethod
    def execute(cls, images, fit_mode="max", pad_mode="replicate",
                remove_black_bars="auto", black_bar_threshold=0.15,
                bar_variance_threshold=0.06):
        frames = list(images.values())
        if not frames:
            return io.NodeOutput(None)

        # strip source pad lines BEFORE fitting
        if remove_black_bars == "auto":
            frames = [detect_and_crop_black_bars(
                f, threshold=black_bar_threshold,
                var_threshold=bar_variance_threshold) for f in frames]

        # target box
        if fit_mode == "first":
            target_w, target_h = frames[0].shape[2], frames[0].shape[1]
        else:
            target_w = max(f.shape[2] for f in frames)
            target_h = max(f.shape[1] for f in frames)

        out = []
        regions = []
        for f in frames:
            if fit_mode == "fill":
                # cover-crop fills the whole box: no pad to crop back
                f = _cover_crop(f, target_w, target_h)
                regions.append((0, f.shape[1], 0, f.shape[2]))
            else:
                _, h_orig, w_orig, _ = f.shape
                f = _fit_resize(f, target_w, target_h)
                _, h_res, w_res, _ = f.shape
                f = _fit_pad(f, target_w, target_h, pad_mode)
                pad_l = (target_w - w_res) // 2
                pad_t = (target_h - h_res) // 2
                # content extent inside the padded box (before any padding)
                regions.append((pad_t, pad_t + h_res, pad_l, pad_l + w_res))
            out.append(f)
        return io.NodeOutput(torch.cat(out, dim=0), regions)


NODE_CLASS_MAPPINGS = {_NODE_ID: H3BatchImages}
NODE_DISPLAY_NAME_MAPPINGS = {_NODE_ID: "H3 Batch Images (no crop)"}
