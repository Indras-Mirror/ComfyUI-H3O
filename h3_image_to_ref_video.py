"""H3 Image to Ref Video — turn one image into a reference-video frame batch.

MiniMax H3 reference videos (ref_videos.ref_video_N on
MiniMaxH3ReferenceToVideo) must be a batched IMAGE with a frame count that
satisfies `n >= 5 and n % 17 == 5` (nodes_minimax_h3.py validates exactly
this). Feeding a single LoadImage straight into ref_video_0 fails that check —
this node expands one image into such a batch by repeating it.

With the default frame_count=5 the output is the minimum valid count: a
"static shot" reference clip. The reference supplies appearance (and, mixed
with RefBoost, identity) while motion comes from the prompt. Larger counts
waste VRAM without adding motion since all frames are identical; raise
frame_count only if you need a longer static anchor.

The frame_count input auto-snaps to the nearest valid value (>= 5, n % 17 == 5),
so 6 -> 5, 21 -> 22 (nearest up), 22 -> 22, 40 -> 39.
"""

import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def register_node(identifier: str, display_name: str):
    def decorator(cls):
        NODE_CLASS_MAPPINGS[identifier] = cls
        NODE_DISPLAY_NAME_MAPPINGS[identifier] = display_name
        return cls
    return decorator

def snap_frame_count(n: int) -> int:
    """Nearest valid H3 ref-video frame count: n >= 5 and n % 17 == 5."""
    n = max(5, int(n))
    if n % 17 == 5:
        return n
    rem = n % 17
    # n = 17k + rem; find k such that 17k + 5 is closest to n.
    below = n + (5 - rem)          # round up direction within same block
    # candidate from previous block (if rem < 5)
    prev = n - (rem - 5) if rem >= 5 else n - (17 + rem - 5)
    if prev < 5:
        return below
    return below if abs(below - n) <= abs(prev - n) else prev

@register_node("H3ImageToRefVideo", "H3 Image to Ref Video")
class H3ImageToRefVideo:
    """Expand a single image into a valid MiniMax H3 reference-video batch.

    Repeats the input image frame_count times (default 5 — the minimum valid
    H3 ref-video length) and returns it as a batched IMAGE ready for
    `ref_videos.ref_video_0` on MiniMaxH3ReferenceToVideo. The frame count
    auto-snaps to the constraint `n >= 5 and n % 17 == 5` enforced by
    MiniMaxH3ReferenceToVideo, so this output always validates.
    """

    CATEGORY = "MiniMax"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "The single image to expand into a reference-video "
                               "batch (wire from any LoadImage / Image Resize node).",
                }),
            },
            "optional": {
                "frame_count": ("INT", {
                    "default": 5, "min": 5, "max": 200, "step": 1,
                    "tooltip": "Number of repeated frames in the output batch. "
                               "Snapped to a valid H3 ref-video length "
                               "(n >= 5, n % 17 == 5). Default 5 is the minimum.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("ref_video",)
    FUNCTION = "expand"
    OUTPUT_NODE = False

    def expand(self, image: torch.Tensor, frame_count: int = 5):
        # image: [B, H, W, C] — take the first frame regardless of B.
        frame = image[0:1]
        n = snap_frame_count(frame_count)
        batch = frame.repeat(n, 1, 1, 1)
        return (batch,)
