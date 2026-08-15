"""MiniMax H3 Reference to Video — batch image input variant.

A sibling of the official ``MiniMaxH3ReferenceToVideo`` (V3 schema node) that
adds a single optional ``images_batch`` IMAGE input. Registers under its OWN
node_id ("H3ReferenceToVideo") — ComfyUI blocks custom nodes from overriding
core node IDs (nodes.py:2336 builds an `ignore` set from already-registered
core IDs), so a same-ID drop-in silently loses the registry slot. This node
sits next to the official one in the node menu.

The schema is derived from the parent at import time, so core updates to the
official node (new inputs, changed tooltips, etc.) propagate automatically.

Batch behavior
--------------
A connected ``images_batch`` tensor [B, H, W, C] is split frame-by-frame and
merged into the reference-image set as ``ref_image_N`` keys, numbered AFTER any
individually wired ``ref_image_*`` slots. So with ref_image_0 wired and a 5-frame
batch, the model sees ref_image_0 + ref_image_1..5 (6 image refs total). The
prompt labels them <Picture 1>..<Picture 6> in that same order — matching how
the H3 Prompt Enhancer numbers batch frames.

Behavior is identical to the official node when images_batch is disconnected,
so you can swap it into any existing workflow in place of the official node.
"""

import copy

from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo

_NODE_ID = "H3ReferenceToVideo"


class MiniMaxH3ReferenceToVideoBatch(MiniMaxH3ReferenceToVideo):
    """Official MiniMax H3 Reference to Video + optional batched ref images."""

    @classmethod
    def define_schema(cls):
        schema = copy.deepcopy(super().define_schema())
        schema.node_id = _NODE_ID  # own id: core-node override is blocked
        schema.display_name = "MiniMax H3 Reference to Video (Batch)"
        # The official node docstring mentions the fixed order; keep it, but note
        # batch frames slot in after the individually-wired ref images.
        schema.inputs = schema.inputs + [
            io.Image.Input(
                "images_batch",
                optional=True,
                tooltip="Batch of reference images (e.g. from BatchImagesNode). "
                        "Each frame becomes a numbered ref image AFTER any wired "
                        "ref_image_* slots (ref_image_0 wired + 5-frame batch -> "
                        "image0..image5). Prompt labels are <Picture 1>..<Picture N> "
                        "in the same order.",
            ),
            io.AnyType.Input(
                "images_batch_regions",
                optional=True,
                tooltip="Per-frame (y0, y1, x0, x1) content regions from the "
                        "H3BatchImages 'content_regions' output — the subject extent "
                        "inside each padded frame. When supplied, padding is cropped "
                        "back off per frame so H3's ref sizing sees each ref's true "
                        "geometry, matching individually-wired refs.",
            ),
        ]
        return schema

    @classmethod
    def execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                ref_image_size="match", ref_images=None, ref_videos=None,
                ref_video_audios=None, ref_audios=None, images_batch=None,
                images_batch_regions=None):
        if images_batch is not None:
            ref_images = dict(ref_images) if ref_images else {}
            frames = images_batch if images_batch.dim() == 4 else images_batch.unsqueeze(0)
            # content-region side-channel (padded common-box layout) — the batch
            # node emits ONE region list for the whole batch, in frame order.
            region_list = None
            if images_batch_regions is not None:
                if (isinstance(images_batch_regions, (list, tuple))
                        and len(images_batch_regions) == 1
                        and isinstance(images_batch_regions[0], (list, tuple))):
                    region_list = images_batch_regions[0]
                elif isinstance(images_batch_regions, (list, tuple)):
                    region_list = images_batch_regions
                elif hasattr(images_batch_regions, "__len__") \
                        and len(images_batch_regions) == 4:
                    region_list = [tuple(images_batch_regions)] * frames.shape[0]
            # find the next free ordinal among existing ref_image_N keys
            used = set()
            for key in ref_images.keys():
                if key.startswith("ref_image_") and key[len("ref_image_"):].isdigit():
                    used.add(int(key[len("ref_image_"):]))
            n = max(used) + 1 if used else 0
            for i, f in enumerate(frames):
                while f"ref_image_{n}" in ref_images:
                    n += 1
                # The official node reads each ref image as [B, H, W, C]
                # (img.shape[1] = H, shape[2] = W) and indexes img[:1] —
                # a 3D batch frame must be unsqueezed to match LoadImage
                # output or the resize math collapses on C as W.
                frame = f.unsqueeze(0) if f.dim() == 3 else f
                # undo the batch stack's padding: crop back to the subject's
                # content extent so H3's per-ref scaling sees native geometry.
                if region_list and i < len(region_list):
                    y0, y1, x0, x1 = region_list[i]
                    frame = frame[:, y0:y1, x0:x1, :]
                ref_images[f"ref_image_{n}"] = frame
                n += 1
        return super().execute(
            clip, vae, audio_vae, prompt, width, height, length,
            ref_image_size=ref_image_size, ref_images=ref_images,
            ref_videos=ref_videos, ref_video_audios=ref_video_audios,
            ref_audios=ref_audios)


NODE_CLASS_MAPPINGS = {_NODE_ID: MiniMaxH3ReferenceToVideoBatch}
NODE_DISPLAY_NAME_MAPPINGS = {_NODE_ID: "MiniMax H3 Reference to Video (Batch)"}
