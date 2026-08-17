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
        # INPUT_IS_LIST: images_batch can arrive as a python LIST of individual
        # frames (from H3BatchImages 'Refs (list)' or the enhancer's
        # ref_images_out) OR a batched tensor — either way each frame becomes
        # its OWN ref_image_N. The executor wraps every input in a list; all
        # scalar inputs are unwrapped in execute() before super().execute().
        schema.is_input_list = True
        schema.inputs = schema.inputs + [
            io.Image.Input(
                "images_batch",
                optional=True,
                tooltip="Reference images — a LIST of individual [1,H,W,C] frames "
                        "(e.g. H3BatchImages 'Refs (list)' or the Prompt Enhancer "
                        "Plus 'ref_images_out') or a batched tensor. Each frame "
                        "becomes a numbered ref image AFTER any wired ref_image_* "
                        "slots (ref_image_0 wired + 5 frames -> image0..image5). "
                        "Prompt labels are <Picture 1>..<Picture N> in the same "
                        "order.",
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

    @staticmethod
    def _un1(v):
        """INPUT_IS_LIST wraps every input in a 1-element list — unwrap scalars."""
        if isinstance(v, tuple) and len(v) == 1 and v[0] is None:
            return None
        return v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v

    @staticmethod
    def _unwrap_dict(d):
        """INPUT_IS_LIST wraps each AUTO-DICT value in a 1-element list too.

        The autogrow dict inputs (ref_images / ref_videos / ref_video_audios /
        ref_audios) arrive as {"ref_video_0": [tensor], ...} under INPUT_IS_LIST
        — each value must be unwrapped or the official node sees a list where a
        tensor belongs. A no-op for direct (test) calls.
        """
        if not d:
            return d
        return {k: (v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v)
                for k, v in d.items()}

    @classmethod
    def execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                ref_image_size="match", ref_images=None, ref_videos=None,
                ref_video_audios=None, ref_audios=None, images_batch=None,
                images_batch_regions=None):
        # INPUT_IS_LIST unwrap — the executor wraps every input in a list.
        clip = cls._un1(clip)
        vae = cls._un1(vae)
        audio_vae = cls._un1(audio_vae)
        prompt = cls._un1(prompt)
        width = cls._un1(width)
        height = cls._un1(height)
        length = cls._un1(length)
        ref_image_size = cls._un1(ref_image_size)
        ref_images = cls._unwrap_dict(cls._un1(ref_images))
        ref_videos = cls._unwrap_dict(cls._un1(ref_videos))
        ref_video_audios = cls._unwrap_dict(cls._un1(ref_video_audios))
        ref_audios = cls._unwrap_dict(cls._un1(ref_audios))
        images_batch_regions = cls._un1(images_batch_regions)

        if images_batch is not None:
            images_batch = cls._un1(images_batch)
            ref_images = dict(ref_images) if ref_images else {}
            # images_batch may arrive as:
            #  - a LIST of individual [1,H,W,C] frames (H3BatchImages 'Refs
            #    (list)' / enhancer ref_images_out) -> frames = the list
            #  - a batched tensor [B,H,W,C] (legacy batch output) -> split here
            # Both produce per-frame refs; each frame is its OWN reference.
            if isinstance(images_batch, (list, tuple)):
                frames = [f.unsqueeze(0) if f.dim() == 3 else f
                          for f in images_batch if f is not None]
            else:
                frames = (images_batch if images_batch.dim() == 4
                          else images_batch.unsqueeze(0))
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
                    region_list = [tuple(images_batch_regions)] * len(frames)
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
