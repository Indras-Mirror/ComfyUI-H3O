# SUPERSEDED 2026-08-17: redundant with H3RefToVideoBatch's internal split
# (h3_ref_to_video_batch.py). Batch is wired DIRECTLY to H3ReferenceToVideo
# (Batch) — the batch-sibling node splits internally. Kept in backup/, not
# deleted, per the h3o-batch-internal packet. Not registered/imported anymore.
"""H3 Batch to Ref Images — split a batch into individual [1,H,W,C] refs.

The H3 reference-to-video node's per-image ref slots (ref_image_N) each want a
single [1,H,W,C] IMAGE tensor, encoded and match-scaled independently. A batch
[ B, H, W, C ] carries one common box, so the naive "wire the batch straight
into images_batch" path either dilutes the subject (padded frames encoded as-is)
or needs the content_regions side-channel to crop padding back off.

This node is the explicit batch -> list conversion: it splits the batch into
individual refs so the ref-to-video path gets per-image tensors with correct
geometry, no regions required. Two modes:

  * With megapixels set on H3BatchImages upstream, every frame is already a
    uniform exact box (bars stripped, properly padded/cropped) — the split is
    a pure unstack, each ref [1,H,W,C] at the exact box. PRIMARY path.
  * With a plain fit+pad batch, supply the optional content_regions output from
    the SAME H3BatchImages node and each frame is cropped back to its content
    extent (native geometry), bit-identical to individually-wired refs.
    FALLBACK path — kept for batches that don't use megapixels.

Frame order is preserved, so ref_image_0..N-1 match the prompt enhancer's
<Picture 1>..<Picture N> numbering when wired in the same order.

Outputs beyond the batch size are None (unwired slots are not evaluated).
"""

from comfy_api.latest import io

_NODE_ID = "H3BatchToRefImages"
# matches the official node's ref_image_N autogrow (prefix "ref_image_",
# min=0, max=9 -> ref_image_0..ref_image_9)
_MAX_REFS = 10


class H3BatchToRefImages(io.ComfyNode):
    """Split a batch IMAGE into individual [1,H,W,C] ref_image_N outputs."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=_NODE_ID,
            display_name="H3 Batch to Ref Images",
            description="Split an images_batch into individual [1,H,W,C] "
                        "reference images for the H3 ref-to-video node.",
            category="image/batch",
            search_aliases=["batch to list", "split batch", "batch to refs",
                            "unstack", "batch to individual"],
            inputs=[
                io.Image.Input(
                    "images_batch",
                    tooltip=("A [B,H,W,C] batch (e.g. from H3BatchImages). "
                             "Each frame becomes one ref_image_N output in "
                             "order."),
                ),
                io.Array.Input(
                    "content_regions",
                    optional=True,
                    tooltip=("OPTIONAL fallback: the H3BatchImages 'Content "
                             "regions' output (same node as images_batch). "
                             "When supplied, each frame is cropped back to "
                             "its content extent, restoring native per-frame "
                             "geometry. Not needed when megapixels is set on "
                             "H3BatchImages — those frames are already a "
                             "uniform exact box."),
                ),
            ],
            outputs=[
                io.Image.Output(f"ref_image_{i}") for i in range(_MAX_REFS)
            ],
        )

    @classmethod
    def execute(cls, images_batch, content_regions=None):
        frames = (images_batch if images_batch.dim() == 4
                  else images_batch.unsqueeze(0))

        # Normalize the region side-channel the same way the batch sibling
        # does (h3_ref_to_video_batch.py:77-86): H3BatchImages emits ONE list
        # for the whole batch; a single 4-tuple applies to every frame.
        region_list = None
        if content_regions is not None:
            if (isinstance(content_regions, (list, tuple))
                    and len(content_regions) == 1
                    and isinstance(content_regions[0], (list, tuple))):
                region_list = content_regions[0]
            elif isinstance(content_regions, (list, tuple)):
                region_list = content_regions
            elif hasattr(content_regions, "__len__") \
                    and len(content_regions) == 4:
                region_list = [tuple(content_regions)] * frames.shape[0]

        out = []
        for i, f in enumerate(frames):
            frame = f.unsqueeze(0) if f.dim() == 3 else f
            if region_list and i < len(region_list):
                y0, y1, x0, x1 = region_list[i]
                frame = frame[:, y0:y1, x0:x1, :]
            out.append(frame)
        # unused outputs beyond the batch size -> None (unwired, not evaluated)
        out += [None] * (_MAX_REFS - len(out))
        return io.NodeOutput(*out)


NODE_CLASS_MAPPINGS = {_NODE_ID: H3BatchToRefImages}
NODE_DISPLAY_NAME_MAPPINGS = {_NODE_ID: "H3 Batch to Ref Images"}
