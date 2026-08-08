"""ComfyUI-H3O — MiniMax H3 node pack by Indra's Mirror.

Aggregates the MiniMax H3 companion nodes in one place:
  - H3PromptEnhancer            (from ComfyUI-BerniniPromptEnhancer)
  - H3AspectRatioDetector       (standalone)
"""

from .h3_prompt_enhancer import H3PromptEnhancer
from .H3AspectRatioDetector import H3AspectRatioDetector

NODE_CLASS_MAPPINGS = {
    "H3PromptEnhancer": H3PromptEnhancer,
    "H3AspectRatioDetector": H3AspectRatioDetector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3PromptEnhancer": "H3 Prompt Enhancer",
    "H3AspectRatioDetector": "H3 Aspect Ratio Detector",
}

# Local-only node: H3RegionAttentionMask lives in the working copy
# (git-ignored — derived from a community sketch, kept out of the
# public repo). Present here it registers; clones fall back to the
# two core nodes.
try:
    from .h3_region_mask import H3RegionAttentionMask  # noqa: E402

    NODE_CLASS_MAPPINGS["H3RegionAttentionMask"] = H3RegionAttentionMask
    NODE_DISPLAY_NAME_MAPPINGS[
        "H3RegionAttentionMask"] = "H3 Region Attention Mask"
except ImportError:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
