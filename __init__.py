"""ComfyUI-H3O — MiniMax H3 node pack by Indra's Mirror.

Aggregates the MiniMax H3 companion nodes in one place:
  - H3PromptEnhancer            (from ComfyUI-BerniniPromptEnhancer)
  - H3AspectRatioDetector       (standalone)
  - H3ImageToRefVideo           (single image -> valid H3 ref-video batch)
  - H3RefBoostChar              (slot-targeted character ref boost)
  - H3RefBudget                 (packed-row budget / repeat calculator)
"""

from .h3_prompt_enhancer import H3PromptEnhancer
from .h3_prompt_enhancer_plus import H3PromptEnhancerPlus
from .H3AspectRatioDetector import H3AspectRatioDetector
from .h3_image_to_ref_video import H3ImageToRefVideo
from .h3_refboost_char import H3RefBoostChar
from .h3_refboost_v import H3RefBoostV
from .h3_ref_to_video_batch import MiniMaxH3ReferenceToVideoBatch
from .h3_embed_cache import H3EmbedCache
from .h3_ref_budget import H3RefBudget
from .h3_batch_images import H3BatchImages
from .h3_tae_preview import H3TAEPreview

NODE_CLASS_MAPPINGS = {
    "H3PromptEnhancer": H3PromptEnhancer,
    "H3PromptEnhancerPlus": H3PromptEnhancerPlus,
    "H3AspectRatioDetector": H3AspectRatioDetector,
    "H3ImageToRefVideo": H3ImageToRefVideo,
    "H3RefBoostChar": H3RefBoostChar,
    "H3RefBoostV": H3RefBoostV,
    "H3RefBudget": H3RefBudget,
    "H3BatchImages": H3BatchImages,
    "H3TAEPreview": H3TAEPreview,
    # Sibling of the official node: adds the images_batch input.
    # (Same-ID override is blocked by ComfyUI's ignore set.)
    "H3ReferenceToVideo": MiniMaxH3ReferenceToVideoBatch,
    "H3EmbedCache": H3EmbedCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3PromptEnhancer": "H3 Prompt Enhancer",
    "H3PromptEnhancerPlus": "H3 Prompt Enhancer Plus",
    "H3AspectRatioDetector": "H3 Aspect Ratio Detector",
    "H3ImageToRefVideo": "H3 Image to Ref Video",
    "H3RefBoostChar": "H3 Ref Boost Char",
    "H3RefBoostV": "H3 Ref Boost V (attention)",
    "H3RefBudget": "H3 Ref Budget",
    "H3BatchImages": "H3 Batch Images (no crop)",
    "H3TAEPreview": "H3 TAE Preview",
    "H3ReferenceToVideo": "MiniMax H3 Reference to Video (Batch)",
    "H3EmbedCache": "H3 Embed Cache",
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
