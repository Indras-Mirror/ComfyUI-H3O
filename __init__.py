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

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
