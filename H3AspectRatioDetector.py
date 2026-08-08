"""H3 Aspect Ratio Detector — feed a LoadImage and pick the Resolution Selector
aspect.

Outputs the exact `aspect_ratio` option string the core "Resolution Selector"
node expects, chosen to match the image's own dimensions (orientation-aware).
Connect the `aspect_ratio` output straight into the Resolution Selector's
aspect_ratio socket and the width/height follow automatically.
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


# Mirrors comfy_extras/nodes_resolution.py AspectRatio enum — kept up to date
# with the core Resolution Selector. Format: (option string, w_ratio, h_ratio)
ASPECT_RATIOS = [
    ("1:1 (Square)", 1, 1),
    ("2:3 (Portrait Photo)", 2, 3),
    ("3:2 (Photo)", 3, 2),
    ("3:4 (Portrait Standard)", 3, 4),
    ("4:3 (Standard)", 4, 3),
    ("9:16 (Portrait Widescreen)", 9, 16),
    ("16:9 (Widescreen)", 16, 9),
    ("21:9 (Ultrawide)", 21, 9),
]

ASPECT_RATIO_NAMES = [opt[0] for opt in ASPECT_RATIOS]


@register_node("H3AspectRatioDetector", "H3 Aspect Ratio Detector")
class H3AspectRatioDetector:
    """Detect the aspect ratio of an input image and emit the matching
    Resolution Selector option, so the H3 i2v canvas automatically matches
    the source image's framing (no manual selection needed).

    Picks the option whose w/h is closest to the image — orientation-aware
    (a 3:4 portrait image selects "3:4 (Portrait Standard)", a 16:9 landscape
    selects "16:9 (Widescreen)"). The COMBO-typed output plugs straight into
    the Resolution Selector's aspect_ratio input.
    """

    CATEGORY = "MiniMax"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "The image whose aspect ratio should drive the canvas "
                               "(wire from any LoadImage / Image Resize node)."
                }),
            },
            "optional": {
                "aspect_ratio": ("STRING", {
                    "default": "",
                    "tooltip": "Optional manual override using the exact Resolution "
                               "Selector option string, e.g. \"16:9 (Widescreen)\". "
                               "Leave empty for auto-detection.",
                }),
            },
        }

    RETURN_TYPES = ("COMBO", "INT", "INT", "FLOAT",)
    RETURN_NAMES = ("aspect_ratio", "width", "height", "ratio",)
    FUNCTION = "detect"

    def detect(self, image: torch.Tensor, aspect_ratio: str = ""):
        _, h, w, _ = image.shape

        if aspect_ratio and aspect_ratio.strip():
            manual = aspect_ratio.strip()
            for name, w_r, h_r in ASPECT_RATIOS:
                if name == manual:
                    return (name, int(w), int(h), float(w / h))

        image_ratio = w / h
        best_name = None
        best_err = float("inf")
        for name, w_r, h_r in ASPECT_RATIOS:
            err = abs(image_ratio - w_r / h_r)
            if err < best_err:
                best_err = err
                best_name = name

        return (best_name, int(w), int(h), float(w / h))