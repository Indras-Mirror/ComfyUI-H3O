"""H3 Ref Budget — packed-sequence row accounting for MiniMax H3 ref2va.

Mirrors the model's exact arithmetic so you know, before sampling, how many
packed rows your refs cost and whether ref_repeat is pushing the sequence past
a sensible budget:

  seq_len = text_len + ref_rows + audio_rows + target_rows

where (comfy/ldm/minimax/model.py PackedLayout + comfy_extras/nodes_minimax_h3.py):
  - target_rows = latent_t * (lat_h//2) * (lat_w//2)   # 2x2 patch rows per frame
  - each image ref adds (th//16//2) * (tw//16//2) rows, x ref_repeat when boosted
  - each video ref adds ref_latent_t * its frame rows (one block, no repeat)
  - audio_rows = audio_t * 2 (stereo)
  - frame_count snaps up to 17k+5; latent_t = 2 + ((fc-5)//17)*5; audio_t = fc/24*40

There is NO hard model cap — the budget is VRAM/attention-time. This node gives
you the number so you can pick ref_repeat / ref count deliberately. Recommended
outputs are advisory only (they assume a flat per-row budget).

Rows are advisory units, not the model's token count; compare relative sizes,
not absolute magnitudes, across runs.
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def register_node(identifier: str, display_name: str):
    def decorator(cls):
        NODE_CLASS_MAPPINGS[identifier] = cls
        NODE_DISPLAY_NAME_MAPPINGS[identifier] = display_name
        return cls
    return decorator

FPS = 24
AUDIO_LATENT_FPS = 40
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344

def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n

def video_latent_t(frame_count):
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2

def temporal_shape(length):
    frame_count = align_frame_count(max(5, length))
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)

def ref_video_frame_rows(frames, width, height):
    """Rows a ref video block occupies. Frame count snaps down to 17k+5 >= 5."""
    n = max(5, int(frames))
    while n % 17 != 5:
        n -= 1
    return video_latent_t(n) * (height // 32) * (width // 32)

@register_node("H3RefBudget", "H3 Ref Budget")
class H3RefBudget:
    """Compute the packed-sequence row budget for an H3 ref2va run.

    Given target dims/length and your ref counts, report how many packed rows
    the refs cost at the current ref_repeat, and what the max repeat / extra
    ref count is under a flat row budget. Purely arithmetic — no model needed.
    """

    CATEGORY = "MiniMax"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1344, "min": 32, "max": 8192, "step": 32,
                                  "tooltip": "Target generation width."}),
                "height": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32,
                                   "tooltip": "Target generation height."}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17,
                                   "tooltip": "Target frame count at 24 fps (snaps up to 17k+5)."}),
                "ref_image_count": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1,
                                            "tooltip": "Number of ref IMAGE blocks (individually wired + batch frames)."}),
                "ref_repeat": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1,
                                       "tooltip": "ref_repeat on the boost node — multiplies image ref rows."}),
                "ref_video_count": ("INT", {"default": 0, "min": 0, "max": 3, "step": 1,
                                            "tooltip": "Number of ref VIDEO blocks."}),
                "ref_video_frames": ("INT", {"default": 5, "min": 5, "max": 60, "step": 1,
                                             "tooltip": "Frames per ref video (snaps down to 17k+5)."}),
                "text_tokens": ("INT", {"default": 8000, "min": 0, "max": 131072, "step": 500,
                                        "tooltip": "Estimated Qwen text-token count of the full enhanced prompt. "
                                                   "The 6-field rv2v prompts typically land 6k-12k."}),
                "row_budget": ("INT", {"default": 0, "min": 0, "max": 500000, "step": 1000,
                                       "tooltip": "Advisory packed-row ceiling. 0 = disabled (recommended outputs "
                                                  "report infinity/no-limit)."}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("seq_len", "target_rows", "ref_rows", "img_rows_per_ref",
                    "video_rows_per_ref", "audio_rows", "recommended_repeat", "over_budget")
    FUNCTION = "calculate"
    OUTPUT_NODE = False

    def calculate(self, width, height, length, ref_image_count, ref_repeat,
                  ref_video_count, ref_video_frames, text_tokens, row_budget):
        frame_count, latent_t, audio_t = temporal_shape(length)
        # 2x2 patch rows per latent frame: (lat//2)^2 with lat = dim//16
        frame_rows = (height // 32) * (width // 32)
        target_rows = latent_t * frame_rows
        audio_rows = audio_t * 2
        img_rows_per_ref = frame_rows  # "match" sizing scales refs to target area
        video_rows_per_ref = ref_video_frame_rows(ref_video_frames, width, height)
        ref_rows = (ref_image_count * ref_repeat * img_rows_per_ref +
                    ref_video_count * video_rows_per_ref)
        seq_len = text_tokens + ref_rows + audio_rows + target_rows

        # recommended max repeat under budget: keep seq_len <= budget
        if row_budget > 0 and ref_image_count > 0:
            base = text_tokens + audio_rows + target_rows + ref_video_count * video_rows_per_ref
            max_repeat = (row_budget - base) // (ref_image_count * img_rows_per_ref)
            recommended_repeat = max(1, min(max_repeat, 8))
            over_budget = seq_len > row_budget
        else:
            recommended_repeat = ref_repeat
            over_budget = False

        return (seq_len, target_rows, ref_rows, img_rows_per_ref,
                video_rows_per_ref, audio_rows, recommended_repeat, over_budget)
