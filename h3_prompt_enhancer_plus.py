"""H3 Prompt Enhancer Plus — enhanced prompt writing with image passthrough.

Inherits all functionality from H3PromptEnhancer but adds:
  1. ref_images_out — passes through reference images in the exact <Picture N>
     order used in the prompt, so wiring them to H3ReferenceToVideo's images_batch
     guarantees the same numbering on both nodes.
  2. enhanced_rules toggle — injects community-derived dialogue, audio, and
     consistency rules (from the Qwen 3.8 H3 prompt-writing guide) into the
     system prompt as an optional patch. Toggleable so you can A/B test.
"""

import torch

from .h3_prompt_enhancer import H3PromptEnhancer


# ═══════════════════════════════════════════════════════════════════════════════
# Enhanced community rules — appended to the system prompt when enabled
# ═══════════════════════════════════════════════════════════════════════════════

H3_ENHANCED_RULES = """\

ENHANCED DIALOGUE RULES:
- Never write vague dialogue instructions such as "they talk", "they argue",
  "she says something", or "he responds". If speech is desired, write the
  exact short line and assign it explicitly to a named speaker:
  <Subject N> (Sx) says clearly: <d>[English] Exact dialogue here.</d>
- Keep dialogue short for 5-7 second clips — one or two brief lines maximum.
- Avoid overlapping speech unless specifically requested.
- If there is NO dialogue, explicitly state:
  "No spoken dialogue. Characters communicate through facial expressions
  and gestures."

ENHANCED AUDIO RULES:
- Explicitly describe ambience, Foley sounds, physical impacts, wind, cloth
  movement, footsteps, breathing, and crowd sound when relevant to the scene.
- For physical actions, name the specific sound: "the metallic clang of a
  sword strike", "soft crunch of footsteps on gravel", "the rustle of fabric
  as she turns".
- For clips with no background music, write: non_diegetic_music: N/A
- Generated dialogue must always be exact text, never paraphrased.

ENHANCED CONSISTENCY RULES:
- Preserve across all shots: character identity (face, hair, body), wardrobe
  (exact clothing items, colors, fit), accessories, props, proportions, and
  environment stability.
- Do not introduce costume changes, hair changes, or location changes between
  shots unless the user explicitly requests them.
- Do not request readable text, subtitles, logos, watermarks, UI elements, or
  exact typography in the video.
- Keep each shot physically plausible — do not overload a short clip with too
  many characters, actions, locations, transformations, or camera movements.
- For dialogue, comedy, greetings, or direct-to-camera performance: prefer one
  continuous shot.
- For action, trailers, fights, chases: use up to 3-4 clear shots in a 6-7
  second clip. Do not compress too many cuts into a short duration."""


class H3PromptEnhancerPlus(H3PromptEnhancer):
    """H3 Prompt Enhancer Plus — enhanced rules + image passthrough.

    Adds ref_images_out (IMAGE batch of all reference images in <Picture N>
    order) and an enhanced_rules toggle for community dialogue/audio/consistency
    rules. Wire ref_images_out to H3ReferenceToVideo's images_batch for
    guaranteed <Picture N> alignment.
    """

    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        base["optional"]["enhanced_rules"] = ("BOOLEAN", {
            "default": True,
            "tooltip": "Inject community-derived dialogue, audio, and "
                       "consistency rules into the system prompt. These "
                       "enforce explicit dialogue lines, detailed Foley/SFX, "
                       "and cross-shot consistency. Turn OFF to use the "
                       "original system prompt only."
        })
        return base

    RETURN_TYPES = ("STRING", "STRING", "IMAGE",)
    RETURN_NAMES = ("h3_prompt", "system_prompt", "ref_images_out",)
    FUNCTION = "enhance_plus"
    CATEGORY = "h3/prompt"

    def enhance_plus(self, prompt, task_type, duration, model,
                     local_backend="off",
                     source_image=None, last_frame_image=None,
                     reference_image_0=None, reference_image_1=None,
                     reference_image_2=None, images_batch=None,
                     source_video=None,
                     temperature=0.7, max_tokens=4096,
                     custom_system_prompt="", custom_model="", api_key=None,
                     advanced_prompt="on",
                     same_subject=False, style_transfer="off",
                     auto_describe=True, auto_describe_max_tokens=4096,
                     seed=-1,
                     enhanced_rules=True):
        # Build the ref images list in the same order enhance() will
        # number them as <Picture N>
        ref_images_ordered = []

        # RI2V scene-as-image: source_image becomes <Picture 1>
        ri2v_scene_as_image = (task_type == "ri2v"
                               and source_image is not None
                               and source_video is None)
        if ri2v_scene_as_image:
            scene_img = (source_image[0] if source_image.dim() == 4
                         else source_image)
            ref_images_ordered.append(
                scene_img.unsqueeze(0) if scene_img.dim() == 3 else scene_img)

        # Source image for i2v/fl2v/l2v becomes <Picture 1>
        if source_image is not None and task_type in ("i2v", "fl2v", "l2v"):
            img = (source_image[0] if source_image.dim() == 4
                   else source_image)
            ref_images_ordered.append(
                img.unsqueeze(0) if img.dim() == 3 else img)

        # Last frame for fl2v becomes <Picture 2>
        if last_frame_image is not None and task_type == "fl2v":
            img = (last_frame_image[0] if last_frame_image.dim() == 4
                   else last_frame_image)
            ref_images_ordered.append(
                img.unsqueeze(0) if img.dim() == 3 else img)

        # Reference images in order
        if task_type in ("r2v", "rv2v", "i2v", "ri2v"):
            for ref in (reference_image_0, reference_image_1,
                        reference_image_2):
                if ref is not None:
                    img = ref[0] if ref.dim() == 4 else ref
                    ref_images_ordered.append(
                        img.unsqueeze(0) if img.dim() == 3 else img)
            if images_batch is not None:
                if images_batch.dim() == 4:
                    for f in images_batch:
                        ref_images_ordered.append(f.unsqueeze(0))
                else:
                    ref_images_ordered.append(
                        images_batch.unsqueeze(0)
                        if images_batch.dim() == 3 else images_batch)

        # Append enhanced rules to custom_system_prompt if enabled
        effective_custom = custom_system_prompt or ""
        if enhanced_rules:
            effective_custom = (effective_custom.rstrip() + "\n"
                                + H3_ENHANCED_RULES
                                if effective_custom.strip()
                                else H3_ENHANCED_RULES)

        # Call the parent's enhance method
        h3_prompt, system_prompt = self.enhance(
            prompt=prompt,
            task_type=task_type,
            duration=duration,
            model=model,
            local_backend=local_backend,
            source_image=source_image,
            last_frame_image=last_frame_image,
            reference_image_0=reference_image_0,
            reference_image_1=reference_image_1,
            reference_image_2=reference_image_2,
            images_batch=images_batch,
            source_video=source_video,
            temperature=temperature,
            max_tokens=max_tokens,
            custom_system_prompt=effective_custom,
            custom_model=custom_model,
            api_key=api_key,
            advanced_prompt=advanced_prompt,
            same_subject=same_subject,
            style_transfer=style_transfer,
            auto_describe=auto_describe,
            auto_describe_max_tokens=auto_describe_max_tokens,
            seed=seed,
        )

        # Build the output batch — all ref images stacked in <Picture N> order
        if ref_images_ordered:
            # Resize all to match the first image's spatial dims so we can stack
            h, w = ref_images_ordered[0].shape[1], ref_images_ordered[0].shape[2]
            matched = []
            for img in ref_images_ordered:
                if img.shape[1] != h or img.shape[2] != w:
                    img = torch.nn.functional.interpolate(
                        img.permute(0, 3, 1, 2),
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)
                matched.append(img)
            ref_batch = torch.cat(matched, dim=0)
        else:
            # No ref images — return a 1x1 dummy so the output type is valid
            ref_batch = torch.zeros(1, 1, 1, 3)

        return (h3_prompt, system_prompt, ref_batch,)
