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
        # The base class appends chain_conversation after context_length, i.e.
        # BEFORE enhanced_rules here. Re-append it after enhanced_rules so it
        # stays the LAST optional input on the Plus node too — saved Plus
        # workflows end their widgets_values at enhanced_rules, and an inserted
        # middle widget would shift that value onto chain_conversation.
        cc = base["optional"].pop("chain_conversation")
        base["optional"]["chain_conversation"] = cc
        return base

    RETURN_TYPES = ("STRING", "STRING", "IMAGE",
                    "STRING", "STRING", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("h3_prompt", "system_prompt", "ref_images_out",
                    "cell_1", "cell_2", "cell_3", "cell_4", "cell_5", "cell_6",)
    FUNCTION = "enhance_plus"
    CATEGORY = "h3/prompt"
    # ref_images_out is a LIST of individual [1,H,W,C] refs (in <Picture N>
    # order) — every frame keeps its own geometry, never stretched/collated.
    # Consumers that declare INPUT_IS_LIST (e.g. H3ReferenceToVideo Batch)
    # receive the whole list in one call; others run once per item.
    # cell_1..cell_6: the six parsed static-cell prompts (ri2i_multi mode only;
    # empty strings for any other task type).
    OUTPUT_IS_LIST = (False, False, True, False, False, False, False, False, False)
    # Accept a LIST on images_batch (and lists on every input) in one call.
    # The executor wraps ALL inputs in lists — each is unwrapped below before
    # the parent enhance() sees it, so the LLM path is unchanged.
    INPUT_IS_LIST = True

    @staticmethod
    def _un1(v):
        """INPUT_IS_LIST wraps every input in a 1-element list — unwrap scalars."""
        return v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v

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
                     editing_frame="on", seed=-1, context_length=-1,
                     chain_conversation="off",
                     enhanced_rules=True):
        # INPUT_IS_LIST unwrap — every input arrives wrapped in a list.
        prompt = self._un1(prompt)
        task_type = self._un1(task_type)
        duration = self._un1(duration)
        model = self._un1(model)
        local_backend = self._un1(local_backend)
        source_image = self._un1(source_image)
        last_frame_image = self._un1(last_frame_image)
        reference_image_0 = self._un1(reference_image_0)
        reference_image_1 = self._un1(reference_image_1)
        reference_image_2 = self._un1(reference_image_2)
        source_video = self._un1(source_video)
        temperature = self._un1(temperature)
        max_tokens = self._un1(max_tokens)
        custom_system_prompt = self._un1(custom_system_prompt)
        custom_model = self._un1(custom_model)
        api_key = self._un1(api_key)
        advanced_prompt = self._un1(advanced_prompt)
        same_subject = self._un1(same_subject)
        style_transfer = self._un1(style_transfer)
        auto_describe = self._un1(auto_describe)
        auto_describe_max_tokens = self._un1(auto_describe_max_tokens)
        editing_frame = self._un1(editing_frame)
        seed = self._un1(seed)
        context_length = self._un1(context_length)
        chain_conversation = self._un1(chain_conversation)
        enhanced_rules = self._un1(enhanced_rules)

        # images_batch may arrive as a LIST of individual frames (from
        # H3BatchImages 'Refs (list)') or a batched tensor — normalize to a
        # batch tensor so the parent LLM path (and the <Picture N> ordering)
        # is unchanged: one call, all images.
        if isinstance(images_batch, (list, tuple)):
            ib_frames = [f.unsqueeze(0) if f.dim() == 3 else f
                         for f in images_batch if f is not None]
            images_batch = torch.cat(ib_frames, dim=0) if ib_frames else None

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
        if task_type in ("r2v", "rv2v", "i2v", "ri2v", "ri2i", "ri2i_multi"):
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
            editing_frame=editing_frame,
            seed=seed,
            context_length=context_length,
            chain_conversation=chain_conversation,
        )

        # ref_images_out — a LIST of individual [1,H,W,C] images in <Picture N>
        # order, each at its OWN geometry (no stretching, no collation). The
        # H3ReferenceToVideo (Batch) node accepts this list and treats every
        # frame as its own reference.
        if task_type == "ri2i_multi":
            cells = _parse_multi_cells(h3_prompt)
            return (h3_prompt, system_prompt, ref_images_ordered, *cells)
        return (h3_prompt, system_prompt, ref_images_ordered,
                "", "", "", "", "", "")


def _parse_multi_cells(raw: str):
    """Split the ri2i_multi JSON reply into six cell prompts.

    Expected: {"cells": [{"view": "front", "prompt": "..."}, ...]} in the fixed
    order front, face, left, right, back, seductive. Tolerates markdown fences,
    chatty prose around the JSON, and missing/extra entries (falls back to the
    raw text in cell_1 when nothing parses, so the user still sees the reply).
    """
    import json
    import re

    cells = ["", "", "", "", "", ""]
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(cleaned[start:end + 1])
        except Exception:
            payload = None
        if isinstance(payload, dict):
            entries = payload.get("cells") or payload.get("prompts") or []
            if not isinstance(entries, list):
                entries = [entries]
            for i, entry in enumerate(entries[:6]):
                if not isinstance(entry, dict):
                    continue
                prompt = (entry.get("prompt") or entry.get("text")
                          or entry.get("detailed_description"))
                if isinstance(prompt, str) and prompt.strip():
                    cells[i] = prompt.strip()
    if not any(cells):
        cells[0] = raw.strip()
    return cells
