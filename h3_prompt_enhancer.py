"""MiniMax H3 Prompt Enhancer — vision LLM prompt enhancement for MiniMax H3.

Produces the structured prompt format that MiniMax H3 expects (the "H3-Context-IR"
replacement): a local LLM reformats the user's raw request into the 3-field base
format (T2V/I2V/FL2V/L2V) or the 6-field full-reference format (R2V), following
the official MiniMax H3 prompt writing guides.

Reuses the OpenRouter API client (_call_openrouter) from nodes.py so we do not
introduce a second LLM backend.
"""

import json
import logging
import os

# ── Reuse the existing LLM client and helpers from nodes.py ─────────────────
from .h3o_shared import (
    LOCAL_MODEL_DEFAULTS,
    OLLAMA_DEFAULT_URL,
    LOCAL_NSFW_PERMISSION,
    JOYCAPTION_STYLE_PROMPT,
    _tensor_to_base64,
    _call_openrouter,
    _call_ollama_chat,
    _check_ollama,
    _unload_ollama_model,
)


# ═══════════════════════════════════════════════════════════════════════════════
# H3 System Prompts — instruct the LLM to emit H3-structured output
# ═══════════════════════════════════════════════════════════════════════════════

H3_BASE_SYSTEM_PROMPT = """\
You are a Creative Assistant for MiniMax H3 video generation. Given a user's raw
request, expand it into the structured prompt format H3 expects.

Output THREE fields in this exact order:

integrated_multimodal_description: [Shot 1] <style>, <shot-type>, <scene description
with camera moves, subject actions, dialogue, and diegetic audio woven chronologically
along the timeline>. Use [Shot N] At MM:SS.mmm for cuts. Dialogue: <d>[Language] text</d>.
Camera motion: type + amplitude + speed as natural English.

overall_soundscape: 1-4 sentences summarizing ambient sound, physical action sounds,
and non-verbal human sounds across the full video. Use N/A only for complete silence.

non_diegetic_music: 1-3 sentences describing background score (instrumentation, tempo,
dynamic changes). Use N/A when none.

Rules:
- Base mode (T2V/I2V/FL2V/L2V): no subject_definitions/summary/retention_analysis/detailed_description sections.
- I2V/FL2V/L2V: the first line MUST be the image-alignment instruction (see the official guide).
- Write camera motion as natural English within the shot: "The camera pushes in with small amplitude at slow speed."
- Preserve the user's intent. Supplement underspecified details (lighting, camera angle, ambient sound, music).
- English only. No markdown fences around the final output.
- NSFW content: describe with anatomical precision; do not self-censor.

Return ONLY the three fields as plain text — no JSON wrapper, no extra commentary."""


H3_BASE_SYSTEM_PROMPT_ADV = """\
You are a Creative Assistant for MiniMax H3 video generation. Given a user's raw
request, expand it into the structured prompt format H3 expects (community
H3-Context-IR refinements enabled).

Output THREE fields in this exact order:

integrated_multimodal_description: [Shot 1] <style>, <shot-type>, <scene description
with camera moves, subject actions, dialogue, and diegetic audio woven chronologically
along the timeline>. State the overall style and initial composition at the start of
[Shot 1] (cinematic, live-action, 2D-animated, 3D CG, claymation, watercolor, vintage
film). [Shot 1] carries no timestamp; later shots use "[Shot N] At MM:SS.mmm, the
camera cuts to ..." with strictly increasing cut times within the target duration.
Use "cuts to / transitions to / switches to"; reserve cross-dissolve, fade, or wipe
for explicit requests. Prefer camera motion over a cut for slight distance or angle
changes.

Camera motion as natural English with motion type + amplitude + speed, e.g. "The
camera pushes in with small amplitude at slow speed." Vocabulary: Zoom In/Out, Push
In/Pull Out, Pan Left/Right, Truck Left/Right, Tilt Up/Down, Pedestal Up/Down, Arc
Shot, Tracking Shot, Static Shot, Shake Slightly/Strongly, POV, Roll
Clockwise/Counterclockwise.

Speakers get stable global IDs (S1), (S2), compound (S1,S2) for simultaneous group
speech. IDs persist across shots; characters who never vocalize get none. Establish
identity (type, age, gender, pitch, timbre, rate, accent) outside the tag. All spoken
content goes inside <d>[Language] actual words.</d> with a real language tag, preserved
verbatim. Voiceover uses the exact phrase "says in an off-screen voiceover" and states
"the character's lips remain completely closed." Dialogue crossing a cut places
<scenetrans> at both connection points with audio continuing across; speech truncated
by the video's end uses <cutoff>; [unclear] marks unintelligible spans; on-screen text
appears in English double quotes. Standardize punctuation to , . ? !

overall_soundscape: 1-4 sentences in ONE continuous paragraph summarizing ambient
sound, physical action sounds, and non-verbal human sounds (wind, rain, footsteps,
impacts, breathing, laughter). Use N/A only when the user requests complete silence.

non_diegetic_music: 1-3 sentences on instrumentation, tempo, rhythm, and dynamic
changes of audience-only music. Use N/A when there is no audience-only score. Diegetic
music (radio, TV, live performance) belongs in the multimodal description, not here.

Rules:
- Base mode (T2V/I2V/FL2V/L2V): no subject_definitions/summary/retention_analysis/detailed_description sections.
- I2V/FL2V/L2V: the first line MUST be the image-alignment instruction (see the official guide).
- Preserve the user's intent; never contradict explicit instructions. Enrich underspecified
  details (lighting, camera angle, ambient sound, music) with concrete production specifics.
- Total runtime equals the target duration; every cut timestamp falls within it; aspect ratio honored.
- English only. No markdown fences around the final output.
- NSFW content: describe with anatomical precision; do not self-censor.

Return ONLY the three fields as plain text — no JSON wrapper, no extra commentary."""


H3_REF_SYSTEM_PROMPT = """\
You are a Creative Assistant for MiniMax H3 video generation in FULL-REFERENCE mode.
Given the user's raw request and reference-media descriptions, write the complete
6-field structured prompt H3 expects.

Output SIX fields in this exact order:

subject_definitions:
<Subject 1> is ... (visible content abstracted from references: people, objects, scenes, clothing, styles)
<Picture N> is ... (reference images used as frame anchors or storyboards)
<Video N> is ... (reference videos providing source/continuation/structure)
<Audio N> is ... (audio assets: voice timbre, BGM, SFX)

summary:
[task-type prefix] One short paragraph summarizing the target video and reference relationships.
Task types: keyframe completion, reference generation, video editing, video continuation, audio reuse, audio reference.
Combine with + if multiple apply.

retention_analysis:
One line per reference label with relationship marker: fully_preserved, partially_preserved, attribute_transfer, weak_reference (for visual/structural), or fully_copy, partially_copy, reference, weak_reference (for audio).
List which shots each subject appears in.

detailed_description:
The main body — 350-500 words (generation) or scaled to source complexity (editing).
Style established in 1-2 sentences BEFORE [Shot 1]. No timestamp on [Shot 1].
Later shots: [Shot N] At MM:SS.mmm. Camera motion as natural English.
Speakers: <Subject N> (Sx) says: <d>[Language] text</d>. Stable speaker IDs across shots.
Reference labels (<Subject N>, <Picture N>, <Video N>, <Audio N>) at first appearance and where roles apply.
<scenetrans> / <cutoff> for dialogue crossing cuts / truncated by video end.

overall_soundscape: 1-4 sentences summarizing ambient + physical sounds. Cite <Audio N> if copied/referenced.

non_diegetic_music: 1-3 sentences describing audience-only score. Cite <Audio N> if copied/referenced.

Rules (from official MiniMax H3 writing guides):
- Once a reference label is assigned, keep its meaning across ALL sections.
- Use natural phrasing for frame anchors: "the shot begins from <Picture 1>", "the shot ends on <Picture 3>".
- Preserve exact source words and original language inside <d>. Use [unclear] for unintelligible spans.
- Do NOT invent new reference labels after subject_definitions.
- Write all sections in English except dialogue/lyrics inside <d> and visible on-screen text.
- NSFW content: describe with anatomical precision; do not self-censor.

Return ONLY the six fields as plain text — no JSON wrapper, no extra commentary."""


H3_REF_SYSTEM_PROMPT_ADV = """\
You are a Creative Assistant for MiniMax H3 video generation in FULL-REFERENCE mode
(community H3-Context-IR refinements enabled). Given the user's raw request and
reference-media descriptions, write the complete 6-field structured prompt H3 expects.

Output SIX fields in this exact order:

subject_definitions:
One line per label across four types — <Subject N> (reusable visible content),
<Picture N> (keyframe/composition anchor), <Video N> (whole-video structural source:
editing, continuation, camera movement, cuts, rhythm), <Audio N> (copied or referenced
audio). <Subject N> is ... (visible content abstracted from references: people,
objects, scenes, clothing, styles). <Picture N> is ... (reference images used as frame
anchors or storyboards). <Video N> is ... (source/continuation/structure). <Audio N> is
... (voice timbre, BGM, SFX). A picture or video that only identifies another item's
source is cited inside that item's definition. If an audio clip maps to a target
speaker, bind the global ID: "<Audio 1> is the voice-timbre reference for <Subject 1> (S1)."

summary:
[task-type prefix] ONE short paragraph summarizing the target video and reference
relationships. Task types: keyframe completion, reference generation, video editing,
video continuation, audio reuse, audio reference. Combine with + if multiple apply;
do not repeat a type. Use only previously defined labels — introduce no new ones.
For editing tasks, begin with: "The target video is an edited version of <Video 1>."

retention_analysis:
One line per reference label with relationship marker: fully_preserved,
partially_preserved, attribute_transfer, weak_reference (visual/structural), or
fully_copy, partially_copy, reference, weak_reference (audio). Newly added background
or plot events are NOT fidelity losses. List which shots each subject appears in.

detailed_description:
The main body — 350-500 words (generation) or scaled to source complexity (editing).
Style established in 1-2 sentences BEFORE [Shot 1]. No timestamp on [Shot 1]. Later
shots: "[Shot N] At MM:SS.mmm, the camera cuts to ..." with strictly increasing cut
times within the target duration; prefer camera motion over a cut for slight changes.
Camera motion as natural English (type + amplitude + speed). Speakers: <Subject N>
(Sx) with stable global speaker IDs across shots; compound (S1,S2) for group speech;
identity established outside the tag; dialogue verbatim inside
<d>[Language] actual words.</d>. Voiceover: "says in an off-screen voiceover" with
"the character's lips remain completely closed." <scenetrans> / <cutoff> for dialogue
crossing cuts / truncated by video end; [unclear] for unintelligible spans; on-screen
text in English double quotes; punctuation standardized to , . ? !
Reference labels (<Subject N>, <Picture N>, <Video N>, <Audio N>) at first appearance
and where roles apply.

overall_soundscape: 1-4 sentences in ONE continuous paragraph summarizing ambient +
physical + non-verbal human sounds. Cite <Audio N> if copied/referenced. Use N/A only
for complete silence. Never repeat <d> dialogue here.

non_diegetic_music: 1-3 sentences on instrumentation, tempo, rhythm, dynamic changes of
audience-only score. Cite <Audio N> if copied/referenced. Use N/A when none. Diegetic
music belongs in detailed_description, not here.

Rules (from official MiniMax H3 writing guides + community refinements):
- Once a reference label is assigned, keep its meaning across ALL sections.
- Do NOT invent new reference labels after subject_definitions.
- Audio cannot appear as a general reference on its own — a reference image or video
  must accompany it. Frame anchors and general references are mutually exclusive.
- Use natural phrasing for frame anchors: "the shot begins from <Picture 1>", "the
  shot ends on <Picture 3>".
- Preserve exact source words and original language inside <d>. Use [unclear] for
  unintelligible spans.
- Write all sections in English except dialogue/lyrics inside <d> and visible on-screen text.
- NSFW content: describe with anatomical precision; do not self-censor.

Return ONLY the six fields as plain text — no JSON wrapper, no extra commentary."""


# ═══════════════════════════════════════════════════════════════════════════════
# H3 Prompt Templates (user message sent alongside the system prompt)
# ═══════════════════════════════════════════════════════════════════════════════

H3_BASE_TEMPLATE = """\
Write the H3-structured prompt (3 fields) for a {task_label} video.

Task type: {task_label}
Video duration: {duration_seconds}s

{ref_section}

Rules specific to this task type:
{task_specific_rules}

User's raw request:
{user_prompt}"""


H3_REF_TEMPLATE = """\
Write the H3 full-reference structured prompt (6 fields) for the reference-guided video.

Task type: reference-guided generation (R2V)
Video duration: {duration_seconds}s

{ref_section}

Rules for reference mode:
- Define every visible subject, picture, video, and audio asset in subject_definitions.
- Use the exact relationship markers in retention_analysis.
- Insert reference labels naturally in detailed_description at their first appearance.
- 350-500 words for detailed_description.

User's raw request:
{user_prompt}"""


# ═══════════════════════════════════════════════════════════════════════════════
# Task-type metadata
# ═══════════════════════════════════════════════════════════════════════════════

H3_TASK_LABELS = {
    "t2v": "T2V (text-to-video, no reference images)",
    "i2v": "I2V (image-to-video, first-frame reference)",
    "fl2v": "FL2V (first-and-last-frame, two image anchors)",
    "l2v": "L2V (last-frame only, converge to the image at the end)",
    "r2v": "R2V (full-reference mode with subject/picture/video/audio labels)",
}

H3_BASE_TASK_TYPES = {"t2v", "i2v", "fl2v", "l2v"}

H3_SAME_SUBJECT_RULE = """\
SAME SUBJECT: ALL reference images show the SAME person/subject from different
angles, poses, or lighting. Analyze them as ONE unified subject — combine identity
features from every reference into a single subject definition, and never treat
image0 / image1 / image2 as different people."""

# Style transfer — references are a REAL person, but the target video is rendered
# in a non-photorealistic art style. Instructs the LLM to keep identity-defining
# features while committing fully to the chosen art style (mini-version of the
# Bernini style_transfer rules, adapted for H3's structured prompt format).
H3_STYLE_TRANSFER_RULES = {
    "anime": """\
STYLE TRANSFER -- ANIME: The reference image(s) show a REAL person, but the target
video is 2D ANIME / cel-shaded style. Describe the reference subject's identity
features (face shape, hair color/style, eye color, distinguishing marks) but instruct
the model to render them as an anime character -- cel-shaded skin, large expressive
eyes, simplified nose/mouth, clean linework, flat color fills. Do NOT describe
photorealistic skin texture, pores, or lighting. Translate real features into anime
equivalents (e.g. 'brown wavy hair' → 'brown wavy anime-style hair'). Set the
visual style statement in [Shot 1] to "2D anime / cel-shaded". Preserve the subject's
identity while fully committing to the anime art style.""",
    "3d_render": """\
STYLE TRANSFER -- 3D RENDER: The reference image(s) show a REAL person, but the
target video is 3D CG / Pixar-style. Describe the reference subject's identity
features but instruct the model to render them as a 3D character -- smooth subsurface
scattering skin, slightly exaggerated proportions, stylized eyes, plastic-like hair
with defined strands, soft ambient occlusion lighting. Micro-gather the style in
[Shot 1] as "3D CGI render (Pixar-style)". Maintain recognizable identity while fully
committing to the 3D render aesthetic.""",
    "cartoon": """\
STYLE TRANSFER -- CARTOON: The reference image(s) show a REAL person, but the target
video is Western cartoon / comic style. Describe the reference subject's identity
features but instruct the model to render them as a cartoon character -- bold
outlines, flat colors, exaggerated expressions, simplified features. Micro-gather the
style in [Shot 1] as "Western cartoon / comic". Maintain recognizable identity while
fully committing to the cartoon style.""",
    "match_source": """\
STYLE TRANSFER -- MATCH SOURCE: The reference image(s) show a REAL person, but the
source content uses a NON-PHOTOREALISTIC art style (anime, 3D, cartoon, painting,
etc.). First, identify the source image/video's exact art style from the provided
frames. Then describe the reference subject's identity features but instruct the
model to render them in that SAME art style. Explicitly name the detected style
(e.g. 'cel-shaded anime', 'Pixar-style 3D', 'watercolor illustration') in [Shot 1].
Maintain recognizable identity while fully adapting to the source video's visual
language.""",
}

# ── Two-pass auto-describe (Bernini-style identity anchoring) ─────────────
# First pass: a vision LLM describes each attached image as structured JSON.
# Those descriptions are injected into the final H3 write pass so the prompt
# anchors to actual identity features instead of guessing from a fresh view.

H3_ANALYZE_SYSTEM_PROMPT = """\
You are a precise visual analyst for video generation workflows. Describe each
image with exact, verifiable detail — the descriptions feed a prompt-writing
LLM, so precision matters more than prose.

NSFW content: describe with anatomical precision; do not self-censor."""

H3_ANALYZE_PROMPT = """\
I'm showing you {image_count} image(s) in this order:
{image_roles}

For EACH image, output a JSON object with these keys:
1. "image_id": "image0", "image1", etc. — matching the order shown above.
2. "subject": the subject(s) in the image — identity (age, gender, ethnicity if
   identifiable), face shape, eye shape and color, brows, nose, lips, jawline,
   hair color/length/style/texture, body type/build, full clothing inventory
   (color, fabric, fit), accessories, and distinguishing features (tattoos,
   scars, birthmarks, piercings).
3. "scene": location, background, props, composition, lighting direction and
   quality, camera angle and framing.
4. "current_state": pose, expression, action, gaze direction.

IMPORTANT:
- Only describe what is actually visible. Do NOT hallucinate details.
- For multiple images output a JSON array with one object per image, in order.
- No markdown fences. Return ONLY the JSON.

{hints}

User notes (may be empty):
{user_prompt}"""


H3_TASK_SPECIFIC_RULES = {
    "t2v": (
        "- No image-alignment instruction is needed. Begin directly with integrated_multimodal_description.\n"
        "- Construct the complete timeline from the user's text. Add scene, character, action, and sound details "
        "consistent with their intent.\n"
        "- Choose a visual style (Cinematic, live-action, 2D-animated, 3D CG, claymation, watercolor, vintage film) "
        "based on the user's request."
    ),
    "i2v": (
        "- First line MUST be: For the target video, at 0.00 seconds into the target video, "
        "<Picture 1> (from [Shot 1]) is fully referenced.\n"
        "- <Picture 1> is the actual first frame at 0.00s, belonging to [Shot 1].\n"
        "- First establish style, subjects, composition, and scene anchors from the image, "
        "then describe the next action.\n"
        "- Structure: first-frame anchor → action onset → continuous development → result or reaction."
    ),
    "fl2v": (
        "- First line MUST be: How the reference pictures align with the target video — "
        "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
        "Picture 2 (from Shot N) aligns with the S.SS-second mark of the target video.\n"
        "- Picture 1 = opening, Picture 2 = ending. Describe the motion path between them.\n"
        "- Prefer a single shot for continuous interpolation. Use multiple shots only when specified.\n"
        "- Structure: first-frame state → observable intermediate changes → progressively narrowing "
        "differences → last-frame state."
    ),
    "l2v": (
        "- First line MUST be: How the reference pictures align with the target video — "
        "<Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.\n"
        "- <Picture 1> is the FINAL frame, belonging to the last [Shot N].\n"
        "- Infer a plausible earlier state, then describe how the scene gradually converges to the reference image.\n"
        "- Structure: plausible preceding state → explicit action and transition path → gradual convergence "
        "in the final shot → last-frame landing."
    ),
    "r2v": (
        "- Full-reference mode with subject_definitions, summary, retention_analysis, detailed_description, "
        "overall_soundscape, non_diegetic_music.\n"
        "- Define all referenced content (subjects, pictures, videos, audio) in subject_definitions.\n"
        "- Use retention_analysis to describe how each referenced item is preserved/transferred/referenced.\n"
        "- detailed_description is the main body — 350-500 words, reference labels at first appearance."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# H3 Prompt Enhancer Node
# ═══════════════════════════════════════════════════════════════════════════════

class H3PromptEnhancer:
    """Enhances user prompts into MiniMax H3 structured format via a vision LLM.

    Produces the structured prompt that H3 expects — the community substitute for
    the hosted "H3-Context-IR" prompt-enhancement system. One node handles all
    H3 task types: a task_type widget (T2V / I2V / FL2V / L2V / R2V) switches
    between the 3-field base format and the 6-field full-reference format.

    Reuses the OpenRouter API client (_call_openrouter) and Ollama fallback from
    nodes.py — no new LLM backend.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Your raw scene description. The LLM expands this into H3's "
                               "structured format with camera moves, shots, audio, and music fields."
                }),
                "task_type": (
                    ["t2v", "i2v", "fl2v", "l2v", "r2v"],
                    {"default": "t2v",
                     "tooltip": "t2v=text-to-video (no images), i2v=first-frame, "
                                "fl2v=first+last frame, l2v=last-frame only, "
                                "r2v=full-reference (6-field output)."}
                ),
                "duration": ("FLOAT", {
                    "default": 5.0, "min": 0.5, "max": 120.0, "step": 0.5,
                    "tooltip": "Target video duration in seconds. Affects shot pacing and cut timing."
                }),
                "model": (
                    ["x-ai/grok-4.3", "x-ai/grok-4-0709",
                     "deepseek/deepseek-chat-v3-0324",
                     "google/gemini-2.5-flash", "openai/gpt-4o"],
                    {"default": "x-ai/grok-4.3"}
                ),
                "local_backend": (
                    ["off", "ollama/joycaption"],
                    {"default": "off",
                     "tooltip": "Use a LOCAL model via Ollama instead of OpenRouter. "
                                "JoyCaption Beta One (7.5GB) for vision-capable enhancement."}
                ),
            },
            "optional": {
                "source_image": ("IMAGE", {
                    "tooltip": "First-frame image for I2V/FL2V, last-frame for L2V. "
                               "For FL2V connect both source_image (first) and last_frame_image."
                }),
                "last_frame_image": ("IMAGE", {
                    "tooltip": "Last-frame image for FL2V only. Leave disconnected for I2V/L2V."
                }),
                "reference_image_0": ("IMAGE", {
                    "tooltip": "Reference image: R2V subject/picture reference, or I2V "
                               "replacement subject (defines <Subject 1>)."
                }),
                "reference_image_1": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "source_video": ("IMAGE", {
                    "tooltip": "Source video as batched frames (e.g. from BerniniLoadVideo). "
                               "Frames are sampled across the timeline and shown to the LLM as "
                               "<Video 1> — the source asset for video editing / video continuation. "
                               "R2V mode only."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Lower = more predictable format. Higher = more creative details."
                }),
                "max_tokens": ("INT", {
                    "default": 4096, "min": 512, "max": 16384, "step": 256,
                    "tooltip": "H3 ref-mode prompts are long (6 fields). Use 4096+ for R2V."
                }),
                "advanced_prompt": (["on", "off"], {
                    "default": "on",
                    "tooltip": "on=enhanced system prompt with community H3-Context-IR "
                               "refinements (camera-motion vocabulary, stable speaker IDs, "
                               "voiceover phrasing, audio-binding rules). off=original "
                               "simpler prompt."
                }),
                "custom_system_prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Overrides the built-in H3 system prompt. Leave empty for default."
                }),
                "custom_model": ("STRING", {
                    "default": "",
                    "tooltip": "Any OpenRouter model ID. Overrides the dropdown when non-empty."
                }),
                "api_key": ("STRING", {"default": ""}),
                # NOTE: new widgets appended LAST so existing saved workflows keep
                # their widgets_values alignment.
                "same_subject": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When ON, tells the LLM that ALL reference images show "
                               "the SAME person from different angles/poses. The prompt "
                               "describes ONE unified subject combining features from all "
                               "references, instead of treating image0/image1/image2 as "
                               "different people. Turn ON when wiring multiple photos of "
                               "the same person for better identity coverage."
                }),
                "style_transfer": (
                    ["off", "anime", "3d_render", "cartoon", "match_source"],
                    {"default": "off",
                     "tooltip": "Stylize the reference person into a non-photorealistic "
                                "art style. 'anime' = 2D cel-shaded, '3d_render' = Pixar/3D "
                                "CG, 'cartoon' = Western cartoon/comic, 'match_source' = "
                                "auto-detect the source video's art style and adapt the "
                                "reference to it. The LLM keeps identity-defining features "
                                "but renders the subject in the chosen style."}
                ),
                "auto_describe": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Two-pass mode (Bernini-style). First pass: a vision LLM "
                               "describes each attached image as structured JSON (identity, "
                               "scene, current_state). Those descriptions are injected into "
                               "the H3 write so distinctive references anchor properly — much "
                               "stronger identity consistency for R2V subjects and I2V "
                               "replacement characters. Costs one extra API call."
                }),
                "auto_describe_max_tokens": ("INT", {
                    "default": 4096, "min": 512, "max": 16384, "step": 256,
                    "tooltip": "Token budget for the auto_describe analysis call. Multi-image "
                               "scenes need 4096+. Lower to 2048 for a single simple image."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("h3_prompt", "system_prompt",)
    FUNCTION = "enhance"
    CATEGORY = "h3/prompt"

    def _format_analysis(self, raw):
        """Parse the auto_describe analysis response into a text block, or None."""
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(parsed, list):
                parts = []
                for entry in parsed:
                    if not isinstance(entry, dict):
                        continue
                    iid = entry.get("image_id", "image?")
                    line = f"{iid}: {entry.get('subject', '')}".strip()
                    scene = entry.get("scene", "")
                    state = entry.get("current_state", "")
                    if scene:
                        line += f" | Scene: {scene}"
                    if state:
                        line += f" | State: {state}"
                    if line.strip():
                        parts.append(line)
                return "\n".join(parts) if parts else None
            if isinstance(parsed, dict) and parsed.get("image_id"):
                line = f"{parsed.get('image_id')}: {parsed.get('subject', '')}".strip()
                if parsed.get("scene"):
                    line += f" | Scene: {parsed['scene']}"
                if parsed.get("current_state"):
                    line += f" | State: {parsed['current_state']}"
                return line if line.strip() else None
            return None
        except (json.JSONDecodeError, TypeError, AttributeError):
            if isinstance(raw, str) and raw.strip():
                text = raw.strip()
                if text.startswith("```"):
                    lines = text.splitlines()
                    if lines and lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    text = "\n".join(lines).strip()
                return text if text.strip() else None
            return None

    def _analyze_images(self, api_fn, api_key, llm_model, image_parts, roles,
                        hints, temperature, max_tokens):
        """First-pass: describe every attached image as structured JSON.

        Returns a formatted text block the write pass uses to anchor identity,
        or None if the analysis failed."""
        if not image_parts:
            return None
        content = list(image_parts)
        text = H3_ANALYZE_PROMPT.format(
            image_count=len(image_parts),
            image_roles="\n".join(
                f"- image{i}: {role}" for i, role in enumerate(roles)),
            hints=hints or "(none)",
            user_prompt="(no notes — infer everything from the images)",
        )
        content.append({"type": "text", "text": text})
        try:
            raw = api_fn(
                api_key, llm_model, H3_ANALYZE_SYSTEM_PROMPT, content,
                temperature=max(temperature - 0.2, 0.0),
                max_tokens=max_tokens,
            )
        except Exception as e:
            logging.warning(f"[H3PromptEnhancer] auto_describe analysis failed: {e}")
            return None
        return self._format_analysis(raw)

    def enhance(self, prompt, task_type, duration, model, local_backend="off",
                source_image=None, last_frame_image=None,
                reference_image_0=None, reference_image_1=None,
                reference_image_2=None, source_video=None,
                temperature=0.7, max_tokens=4096,
                custom_system_prompt="", custom_model="", api_key=None,
                advanced_prompt="on",
                same_subject=False, style_transfer="off",
                auto_describe=True, auto_describe_max_tokens=4096):
        """Enhance a user prompt into H3 structured format."""

        # ── Resolve model ─────────────────────────────────────────────────
        llm_model = (custom_model.strip()
                     if custom_model and custom_model.strip()
                     else model)

        # ── Local backend detection ───────────────────────────────────────
        use_local = local_backend != "off"
        if use_local:
            defaults = LOCAL_MODEL_DEFAULTS.get(local_backend, {})
            ollama_url = defaults.get("ollama_url", OLLAMA_DEFAULT_URL)
            ollama_model = defaults.get("ollama_model", "")
            if not ollama_model:
                raise ValueError(
                    f"No ollama_model for local_backend='{local_backend}'.")
            logging.info(
                f"[H3PromptEnhancer] Ollama backend: {local_backend} "
                f"model={ollama_model} url={ollama_url}")
            if not _check_ollama(ollama_url):
                raise RuntimeError(
                    f"Ollama not reachable at {ollama_url}. "
                    "Start it with: systemctl start ollama")

            def _local_api(api_key, model, system_prompt, user_content,
                           timeout=180, temperature=0.7, max_tokens=4096,
                           retries=2):
                return _call_ollama_chat(
                    ollama_url, ollama_model, user_content, system_prompt,
                    temperature=temperature, max_tokens=max_tokens,
                    timeout=timeout)
            api_fn = _local_api
        else:
            api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "No API key provided. Set OPENROUTER_API_KEY env var "
                    "or connect a key to the api_key input.")
            api_fn = None  # _call_openrouter is used directly

        # ── Select system prompt ──────────────────────────────────────────
        use_adv = advanced_prompt == "on"
        if custom_system_prompt and custom_system_prompt.strip():
            system_prompt = custom_system_prompt.strip()
        elif task_type in H3_BASE_TASK_TYPES:
            system_prompt = (H3_BASE_SYSTEM_PROMPT_ADV if use_adv
                             else H3_BASE_SYSTEM_PROMPT)
        else:
            system_prompt = (H3_REF_SYSTEM_PROMPT_ADV if use_adv
                             else H3_REF_SYSTEM_PROMPT)

        # Local models need explicit NSFW permission
        if use_local and not (custom_system_prompt and custom_system_prompt.strip()):
            system_prompt = (LOCAL_NSFW_PERMISSION + " " +
                             JOYCAPTION_STYLE_PROMPT + " " +
                             system_prompt)

        # ── Build vision content (images sent to the LLM) ─────────────────
        user_content = []
        ref_section = ""
        role_list = []  # parallel to user_content image parts, for auto_describe

        # Source image (first-frame for I2V/FL2V, last-frame for L2V)
        if source_image is not None and task_type in ("i2v", "fl2v", "l2v"):
            if source_image.dim() == 4:
                b64 = _tensor_to_base64(source_image[0])
            else:
                b64 = _tensor_to_base64(source_image)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })
            frame_word = "LAST frame" if task_type == "l2v" else "first frame"
            role_list.append(f"{frame_word} of the target video (Picture 1 anchor)")
            ref_section += ("A reference image (Picture 1) is attached. "
                            f"It shows the {frame_word} of the video.\n\n")

        # Source video (R2V video editing/continuation) — sampled frames shown
        # as <Video 1>. Per the official guide, <Video N> is reserved for
        # whole-video relationships (editing, continuation, camera/cut structure).
        if source_video is not None and task_type == "r2v":
            if source_video.dim() == 4:
                n_frames = source_video.shape[0]
                n_sample = min(n_frames, 6)
                idxs = sorted({int(round(i * (n_frames - 1) / (n_sample - 1)))
                               for i in range(n_sample)}) if n_frames > 1 else [0]
            else:
                n_frames, idxs = 1, [0]
            for i in idxs:
                frame = source_video[i] if source_video.dim() == 4 else source_video
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + _tensor_to_base64(frame)}
                })
                role_list.append(
                    f"frame {i + 1}/{len(idxs)} of the SOURCE video (<Video 1>)")
            ref_section += (
                f"A source video (<Video 1>) is attached as {len(idxs)} frames "
                f"sampled across its {n_frames}-frame timeline. It is the SOURCE video "
                "for the target video. If the user asks to edit or continue it, use task "
                "type [video editing] or [video continuation] and begin the summary with: "
                "The target video is an edited version of <Video 1>.\n\n")

        # Last-frame image (FL2V only)
        if last_frame_image is not None and task_type == "fl2v":
            if last_frame_image.dim() == 4:
                b64 = _tensor_to_base64(last_frame_image[0])
            else:
                b64 = _tensor_to_base64(last_frame_image)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })
            role_list.append("LAST frame of the target video (Picture 2 anchor)")
            ref_section += ("A second reference image (Picture 2) is attached. "
                            "It shows the LAST frame of the video.\n\n")

        # Reference images — R2V full-reference mode, and I2V replacement subjects.
        # In R2V they may be labeled Picture or Subject; in I2V they always define
        # <Subject N> replacement characters whose appearance must drive the
        # integrated_multimodal_description (note: the i2v gen node has no ref_image
        # conditioning — only the prompt text reaches the model).
        if task_type in ("r2v", "i2v"):
            ref_images = []
            for ref in (reference_image_0, reference_image_1, reference_image_2):
                if ref is not None:
                    ref_images.append(ref)
            for i, ref_img in enumerate(ref_images):
                if ref_img.dim() == 4:
                    b64 = _tensor_to_base64(ref_img[0])
                else:
                    b64 = _tensor_to_base64(ref_img)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                })
                if task_type == "i2v":
                    role_list.append(
                        f"reference image defining <Subject {i + 1}> (REPLACEMENT character)")
                    ref_section += (
                        f"Reference image {i} is attached and defines <Subject {i+1}> — "
                        "the REPLACEMENT character. State <Subject "
                        f"{i+1}>'s exact appearance from this image at the start of "
                        "integrated_multimodal_description (before the action), and "
                        "drive all of its actions from the user's request. The first "
                        "frame anchor (Picture 1) shows the ORIGINAL character being "
                        "replaced.\n")
                else:
                    role_list.append(
                        f"reference image (label as Picture {i + 1} or Subject {i + 1})")
                    ref_section += (f"Reference image {i} is attached "
                                    f"(label as Picture {i+1} or Subject {i+1}).\n")
            if ref_images:
                ref_section += "\n"

        # ── Same-subject + style-transfer directives ─────────────────────
        # Appended to the reference section so the LLM merges multiple photos
        # of one person into a single subject and/or renders the reference in a
        # non-photorealistic art style (Bernini-style controls).
        extra_rules = []
        has_refs = any(r is not None for r in
                       (reference_image_0, reference_image_1, reference_image_2))
        if same_subject and has_refs:
            extra_rules.append(H3_SAME_SUBJECT_RULE)
        if style_transfer and style_transfer != "off":
            rule = H3_STYLE_TRANSFER_RULES.get(style_transfer)
            if rule:
                extra_rules.append(rule)
        if extra_rules:
            ref_section += "\n" + "\n\n".join(extra_rules) + "\n\n"

        # ── Auto-describe first pass (Bernini-style identity anchoring) ──
        # The analysis LLM describes every attached image as JSON; the result
        # is injected into the ref section so the write pass anchors identity
        # from explicit features instead of a fresh look at the images.
        if auto_describe and user_content:
            image_parts = [c for c in user_content if c.get("type") == "image_url"]
            if image_parts and len(role_list) == len(image_parts):
                hints = []
                if same_subject and has_refs:
                    hints.append("NOTE: ALL reference images show the SAME person "
                                 "from different angles/poses — analyze as ONE "
                                 "unified subject, merging features.")
                if style_transfer and style_transfer != "off":
                    hints.append("NOTE: identity features will be translated into a "
                                 "non-photorealistic art style — emphasize "
                                 "style-independent identity features (face shape, "
                                 "hair, eye color, distinguishing marks).")
                caller = api_fn if use_local else _call_openrouter
                api_k = "" if use_local else api_key
                analysis = self._analyze_images(
                    caller, api_k, llm_model, image_parts, role_list,
                    "\n".join(hints), temperature, auto_describe_max_tokens)
                if analysis:
                    ref_section += ("--- Context Analysis (image descriptions from "
                                    "the first pass) ---\n"
                                    f"{analysis}\n--- End Analysis ---\n\n")

        # ── Build the user message ────────────────────────────────────────
        task_label = H3_TASK_LABELS.get(task_type, task_type)
        task_rules = H3_TASK_SPECIFIC_RULES.get(task_type, "")

        if task_type in H3_BASE_TASK_TYPES:
            user_text = H3_BASE_TEMPLATE.format(
                task_label=task_label,
                duration_seconds=f"{duration:.1f}",
                ref_section=ref_section,
                task_specific_rules=task_rules,
                user_prompt=prompt if prompt.strip() else "(no user notes — infer everything from the images)",
            )
        else:
            user_text = H3_REF_TEMPLATE.format(
                duration_seconds=f"{duration:.1f}",
                ref_section=ref_section,
                user_prompt=prompt if prompt.strip() else "(no user notes — infer everything from the images)",
            )

        # ── Send to LLM ───────────────────────────────────────────────────
        caller = api_fn if use_local else _call_openrouter
        api_k = "" if use_local else api_key

        raw_output = caller(
            api_k, llm_model, system_prompt,
            [{"type": "text", "text": user_text}] if not user_content else
            user_content + [{"type": "text", "text": user_text}],
            temperature=temperature, max_tokens=max_tokens,
        )

        # Clean up Ollama VRAM
        if use_local:
            _unload_ollama_model(ollama_url, ollama_model)

        # ── Return ────────────────────────────────────────────────────────
        return (raw_output, system_prompt,)


# ═══════════════════════════════════════════════════════════════════════════════
# ComfyUI registration (used by __init__.py after import)
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "H3PromptEnhancer": H3PromptEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3PromptEnhancer": "H3 Prompt Enhancer",
}
