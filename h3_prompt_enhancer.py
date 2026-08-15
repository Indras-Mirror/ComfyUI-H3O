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
import re

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
- ANY vocalization (moans, groans, laughter, screams, gasps) belongs to a speaker with a stable ID — never attribute
  such sounds to a nameless subject. Nonverbal vocalizations are described in prose OUTSIDE <d>, with the speaker's (Sx) ID.
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

Speakers get stable global IDs (S1), (S2), compound (S1,S2) for simultaneous
speech. IDs persist across shots; characters who never vocalize get none. ANY
vocalization (groans, moans, laughter, screams, gasps) counts as vocalizing: the
source still needs the (Sx) ID, with the nonverbal sound described as plain
prose outside <d>. Establish identity (type, age, gender, pitch, timbre, rate,
accent) outside the tag. All spoken content goes inside <d>[Language] actual
words.</d> with a real language tag, preserved verbatim; tildes, ellipses, and
other decorative punctuation inside <d> are kept exactly as the user wrote
them. Voiceover uses the exact phrase "says in an off-screen voiceover" and
states "the character's lips remain completely closed." Dialogue crossing a cut
places <scenetrans> at both connection points with audio continuing across;
speech truncated by the video's end uses <cutoff>; [unclear] marks
unintelligible spans; on-screen text appears in English double quotes.
Preserve user-provided dialogue verbatim — do not rewrite, normalize, or strip
its punctuation.

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
- If the user assigns speaker IDs like (S1) or (S2), keep them exactly: same numbers, same
  people, same vocal order. Assign the next free ID only to vocal characters the user never names.
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
- ANY vocalization (moans, groans, laughter, screams, gasps) belongs to a speaker with a stable (Sx) ID —
  never attribute such sounds to a nameless subject. Nonverbal vocalizations go as prose OUTSIDE <d>.
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
... (voice timbre, BGM, SFX). If an audio clip maps to a target
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
or plot events are NOT fidelity losses. In character swap every reference picture is
attribute_transfer — never weak_reference; no picture is "not appearing". List which
shots each subject appears in.

detailed_description:
The main body — 350-500 words (generation) or scaled to source complexity (editing).
Style established in 1-2 sentences BEFORE [Shot 1]. No timestamp on [Shot 1]. Later
shots: "[Shot N] At MM:SS.mmm, the camera cuts to ..." with strictly increasing cut
times within the target duration; prefer camera motion over a cut for slight changes.
Camera motion as natural English (type + amplitude + speed). Speakers: <Subject N>
(Sx) with stable global speaker IDs across shots; compound (S1,S2) for group speech;
identity established outside the tag; dialogue verbatim inside
<d>[Language] actual words.</d>. ANY vocalization (groans, moans, laughter, screams,
gasps) is a vocal event: the source must have or be given an (Sx) ID, with the
nonverbal sound described as plain prose — such utterances are never voiced by a
nameless subject. Voiceover: "says in an off-screen voiceover" with
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
- If the user assigns speaker IDs like (S1) or (S2), keep them exactly: same numbers, same
  people, same vocal order. Assign the next free ID only to vocal characters the user never names.
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


# ── RV2V subject-mode blocks (injected into H3_RV2V_SYSTEM_PROMPT_ADV) ──────
# same_subject=False (default): each ref image is its own replacement character.
# same_subject=True: all ref images are ONE person (multiple views merged into
# a single <Subject 1>) — the batch use case.
H3_RV2V_SEMANTICS_SEPARATE = (
    "- The reference images (<Picture 1>, <Picture 2>, ...) are REPLACEMENT "
    "characters: each picture defines the appearance of a character to swap into "
    "<Video 1>'s scene (<Picture 1> -> the source video's existing character, or "
    "a new character the user wants).\n"
    "- NEVER treat the pictures as the same person as each other unless the user "
    "says so. Each picture anchors its own replacement character — always bind the "
    "character's appearance to its exact <Picture N> label, never to an invented "
    "name the model cannot connect to the image.\n"
    "- You may give each character a speaking name (<Subject N>) for speaker tags, "
    "but ALWAYS define it as \"<Subject N> is the person from <Picture N> with ...\"."
)

H3_RV2V_SEMANTICS_SAME = (
    "- The reference images (<Picture 1>, <Picture 2>, ...) are ALL THE SAME person — "
    "multiple views (face, body, tattoos, outfit, back-view) of ONE replacement "
    "character who swaps into <Video 1>'s scene. <Picture 1> is the primary view; "
    "<Picture 2>+ are additional angles/detail of the SAME person.\n"
    "- NEVER split the pictures into separate people: they define ONE unified "
    "character. Give that single character ONE speaking name (<Subject 1>) and "
    "define it as \"the unified person from <Picture 1>, with additional views in "
    "<Picture 2>, <Picture 3>, ...\". Never <Subject 2>, <Subject 3>, etc."
)

H3_RV2V_DEFS_SEPARATE = (
    "Each reference image anchors its own replacement character: \"<Subject 1> is "
    "the person from <Picture 1> with [identity features]\" — the <Picture N> "
    "label is what binds the description to the image."
)

H3_RV2V_DEFS_SAME = (
    "<Subject 1> is the ONE unified person whose appearance merges <Picture 1> "
    "plus every additional view (<Picture 2>, ...): face, hair, body, skin tone, "
    "distinguishing marks, clothing. Do NOT create <Subject 2>, <Subject 3>, etc. — "
    "all reference images feed the single character anchored at <Picture 1>."
)

H3_RV2V_RETENTION_SEPARATE = (
    "Character-swap retention (REQUIRED): mark <Video 1> as partially_preserved "
    "(scene, framing, lighting, camera preserved; the character being replaced is "
    "swapped out). Each REPLACEMENT character (<Subject N> from <Picture N>) is "
    "attribute_transfer — transfer face, body, hair style, hair color, skin tone, "
    "clothing, and makeup from the reference. Mark each <Picture N> as "
    "attribute_transfer TOO — every reference picture is a STRONG identity source "
    "whose features must be transferred; never mark an identity picture as "
    "weak_reference. Do NOT name the replaced character as its own <Subject N>; "
    "describe it only as the person being replaced inside <Video 1>'s definition."
)

H3_RV2V_RETENTION_SAME = (
    "Character-swap retention (REQUIRED): mark <Video 1> as partially_preserved "
    "(scene, framing, lighting, camera preserved; the character being replaced is "
    "swapped out). <Subject 1> (the unified person from all reference images) is "
    "attribute_transfer — transfer face, body, hair style, hair color, skin tone, "
    "clothing, and makeup from the reference. Mark each <Picture N> as "
    "attribute_transfer TOO — every reference picture is a STRONG identity source "
    "whose features must be transferred; never mark an identity picture as "
    "weak_reference. Do NOT name the replaced character as its own <Subject N>; "
    "describe it only as the person being replaced inside <Video 1>'s definition."
)

H3_RV2V_RULES_SEPARATE = (
    "- Each reference IMAGE (<Picture N>) anchors a SEPARATE replacement character "
    "unless the user says they're the same person.\n"
    "- Do NOT define the ORIGINAL character from <Video 1> as its own <Subject N>. "
    "Describe it only as the person being replaced inside <Video 1>'s definition "
    "(e.g. \"the woman in <Video 1>, fully replaced\"). Use attribute_transfer for "
    "each replacement character.\n"
    "- REFERENCES ARE STRONG (REQUIRED): every reference picture carries identity "
    "that must reach the render — mark EVERY <Picture N> as attribute_transfer. "
    "There is NO weak_reference in character swap — no picture is 'identity source "
    "only', none is 'not appearing'. If a picture shows a body part, tattoo, or "
    "outfit detail, that detail is transferred into its <Subject N>'s definition."
)

H3_RV2V_RULES_SAME = (
    "- All reference IMAGES (<Picture 1>..<Picture N>) define the SAME <Subject 1> — "
    "one unified person, multiple views. Never define separate per-image subjects.\n"
    "- Do NOT define the ORIGINAL character from <Video 1> as its own <Subject N>. "
    "Describe it only as the person being replaced inside <Video 1>'s definition "
    "(e.g. \"the woman in <Video 1>, fully replaced\"). Use attribute_transfer for "
    "the replacement character.\n"
    "- REFERENCES ARE STRONG (REQUIRED): every reference picture carries identity "
    "that must reach the render — mark EVERY <Picture N> as attribute_transfer. "
    "There is NO weak_reference in character swap — no picture is 'identity source "
    "only', none is 'not appearing'. If a picture shows a body part, tattoo, or "
    "outfit detail, that detail is transferred into <Subject 1>'s definition."
)

H3_RV2V_SYSTEM_PROMPT_ADV = """\
You are a Creative Assistant for MiniMax H3 video generation in REFERENCE-IMAGE-AS-VIDEO
to video (RV2V) mode — the H3 equivalent of Bernini subject-to-video (R2V). Given the
user's raw request and reference media, write the complete 6-field structured prompt H3 expects.

Key RV2V semantics:
- The static reference video (<Video 1>) carries the SCENE: its setting, framing,
  lighting, background, camera movement, and any characters that stay unchanged.
{subject_semantics}
- Task type is [reference generation]. The target video keeps <Video 1>'s scene and
  replaces characters with the reference-image subjects.

Output SIX fields in this exact order:

subject_definitions:
One line per label. <Video 1> is ... (the static scene reference clip — its setting,
framing, lighting, and unchanged characters are the target video's scene).
{subject_defs}
<Audio N> is ... (audio assets if provided).

summary:
[reference generation] ONE short paragraph: the target video keeps <Video 1>'s scene
while <Subject N> (from the <Picture N> reference images) replaces the scene's character(s). If
the source video's scene has characters to replace, say which replaces which.

retention_analysis:
One line per reference label with relationship marker: <Video 1> is partially_preserved
(scene, framing, lighting, camera preserved; characters listed get replaced or stay).
{subject_retention}
List which shots each subject appears in.

detailed_description:
The main body — 350-500 words. Style established in 1-2 sentences BEFORE [Shot 1].
No timestamp on [Shot 1]. Later shots: "[Shot N] At MM:SS.mmm, the camera cuts to ..."
with strictly increasing cut times within the target duration; prefer camera motion over a cut
for slight changes. Camera motion as natural English (type + amplitude + speed). Speakers:
<Subject N> (Sx) with stable global speaker IDs across shots; compound (S1,S2) for group
speech; identity established outside the tag; dialogue verbatim inside
<d>[Language] actual words.</d>. ANY vocalization (groans, moans, laughter, screams,
gasps) is a vocal event: the source must have or be given an (Sx) ID, with the
nonverbal sound described as plain prose. Voiceover: "says in an off-screen voiceover" with
"the character's lips remain completely closed." <scenetrans> / <cutoff> for dialogue
crossing cuts / truncated by video end; [unclear] for unintelligible spans; on-screen
text in English double quotes; punctuation standardized to , . ? !
Reference labels (<Subject N> bound to <Picture N>, <Video 1>, <Audio N>) at first appearance
and where roles apply. Describe each replacement subject's identity from ITS reference
image with enough specificity that the renderer binds to the right picture — ALWAYS name the
exact <Picture i> label that carries that subject (face shape, eyes, hair, body, skin tone,
distinguishing marks). The <Picture N> label is the ANCHOR; the text
strengthens it. Keep <Video 1>'s scene, framing, lighting, and camera.

overall_soundscape: 1-4 sentences in ONE continuous paragraph summarizing ambient +
physical + non-verbal human sounds. Cite <Audio N> if copied/referenced. Use N/A only
for complete silence. Never repeat <d> dialogue here.

non_diegetic_music: 1-3 sentences on instrumentation, tempo, rhythm, dynamic changes of
audience-only score. Cite <Audio N> if copied/referenced. Use N/A when none. Diegetic
music belongs in detailed_description, not here.

Rules (from official MiniMax H3 writing guides + community refinements):
- Once a reference label is assigned, keep its meaning across ALL sections.
{subject_rules}
- <Video 1> supplies the SCENE, not a character identity. A replaced character's appearance
  comes from its <Picture N> reference image, not from <Video 1> — the <Picture N>
  label is the ONLY image anchor the renderer sees, so every identity statement must
  name the exact <Picture i> that carries it.
- If the user assigns speaker IDs like (S1) or (S2), keep them exactly: same numbers, same
  people, same vocal order. Assign the next free ID only to vocal characters the user never names.
- Do NOT invent new reference labels after subject_definitions.
- Audio cannot appear as a general reference on its own — a reference image or video
  must accompany it. Frame anchors and general references are mutually exclusive.
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

H3_RV2V_TEMPLATE = """\
Write the H3 full-reference structured prompt (6 fields) for a reference-image-as-video
to video generation (RV2V).

Task type: reference generation (RV2V) — the static reference video (<Video 1>) provides
the SCENE; the reference images (<Picture 1>, <Picture 2>, ...) provide the REPLACEMENT characters.
Video duration: {duration_seconds}s

{ref_section}

Rules for RV2V reference mode:
- Task type in summary is [reference generation].
- <Video 1> is a STATIC scene reference: all frames show the same fixed shot — its setting,
  framing, lighting, background, and camera are the target video's scene. The target video
  is NOT free to invent a new scene.
- REMEMBER: the renderer binds identity through the <Picture N> labels only. Always write
  "<Subject N> is the person from <Picture N>" — never describe a character without naming
  the <Picture N> that anchors them.

{subject_mode_rules}
- Scene elements NOT tied to a replaced character are preserved as-is: background, lighting,
  camera movement, props, and any characters that stay.
- detailed_description is the main body — 350-500 words, reference labels at first appearance.
- In retention_analysis mark <Video 1> as partially_preserved (scene/framing/lighting kept).

User's raw request:
{user_prompt}"""

H3_RV2V_TEMPLATE_SAME_SUBJECT = """\
Write the H3 full-reference structured prompt (6 fields) for a reference-image-as-video
to video generation (RV2V).

Task type: reference generation (RV2V) — the static reference video (<Video 1>) provides
the SCENE; the reference images (<Picture 1>, <Picture 2>, ...) provide the REPLACEMENT characters.
Video duration: {duration_seconds}s

{ref_section}

Rules for RV2V reference mode:
- Task type in summary is [reference generation].
- <Video 1> is a STATIC scene reference: all frames show the same fixed shot — its setting,
  framing, lighting, background, and camera are the target video's scene. The target video
  is NOT free to invent a new scene.
- REMEMBER: the renderer binds identity through the <Picture N> labels only. Always write
  "<Subject N> is the person from <Picture N>" — never describe a character without naming
  the <Picture N> that anchors them.

{subject_mode_rules}
- Scene elements NOT tied to a replaced character are preserved as-is: background, lighting,
  camera movement, props, and any characters that stay.
- detailed_description is the main body — 350-500 words, reference labels at first appearance.
- In retention_analysis mark <Video 1> as partially_preserved (scene/framing/lighting kept).

User's raw request:
{user_prompt}"""

H3_RV2V_SUBJECT_RULES_SEPARATE = """\
- Each reference image anchors a REPLACEMENT character: "<Subject 1> is the person from
  <Picture 1> with [full appearance — face, hair, body, skin, clothing, distinguishing marks]"
  — the character whose attributes replace the corresponding character in <Video 1>'s scene.
- In retention_analysis mark each replacement character as attribute_transfer — transfer
  the face, body, hair style, hair color, skin tone, clothing, and makeup from the reference.
  Mark EVERY <Picture N> as attribute_transfer — every reference picture is a STRONG
  identity source whose features must reach the render. NEVER use weak_reference for a
  character reference picture. Do NOT name the replaced character as its own subject —
  describe it only inside <Video 1>'s definition."""

H3_RV2V_SUBJECT_RULES_SAME = """\
- ALL reference images show the SAME person (multiple views — face, body, detail).
  <Picture 1> defines the ONE unified subject: "<Subject 1> is the person from <Picture 1>
  with [full appearance]". The remaining images (<Picture 2>, ...) are ADDITIONAL VIEWS of
  that same <Subject 1> — merge their features into its single definition. NEVER split into
  separate per-image subjects — every image is a view of the same single person.
- In retention_analysis mark <Subject 1> as attribute_transfer — transfer the face, body,
  hair style, hair color, skin tone, clothing, and makeup from the reference pictures.
  Mark EVERY <Picture N> as attribute_transfer — all reference pictures are STRONG
  identity sources and must feed the transfer. NEVER use weak_reference for a character
  reference picture. Do NOT name the replaced character as its own subject — describe it
  only inside <Video 1>'s definition."""


# ═══════════════════════════════════════════════════════════════════════════════
# Task-type metadata
# ═══════════════════════════════════════════════════════════════════════════════

H3_TASK_LABELS = {
    "t2v": "T2V (text-to-video, no reference images)",
    "i2v": "I2V (image-to-video, first-frame reference)",
    "fl2v": "FL2V (first-and-last-frame, two image anchors)",
    "l2v": "L2V (last-frame only, converge to the image at the end)",
    "r2v": "R2V (full-reference mode with subject/picture/video/audio labels)",
    "rv2v": "RV2V (static reference video = scene, reference images = replacement characters)",
    "ri2v": "RI2V (reference images to video — scene from ref image or source video, character identity from ref images)",
}

H3_BASE_TASK_TYPES = {"t2v", "i2v", "fl2v", "l2v"}
H3_RV2V_TASK_TYPES = {"rv2v"}
# ri2v reuses the proven r2v 6-field template (LLM-chosen retention), with a
# scene-labeled source video and a replacement-preservation override. It is
# deliberately NOT in H3_RV2V_TASK_TYPES so it routes through the r2v branch.
H3_RI2V_TASK_TYPES = {"ri2v"}

H3_REF_STRONG_RULE = """\
REFERENCES ARE STRONG (REQUIRED, character swap): every reference picture is an
identity source whose features MUST reach the target. In retention_analysis mark
EVERY character reference <Picture N> as attribute_transfer — never weak_reference.
No picture is "identity source only", "not reproduced as a frame", or "not
appearing in the scene". When a picture shows a body part, tattoo, clothing, or
detail, that detail is transferred into the <Subject N> definition it anchors."""

H3_SAME_SUBJECT_RULE = """\
SAME SUBJECT: ALL reference images show the SAME person/subject from different
angles, poses, or lighting. Analyze them as ONE unified subject — combine identity
features from every reference into a single subject definition, and never treat
<Picture 1> / <Picture 2> / <Picture 3> as different people.
REFERENCES ARE STRONG: every reference picture feeds the subject's identity —
mark EVERY <Picture N> as attribute_transfer, never weak_reference. No picture is
"identity source only" or "not appearing in the scene"; each contributes its
face/body/outfit details to the unified <Subject 1>."""

H3_RV2V_SAME_SUBJECT_RULE = """\
SCENE/REPLACEMENT RULE: the static reference video (<Video 1>) provides the SCENE —
its setting, framing, lighting, background, and camera. The reference images
(<Picture 1>, <Picture 2>, ...) define the REPLACEMENT characters that appear in that scene.
Each reference image anchors its OWN replacement character (<Picture 1> -> <Subject 1>,
<Picture 2> -> <Subject 2>, ...) unless the user explicitly says two images are the same person.
Never merge the reference images into one subject, and never treat <Video 1>'s
content as the identity of a replaced character — it is the scene."""

# RI2V retention recipe — character-swap structure matching the community
# minimaxH3Character workflow: replacement = attribute_transfer (transfer
# face/body/hair/clothing from reference), picture refs = attribute_transfer,
# all references STRONG (never weak_reference),
# source video = partially_preserved, replaced character NOT named as own subject.
H3_RI2V_RULE = """\
CHARACTER-SWAP RETENTION (REQUIRED): mark the source video <Video N> as
partially_preserved — its scene, structure, framing, camera, timing, and background
are kept, while the character being replaced is swapped out. Mark the REPLACEMENT
character(s) from the reference pictures as attribute_transfer — transfer the
face, body, hair style, hair color, skin tone, clothing, and makeup from the
reference picture onto the character in the video. Describe their full appearance
in subject_definitions. Mark each reference <Picture N> as attribute_transfer TOO —
every reference picture is a STRONG identity source; never mark an identity
picture as weak_reference. Do NOT name the replaced character as its own
<Subject N> — describe it only as the person being replaced inside <Video N>'s
definition."""

# RI2V scene-as-image variant: scene comes as <Picture 1> instead of <Video N>
H3_RI2V_RULE_SCENE_IMAGE = """\
CHARACTER-SWAP RETENTION (REQUIRED): <Picture 1> provides the scene — mark it
fully_preserved in retention_analysis (environment, framing, lighting, composition
all preserved). Mark the REPLACEMENT character(s) from the other reference
pictures as attribute_transfer — transfer the face, body, hair style, hair color,
skin tone, clothing, and makeup from the reference picture onto the character in
the scene. Describe their full appearance in subject_definitions. Mark each
character reference <Picture N> (N > 1) as attribute_transfer TOO — every
reference picture is a STRONG identity source; never mark an identity picture as
weak_reference."""

# Same-subject rule variant when scene is the first ref image
H3_SAME_SUBJECT_RULE_SCENE_IMAGE = """\
SAME SUBJECT: ALL character reference images (<Picture 2> onward) show the SAME
person/subject from different angles, poses, or lighting. Analyze them as ONE
unified subject — combine identity features from every character reference into a
single subject definition. <Picture 1> is the SCENE (environment/setting),
NOT a character — do not merge it into the person's identity. NEVER split
<Picture 2>, <Picture 3>, etc. into separate per-image subjects — every character
image is a view of the same single person.
REFERENCES ARE STRONG: every character reference picture feeds the subject's
identity — mark EVERY character <Picture N> as attribute_transfer, never
weak_reference. No picture is "identity source only" or "not appearing";
each contributes its face/body/outfit details to the unified <Subject 1>."""

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
LLM, so precision matters more than prose. When the user's notes assign speaker
IDs like (S1) or (S2), bind each ID to the matching person you see, based on
physical identity.

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
5. "speaker_label": if the user's notes below assign this person a speaker ID such
   as (S1) or (S2), repeat that exact ID here so the write pass can bind identity
   to the speaker. Match by physical identity (clothing, hair, body type,
   position) between the notes and the image. If the notes assign NO speaker ID
   to this person, omit the key or use null. NEVER invent a (Sx) the notes do
   not assign.

IMPORTANT:
- Only describe what is actually visible. Do NOT hallucinate details.
- For multiple images output a JSON array with one object per image, in order.
- No markdown fences. Return ONLY the JSON.

Directive:
- If the user's notes below assign speaker IDs like (S1), (S2), or (S3), treat
  every ID as fixed: do not renumber, re-label, or drop any of them in your
  descriptions.

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
    "rv2v": (
        "- Full-reference mode with subject_definitions, summary, retention_analysis, detailed_description, "
        "overall_soundscape, non_diegetic_music.\n"
        "- <Video 1> is a STATIC SCENE reference: its setting, framing, lighting, background, "
        "and camera movement are the target video's scene.\n"
        "- Each reference IMAGE defines a REPLACEMENT character: <Subject 1> is the person from "
        "<Picture 1>, <Subject 2> from <Picture 2>, etc. — the character swapped into the scene.\n"
        "- Scene elements not tied to a replaced character are preserved as-is: background, "
        "lighting, camera, props, and characters that stay.\n"
        "- Never merge the reference images into one subject unless the user says they're the "
        "same person. Never use the ref-video's content as a replaced character's identity.\n"
        "- detailed_description is the main body — 350-500 words, reference labels at first "
        "appearance, each replacement subject's identity described from ITS reference image."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# H3 Prompt Enhancer Node
# ═══════════════════════════════════════════════════════════════════════════════

class H3PromptEnhancer:
    """Enhances user prompts into MiniMax H3 structured format via a vision LLM.

    Produces the structured prompt that H3 expects — the community substitute for
    the hosted "H3-Context-IR" prompt-enhancement system. One node handles all
    H3 task types: a task_type widget (T2V / I2V / FL2V / L2V / R2V / RV2V) switches
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
                    ["t2v", "i2v", "fl2v", "l2v", "r2v", "rv2v", "ri2v"],
                    {"default": "t2v",
                     "tooltip": "t2v=text-to-video (no images), i2v=first-frame, "
                                "fl2v=first+last frame, l2v=last-frame only, "
                                "r2v=full-reference (6-field output), "
                                "rv2v=static ref-video = scene, reference images = "
                                "replacement characters (Bernini RV2V semantics)."}
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
                "images_batch": ("IMAGE", {
                    "tooltip": "Batch of reference images (e.g. from BatchImagesNode). "
                               "Each frame becomes a reference image, numbered in order "
                               "after any wired reference_image_0/1/2. A 6-frame batch "
                               "with no individual slots wired gives image0..image5. "
                               "Works in R2V / RV2V / I2V modes."
                }),
                "source_video": ("IMAGE", {
                    "tooltip": "Source video as batched frames (e.g. from BerniniLoadVideo). "
                               "Frames are sampled across the timeline and shown to the LLM as "
                               "<Video 1> — the source asset for video editing / video continuation. "
                               "R2V mode only. For RV2V, wire the static scene ref-video here "
                               "(e.g. from H3 Image to Ref Video) — the LLM keeps its scene "
                               "and swaps in the reference-image characters."
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
                    "tooltip": "APPENDED to the selected built-in H3 system prompt as a "
                               "patch (e.g. copy/consistency overrides) — the base format "
                               "and speaker rules are always kept. Leave empty for the built-in "
                               "prompt alone. For a full replacement, paste the entire prompt."
                }),
                "custom_model": ("STRING", {
                    "default": "",
                    "tooltip": "Any OpenRouter model ID. Overrides the dropdown when non-empty."
                }),
                "api_key": ("STRING", {"default": ""}),
                # NOTE: new widgets appended LAST so existing saved workflows keep
                # their widgets_values alignment.
                "same_subject": (["auto", "on", "off"], {
                    "default": "auto",
                    "tooltip": "Whether ALL reference images show the SAME person from "
                               "different angles/poses (merged into ONE unified <Subject 1>)\n"
                               "  auto = honor the old boolean switch AND the raw prompt's "
                               "phrasing ('same subject' / 'same person' in the user text)\n"
                               "  on   = force same-subject merge scaffold\n"
                               "  off  = force per-image subjects (each picture anchors its "
                               "own replacement character). Use to A/B the same-subject "
                               "scaffold against per-image."
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
                # NOTE: must remain the LAST optional input — saved workflows'
                # widgets_values arrays append in order, so trailing entries are
                # safe to miss before this one.
                "seed": ("INT", {
                    "default": -1, "min": -1, "max": 2147483647, "step": 1,
                    "tooltip": "Sampling seed sent to the LLM backend. -1 = fresh random "
                               "seed every run (new prompt each queue; only works when the "
                               "provider honours a seed field), 0+ = fixed & reproducible "
                               "output. Overrides nothing else — temperature still applies."
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
                    lbl = entry.get("speaker_label")
                    if lbl and str(lbl).strip():
                        line += f" | Speaker: {lbl}"
                    if line.strip():
                        parts.append(line)
                return "\n".join(parts) if parts else None
            if isinstance(parsed, dict) and parsed.get("image_id"):
                line = f"{parsed.get('image_id')}: {parsed.get('subject', '')}".strip()
                if parsed.get("scene"):
                    line += f" | Scene: {parsed['scene']}"
                if parsed.get("current_state"):
                    line += f" | State: {parsed['current_state']}"
                lbl = parsed.get("speaker_label")
                if lbl and str(lbl).strip():
                    line += f" | Speaker: {lbl}"
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
                        hints, temperature, max_tokens, user_prompt="",
                        seed=-1):
        """First-pass: describe every attached image as structured JSON.

        The caller's raw prompt is passed through (as user notes) so any speaker
        IDs or naming the user assigned can be bound to the right identities in
        the images. Returns a formatted text block the write pass uses to anchor
        identity, or None if the analysis failed."""
        if not image_parts:
            return None
        content = list(image_parts)
        text = H3_ANALYZE_PROMPT.format(
            image_count=len(image_parts),
            image_roles="\n".join(
                f"- image{i}: {role}" for i, role in enumerate(roles)),
            hints=hints or "(none)",
            user_prompt=user_prompt.strip() or
                       "(no user notes — infer everything from the images)",
        )
        content.append({"type": "text", "text": text})
        try:
            raw = api_fn(
                api_key, llm_model, H3_ANALYZE_SYSTEM_PROMPT, content,
                temperature=max(temperature - 0.2, 0.0),
                max_tokens=max_tokens,
                seed=seed,
            )
        except Exception as e:
            logging.warning(f"[H3PromptEnhancer] auto_describe analysis failed: {e}")
            return None
        return self._format_analysis(raw)

    def enhance(self, prompt, task_type, duration, model, local_backend="off",
                source_image=None, last_frame_image=None,
                reference_image_0=None, reference_image_1=None,
                reference_image_2=None, images_batch=None, source_video=None,
                temperature=0.7, max_tokens=4096,
                custom_system_prompt="", custom_model="", api_key=None,
                advanced_prompt="on",
                same_subject="auto", style_transfer="off",
                auto_describe=True, auto_describe_max_tokens=4096,
                seed=-1):
        """Enhance a user prompt into H3 structured format."""

        # ── Resolve same_subject (tri-state) ───────────────────────────────
        # "on" forces the unified-subject scaffold, "off" forces per-image
        # subjects, "auto" honors the prompt's phrasing ("same subject" /
        # "same person" in the raw user text). Backward compatible: the old
        # boolean True/False still resolves to on/off.
        if same_subject is True or same_subject == "on":
            same_subject = True
        elif same_subject is False or same_subject == "off":
            same_subject = False
        elif same_subject == "auto":
            if prompt and re.search(
                    r"\b(same subject|same person|all (?:the )?same|one (?:and )?the same)\b",
                    prompt, re.IGNORECASE):
                same_subject = True
                logging.info("[H3PromptEnhancer] same_subject auto-detected from "
                             "prompt phrasing")
            else:
                same_subject = False

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
                           retries=2, seed=None):
                return _call_ollama_chat(
                    ollama_url, ollama_model, user_content, system_prompt,
                    temperature=temperature, max_tokens=max_tokens,
                    timeout=timeout, seed=seed)
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
        if task_type in H3_BASE_TASK_TYPES:
            system_prompt = (H3_BASE_SYSTEM_PROMPT_ADV if use_adv
                             else H3_BASE_SYSTEM_PROMPT)
        elif task_type in H3_RV2V_TASK_TYPES:
            system_prompt = H3_RV2V_SYSTEM_PROMPT_ADV.format(
                subject_semantics=(H3_RV2V_SEMANTICS_SAME if same_subject
                                   else H3_RV2V_SEMANTICS_SEPARATE),
                subject_defs=(H3_RV2V_DEFS_SAME if same_subject
                              else H3_RV2V_DEFS_SEPARATE),
                subject_retention=(H3_RV2V_RETENTION_SAME if same_subject
                                   else H3_RV2V_RETENTION_SEPARATE),
                subject_rules=(H3_RV2V_RULES_SAME if same_subject
                               else H3_RV2V_RULES_SEPARATE),
            )
        else:
            system_prompt = (H3_REF_SYSTEM_PROMPT_ADV if use_adv
                             else H3_REF_SYSTEM_PROMPT)

        # custom_system_prompt APPENDS to the built-in prompt as a patch
        # (copy/consistency overrides), so the base format + speaker rules are
        # never lost. For a true full replacement, write the entire prompt here.
        if custom_system_prompt and custom_system_prompt.strip():
            system_prompt = (system_prompt + "\n\n" +
                             custom_system_prompt.strip())

        # Local models need explicit NSFW permission
        if use_local:
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

        # Source video — R2V video editing/continuation: sampled frames shown
        # as <Video 1>. Per the official guide, <Video N> is reserved for
        # whole-video relationships (editing, continuation, camera/cut structure).
        # RV2V: source_video is a STATIC SCENE reference (a repeated image,
        # e.g. from H3 Image to Ref Video) — same <Video 1> label, but the LLM is
        # told it defines identity ONLY, never editing/continuation.
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
        elif source_video is not None and task_type == "ri2v":
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
                    f"frame {i + 1}/{len(idxs)} of the SOURCE video (<Video 1>) — scene and structure")
            ref_section += (
                f"A source video (<Video 1>) is attached as {len(idxs)} frames "
                f"sampled across its {n_frames}-frame timeline. <Video 1> supplies the "
                "target video's scene, framing, lighting, camera, timing, body positions, "
                "and structure — all preserved. The character being replaced appears IN "
                "<Video 1>, but do NOT define that original character as its own subject; "
                "describe it only as the person being replaced inside <Video 1>'s "
                "definition. In retention_analysis mark <Video 1> partially_preserved "
                "(scene/structure kept, replaced character fully replaced).\n\n")
        elif source_video is not None and task_type == "rv2v":
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
                    f"frame {i + 1}/{len(idxs)} of the STATIC SCENE reference (<Video 1>)")
            ref_section += (
                f"A static reference video (<Video 1>) is attached as {len(idxs)} frames "
                f"sampled across its {n_frames}-frame timeline. ALL frames show the SAME "
                "fixed shot — it is a single scene frame repeated as a short clip. <Video 1> "
                "supplies the target video's scene, framing, lighting, camera, timing, body "
                "positions, and structure — all preserved. The character being replaced "
                "appears IN <Video 1>, but do NOT define that original character as its own "
                "subject; describe it only as the person being replaced inside <Video 1>'s "
                "definition (e.g. \"the woman in <Video 1>, fully replaced\"). In "
                "retention_analysis mark <Video 1> as partially_preserved (scene/structure "
                "kept, replaced character swapped out). Mark the REPLACEMENT character(s) "
                "from the reference images as attribute_transfer — transfer face, body, hair, "
                "skin tone, clothing from the reference.\n\n")

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
        if task_type in ("r2v", "rv2v", "i2v", "ri2v"):
            ref_images = []
            # RI2V scene-as-image: source_image becomes the scene (<Picture 1>),
            # prepended before character ref images so numbering aligns with
            # the H3RefToVid node (where the same image wires to ref_image_0).
            if (task_type == "ri2v" and source_image is not None
                    and source_video is None):
                scene_img = (source_image[0] if source_image.dim() == 4
                             else source_image)
                ref_images.append(scene_img)
            for ref in (reference_image_0, reference_image_1, reference_image_2):
                if ref is not None:
                    ref_images.append(ref)
            # Batch input: each frame becomes a numbered reference image,
            # continuing after any individually-wired slots.
            if images_batch is not None:
                if images_batch.dim() == 4:
                    for f in images_batch:
                        ref_images.append(f)
                else:
                    ref_images.append(images_batch)
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
                elif task_type == "rv2v":
                    if same_subject:
                        # All ref images are ONE person (multiple views: face,
                        # body, details). <Picture 1> is the primary view; the
                        # rest are additional views feeding the SAME <Subject 1>.
                        if i == 0:
                            role_list.append(
                                f"reference image <Picture {i + 1}> defining <Subject 1> "
                                "(the ONE unified person; other images are extra views)")
                            ref_section += (
                                f"Reference image <Picture {i + 1}> is attached and defines "
                                "<Subject 1> — the SINGLE unified person who replaces the "
                                "original character. Describe <Subject 1>'s full appearance "
                                f"from <Picture {i + 1}>: face shape, eyes, hair color/style, "
                                "body type, skin tone, clothing, and distinguishing marks. "
                                "Subsequent reference images are ADDITIONAL VIEWS of this "
                                "SAME person — merge their features into <Subject 1>, never "
                                "create new subjects. In retention_analysis mark <Subject 1> "
                                "and EVERY <Picture N> as attribute_transfer (strong identity "
                                "sources; never weak_reference).\n")
                        else:
                            role_list.append(
                                f"reference image <Picture {i + 1}> (additional view of "
                                "<Subject 1> — same person, different angle/detail)")
                            ref_section += (
                                f"Reference image <Picture {i + 1}> is an ADDITIONAL VIEW of "
                                "the SAME <Subject 1> (the same person as <Picture 1> — this "
                                "shot shows another angle / body detail / outfit of the ONE "
                                "unified subject). Merge its features into <Subject 1>'s "
                                "definition: add anything new this view reveals (body detail, "
                                "tattoos, build, hair back-view, etc.). Do NOT create a new "
                                "subject.\n")
                    else:
                        role_list.append(
                            f"reference image <Picture {i + 1}> defining <Subject {i + 1}> "
                            "(REPLACEMENT character into <Video 1>'s scene)")
                        ref_section += (
                            f"Reference image <Picture {i + 1}> is attached and defines "
                            f"<Subject {i + 1}> — the REPLACEMENT character to swap into "
                            f"<Video 1>'s scene. Describe <Subject {i + 1}>'s full "
                            f"appearance from <Picture {i + 1}>: face shape, eyes, hair "
                            "color/style, body type, skin tone, clothing, and distinguishing "
                            "marks. In retention_analysis mark <Subject "
                            f"{i + 1}> AND <Picture {i + 1}> as attribute_transfer (strong "
                            "identity source; never weak_reference). The scene, "
                            "framing, lighting, and camera come from <Video 1>.\n")
                elif task_type == "ri2v":
                    ri2v_scene_as_image = (source_video is None
                                           and source_image is not None)
                    if ri2v_scene_as_image and i == 0:
                        role_list.append(
                            f"reference image <Picture {i + 1}> — SCENE "
                            "(environment, framing, lighting)")
                        ref_section += (
                            f"Reference image <Picture {i + 1}> provides the SCENE — "
                            "the environment, setting, framing, lighting, and "
                            "composition for the target video. Define it as a "
                            "scene/environment in subject_definitions (e.g. "
                            f"'<Subject N> is the setting in <Picture {i + 1}>, "
                            "with...'). In retention_analysis mark it as "
                            "fully_preserved (scene preserved). Do NOT derive "
                            "any character's identity from this image.\n")
                    elif same_subject:
                        is_primary = ((not ri2v_scene_as_image and i == 0)
                                      or (ri2v_scene_as_image and i == 1))
                        scene_label = ("<Picture 1>'s" if ri2v_scene_as_image
                                       else "<Video 1>'s")
                        if is_primary:
                            role_list.append(
                                f"reference image <Picture {i + 1}> defining the REPLACEMENT "
                                "character's full appearance (primary view)")
                            ref_section += (
                                f"Reference image <Picture {i + 1}> is attached and defines "
                                "the REPLACEMENT character (<Subject 1>) who appears in "
                                f"{scene_label} scene. Describe <Subject 1>'s full appearance "
                                f"from <Picture {i + 1}>: face shape, eyes, hair color/style, "
                                "body type, skin tone, clothing, and distinguishing marks. "
                                "Subsequent reference images are ADDITIONAL VIEWS of this "
                                "SAME person — merge their features into <Subject 1>, never "
                                "create new subjects. In retention_analysis mark <Subject 1> "
                                "and EVERY <Picture N> as attribute_transfer (strong identity "
                                "sources; never weak_reference).\n")
                        else:
                            role_list.append(
                                f"reference image <Picture {i + 1}> (additional view of "
                                "<Subject 1> — same person, different angle/detail)")
                            ref_section += (
                                f"Reference image <Picture {i + 1}> is an ADDITIONAL VIEW of "
                                "the SAME <Subject 1>. Merge any new details this view "
                                "reveals (body, tattoos, build, back-view, outfit details) "
                                "into <Subject 1>'s definition. Do NOT create a new subject.\n")
                    else:
                        subj_n = i + 1 if not ri2v_scene_as_image else i
                        scene_label = ("<Picture 1>'s" if ri2v_scene_as_image
                                       else "<Video 1>'s")
                        role_list.append(
                            f"reference image <Picture {i + 1}> defining the REPLACEMENT "
                            "character's full appearance")
                        ref_section += (
                            f"Reference image <Picture {i + 1}> is attached and defines the "
                            f"REPLACEMENT character (<Subject {subj_n}>) who appears in "
                            f"{scene_label} scene. Describe <Subject {subj_n}>'s full appearance "
                            f"from <Picture {i + 1}>: face shape, eyes, hair color/style, "
                            "body type, skin tone, clothing, and distinguishing marks. In "
                            f"retention_analysis mark <Subject {subj_n}> AND <Picture {i + 1}> "
                            "as attribute_transfer (strong identity source; never "
                            "weak_reference).\n")
                else:
                    role_list.append(
                        f"reference image (label as Picture {i + 1} or Subject {i + 1})")
                    ref_section += (f"Reference image {i} is attached "
                                    f"(label as Picture {i+1} or Subject {i+1}).\n")
            if ref_images:
                ref_section += "\n"
                # Hard bound on <Picture N> labels: the renderer binds pictures by
                # ORDER of the attached refs only — no other picture label exists.
                # Stops the LLM inventing <Picture 6>, <Picture 9>, etc. when the
                # user attached 5 images (phantom labels steer identity nowhere).
                n_p = len(ref_images)
                pic_list = ", ".join(f"<Picture {i + 1}>" for i in range(n_p))
                ref_section += (
                    f"EXACTLY {n_p} reference image(s) are attached, in this order: "
                    f"{pic_list}. Use NO other <Picture N> label — any picture "
                    f"beyond <Picture {n_p}> does not exist in this generation and "
                    "MUST NOT appear in subject_definitions, retention_analysis, "
                    "or detailed_description. Every listed picture MUST appear in "
                    "retention_analysis; none is weak_reference, none is 'not "
                    "appearing'.\n\n")

        # ── Same-subject + style-transfer directives ─────────────────────
        # Appended to the reference section so the LLM merges multiple photos
        # of one person into a single subject and/or renders the reference in a
        # non-photorealistic art style (Bernini-style controls).
        extra_rules = []
        has_refs = (any(r is not None for r in
                        (reference_image_0, reference_image_1, reference_image_2))
                    or images_batch is not None)
        # RV2V: static ref-video = scene, ref images = replacement characters.
        # Scene video and images are DIFFERENT roles. BUT when same_subject is
        # ON, the user is saying all ref images are ONE person (multiple views
        # of a single character) — that intent wins over the per-image rule.
        ri2v_scene_mode = (task_type in H3_RI2V_TASK_TYPES
                           and source_video is None
                           and source_image is not None and has_refs)
        if same_subject and has_refs:
            if ri2v_scene_mode:
                extra_rules.append(H3_SAME_SUBJECT_RULE_SCENE_IMAGE)
            else:
                extra_rules.append(H3_SAME_SUBJECT_RULE)
        if task_type in H3_RI2V_TASK_TYPES and has_refs and source_video is not None:
            extra_rules.append(H3_RI2V_RULE)
        elif ri2v_scene_mode:
            extra_rules.append(H3_RI2V_RULE_SCENE_IMAGE)
        elif (task_type in H3_RV2V_TASK_TYPES and has_refs
              and source_video is not None and not same_subject):
            extra_rules.append(H3_RV2V_SAME_SUBJECT_RULE)
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
                if ri2v_scene_mode:
                    hints.append("NOTE: <Picture 1> is the SCENE (setting, framing, "
                                 "lighting); subsequent reference images are "
                                 "REPLACEMENT character(s) for that scene.")
                if same_subject and has_refs:
                    if ri2v_scene_mode:
                        hints.append("NOTE: ALL character reference images "
                                     "(<Picture 2> onward) show the SAME person "
                                     "from different angles/poses/body views — "
                                     "analyze as ONE unified subject. <Picture 1> "
                                     "is the SCENE, not a character.")
                    else:
                        hints.append("NOTE: ALL reference images show the SAME person "
                                     "from different angles/poses/body views — analyze "
                                     "as ONE unified subject, merging features (face, "
                                     "body, tattoos, build) into a single definition.")
                elif task_type in H3_RV2V_TASK_TYPES and source_video is not None:
                    hints.append("NOTE: <Video 1> is the SCENE (setting, framing, "
                                 "lighting); each reference image is a separate "
                                 "REPLACEMENT character for that scene.")
                if style_transfer and style_transfer != "off":
                    hints.append("NOTE: identity features will be translated into a "
                                 "non-photorealistic art style — emphasize "
                                 "style-independent identity features (face shape, "
                                 "hair, eye color, distinguishing marks).")
                caller = api_fn if use_local else _call_openrouter
                api_k = "" if use_local else api_key
                analysis = self._analyze_images(
                    caller, api_k, llm_model, image_parts, role_list,
                    "\n".join(hints), temperature, auto_describe_max_tokens,
                    user_prompt=prompt, seed=seed)
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
        elif task_type in H3_RV2V_TASK_TYPES:
            rv2v_subject_rules = (
                H3_RV2V_SUBJECT_RULES_SAME if same_subject
                else H3_RV2V_SUBJECT_RULES_SEPARATE)
            rv2v_template = (H3_RV2V_TEMPLATE_SAME_SUBJECT if same_subject
                             else H3_RV2V_TEMPLATE)
            user_text = rv2v_template.format(
                duration_seconds=f"{duration:.1f}",
                ref_section=ref_section,
                subject_mode_rules=rv2v_subject_rules,
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
            seed=seed,
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
