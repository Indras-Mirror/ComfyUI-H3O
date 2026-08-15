# ComfyUI-H3O

MiniMax H3 node pack by **Indra's Mirror** — a home for Mal's MiniMax H3
companion nodes.

## Nodes

- [**H3PromptEnhancer**](#h3promptenhancer) — vision-LLM prompt enhancement
  into MiniMax H3's structured prompt format
- [**H3AspectRatioDetector**](#h3aspectratiodetector) — auto-match the
  Resolution Selector canvas to a source image's aspect ratio
- [**H3RefBoostV**](#h3refboostv) — attention-level reference V-boost +
  source V-damp (the strong identity lever, norm-cancellation-proof)

---

## H3PromptEnhancer

The community "H3-Context-IR" replacement. It reformats a raw user request into
the exact structured prompt format MiniMax H3 expects, using a vision-capable
LLM that actually **sees** your reference images and source video frames.

### Output format

The output schema switches on the **task_type** widget:

| task_type | Meaning | Output fields |
|-----------|---------|---------------|
| `t2v` | text-to-video, no images | 3 fields |
| `i2v` | image-to-video (first-frame anchor) | 3 fields |
| `fl2v` | first + last frame anchors | 3 fields |
| `l2v` | last-frame only (converge to image) | 3 fields |
| `r2v` | full-reference mode (subject/picture/video/audio labels) | 6 fields |
| `rv2v` | static ref-video = scene, images = replacement characters (legacy two-subject) | 6 fields |
| `ri2v` | **reference images replace a character in a source video, scene/structure preserved** | 6 fields |

**`ri2v` is the recommended character-swap mode.** It reuses the proven `r2v`
6-field template and applies the empirically-correct retention recipe that made
`video_minimax_h3_r2v_H3Prompt_VSR_Mask.json` work: the replacement character is
`fully_preserved`, each `<Picture N>` face/hair source is `weak_reference`, the
source `<Video N>` is `partially_preserved` (scene/structure kept, replaced
character fully replaced), and the original character is **never named as its
own subject**. This is the opposite of `rv2v`'s two-subject recipe
(original=`weak_reference`, replacement=`attribute_transfer`), which renders the
original unchanged.

**3-field (base)** format:

1. `integrated_multimodal_description` — `[Shot 1]` style + shot type + scene
   description with camera moves, subject actions, dialogue, and diegetic audio
   woven chronologically along the timeline. Cuts are marked `[Shot N] At
   MM:SS.mmm`. Dialogue uses `<d>[Language] text</d>`.
2. `overall_soundscape` — 1-4 sentences of ambient sound, physical action
   sounds, and diegetic audio.
3. `non_diegetic_music` — background music: genre, tempo, instrumentation,
   mood, where it starts/ends.

**6-field (R2V reference)** format:

1. `subject_definitions` — one line per label: `<Subject N>` (reusable visible
   content), `<Picture N>` (frame anchor / storyboard), `<Video N>` (whole-video
   structural source: editing, continuation, camera, cuts, rhythm), `<Audio N>`
   (voice timbre, BGM, SFX).
2. `summary` — one paragraph tying the target video to the references.
3. `retention_analysis` — what is preserved / transferred / referenced from
   each reference item.
4. `detailed_description` — the main body, 350-500 words, reference labels at
   first appearance.
5. `overall_soundscape` — as above.
6. `non_diegetic_music` — as above.

### How it works (two passes)

**Pass 1 — auto-describe (identity anchoring).** When `auto_describe` is on and
any image is attached, the node first calls the LLM in *analysis* mode with a
strict JSON schema. Every attached image (source frames, references, first/last
frames) is labeled (`image0`, `image1`, …) and described as:

- `subject` — identity, face shape, eyes, hair, body type, full clothing
  inventory, accessories, distinguishing features (tattoos, scars, piercings)
- `scene` — location, background, props, composition, lighting, camera angle
- `current_state` — pose, expression, action, gaze direction

That analysis is injected into the write pass inside a
`--- Context Analysis ---` block, so the final prompt anchors on **explicit
identity features** instead of the writing LLM re-guessing from a fresh look.
This is the main lever for strong R2V subject consistency and I2V replacement
characters. Costs one extra API call per run; budget it with
`auto_describe_max_tokens` (4096+ for multi-image scenes, 2048 for a single
simple image).

**Pass 2 — the write.** The node assembles the final user message:

- **Vision content**: source image (`<Picture 1>` anchor for I2V/FL2V/L2V),
  last-frame image (`<Picture 2>` for FL2V), source video sampled to ≤6 frames
  across its timeline (shown as `<Video 1>` — reserved by the official guide
  for whole-video relationships: editing, continuation, camera/cut structure),
  and up to 3 reference images (R2V subjects/pictures, or I2V replacement
  `<Subject N>` definitions).
- **Task framing**: the H3 task label plus task-specific rules injected per
  type — e.g. i2v *must* open with the "at 0.00 seconds, &lt;Picture 1&gt; is
  fully referenced" line; r2v sets the 350-500 word detailed_description rule.
- **Directives**: optional `same_subject` (all references are one person →
  merge identity across images) and `style_transfer` (render the reference
  subject in `anime` / `3d_render` / `cartoon` / `match_source` style while
  keeping identity-defining features).
- **Your raw prompt** rides at the end as the user notes.

That whole message (images + text) goes to the LLM with the selected system
prompt, and the structured result is returned on the `h3_prompt` output — wire
it straight into the H3 node's prompt input. `system_prompt` is the second
output, for inspection/debugging.

### Model / backend

- **OpenRouter (default)** — `model` dropdown (Grok, DeepSeek, Gemini, GPT-4o)
  or a free-form `custom_model` ID. API key from the `api_key` input or the
  `OPENROUTER_API_KEY` environment variable. **Never stored in the repo.**
- **Local Ollama (JoyCaption Beta One, 7.5GB)** — `local_backend =
  "ollama/joycaption"`. Requires Ollama running (`systemctl start ollama`).
  The node auto-appends NSFW permission + JoyCaption prose-style prompts for
  local models, and unloads the model from VRAM after the call.

### Other controls

| Widget | Default | What it does |
|--------|---------|--------------|
| `temperature` | 0.7 | Lower = predictable format, higher = creative detail |
| `max_tokens` | 4096 | R2V prompts are long (6 fields) — keep ≥4096 |
| `advanced_prompt` | on | Community H3-Context-IR refinements: camera-motion vocabulary, stable speaker IDs, voiceover phrasing, audio-binding rules |
| `custom_system_prompt` | "" | Full override of the built-in system prompt |
| `duration` | 5.0s | Drives shot pacing / cut timing in the template |
| `seed` | -1 | -1 = fresh random each run (new prompt per queue), 0+ = fixed & reproducible. Sent to the LLM backend alongside temperature. |

### Quick example

```
User prompt: "a man jumps off a cliff into a lake, slow motion"
task_type: t2v, duration: 5.0

→ h3_prompt:
integrated_multimodal_description:
[Shot 1] Cinematic, live-action, wide shot of a cliff top at golden hour...
camera: slow push-in, 0.2x amplitude...
[Shot 2] At 00:03.500, the camera cuts to a low angle...
  ...
overall_soundscape: ...
non_diegetic_music: ...
```

---

## H3AspectRatioDetector

Feed any `IMAGE` (a `LoadImage` or image-resize output) and it emits the exact
`aspect_ratio` option string the core **Resolution Selector** node expects —
picked to match the image's own dimensions, orientation-aware (a 3:4 portrait
selects "3:4 (Portrait Standard)", a 16:9 landscape selects "16:9
(Widescreen)"). Plug the `aspect_ratio` output into the Resolution Selector
socket and the canvas width/height follow automatically — no manual selection.

Outputs: `aspect_ratio` (COMBO string), `width`, `height`, `ratio` (float w/h).
The optional `aspect_ratio` string input is a manual override, e.g.
`"21:9 (Ultrawide)"`.

---

## H3RefBoostV

The **attention-level** identity lever. It scales the **VALUE rows** inside
self-attention — the one place H3's normalization stack can't cancel a scale.

### Why not just `ref_scale`?

Latent-level amplitude scaling (`H3RefBoostChar.ref_scale` / `.source_scale`)
multiplies the conditioning latent, which then passes through
`video_patch_proj` (a Linear **with bias**) and the DiT's RMSNorm. RMSNorm is
scale-invariant, so only a subtle "bias-mediated" residual survives — that's
why amplitude dials are weak. Bernini RefBoost V2/V3 moved off amplitude
scaling for exactly this reason.

The value projection is different: `out_i = Σ_j attn_ij · v_j`. Scaling the V
rows of a reference makes it proportionally louder in every token update, and
no LayerNorm/RMSNorm on Q or K can undo it (Q and K are RMSNorm'd; V is not).

### What it does

- `ref_v_boost` — multiply the V rows of the **selected character image refs**
  up (default 1.3). 1.15-1.3 = noticeable, 1.4-1.6 = strong, >1.8 risks
  oversaturation.
- `source_v_scale` — multiply the V rows of the **source ref-video** down
  (default 0.8). Weakens the source person so the replacement wins. 0.7-0.9
  typical.
- `ref_slots` — which image refs are the character ("0", "0,1", "all").
- `schedule` / `step_threshold` — sigma ramp; same semantics as `H3RefBoostChar`.

Both scales ramp from neutral (1.0) at high sigma to their target at low sigma,
so the structure pass is untouched and the identity pass does the work.

### How it works (no core edit)

A `DIFFUSION_MODEL` wrapper resolves each ref block's absolute row range in the
packed sequence once per run (from `PackedLayout.segments`), then each step
writes `{ranges + ramped scales}` into `transformer_options`.
`comfy.ldm.minimax.model.Attention.forward` is replaced once at `apply()` time
with a **class-level** patch (runtime monkeypatch, not a core file edit); when
the key is absent it delegates to the original forward untouched, so every other
H3 workflow is unaffected.

### Sol-Attn compatibility (fixed 2026-08-14)

An earlier instance-level patch was silently bypassed by Sol-Attn: its
`_compose_module_patch` wraps *instance* `attn.forward` attrs and, for eligible
calls, runs `type(module).forward` — the original — so V-scaling never executed
(renders looked identical to no boost). The class-level patch has no instance
attr, so Sol-Attn's compose hook skips it: our V-scaling runs every call and
the scaled q/k/v still reach `optimized_attention`, where Sol-Attn's override
dispatches eligible calls to its kernel. Both compose. Patch-Sage composition
untested.

### Placement

Place it **before Spectrum** in the chain (both are DIFFUSION_MODEL wrappers;
Spectrum must observe the post-expansion layout).

### Status

Unit-tested (24/24) against the real `PackedLayout` and the real `Attention`
module (with stubbed GPU kernels), including Sol-Attn-bypass proof.
**Render-test of the class-level patch pending** (the pre-fix renders did not
exercise the V-scaling at all).

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Indras-Mirror/ComfyUI-H3O.git
```

Restart ComfyUI.

## API Keys

**No API keys are stored in this repository.** Keys are passed per-node via the
`api_key` input, or read from the `OPENROUTER_API_KEY` environment variable.

## License

See [LICENSE](LICENSE).
