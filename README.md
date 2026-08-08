# ComfyUI-H3O

MiniMax H3 node pack by **Indra's Mirror** — a home for Mal's MiniMax H3
companion nodes.

## Nodes

### H3PromptEnhancer
Vision-LLM prompt enhancement for MiniMax H3. Reformats a raw user request into
the structured prompt format H3 expects:

- **Base format (T2V / I2V / FL2V / L2V):** 3 fields —
  `integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`
- **Full-reference format (R2V):** 6 fields — `subject_definitions`, `summary`,
  `retention_analysis`, `detailed_description`, `overall_soundscape`,
  `non_diegetic_music`

Backends:
- **OpenRouter** (Grok, DeepSeek, Gemini, etc.) — API key from the node input
  or the `OPENROUTER_API_KEY` environment variable
- **Local Ollama (JoyCaption)** — `local_backend = "ollama/joycaption"`

Outputs `h3_prompt` (structured text) and `system_prompt` (for inspection).
Wires straight into the H3 workflow's prompt slot.

### H3AspectRatioDetector
Feed a `LoadImage`/`Image Resize` output and it emits the exact
`aspect_ratio` option string the core **Resolution Selector** node expects,
matched to the image's own dimensions (orientation-aware). Plug the
`aspect_ratio` output into the Resolution Selector socket and width/height
follow automatically.

### H3RegionAttentionMask
Real spatial/temporal regional control for MiniMax H3 DiT self-attention —
the *correct* version of the popular `ComfyUI-MiniMaxH3-AttentionMask` node.

**Important**: that node uses `set_model_attn1_patch`, which is a silent no-op
on MiniMax H3 (H3's `Attention` class never dispatches `attn1_patch` — it calls
`optimized_attention` directly). This node hooks H3 correctly via
`optimized_attention_override` (the same hook `SolAttnPatch` uses) and
geometrically aligns the mask to H3's packed video token layout.

Feed it any `MASK` (SAM2 / SAM3 / RMBG / LoadImage Mask output):

- **preserve_foreground** — damp background QUERY tokens: background attends
  weakly, freeing it to change (background replacement drifts, foreground
  stays clean and locked).
- **suppress_background** — damp background KV columns: no query can read the
  background (hold a character while suppressing the background).

Composes with SolAttnPatch / Spectrum / FirstBlockCache via the standard
`optimized_attention_override` chain. `strength` fades 0..1; `sigma_start` /
`sigma_end` gate by sampling sigma (1.0/0.0 = always on; try 0.8/0.2 for a
warm-up dense schedule).

## Workflow

`video_minimax_h3_r2v_H3Prompt_VSR_Mask.json` — a copy of the R2V VSR workflow
with the node wired into the model chain:

```
UNETLoader → PathchSageAttentionKJ → SolAttnPatch → FirstBlockCache → RefBoost
            → H3RegionAttentionMask → SpectrumApplyMiniMaxH3 → sampler
```

The `Region Mask` LoadImage placeholder expects `region_mask.png` in your
ComfyUI `input/` folder; wire any real mask source (SAM2/SAM3) in its place.

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
