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
