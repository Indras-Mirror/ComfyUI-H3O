# RESUME — Klein Character Sheet Project (continue here)

**Repo:** /media/mal/Crucible/AI-ART/ComfyUI (= /home/mal/AI/ComfyUI bind-mount, branch master @ `14b05228c` = comfy-core 0.30, frontend 1.47.12)
**H3O pack:** /media/mal/Crucible/AI-ART/ComfyUI/custom_nodes/ComfyUI-H3O (branch `ri2v-prompt-upgrade`)
**ComfyUI:** currently DOWN (user stopped it). qwen aggressive server RUNNING on 8099 (22.6GB — stop it before booting ComfyUI; 24GB card can't hold both).

════════ 1. THE CURRENT TASK ════════
Build a **character sheet** for Mal's character aliclo using **Flux-2 Klein**
(text-to-image + native klein-edit reference conditioning). Workflow:
`ComfyUI/user/default/workflows/Character Sheet Klein - Flux2.json`.
Status: **v5 REWIRED, STATICALLY VALIDATED — ready for a GPU test run.**

════════ 2. THE ROOT CAUSE (finally diagnosed) ════════
The v3/v4 "random panels" bug had TWO stacked causes, both now fixed:

1. **The ref conditioning was dead code.** The hashed shim nodes
   (`ComfyUI-KleinRefNodes`, classes `27eacb9f-...` / `93041a64-...`) attached
   the ref latent under conditioning key **`reference_latent` (singular)** —
   but comfy 0.30's consumers read **`reference_latents` (plural)**:
   `model_base.py:1036` (`kwargs.get("reference_latents")` → CONDList
   `ref_latents`) and the Flux2 model forward `ref_latents=` kwarg
   (`comfy/ldm/flux/model.py:344`). The singular key never matched → the model
   never saw the refs. The handoff's "nothing consumes it" was RIGHT but the
   mechanism was: key-name mismatch + no `reference_latents_method` + no
   model patch.
2. **Full-range sigmas destroyed the img2img init.** Flux2Scheduler emits
   timesteps 1→0 (nodes_flux.py:216-224, `generalized_time_snr_shift`,
   mu≈2.29 at 4096 tokens) — no denoise param. VAEEncode init into a full
   schedule = pure-noise start = same as EmptyFlux2LatentImage.

**The official mechanism** (verified against Comfy-Org's klein-9b KV edit
template, fetched to /tmp/wf_image_flux2_klein_9b_kv_image_edit.json): the
real hashed nodes ARE official core nodes (newer comfy than 0.30) that set
`reference_latents` + patch the model. 0.30 ships the same mechanism under
native names:
  `ReferenceLatent` (nodes_edit_model.py:6 — `conditioning_set_values(
  conditioning, {"reference_latents": [latent["samples"]]}, append=True)`) +
  `FluxKontextImageScale` (kontext-resolution ref prep) +
  `FluxKontextMultiReferenceLatentMethod` (`reference_latents_method`,
  options offset/index/uxo/uno/index_timestep_zero) + `FluxKVCache`
  (KV-caches ref tokens; patches `default_ref_method`="index_timestep_zero").
Flux2 arch has `FluxParams.default_ref_method="offset"` + `ref_index_scale`
defaults (flux/model.py:41-42) so no AttributeError risk.
Research (klein best practices): identity comes from the ref CONDITIONING,
NOT img2img init; 9B base wants euler + Flux2Scheduler ~20 steps + guidance
3.5-5 (model default 3.5, no FluxGuidance node needed); CFG 1; refs ~1MP
aspect-preserved, ≤4 chained, same VAE; lowering denoise much below 1.0
loses coherence.

════════ 3. WHAT v5 DOES (119 nodes / 178 links) ════════
- **7 panels:** front, face, left, right, back, **feet (NEW — user request)**,
  seductive. Enhancer `klein_sheet` task now emits **cell_1..cell_7** (10
  outputs; H3PromptEnhancerPlus RETURN_TYPES/NAMES + OUTPUT_IS_LIST + fallback
  tuple + `_parse_multi_cells` all extended 6→7; system prompt
  `H3_KLEIN_SHEET_SYSTEM_PROMPT` = 7 views incl. feet; suite 187/187 green).
- **All 3 refs condition EVERY panel** (user's "all cells full understanding"):
  shared LoadImages 1006/1007/1008 → 3× FluxKontextImageScale → 3× shared
  VAEEncode (1013/1014/1015) → per panel chained ReferenceLatent ×3 →
  FluxKontextMultiReferenceLatentMethod("index_timestep_zero") → CFGGuider
  positive. This is the REAL klein-edit multi-ref instance.
- **FluxKVCache (1022)** on the model path (lora → KV → all guiders) — needed
  with 3 ~1MP refs; matches the official template.
- **Partial denoise init (fallback + structural seed):** per-panel init
  VAEEncode from view-appropriate ref squared to 1024×1024 center-crop
  (ImageScale lanczos) → sampler latent_image; sigmas = Flux2Scheduler
  (20, 1024, 1024) → **SplitSigmasDenoise denoise=0.85** → low → sampler.
  At 0.85 the init ≈ empty-latent behavior (identity is ref-driven) but
  survives as a weak structural seed + fallback if ref conditioning misbehaves.
  **TUNING KNOB:** drag denoise down toward 0.4 for stronger pose lock (some
  coherence cost per research); steps 20; cfg 1.
- CFGGuider negative = ConditioningZeroOut of the panel text cond.

════════ 3b. ⛔ ALL-BLACK OUTPUT BUG — FOUND & FIXED (GPU bisect, 2026-08-20) ════════
The first v5 test produced ALL-BLACK frames (pure zeros) on every panel. GPU
bisect (single-panel API prompts, same seed, static text) nailed it:

- ctrl_E (4 loras via Power Lora Loader): BLACK
- ctrl_F (NO loras): NORMAL          -> the lora stack was the killer
- ctrl_I/J (KLEIN-Unchained alone via standard LoraLoader / rgthree): NORMAL
- ctrl_K (BFS + snofs): NORMAL       <- user's recipe, PROVEN
- ctrl_L (Unchained alone): NORMAL
- ctrl_M (turn2real alone): NORMAL   -> all 4 loras work INDIVIDUALLY

CONCLUSION: the 4-lora merge (Unchained 0.6 + BFS 1.0 + snofs 0.4 + turn2real
1.0) produces zeros/NaN on comfy 0.30 — a COMBINATION bug (additive fp16
merges), not any single lora file. Same 4-lora stack worked at 18:00 on the
previous ComfyUI instance (driver/pack state differs — rgthree pack files
re-touched 17:41 during the 0.33 chaos).

FIX APPLIED: workflow now uses BFS (1.0) + snofs (0.4) only (generator lora
list). 3-lora combos (+Unchained / +turn2real) tested — see ctrl_N/O results
before re-adding a third lora. User's earlier observation that ctrl_* people
"don't look like the reference" is EXPECTED — those were ref-less isolation
tests; the identity test is the full workflow with the native refs.

════════ 3c. H3 SHEET UPGRADED TO 8 CELLS (ri2i_multi8 mode) ════════
User's best sheets come from the Minimax H3 sheet ("Character Sheet RI2I -
Minimax H3.json"). Upgraded 6 -> 8 cells (front, face, left, right, back,
FEET, HANDS, seductive) so the OrbitSheetsContactSheet grid is a perfect 2x4
rectangle (7 cells would leave a black empty box — user's constraint).
- New task type "ri2i_multi8": H3_RI2I_MULTI8_SYSTEM_PROMPT(_ADV) +
  H3_RI2I_MULTI8_TEMPLATE + label + rules-dict entry + combo + prompt
  selection + ref paths. Default ri2i_multi stays 6 cells (existing workflows
  keep alignment; cell order is fixed).
- Plus node now emits cell_1..cell_8 (11 outputs, _parse_multi_cells 8).
- Workflow patched via /tmp/patch_h3_sheet_8cell.py: panels 7-8 cloned from
  panel 6 (#38-43 -> #82-87 feet / #88-93 hands), enhancer task widget ->
  ri2i_multi8, H3BatchImages #54 image6/image7 wired, contact columns 3 -> 4.
  85 nodes / 161 links, validated zero errors, backup
  /tmp/Character_Sheet_H3_6cell_backup.json. Tests 187/187 green.

════════ 3d. KREA 2 (answer: YES, it's an edit model) ════════
comfy 0.30 supports Krea2 (supported_models.py:1887, image_model krea2,
Krea2Tokenizer qwen3vl_4b) and its forward takes ref_latents + supports
index_timestep_zero + sets reference_image_num_tokens (krea2/model.py:276,
329) — SAME native mechanism as klein, SAME wiring (ReferenceLatent chain +
FluxKontextMultiReferenceLatentMethod). NOT downloaded; needs: krea2 model
(~15-25GB), qwen3vl-4b TE, uses Wan21 latent format (wan_2.1_native_vae
already in models/vae). Memory factor 2.2. Viable alternate if klein
identity misses. Wan 2.2 local files are T2V video — wrong tool.

════════ 4. VERIFICATION DONE (no GPU used) ════════
- `/tmp/validate_klein_sheet.py`: node types all known, every link's slot/type
  match, per-node required inputs present, sampler topology checks (sigmas←
  SplitSigmasDenoise low, latent←VAEEncode, guider←CFGGuider, noise←
  RandomNoise, sampler←KSamplerSelect), enhancer 10 outputs + all 7 cells +
  3 ref inputs wired, id bookkeeping. **PASSED, zero errors.**
- ri2v_batch.convert_workflow → 119 API nodes, 7 samplers, 21 ReferenceLatent,
  0 orphans. (NOTE: the H3O helper's old-format conversion DROPS widget
  values — that's a helper limitation; the real frontend maps widgets_values
  positionally, proven by v3's Flux2Scheduler running at 4 steps with the same
  empty-inputs serialization. denoise 0.85 / steps 20 / cfg 1 WILL be applied.)
- H3O suite: 187/187 green. `_parse_multi_cells` 7-cell parse verified.
- Model files verified on disk: flux-2-klein-9b-Q8_0.gguf (models/unet),
  qwen_3_8b_fp8mixed.safetensors (models/text_encoders — CLIPLoader searches
  there), Flux-2-VAE.safetensors (models/vae), 4× klein loras (models/loras).
- Backup of the pre-v5 file: /tmp/Character_Sheet_Klein_v4_backup.json.

════════ 5. NEXT STEPS (GPU test, then tune) ════════
1. Stop qwen on 8099, boot ComfyUI, reload the workflow, queue once.
   Expect: node_errors:{} and 7 panels in output/KleinSheet/<view>_0000N_.png.
2. If panels miss identity: raise denoise→1.0 (pure ref-driven) and/or verify
   reference_latents flows (guidance stays 3.5). If poses too locked: lower
   denoise toward 0.4-0.6.
3. When panels land: commit klein_sheet task (7 cells) + refresh cadence +
   the pack + workflow to ri2v-prompt-upgrade (user's call), bank in Amethyst
   (projects/minimax-h3-node-pack.md).

════════ 6. SESSION HISTORY / GOTCHAS (unchanged) ════════
- ComfyUI 0.30 locked (0.33 broke WAS nodes: sklearn cache + time_shift_slope
  + GPU wedge). NEVER kill ComfyUI mid-CUDA-gen (driver wedge needs reboot).
- qwen (8099, 22.6GB) and ComfyUI can't coexist on the 24GB card.
- ⛔ NSFW HEADLESS ONLY: never open/render/view NSFW; visionproxy/PIL only.
- User's in-UI edits beat disk — reload after scripted edits; bump
  last_node_id/last_link_id when adding nodes.
- Enhancer: local_backend set → API (grok) NOT called — one backend only.
- Generator: /tmp/build_klein_sheet_wf.py (v5; /tmp gets cleared — the JSON
  is the artifact; the generator pulls the enhancer node from
  video_minimax_h3_ri2v_VSR_mask_win.json).

════════ 7. VERIFICATION CHANNELS ════════
- Running ComfyUI object_info: curl -s http://127.0.0.1:8188/object_info
- Workflow→API: H3O experiments/ri2v_batch.py convert_workflow() (widget-drop
  caveat above)
- Vision: scripts/analyze_image.py (stats) + describe_image client
  (score_montages.mjs pattern; OPENROUTER key in that file).
- Outputs: output/KleinSheet/<view>_0000N_.png.
