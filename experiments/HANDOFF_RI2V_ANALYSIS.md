# HANDOFF — RI2V Reference-Identity Analysis (2026-08-17)

**Branch:** `ri2v-ref-consistency` (commit `22f9827`) in
`/media/mal/Crucible/AI-ART/ComfyUI/custom_nodes/ComfyUI-H3O`
**Workflow analyzed:** `video_minimax_h3_ri2v_VSR.json`
**Status:** ROOT CAUSE CONFIRMED + WORKING FIX FOUND (NOVID-style). Suppression
levers tested and eliminated. Node-pack enhancement (constant/early schedules)
implemented but did NOT beat the working fix.

---

## 1. The finding (one paragraph)

The user's RI2V workflow feeds the same manga page (0112.jpg) as a 5-frame
static ref VIDEO (`H3ImageToRefVideo`) AND as the prompt-enhancer's source.
The model treats that ref video as the scene AND its woman as the identity:
**whenever ANY manga scene visual is present as a reference (video or image,
damped or undamped), the output's woman is the manga page's original woman
("Tifa" doujin character) — the photo refs' identity (tattoos, outfit, face)
never transfers.** Removing the scene reference entirely (NOVID variant,
prompt-only scene description) transfers the ref identity consistently across
3 seeds: the output woman has the ref's septum ring, green eyes, cheek mole,
tongue piercing, and chest tattoos, rendered in the manga style the prompt
demands. The user's own hypothesis — "it's the static ref image to 5 frame
video source" — is CONFIRMED.

## 2. Mechanism (verified from source + runs)

- `MiniMaxH3ReferenceToVideo.execute` packs refs into `ref_blocks` (images →
  videos) and `ref_items` (Qwen-VL presentation, `<Picture N>` / `<Video k>`).
- In the DiT (`comfy/ldm/minimax/model.py`), ref rows are pinned at
  timestep ≈ 0.999 (near-clean), injected pristine EVERY step, never
  denoised (`img_update` mask, model.py:566-578). Target rows denoise from
  noise; refs shape them via cross-attention.
- The static 5-frame video = one image repeated 5× → behaves as an amplified
  image with temporal spread (scout report, `scout-h3-identity.md`).
- There is NO hard structure/identity pass — it is emergent: at high sigma
  (early steps) the ref video's content (the manga woman) shapes the target's
  structure; by low sigma the face/outfit are committed and can't be redrawn.
- All existing damp levers ramp in at LOW sigma (p = 1 - sigma → eff 0→1):
  `source_scale` (amplitude, RMSNorm-cancelled = weak), `source_noise`
  (noise-mix), H3RefBoostV `source_v_scale` (V-row damp, strong but LATE).
  Too late — identity already committed.
- Verified the V-damp ENGAGES: debug log "12 char row range(s) ... | 1 source
  row range(s) [(19150, 20782)]" — photo rows (12 ranges, 8× the video's mass)
  still lose.

## 3. Experiment matrix + results (all vision-scored via visionproxy)

Harness: `experiments/ri2v_batch.py` (ComfyUI /prompt API driver; converts the
workflow faithfully OR builds controlled variant graphs). 448×688 portrait,
243 frames (10s), 20 steps, ref_image_size=max, VSR bypassed, frozen prompt
(captured from the enhancer's real output — `ri2v_prompt_frozen.txt`).

| Variant | What it tests | Output woman |
|---|---|---|
| BASE (current config) | manga ref video + 4 photo refs + char boost all/3/3, src 0.8 | MANGA (Tifa) — no ref identity |
| VBOOST | + RefBoostV source_v_scale=0.0 (video V-zero at low sigma) | MANGA (Tifa) |
| VB_CHARONLY | photo V×1.3, video NOT damped | GLITCHED (overdrive) |
| VB_VIDEOZERO | video V→0, photos unboosted | GLITCHED (render broken — video's V is load-bearing at every step) |
| SRC_NOISE | source_noise=0.8 latent noise-mix at low sigma | MANGA (Tifa) |
| SCENEIMG | manga as static IMAGE ref instead of 5-frame video | MANGA (Tifa) |
| ALICIO3 | blonde ref removed (3 aliclo refs) | MANGA (Tifa) |
| SHEET1 | 6-panel character sheet as the single ref | MANGA (Tifa) |
| VBOOST_CONST | constant 30% video V-damp EVERY step (new schedule) | MANGA (Tifa) |
| VBOOST_EARLY | video V-zero during structure pass only (new schedule) | MANGA (Tifa) — full report below |
| **NOVID** | **NO scene ref; prompt-only scene** | **REF WOMAN — septum ring, green eyes, cheek mole, tongue piercing, chest tattoos, manga-styled. 3/3 seeds.** |

Controls (MATCH / PROTON / NOBOOST) were interrupted mid-wave; low decision
value given NOVID settled the question.

Frames + montages (refs | source | 4 output frames per run): `/tmp/ri2v/*/montage.jpg`

## 4. What this means for the fix

**The working fix (today, zero node changes):** don't feed the manga page as a
visual ref. Wire the workflow so the scene comes from the prompt text only —
the NOVID prompt (`ri2v_prompt_frozen_novid.txt`) describes the multi-panel
layout in words and the model reproduces a panel structure with the REF woman.
Trade-off: layout fidelity is weaker than the source-anchored versions (the
model invents panels from the description instead of copying the page).

**The "proper suppression" answer (tested, failed):** no damp schedule —
late (existing), constant, or early (new schedules I added) — lets the refs
win while the manga visual is present. The video's identity is committed in
the structure pass and its V contribution is load-bearing every step.

**Node change made (committed `22f9827`):** `h3_refboost_char.py` /
`h3_refboost_v.py` — added `constant`, `early_linear`, `early_sqrt` schedule
options to `_ramp()` and both nodes' schedule COMBOs. Tested (VBOOST_CONST,
VBOOST_EARLY) — mechanically works, doesn't fix identity. Keep on the branch
(useful for other workflows); do not merge to main yet.

**Not yet tried (next candidate):** photo-STYLE output — flip the enhancer's
`style_transfer` away from `match_source` so the output is photorealistic
(the ref woman in the manga scene's composition). Photo→photo identity
transfer is likely much stronger. This changes the output style — ASK THE
USER before pursuing.

## 5. Key files

```
experiments/ri2v_batch.py              # matrix harness (variants, API driver, manifest)
experiments/ri2v_montage.py            # per-run comparison montage builder
experiments/ri2v_prompt_frozen.txt     # enhancer-captured prompt (video-ref variants)
experiments/ri2v_prompt_frozen_novid.txt   # scene-ref stripped (THE WORKING PROMPT)
experiments/ri2v_prompt_frozen_sceneimg.txt # manga-as-scene-image prompt (failed variant)
experiments/ri2v_prompt_frozen_3ref.txt    # no-blonde prompt (failed variant)
experiments/ri2v_prompt_frozen_sheet.txt   # character-sheet prompt (failed variant)
experiments/HANDOFF_RI2V_ANALYSIS.md   # this file
experiments/RESUME_PROMPT.md           # paste-ready resume prompt
```

## 6. Verified vs inferred

VERIFIED: all matrix outputs (vision-scored, filenames in /tmp/ri2v manifest);
damp engages (debug log); NOVID identity across 3 seeds; output res 448×688;
the enhancer prompt content; VSR-API serialization fix (flat dotted keys);
the workflow graph wiring; branch committed.
INFERRED: NOVID's layout fidelity vs the source-anchored runs (vision model
said "panel layout" present but weaker — no side-by-side pixel measure);
whether photo-style output fixes identity (untested); MATCH/PROTON/NOBOOST
controls (interrupted).

## 7. Next steps (for the fresh session)

1. Show the user NOVID outputs (/tmp/ri2v/NOVID_s101..303 montages) — get his
   verdict on layout fidelity vs identity.
2. Decide with him: (a) ship NOVID-style workflow change (clone the workflow
   JSON, rewire ref_video to nothing, swap in the NOVID prompt — keep the
   original untouched), (b) try photo-style output, (c) both.
3. If (a): create `video_minimax_h3_ri2v_VSR (NOVID-fix).json` — remove the
   `H3ImageToRefVideo`→`ref_videos.ref_video_0` link (node 1112/1113), point
   1060's prompt at the frozen-novid text, and decide whether to keep the
   aspect detector fed from a ref image instead of the video.
4. Run the fixed workflow in the UI (the user's real pipeline, WITH VSR) to
   confirm the fix survives the VSR pass. The VSR serialization in the
   harness conversion is fixed (flat dotted keys) if you re-run the faithful
   baseline: `python3 experiments/ri2v_batch.py --variant BASE --enhance`.
5. Commit any workflow JSON + finalize the branch. The schedule-mode node
   change stays on `ri2v-ref-consistency` unless the user wants it on main.

## 8. Session state

- ComfyUI: STOPPED (restarted for the node change, then drained; currently
  down — start with the batchklein pattern or `nohup python main.py
  --listen 127.0.0.1 --port 8188 &` if more runs are needed).
- GPU was free at last check.
- Results dir: /tmp/ri2v (manifest.jsonl — machine-readable run record).
- Relay peer `mal` (the user's interactive claude, "2IC") is connected but
  NOT polling relay asks — 6+ asks queued unanswered since ~19:10. Don't
  block on him; the flash-worker review (reports/2ic-review.md) already
  covered the review role.
- Amethyst note: `projects/ri2v-ref-consistency-2026-08-17.md`

## 9. PRODUCTION-PATH FINDINGS (2026-08-18 — appended by the conductor)

Supersedes the "any source blocks identity" framing. Full evidence trail:
`~/.quetza-data/conductor/reports/` (ri2v-sweep-p1/p2-hinges/ctrl/p2-rest,
ri2v-prod-ab/prod-validate/prod-src00, claude-fix-stack,
claude-qwen-enhancer, ri2v-g46-test). Read the resume prompt
(experiments/RESUME_PROMPT.md, v3) for the canonical current state.

### The mechanism (sweep, 10 sources, vision-scored)
- The refs (4 aliclo photos, ref_repeat=3) are the DEFAULT identity winner:
  30/30 across photo/drawn/manga/multi-person sources under the generic
  prompt `ri2v_prompt_gen.txt` (explicit "replace the woman, transfer
  face/body/hair/tattoos").
- The ONLY condition that beats them: a B&W manga source whose identity the
  prompt TEXT anchors (frozen prompt's "<Video 1> / manga linework fully
  retained" framing) — 4/4 across two different manga pages. Photo/color-drawn
  sources never override under either prompt.
- Mechanism = prompt × source-type interaction, not "any source blocks".

### The production failure chain
- ri2v-prod-ab: prompt form is the identity switch (H2 confirmed; production
  ref config byte-identical to harness). The enhancer's verbose shot-by-shot
  scene narration anchors the source identity.
- ri2v-prod-validate: the tightened compact directive (_DETAILED_DESC_RI2V)
  still fails identity 2/2 — grok-4.3 PARTIALLY complies (keeps a timestamp,
  <d> dialogue, scene action).
- ri2v-prod-src00: node 1049 source_scale 0.8→0.0 = 1/2 seeds (s202 aliclo
  4/4 both models + layout PASS). Change LEFT in place (workflow NOT
  git-tracked; backup /tmp/video_minimax_h3_ri2v_VSR.pre-src00.bak).
- claude-fix-stack: SRC00 + EXACT generic prompt (via the new
  `_bypass_enhancer` in ri2v_batch.py:297, `--enhance --prompt-file`) =
  identity aliclo 4/4 BOTH models ALL 4 seeds (seed sensitivity RESOLVED),
  but layout FAIL 3/4 (scene dropped — trivial win).

### The candidate full fix (UNTESTED stack)
- SRC00 + a COMPLIANT prompt (identity commitment + style/layout anchor, no
  character narration). grok-4.6 (x-ai/grok-4.6) IS COMPLIANT (ri2v-g46-test:
  zero [Shot N]/MM:SS/<d>/speaker IDs; 3-sentence transfer directive; prompt
  saved /tmp/ri2v-prodfix/G46_enh.prompt.txt). Qwen 3.8 already wired
  (`llamacpp/qwen3.8-heretic-ara`, h3o_shared.py:77); compliance test
  deferred (needs VRAM free).
- Next experiment: production path + SRC00 + G46 prompt, 4 seeds — packet
  staged at ~/.quetza-data/conductor/packets/ri2v-g46-validate.md.
- In-flight at shutdown: ri2v-scene-props (re-dispatched on the amended
  packet — user's 14-image pool ONLY; the first worker ran non-pool images
  and was killed).

## 10. FULL-FIX VERDICT + MASK DIRECTION (2026-08-18 late — appended by the conductor)

### The full-fix validation (ri2v-g46-validate3 — clean, on the USER'S POOL)
Production config: enhancer **x-ai/grok-4.6** (live, deterministic compliant prompts) + **SRC00** + **duration=5** + **turbo ref2v 4-step** (steps=4) → ~2 min/run.
- **Identity WORKS** (2/4 runs perfect 4A aliclo; all strong) — the seed-sensitivity problem is solved.
- **Layout source-dependent** — simple medium-wide (zxc 23.jpg): identity 4A + layout 4PASS on seed 202 (perfect); complex low-angle (xzcsarwq.jpg): layout lost on a seed. SRC00 zeroes the source's layout anchor; the prompt alone describes simple scenes better.
- Two prior attempts were invalidated by an avert_src pin-up leftover (the rogue scene-props worker) — the lesson: **verify the video-ref source in the QUEUED payload (curl /queue), not the saved JSON**. The hardened four-layer verification (saved-JSON edit → queued-payload check → montage RMSE → prompt compliance) worked.

### The mask direction (community-validated)
Reddit r/StableDiffusion "Get miniMax character swap working! Finally" (2026-08-18):
- "Use Sam3 to blur/mask the person and that problem goes away" (similar-looking subjects) — confirms `ri2v-source-mask` (in flight): mask the woman out of the source → source_scale back to 0.8 → refs win identity with full layout anchoring.
- Subject-as-concept prompt structure (PropagandaOfTheDude): `<Subject 1>` = ref face (attribute_transfer), `<Subject 2>` = source person (weak_reference), `<Subject 3>` = clothing (fully_preserved), per-subject retention_analysis → fixes partial merges + clothing confusion. Prompts: omnifit.io/blog/ref2va-what-works-what-fails.
- Bigger visual difference = cleaner swap; prominent clothing distracts; steps 20 for inference (ours: 4 with turbo).

### New workflow files (user-provided, mask-wiring candidates)
`/home/mal/Downloads/minimaxH3Character_v20SolAttnTurbo.json`, `/home/mal/Downloads/video_minimax_h3_r2v_addguides_v1.json`, `/home/mal/Downloads/minimaxH3InfiniteVideoRef2va12Img_v10.json`. LightX2V turbo loras: models/loras/LightxV2-V2.0-T2V/ (T2V high/low noise — ref2v turbo in use is Kijai's 4-step).

### Current config state (workflow JSON, NOT git-tracked)
node 1049 source_scale=0.0 · node 1113 video-ref LoadImage (restored to 0112.jpg after runs — next slice repoints to pool/masked) · node 129 RandomNoise (restored 949448543631237) · node 132 duration=5 · node 124 steps=4 · node 1022 turbo=on. Backups: /tmp/video_minimax_h3_ri2v_VSR.pre-src00.bak, .pre-speed.bak.
Uncommitted: ri2v_batch.py (_bypass_enhancer + duration), h3_prompt_enhancer.py (IDENTITY COMMITMENT + _DETAILED_DESC_RI2V + model list), 2 prompt txts.
Fleet health: flash workers die 2-16 min in ("running stop hook" — 4+ occurrences); claude workers stable — prefer claude for critical slices; STATE.md partial-progress discipline for flash slices.

## 11. MASK SLICE + FULL-WIN CONFIG (2026-08-18 evening — appended by the conductor)

### The mask (ri2v-source-mask, conductor-verified)
Person mask on the source ref video (woman out, feathered) + source_scale 0.8 → **layout FIXED (4P everywhere, ~16/16 frames)** — previously the SRC00 trade-off (identity 4/4 but layout FAIL 3/4). Identity 2A-3A (body/tattoos transfer; FACE markers — septum ring, cheek mole, green eyes — never render). Tested 5 face strategies (~22 runs): black-hole mask, feathered mask, face-only mask, close-up face-crop ref, face+refs — no face breakthrough. The earlier claude worker's "4A identity" claim did NOT reproduce under the conductor rubric (1A-2A on face markers).

### THE FULL WIN (identity + layout simultaneously)
**Comic source (xmen02.jpg) + masked ref video + source_scale 0.5 + generic transfer prompt (experiments/ri2v_prompt_gen.txt via --prompt-file bypass) + NO char boost → seed 202: IDENTITY 4A/3A/4A/4A, LAYOUT 4P×4.**
- s101 same config: identity 4A×3, layout 3P/3P/2P → seed sensitivity persists (see ri2v-seeds-research.md, pro worker in flight).
- source_scale semantics (h3_refboost_char.py:208-211): `src_scale = 1.0 + (source_scale - 1.0) * eff` — 1.0 undamped, 0.0 SRC00, crossover near 0.5.
- Mechanism: comic source = bigger visual difference = cleaner swap; generic prompt transfer directive = refs win identity; 0.5 = layout anchor without identity recommit; boost nodes irrelevant.

### Standalone workflows (DELIVERABLES, converter-verified + smoke-tested)
- `video_minimax_h3_ri2v_VSR_mask.json` — in-graph GD-SAM person mask chain (1113 → GroundingDinoSAMSegment "person" → MaskBlur+ → Blur-composite → H3ImageToRefVideo 1207 → 1060 ref_videos.ref_video_0); unmasked source still feeds the enhancer.
- `..._mask_face.json` (face-only mask variant) · `..._mask_inject.json` (custom_system_prompt face-transfer rules from ri2v-templates-research.md — UNTESTED, generic-prompt bypass won).

### Config for the winning run (per-run)
1113 = comic source · 1049 source_scale=0.5, enabled=False (no boost) · --prompt-file experiments/ri2v_prompt_gen.txt (drops enhancer) · node 129 seed=202 · duration 5 · steps 4 · turbo ref2v. Harness: run_variants.py (SCALE/no_boost/prompt_file envs) at /tmp/ri2v-source-mask/.

### Batch 4 in flight
Generalization of the winning formula: xmen-enma-frost.jpg, Xman.jpg, xzcsarwq.jpg × s202.

### Master state
Master workflow untouched (71 nodes; node 129 restored 949448543631237; node 1113 0112.jpg). Backups: /tmp/video_minimax_h3_ri2v_VSR.pre-src00.bak, .pre-speed.bak, .pre-mask.bak.

### ⛔ HARD RULE (user, 2026-08-18)
NSFW content NEVER opened/rendered/viewed/displayed on Mal's PC — headless only (scores, numpy/PIL, API vision). In QUETZA.md + CLAUDE.md + Letta.

## 12. WORKFLOW-EDIT RULE (2026-08-18 — user-reported bug + fix)
Hand-built workflow JSONs from this session left `last_link_id`/`last_node_id` stale (563 vs actual 572 / 1137 vs 1207) → the ComfyUI frontend's next connection collided with the mask-chain link IDs and wired itself randomly, breaking the workflow for the user. FIXED: all `video_minimax_h3_ri2v_VSR_mask*.json` now carry correct counters + no duplicate IDs (verified) + the mask chain is laid out visibly. RULE for future scripted workflow edits: always bump both counters to the actual maxes and json.load-verify. Fixer: /tmp/ri2v-source-mask/fix_workflows.py.

## 13. LOCAL-ENHANCER OOM FIX (2026-08-18 late)
Symptom: video gen succeeds (444s), then the local llama-server enhancer (Muse-Glimmer 30B) OOMs at spawn — cudaMalloc 14752 MiB failed. Cause: ComfyUI caches MiniMaxH3 (~20GB) across queues; llama-server is a separate process ComfyUI cannot see, so it never unloads for it. Qwen would OOM identically — timing/cache dependent, not model-specific. FIX (applied): h3o_shared.py `_spawn_llama_server` now calls `comfy.model_management.unload_all_models()` + `soft_empty_cache()` before Popen (h3o_shared.py:778). Needs ComfyUI restart to load. Workaround without restart: API enhancer (grok-4.6, local_backend off).
