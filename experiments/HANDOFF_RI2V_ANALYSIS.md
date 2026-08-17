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
