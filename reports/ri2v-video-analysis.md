# RI2V video analysis — Nerdy Rodent "quicker MiniMax H3 video generations" (t=344s)

Source: `/tmp/ri2v-video/transcript.txt`. Full cleaned notes: `reports/ri2v-video-transcript.md`. No re-fetch; analysis only.

## Technique summary (in video order)
1. **Turbo LoRAs** — 4- and 8-step versions for first/last/reference models (Kijai's collection, Dr. Bath's Larry VRH conversions; author prefers Light X2V). Image clarity holds; audio quality can drop — fine with own audio. Ideal for quick prototype runs.
2. **"e r s d e k" sampler** (likely euler_d) — slightly better audio gen with/without LoRA.
3. **Sage attention** — old PathchSageAttentionKJ vs newer H3 memory-efficient Sage patch; Windows fallback: Model Attention Backend + comfy-kitchen attention (speed without installs).
4. **MiniMaxH3SigmaShift** — shift-value control (video 12 / audio 4) like image-gen workflows.
5. **Low VRAM attention + chunk feed-forward** nodes — OOM relief for long/high-res video.
6. **Custom keyframes** — H3 motion context multi-ref: 2-5+ images at *any* position (not just first/last), position values edited per frame; each keyframe needs its own conditioning set; a story prompt joins them.
7. **Lip-sync** — reference model + audio + prompt, media in triangular brackets; VHS_VideoCombine pipes raw audio back for crispness.
8. **Video extend** — ~15s limit per gen; extension prompts (+5s/+7s, repeatable) via first+last model; **video masked motion context context_length=39** ("typically the best choice in all cases"); context frames trimmed after each segment; KJ image-batch-extend-with-overlap for smooth blends; H3 assemble-existing-video for concatenated audio. Seams "pretty smooth".
9. **Global sampler/step inputs** — flip 20 ↔ 6-8 steps in one place when switching turbo/non-turbo.
10. **Manual/automatic masks on reference image loading** — exclude unwanted regions ("MiniMax will often include the clouds because they look like part of her head — if you don't want the clouds, simply don't send them in"); segments clothing/bg/fg.

## The [5:44] section (t=344s) — quoted
"[5:38] >> Oh. [5:39] >> Oh. A rodent. [5:50] >> Not bad at all, eh? Lots of custom keyframe fun."
The moment is the tail of the **3-frame custom keyframe demo** — images at beginning/middle/end joined by a "woman walking along, finding the rodent, picking him up" story prompt (the keyframe setup: [4:36]-[5:28]; demo playback [5:38]-[5:52]; 5:52 onward pivots to lip-sync via the reference model).

## Recommendations for the H3 RI2V workflow
**(a) Testing speed**
- Turbo LoRA variant: harness hardcodes `steps: 20` (ri2v_batch.py:113) with no LoRA; wire a LoRA loader + `steps` cfg key (20→4/8) for the planned ref2v 4-step turbo run. Flip point is `BasicScheduler` `steps`/`scheduler` in `_sampler` (ri2v_batch.py:107).
- 5s videos: `snap_length` already handles `duration` (ri2v_batch.py:67,158) — `duration: 5.0` → ~124 frames @24fps. No code change.
- Sampler A/B: current chain uses `res_multistep` (ri2v_batch.py:115); the video's euler_d claim targets audio — only relevant if audio is ever scored.
- Sage already in chain (PathchSageAttentionKJ); cheap A/B with the newer H3 memory-efficient Sage patch node.

**(b) Identity-transfer quality**
- **Custom keyframes are the top borrow**: harness feeds 4 static refs + a 5-frame static ref video (`H3ImageToRefVideo`); instead place keyframes at real temporal positions (e.g. frames 0/62/124 of the 5s video) with per-frame conditioning — anchors identity mid-video where the static ref video can drift. Video's rule: each keyframe gets its own conditioning set.
- **Masked refs**: the cloud-head example maps directly to our aliclo refs / 0112.jpg source — background bleed is a known identity confound. Add a mask/crop path (LoadImage mask input → masked crop before `H3ImageToRefVideo`/`ref_images`) to strip background/clothing from refs.
- Story-prompt-joining-frames matches our frozen prompt files; triangular-bracket media embedding is already how refs are passed.

**(c) Not applicable + why**
- Video-extend pipeline (context_length 39, H3 assemble-existing-video, KJ overlap): harness is single-segment; only a follow-up if 15s+ chained runs are planned.
- Lip-sync/raw-audio pipe: our scoring is visual-identity only.
- SigmaShift: already applied (shift_video 12, shift_audio 4).
- Low VRAM attention/chunk feed-forward: no OOM observed; revisit only if resolution/length grows.

**Differences vs ours:** video runs 6-8 steps + turbo LoRAs + masked refs; harness runs 20 steps, no LoRA, unmasked full refs (`ref_image_size: max`), single-shot gen.

**Verdict:** the most valuable borrowable technique is masked reference loading — the cheapest lever on identity-transfer quality, since background bleed in the aliclo refs is a known confound and it needs no model/code changes to test (LoadImage mask input).
