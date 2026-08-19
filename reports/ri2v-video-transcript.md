# RI2V video transcript analysis — "MiniMax H3 Turbo + Custom Frames + Video Extend in ComfyUI" (Nerdy Rodent, 13:56, jkMf7IAcm7c)

Full cleaned transcript: `/tmp/ri2v-video/transcript.txt`.

## Techniques in the video
1. **Turbo LoRAs** (4- and 8-step) for first/last/reference models (Kijai's collection, Dr. Bath's Larry VRH conversions; author prefers Light X2V). Image clarity holds, audio quality can suffer — fine when you pipe your own audio. Ideal for quick prototype runs.
2. **"e r s d e k" sampler** (auto-caption; likely euler_d) — slightly better audio generation with a LoRA.
3. **Sage attention** (old PathchSageAttentionKJ vs newer H3 memory-efficient Sage patch); Windows fallback: Model Attention Backend with comfy-kitchen attention — speed without installs.
4. **MiniMaxH3SigmaShift** — shift-value control (video 12 / audio 4) like image-gen workflows.
5. **Low VRAM attention + chunk feed forward** nodes — OOM relief for long/high-res video.
6. **Custom keyframes**: H3 motion context multi-ref — 2-5+ images at *any* position (not just first/last), position values edited per frame; each keyframe needs its own conditioning set; a story prompt joins them.
7. **Lip-sync**: reference model + audio + prompt, media embedded in triangular brackets; VHS_VideoCombine pipes raw audio back for crispness.
8. **Video extend**: ~15s limit per gen; extension prompts (+5s, +7s, repeatable) through first+last model; **video masked motion context context_length=39** ("best choice in all cases"); context frames trimmed after each segment; KJ image-batch-extend-with-overlap for smooth blends; H3 assemble-existing-video for concatenated audio. Seams "pretty smooth" → 30-40s consistency.
9. **Global sampler/step inputs** across multiple KSamplers — flip 20 ↔ 6-8 steps in one place when switching turbo/non-turbo.
10. **Manual/automatic masks on reference image loading** — exclude unwanted regions ("MiniMax often includes the clouds because they look like part of her head — if you don't want the clouds, simply don't send them in"); segments clothing/bg/fg.

## At 5:44 (t=344s)
[5:38] ">> Oh." [5:39] ">> Oh. A rodent." [5:50] ">> Not bad at all, eh? Lots of custom keyframe fun." — the exact moment is the tail of the **3-frame custom keyframe demo** (beginning/middle/end images joined by a "woman walking along, finding the rodent, picking him up" prompt). 5:52 onward pivots to lip-sync via the reference model.

## Recommendations for the H3 RI2V workflow
**Speed (a)**
- Add a turbo-LoRA variant: harness skips the LoRA loader entirely ("all off") and hardcodes `steps: 20` in `_sampler` (ri2v_batch.py:107-113). Kijai's 4/8-step turbo LoRAs target the reference model directly — wire the LoRA loader + a `steps` cfg key (20→4/8) for prototype sweeps. Flip point is `BasicScheduler` `steps`/`scheduler` in `_sampler`.
- 5s videos: `snap_length` already supports it — add `duration: 5.0` to a variant (→ 124 frames @24fps ≈ 5.2s). No code change needed.
- If audio is ever scored, A/B `euler_d` vs `res_multistep` (video's "e r s d e k" claim); current chain uses `res_multistep`.
- Sage attention already in the model chain (`PathchSageAttentionKJ` sage_attention auto + allow_compile); video suggests trying the newer H3 memory-efficient Sage patch — cheap A/B.
**Identity (b)**
- Custom keyframes are the biggest transfer: harness feeds 4 static `ref_images.ref_image_N` + a 5-frame static ref video (`H3ImageToRefVideo`). Instead, place keyframes at real temporal positions (e.g. frames 0/62/124 of the 5s video) with per-frame conditioning — directly anchors identity mid-video where the static ref video can drift. Each keyframe needs its own conditioning set (video's rule).
- Masked refs: the cloud example maps to the harness's refs (aliclo set, source 0112.jpg) — background bleed is a known identity confound. Add mask/crop path (LoadImage mask input → masked crop before `H3ImageToRefVideo` / `ref_images`) to strip background/clothing regions from refs.
- Prompt: video's story-prompt-joining-frames matches the frozen prompt files; the triangular-bracket media convention is already how refs are embedded.
**Not applicable (c)**
- Video-extend pipeline (context_length 39, H3 assemble-existing-video, KJ overlap) — harness is single-segment; VSR bypass already notes "identity decided at base gen". Log as follow-up only if 15s+ chained runs are planned.
- Lip-sync/raw-audio pipe — visual identity scoring only.
- SigmaShift — already applied (shift_video 12, shift_audio 4).

## Differences vs our workflow
- Video: 6-8 steps + turbo LoRAs, masked ref loading, multi-segment extend. Harness: 20 steps, no LoRA, unmasked full refs (`ref_image_size: max`), single-shot `H3ReferenceToVideo` gen.
