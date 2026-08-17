# RESUME PROMPT — paste into a fresh session

```
Continuing the MiniMax H3 RI2V reference-identity analysis. Everything you
need is on disk — read these FIRST, in order:

1. /media/mal/Crucible/AI-ART/ComfyUI/custom_nodes/ComfyUI-H3O/experiments/HANDOFF_RI2V_ANALYSIS.md
   (the full handoff: root cause, evidence table, mechanism, fix status)
2. /media/mal/Crucible/AI-ART/ComfyUI/custom_nodes/ComfyUI-H3O/experiments/RESUME_PROMPT.md
3. ~/.quetza-data/conductor/reports/scout-h3-identity.md (H3 ref-block mechanics)
4. ~/.quetza-data/conductor/reports/2ic-review.md (adversarial matrix review)
5. ~/.quetza-data/amethyst/vault/projects/ri2v-ref-consistency-2026-08-17.md

STATUS IN 6 LINES:
- Root cause CONFIRMED: the 5-frame static manga ref-video (0112.jpg) — and
  ANY manga scene visual ref, image or video — outvotes the photo refs for
  identity at every denoising step. The output's woman is the manga page's
  original woman; the refs' tattoos/outfit/face never transfer.
- WORKING FIX: NOVID variant — no scene ref, prompt-only scene description
  (experiments/ri2v_prompt_frozen_novid.txt). Ref identity transfers in 3/3
  seeds (septum ring, green eyes, cheek mole, tongue piercing, chest tattoos,
  manga-styled). Verified by vision scoring.
- FAILED levers (all tested): source_scale, source_noise, RefBoostV V-damp
  (late AND new constant/early schedules), SCENEIMG, ALICIO3 (no blonde),
  SHEET1 (character sheet). None transfer identity while a manga visual is
  present. VB_CHARONLY/VB_VIDEOZERO glitch.
- Node change committed on branch `ri2v-ref-consistency` (22f9827): added
  constant/early_linear/early_sqrt damp schedules to H3RefBoostChar/V. Works
  mechanically, doesn't fix this case. Keep on branch; don't merge to main.
- EXPERIMENT HARNESS: experiments/ri2v_batch.py — builds ComfyUI /prompt API
  graphs (faithful conversion + controlled variants), 448×688, 243 frames,
  20 steps, frozen prompt, VSR bypassed. Results + montages: /tmp/ri2v/.
- NEXT DECISION (needs the user): (a) ship NOVID-style workflow clone (wire
  out the ref_video, swap the prompt, keep the original workflow untouched),
  (b) try photo-style output (flip style_transfer off match_source — identity
  transfer photo→photo is likely much stronger but changes output style),
  or (c) both. Then run the fixed workflow WITH VSR in the UI to confirm.

RELAY: the user's interactive claude (peer "mal") is connected but does NOT
poll relay asks — don't block on him; use qc-dispatch flash workers instead.

ComfyUI was stopped at handoff. Start it before more runs:
cd /media/mal/Crucible/AI-ART/ComfyUI && source venv/bin/activate &&
nohup python main.py --listen 127.0.0.1 --port 8188 > /tmp/comfyui.log 2>&1 &

DO NOT re-run the full matrix — the question is answered. Only run what the
user's decision requires.
```
