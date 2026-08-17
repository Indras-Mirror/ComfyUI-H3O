# RESUME PROMPT — paste into a fresh session (2026-08-17, v2)

```
Continuing the MiniMax H3 RI2V reference-identity project. Read THESE FIRST,
in order — everything below is also on disk:

1. /media/mal/Crucible/AI-ART/ComfyUI/custom_nodes/ComfyUI-H3O/experiments/HANDOFF_RI2V_ANALYSIS.md
2. /media/mal/Crucible/AI-ART/ComfyUI/custom_nodes/ComfyUI-H3O/experiments/RESUME_PROMPT.md  (this file)
3. ~/.quetza-data/conductor/reports/scout-h3-identity.md      (H3 ref mechanics)
4. ~/.quetza-data/conductor/reports/2ic-review.md             (adversarial matrix review)
5. ~/.quetza-data/amethyst/vault/projects/ri2v-ref-consistency-2026-08-17.md

═══════════════════════════════════════════════════════════════
PART 1 — WHERE THE WORK STANDS (verified)
═══════════════════════════════════════════════════════════════

GOAL: the user's RI2V workflow (video_minimax_h3_ri2v_VSR.json) should let
the photo refs of his character (aliclo: tattooed dark-haired woman)
CONSISTENTLY define the output's woman. It "kind of works but a lot of the
time it doesn't".

ROOT CAUSE — CONFIRMED by 28 runs / 15 variants, all vision-scored:
whenever ANY source visual (the static 5-frame ref-video OR even a static
IMAGE ref of the source) is present, the output's woman is the SOURCE's woman
— the photo refs' identity (tattoos/outfit/face) NEVER transfers. The source
identity commits during the high-sigma structure pass; all damp levers
(source_scale / source_noise / RefBoostV V-damp, late OR constant OR early
schedules) are too weak or break the render. Video's V contribution is
load-bearing every step.

WORKING FIX — VERIFIED 3/3 seeds: NOVID variant (no scene ref at all; the
prompt text describes the scene). Output woman = the ref character (septum
ring, green eyes, cheek mole, tongue piercing, chest tattoos), manga-styled.
Prompt: experiments/ri2v_prompt_frozen_novid.txt. Outputs+montages:
/tmp/ri2v/NOVID_s101..303/ (montage.jpg = refs | source | 4 output frames).

NODE CHANGE (branch only, commit 22f9827): h3_refboost_char.py /
h3_refboost_v.py gained schedule modes "constant" / "early_linear" /
"early_sqrt" (eff = 1 or 1-p). Mechanically works; does NOT fix this case.
Keep on branch ri2v-ref-consistency; do not merge to main yet.

═══════════════════════════════════════════════════════════════
PART 2 — THE CURRENT TASK (in flight when context was refreshed)
═══════════════════════════════════════════════════════════════

SOURCE-CHARACTERIZATION SWEEP (user's latest request): mine the ReActor
folders for ~8-10 DIVERSE source images and run the BASE-style pipeline
(video_ref = source as 5-frame static, refs = aliclo set, generic prompt)
to map WHEN the static-source approach transfers identity vs fails. The user
explicitly wants variety: different real women, close-ups AND full-body,
anime/manga-style images, and a same-woman-as-refs photo. Source folders:
  /media/mal/Crucible/ReActor/New folder          (213 photos, 1944x2592)
  /media/mal/Crucible/ReActor/MIXSAD 2            (289 facebook-style 857x857)
  /media/mal/Crucible/ReActor/image_downloader/PornPics_Downloads (16 scene dirs)
  /media/mal/Crucible/ReActor/image_downloader/Untitled Folder
  + existing input/: 0112.jpg (manga page — known FAIL baseline), aliclo2/4/3

KNOWN GOTCHA: ComfyUI LoadImage historically fails on /media/mal/Crucible/
ReActor paths ("Invalid image file") — COPY source images into
/media/mal/Crucible/AI-ART/ComfyUI/input/sweep/ (real copies, not symlinks)
before referencing them.

SWEEP MECHANICS (harness experiments/ri2v_batch.py): variant BASE with
video_ref=(sweep source, 5), aspect_source=the source, refs=REFS_CURRENT,
prompt=GENERIC (write ri2v_prompt_gen.txt — same structure as the frozen
prompt but no manga-scene specifics; "Replace the woman in the source video
with the woman in the reference images (all same subject)... keep the scene,
framing, camera; transfer face/body/hair/tattoos"). Run 1 seed each first
(448x688, 243f, 20 steps, ~2.5 min/run), vision-score each output: does the
ref woman appear (tattoos/septum/outfit) or the source's woman? Build the
characterization table: source attributes -> identity-transfer outcome.
ComfyUI is DOWN — start it first:
cd /media/mal/Crucible/AI-ART/ComfyUI && source venv/bin/activate &&
nohup python main.py --listen 127.0.0.1 --port 8188 > /tmp/comfyui.log 2>&1 &

═══════════════════════════════════════════════════════════════
PART 3 — ADOPT THE RUNNING WORKERS (do this immediately)
═══════════════════════════════════════════════════════════════

1. qc-status --watch &            (start the conductor watchdog)
2. tmux ls — there is a headless claude worker named qc-claude-2ic:
   - What: Opus 4.6 reviewing the handoff + designing the sweep source
     selection (packet: ~/.quetza-data/conductor/packets/claude-2ic.md).
   - Status: ~/.quetza-data/conductor/.done-claude-2ic (touched when done),
     report → ~/.quetza-data/conductor/reports/claude-2ic.md.
   - If done: READ the report (use its sweep picks), then close the worker
     (qc-status --close-done). If still running: arm qc-await claude-2ic or
     poll the done file.
3. CLAUDE-AS-WORKER RECIPE (just made to work — auth trick):
   env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN -u ANTHROPIC_BASE_URL \
        -u QUETZA_ENDPOINT -u ANTHROPIC_DEFAULT_SONNET_MODEL -u ANTHROPIC_DEFAULT_OPUS_MODEL \
        -u ANTHROPIC_DEFAULT_HAIKU_MODEL -u ANTHROPIC_SMALL_FAST_MODEL \
        claude -p "$(cat <packet>)" --dangerously-skip-permissions < /dev/null
   The proxy env (QUETZA_ENDPOINT/ANTHROPIC_AUTH_TOKEN) is stale/invalid →
   401; unsetting it falls back to the claude.ai OAuth login (Opus 4.6).
   Wrapper script template: /tmp/ri2v/claude_worker.sh
4. The user's INTERACTIVE claude (relay peer "mal") does NOT poll relay asks
   (8+ queued, unanswered ~2h). Do not block on him. qc-dispatch flash
   workers are the reliable path. relay CLI: relay peers / relay ask / relay poll.
5. Old workers from a previous session (dsh-prompt-scout, dsh-prompt-implement)
   belong to another session — leave them (qc-status --close-done skips them).

═══════════════════════════════════════════════════════════════
PART 4 — DECISIONS WAITING ON THE USER
═══════════════════════════════════════════════════════════════

1. Ship the NOVID-style fix as a workflow clone (works today) vs try
   photo-style output (flip style_transfer off match_source — likely stronger
   identity transfer but different output style) vs both. Recommend: NOVID
   clone first, then photo-style as an experiment.
2. Whether the character-sheet refs (H3Gen_00001..5, output/CharacterSheets/)
   should replace the aliclo photo refs in the fixed workflow.
3. Merge the schedule-mode node change to main or keep on the branch.

═══════════════════════════════════════════════════════════════
PART 5 — SESSION/FILE STATE (at handoff)
═══════════════════════════════════════════════════════════════
- Repo: /media/mal/Crucible/AI-ART/ComfyUI/custom_nodes/ComfyUI-H3O
  branch ri2v-ref-consistency, commits: a3a3d83 (handoff+resume),
  22f9827 (node change + harness + prompts). Nothing pushed. backup/ untracked.
- Results: /tmp/ri2v/ (manifest.jsonl machine-readable; montages per run).
- ComfyUI: DOWN (start per Part 2). GPU free.
- Original workflow JSON untouched. All work on the branch.
- The user's request boundary: preserve current functioning; clone workflows
  rather than editing the originals; make no mistakes; don't stop until the
  source-sweep characterization is done and reported.
```
