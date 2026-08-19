# RESUME PROMPT — paste into a fresh session (2026-08-18, v3 — post-production-fix)

```
Continuing the MiniMax H3 RI2V reference-identity project. Repo:
/media/mal/Crucible/AI-ART/ComfyUI/custom_nodes/ComfyUI-H3O
(= /home/mal/AI/ComfyUI-H3O, same tree via symlink; branch ri2v-ref-consistency).

Read THESE FIRST, in order — everything below is also on disk:
1. experiments/HANDOFF_RI2V_ANALYSIS.md                (root-cause analysis)
2. experiments/RESUME_PROMPT.md                        (this file)
3. ~/.quetza-data/conductor/reports/ri2v-sweep-p1.md   (sweep phase 1)
4. ~/.quetza-data/conductor/reports/ri2v-sweep-p2-hinges.md
5. ~/.quetza-data/conductor/reports/ri2v-sweep-ctrl.md
6. ~/.quetza-data/conductor/reports/ri2v-sweep-p2-rest.md
7. ~/.quetza-data/conductor/reports/ri2v-prod-ab.md    (production path disambiguation)
8. ~/.quetza-data/conductor/reports/ri2v-prod-validate.md
9. ~/.quetza-data/conductor/reports/ri2v-prod-src00.md (SRC00 lever)
10. ~/.quetza-data/conductor/reports/claude-fix-stack.md (stack verdict)
11. ~/.quetza-data/conductor/reports/claude-qwen-enhancer.md (Qwen wiring)
12. ~/.quetza-data/conductor/reports/ri2v-g46-test.md  (grok-4.6 compliance — COMPLIANT)
13. ~/.quetza-data/conductor/packets/ri2v-scene-props.md (IN-FLIGHT slice packet)

GOAL: the user's RI2V workflow (video_minimax_h3_ri2v_VSR.json) should let the
photo refs of his character (aliclo: tattooed dark-haired woman, septum ring,
green eyes) CONSISTENTLY define the output's woman. "Kind of works but a lot
of the time it doesn't."

═══════════════════════════════════════════════════════════════
PART 1 — THE FINDING SO FAR (diagnosis evolution, all vision-scored)
═══════════════════════════════════════════════════════════════

1. ORIGINAL HYPOTHESIS — "any source blocks identity". Root cause was
   confirmed as identity COMMITMENT during the high-sigma structure pass: the
   source woman (static 5-frame ref-video or image ref) wins over the photo
   refs. NOVID variant (no scene ref; prompt describes the scene) fixed it
   3/3 seeds (commit a3a3d83). All damp levers (source_scale / source_noise /
   RefBoostV V-damp, late/constant/early schedules) were too weak or broke the
   render. Video's V contribution is load-bearing every step.

2. SWEEP (10 sources, harness experiments/ri2v_batch.py --source, generic
   prompt ri2v_prompt_gen.txt) — the mechanism is prompt × source-type, NOT
   "any source blocks identity":
   - Generic prompt: refs win 30/30 across ALL source types (photo full-body,
     tattooed-similar, multi-person, Western comic, B&W manga, color anime,
     male manga, FACE close-up, aliclo-same-woman, manga control 0112). The
     generic prompt's explicit "replace the woman… transfer face/body/hair/
     tattoos" instruction overrides source identity under every source type.
   - Frozen prompt (manga-anchored scene text) + B&W manga page = SOURCE WINS
     (0112 2/2 control C1; different-series manga #7 2/2 control C3). Manga
     scene-anchoring is GENERAL, not 0112/Tifa-wording-specific.
   - Photo/color-drawn sources never override under either prompt (FACE 2/2
     frozen; 24/24 generic). Visual identity alone never overrides; the frozen
     prompt's scene anchoring is the lever. All claude-2ic SOURCE-WINS
     predictions are refuted under the prompt actually used.
   - Verdicts are prompt-attributed: frozen-prompt matrix ≠ generic-prompt
     sweep; never compare directly.

3. PRODUCTION VALIDATION (ri2v-prod-ab.md) — H2 CONFIRMED: prompt form is the
   identity switch. Production ref config (RefBoostChar 1049, video_ref,
   ref_image_size=max, batch refs 0.5MP pad+fit) is byte-identical to the
   harness — H1 (config-identity) DEAD. Run A: same seeds/source/refs, only
   the prompt changed (production vs generic) → FAIL vs TRANSFER. The
   production prompt's verbose shot-by-shot scene narration anchors the source
   identity.

4. TIGHTENED ENHANCER PROMPT (ri2v-prod-validate.md) — _DETAILED_DESC_RI2V
   compact directive (h3_prompt_enhancer.py:280, replace at :1722) STILL FAILS
   identity 2/2: grok-4.3 partially complies — drops [Shot N]/timecodes but
   keeps a timestamp, <d> dialogue, and scene action (deepthroat/cum). The
   residual scene narration still anchors source identity.

5. SRC00 (ri2v-prod-src00.md) — node 1049 source_scale 0.8→0.0, one widget.
   1/2 seeds: s202 = aliclo 4/4 both models + layout PASS (either-seed rule);
   s101 still manga 0/4. Improvement real but noise-seed sensitive. CHANGE
   LEFT IN PLACE (workflow JSON not git-tracked; backup path below).

6. STACK VERDICT (claude-fix-stack.md) — SRC00 + EXACT generic prompt via new
   enhancer bypass (_bypass_enhancer, ri2v_batch.py:297):
   - Identity: aliclo 4/4 BOTH models (grok-4.3 + gemini-2.5-flash) ALL 4
     seeds → seed sensitivity RESOLVED, deterministic identity.
   - Layout: FAIL 3/4 (scene dropped — trivial win: generic prompt was already
     layout-drift; the verbose production prompt was what held layout).
   - Below the ≥3/4 consistency bar → NOT the full fix. Trade-off made
     explicit: generic prompt = deterministic identity, lost scene; verbose
     enhancer prompt = scene held, identity 1/2.

7. CURRENT HYPOTHESIS — the full fix is SRC00 + a COMPLIANT prompt: identity
   commitment + style/layout anchor, NO character narration. grok-4.6 IS
   compliant (ri2v-g46-test.md, COMPLIANT): 3-sentence transfer directive
   (zero [Shot N], zero MM:SS, zero <d>, zero speaker IDs, zero camera-motion
   words; names the marks; "all 4 pictures attribute_transfer"; retains male
   as <Subject 2> fully_preserved). Untested stacked with SRC00 on identity+
   layout — that is the next experiment. Qwen 3.8 compliance DEFERRED.

═══════════════════════════════════════════════════════════════
PART 2 — IN-FLIGHT (as of 2026-08-18 ~05:00)
═══════════════════════════════════════════════════════════════

1. ri2v-scene-props — RUNNING NOW. Source-property matrix on the PRODUCTION
   path (that's where seed sensitivity lives; the harness showed 30/30 under
   the generic prompt). Packet: ~/.quetza-data/conductor/packets/ri2v-scene-props.md
   - Pool: 14 user-NARROWED images from ComfyUI/input/ ONLY (xzcsarwq.jpg,
     xzvgdfsad.jpg, yumiyimt.jpg, zc.jpg, zxc.jpg, "zxc3 234 .jpg", "zxc 23.jpg",
     "zxc 231asdasf.jpg", "zxc 324 srfafa.jpg", "zxc 3243asd.jpg",
     "zxcgd sg sdgsdg.jpg", zxcvcx.jpg, zxcvxc.jpg, zxcxzcsasd.jpg). NO Rubp images,
     NO Xman/xmen*, NO 0112.jpg. NOTE: the first scene-props worker ACKed the pool
     but ran unrelated images (close_src/small_src/avert_src/multi_src/nopp_src
     hash-matched to NOTHING in the pool) — it was KILLED 2026-08-18 ~05:10 and the
     slice re-dispatched on the amended packet. Verify sources by sha256 against the
     pool before scoring.
   - 5 cells × 2 seeds: CLOSE / SMALL / AVERT / MULTI / NOPP. Drawn sources OK.
   - Cell sources copied to input/sweep/: close_src.jpg, small_src.png,
     avert_src.jpg, multi_src.jpg, nopp_src.png (COPIES, sha256-verified).
   - Stack under test: SRC00 + exact generic prompt via --enhance --prompt-file
     experiments/ri2v_prompt_gen.txt, production path (0112 source slot
     repointed per cell; video-ref LoadImage node 1113 → H3ImageToRefVideo 1112,
     frame_count=5; batch refs 4×aliclo 0.5MP pad+fit).
   - PROGRESS: CLOSE (s101/s202) + SMALL (s101/s202) done → outputs
     MinimaxH3_00107..00110 (ComfyUI/output/MinimaxH3/), montages in
     /tmp/ri2v-scene-props/{CLOSE,SMALL}/. AVERT in progress (dir created
     empty; RandomNoise node 129 currently 101). MULTI + NOPP pending.
   - Report NOT yet landed (reports/ri2v-scene-props.md missing), done file
     not touched. Conclusion contract: which properties block/allow transfer;
     if ALL transfer → face properties don't matter in production either,
     residual variance is seed/model-level.

2. ri2v-g46-test — DONE. Report landed (ri2v-g46-test.md), verdict COMPLIANT
   (details Part 1 §7), .done-ri2v-g46-test touched. grok-4.6 = x-ai/grok-4.6
   already in the enhancer dropdown (h3_prompt_enhancer.py:1318-1321).

3. Qwen 3.8 compliance test — DEFERRED (GPU busy with scene-props worker,
   ~18.9/24.5GB used; Qwen needs ~16-18GB). Run when GPU frees: 0112.jpg + 4
   aliclo refs via the enhancer (llamacpp/qwen3.8-heretic-ara), save prompt to
   /tmp/ri2v-prodfix/QWEN_enh.prompt.txt, check compliance same as g46.

═══════════════════════════════════════════════════════════════
PART 3 — THE UNCOMMITTED CHANGES THAT MUST NOT BE LOST (the work product)
═══════════════════════════════════════════════════════════════

Repo: /media/mal/Crucible/AI-ART/ComfyUI/custom_nodes/ComfyUI-H3O
(branch ri2v-ref-consistency, latest commit 6e55ca9, nothing pushed)

1. experiments/ri2v_batch.py — +56/-1 vs HEAD.
   _bypass_enhancer() at :297: with --enhance AND --prompt-file both set,
   removes the H3PromptEnhancerPlus node from the converted graph, pins node
   1060 prompt + ShowText preview to the file's text, repoints 1060
   images_batch to the batch refs node (enhancer ref_images_out is a
   passthrough), drops enhancer-only nodes. --prompt-file flag at :513,
   wiring at :534. Without --prompt-file, --enhance unchanged (live enhancer).
   This is the byte-identical prompt A/B path — keep it.

2. h3_prompt_enhancer.py — +98/-2 vs HEAD.
   IDENTITY COMMITMENT block appended to H3_RI2V_RULE (supersedes the 350-500
   word instruction; marks every reference picture attribute_transfer, names
   the replaced character only inside <Video N>'s definition) +
   _DETAILED_DESC_RI2V compact directive (:280 def, replaced into the system
   prompt at :1722 for task_type in H3_RI2V_TASK_TYPES): 1-3 sentences, style
   + layout + identity transfer only, no narration/timestamps/dialogue.
   NOTE: grok-4.3 partially ignores this (see Part 1 §4); grok-4.6 complies.

3. The workflow JSON — NOT git-tracked (lives in the workflow dir):
   /media/mal/Crucible/AI-ART/ComfyUI/user/default/workflows/video_minimax_h3_ri2v_VSR.json
   node 1049 H3RefBoostChar source_scale = 0.0 (widgets_values idx 9 — the
   SRC00 lever). Backup: /tmp/video_minimax_h3_ri2v_VSR.pre-src00.bak
   (134288 B, CONFIRMED on disk 2026-08-18 04:14). If the JSON is ever lost,
   re-apply source_scale=0.0 to node 1049 and verify with json.load.

4. The two prompt txts (experiments/):
   - ri2v_prompt_frozen_novid.txt — +3/-5: summary rewritten as
     [video generation] (no <Picture 1> scene anchor), retention_analysis
     drops the partially_preserved scene block.
   - ri2v_prompt_frozen_sheet.txt — +2/-1: sheet-variant edit.
   Also present (committed): ri2v_prompt_gen.txt (the generic prompt used for
   all sweep + stack + scene-props runs) and ri2v_prompt_frozen.txt (original).

5. RandomNoise node 129 (workflow JSON): seed-widget procedure — set per run,
   restore to 949448543631237 after. CURRENTLY 101 (mid scene-props AVERT
   run — the scene-props worker restores it after; verify before any new run).

Other dirty state: backup/ untracked dir in the repo (contains
h3_batch_to_list.py — prior work backup). Untouched: h3_refboost_char.py /
h3_refboost_v.py schedule modes (committed in 22f9827, branch-only, not
merged to main).

═══════════════════════════════════════════════════════════════
PART 4 — DECISIONS WAITING ON THE USER
═══════════════════════════════════════════════════════════════

(a) Keep SRC00 (source_scale=0.0) as the base fix? Current evidence says yes
    (identity deterministic 4/4 with the generic prompt; only layout suffers).
(b) Which enhancer model for production? grok-4.6 (COMPLIANT per g46 test —
    untested stacked with SRC00), Qwen 3.8 (wired at h3_prompt_enhancer.py:
    1324-25 + h3o_shared.py:77, models at /media/mal/NVME1TB/Models/Qwen3.8/,
    compliance untested), or pin the generic prompt via --prompt-file bypass
    (deterministic but loses the scene). Next experiment to run: SRC00 +
    grok-4.6 prompt, 4 seeds, dual-model score.
(c) Merge the uncommitted code (ri2v_batch.py bypass + h3_prompt_enhancer.py
    override)? Recommendation: commit on the branch once scene-props lands;
    do NOT merge to main yet (schedule-mode node change also branch-only).

═══════════════════════════════════════════════════════════════
PART 5 — SESSION/FILE STATE (at handoff)
═══════════════════════════════════════════════════════════════

- Repo: /media/mal/Crucible/AI-ART/ComfyUI/custom_nodes/ComfyUI-H3O
  (= /home/mal/AI/ComfyUI-H3O). Branch ri2v-ref-consistency.
  git status: M experiments/ri2v_batch.py, M experiments/ri2v_prompt_frozen_novid.txt,
  M experiments/ri2v_prompt_frozen_sheet.txt, M h3_prompt_enhancer.py, ?? backup/.
  Nothing committed beyond 6e55ca9; nothing pushed.
- Relay/session: conductor session 7548.32907 (watchdog heartbeat
  session=7548.32907). Worker peer names: qc-ri2v-scene-props,
  qc-ri2v-g46-test, qc-resume-update. CLI: relay peers / ask / reply / poll /
  history (relay binary at ~/.local/bin/relay). The user's interactive claude
  peer does NOT poll relay asks — use workers or wait for the conductor.
- Watchdog: qc-status --watch & (running; ~/.local/bin/qc-status, qc-await).
- ComfyUI: UP on 127.0.0.1:8188 (v0.30.0, PID 9184 since 04:11, GPU loaded by
  the scene-props pipeline). If it dies, restart:
  cd /media/mal/Crucible/AI-ART/ComfyUI && source venv/bin/activate &&
  nohup python main.py --listen 127.0.0.1 --port 8188 > /tmp/comfyui.log 2>&1 &
- Results / montages:
  - /tmp/ri2v/ (sweep p1/p2a; manifest.jsonl per dir, montage.jpg per run)
  - /tmp/ri2v/ctrl/ (control arm C1/C2/C3, frozen prompt)
  - /tmp/ri2v-prodab/ (prod-ab Run A + vision tooling)
  - /tmp/ri2v-prodfix/ (validate ENH_s101/202 + src00; vision_score.py —
    grok-4.3 + gemini-2.5-flash dual scorer; G46_enh.prompt.txt)
  - /tmp/ri2v-fixstack/ (stack ENH_s101..404 + grok/gemini score files)
  - /tmp/ri2v-scene-props/ (in-flight cells; CLOSE/SMALL done)
  - ComfyUI/output/MinimaxH3/ (raw outputs; 00107+ = scene-props runs)
- GPU: single 4090, ~60s model swap. One pipeline at a time. Qwen compliance
  test waits for GPU free.
- User decision needed before next production runs: Part 4 (a)/(b)/(c).
```

Commit recommendation: once ri2v-scene-props lands, commit the harness
(ri2v_batch.py) + enhancer (h3_prompt_enhancer.py) + prompt txts on the
branch, and record the workflow JSON change (source_scale=0.0) in the commit
message — the JSON itself is not git-tracked and only exists in the workflow
dir + /tmp backup.

## v4 — 2026-08-18 LATE: full-fix PARTIAL verdict + mask direction + community validation

### The verdict (ri2v-g46-validate3, clean run on the user's pool)
- Config now: enhancer model **x-ai/grok-4.6** (live, deterministic compliant prompts) + **SRC00** (source_scale=0.0) + **duration=5** + **turbo ref2v 4-step** (steps=4). Speed: ~2 min/run, 124 frames.
- **Identity WORKS**: 2/4 runs perfect aliclo 4A, all runs strong (grok-4.6 prompt compliance confirmed; zero narration markers).
- **Layout is source-dependent**: simpler medium-wide composition (zxc 23.jpg) got identity 4A + layout 4PASS (seed 202 = perfect); complex low-angle reverse cowgirl (xzcsarwq.jpg) lost layout on a seed despite strong identity.
- **Root cause of the ceiling:** SRC00 zeroes the source's visual boost → layout leans entirely on the prompt → simple scenes describe faithfully, complex ones don't. Best run: A/202 (identity 4A + layout 4PASS).

### The direction (user + community validated)
- **Source masking is the next fix candidate** (slice `ri2v-source-mask`, dispatched on claude): mask the woman OUT of the source (Sam3/SegFormer) → no competing identity → source_scale can go BACK to 0.8 (full layout anchoring). Reddit r/StableDiffusion thread ("Get miniMax character swap working! Finally") independently confirms: "Use Sam3 to blur/mask the person and that problem goes away" for similar-looking subjects.
- **Prompt-structure innovation from the same thread** (PropagandaOfTheDude): subject-as-CONCEPTS — split the swap: `<Subject 1>` = ref face/hair (attribute_transfer), `<Subject 2>` = source person (weak_reference, "overall actions retained"), `<Subject 3>` = clothing ref (fully_preserved), etc. — prevents partial merges + clothing confusion. Reference prompts: https://omnifit.io/blog/ref2va-what-works-what-fails
- **Community notes:** bigger visual difference = cleaner swap; prominent clothing distracts the model; steps 20 for inference (we run 4 with turbo).

### New workflow files (user-provided, for mask wiring)
- `/home/mal/Downloads/minimaxH3Character_v20SolAttnTurbo.json`
- `/home/mal/Downloads/video_minimax_h3_r2v_addguides_v1.json`
- `/home/mal/Downloads/minimaxH3InfiniteVideoRef2va12Img_v10.json`
- LightX2V turbo loras: `models/loras/LightxV2-V2.0-T2V/` (high/low noise, T2V only — ref2v turbo in use is Kijai's 4-step).

### Worker adoption for a fresh session
- `qc-status --watch` first; ComfyUI restart: `cd /media/mal/Crucible/AI-ART/ComfyUI && source venv/bin/activate && nohup python main.py --listen 127.0.0.1 --port 8188 > /tmp/comfyui.log 2>&1 &`.
- In-flight at v4: `ri2v-source-mask` (claude; Sam3 mask → source_scale 0.8 → 4 runs on zxc 23.jpg + xzcsarwq.jpg masked; STATE.md at /tmp/ri2v-source-mask/ if created). Staged next: the 14-pool scene-props matrix at the new speed; a source_scale-balance test (0.3-0.5) if the mask under-delivers; the subject-as-concept prompt slice.
- Verification discipline (hard-won): the video-ref source must be verified in the QUEUED payload (curl /queue), not just the saved JSON — a rogue slice's avert_src leftover invalidated two validation attempts.
- Flash workers have been dying at 2-16 min with "running stop hook" (4+ times) — claude workers are stable; prefer claude for critical slices, write STATE.md partial progress for flash slices.

## v5 — 2026-08-18: HARD RULE + mask test results

### ⛔ HARD RULE (user, explicit — CANNOT/MUST NOT)
NSFW content is NEVER opened/rendered/viewed/displayed on Mal's PC (no Artifact
panels, no Read image previews, no image viewers/browser tabs). ALLOWED headless:
desktop-agent screenshot/analyze, visionproxy/OpenRouter vision analysis (API),
numpy/PIL numerical checks on disk. Verification = scores/numbers, never opening the
file visually. Applies to all agents/workers/wrappers. Stored in QUETZA.md + CLAUDE.md
+ Letta.

### ri2v-source-mask — first results (GD-SAM mask + source_scale 0.8)
- Masks verified headless (geometry + API vision): woman blacked, man/background kept.
- zxc23 (simple): LAYOUT 4P×8 FIXED (was 1P in stack test); IDENTITY 1A×7 + 2A×1 (generic).
- xzcsarwq (complex): LAYOUT 4P×4; IDENTITY 3A/3A/2A/1A (strong refs on f1/f2).
- Mechanism: mask restores layout anchoring; refs still lose to the masked ref-video's
  structural dominance at source_scale 0.8 → NEXT: source_scale balance test 0.3/0.5
  on zxc23 (staged in the v4 docs). run_mask.py SCALE env override ready.
- grok-4.6 prompts: subject-as-concept structure, full compliance (zero narration
  markers; retention_analysis: "<Video 1> partially_preserved — scene/structure/
  sofa layout + <Subject 2> retained; only the replaced woman is swapped").

## v6 — 2026-08-18 evening: THE FULL WIN CONFIG (identity + layout)

Paste this into a fresh session to continue from the winning point.

### THE CONFIG (proven: identity 4A/3A/4A/4A + layout 4P×4, seed 202)
**Comic source (xmen02.jpg) + masked ref video (in-graph GD-SAM person mask) + source_scale 0.5 + generic transfer prompt (experiments/ri2v_prompt_gen.txt via --prompt-file, drops the enhancer) + NO char boost (1049 enabled=False).**
- s101 at same config: identity 4A×3 but layout 3P/3P/2P → seed sensitivity is the remaining issue (pro worker researching: ri2v-seeds-research.md).
- source_scale = `1.0 + (scale-1.0)*eff` (h3_refboost_char.py:208-211): 1.0 undamped / 0.5 crossover / 0.0 SRC00.
- How to run: `/tmp/ri2v-source-mask/run_variants.py` with RUNS='[{"source":"xmen02.jpg","seed":202,"tag":"X","scale":0.5,"no_boost":true,"prompt_file":"experiments/ri2v_prompt_gen.txt"}]' WF=video_minimax_h3_ri2v_VSR_mask.json
- Frame scoring: `/tmp/ri2v-source-mask/score_frames.py` (OpenRouter grok-4.3, rubric in score_prompt.txt).

### Where everything stands
- Layout: FIXED by the mask (any mask + source_scale 0.8 → 4P everywhere). Identity: FIXED by comic source + 0.5 + generic prompt (4A). Both together: seed 202 only — seed sensitivity remains.
- Face strategies that did NOT break the ceiling alone: black-hole mask, feathered mask, face-only mask, close-up face ref (aliclo_face.png), face+refs (~22 runs).
- Research landed: ri2v-mask-research.md (SAM3 packs, workflow origins), ri2v-templates-research.md (face-swap prompt structure — "facial likeness" subject lines + verbatim feature retention + source-as-motion-only; custom_system_prompt injection built but UNTESTED in ..._mask_inject.json).
- Deliverables: video_minimax_h3_ri2v_VSR_mask.json / _mask_face.json / _mask_inject.json (ComfyUI/user/default/workflows/).
- Workers: ri2v-seeds-research (pro, in flight). claude quota resets 8pm Melbourne.
- ⛔ NSFW HEADLESS RULE: never open/render/view NSFW on Mal's PC (QUETZA.md/CLAUDE.md/Letta).

## v7 — 2026-08-18: WORKFLOW-EDIT RULE (bug fix — user-reported)

**When hand-editing a workflow JSON, ALWAYS bump `last_node_id` and `last_link_id` to the actual maxes** after adding nodes/links. The ComfyUI frontend assigns the user's NEXT connection `last_link_id + 1` — a stale counter collides with the added link IDs and connections wire themselves into random slots (user-reported: "connections start inputting themselves in randomly"). Fix applied 2026-08-18 to all generated workflows (mask / mask_face / mask_faceref / mask_inject / mask_win): counters now (1207, 572) / win (1213, 572), no duplicate IDs, all convert-verified. New mask-chain nodes also repositioned out of (0,0) into a readable column.
Rule: after ANY scripted workflow edit, set `wf['last_node_id'] = max(node ids)` and `wf['last_link_id'] = max(link ids)` and verify `json.load` round-trips. Reference fixer: /tmp/ri2v-source-mask/fix_workflows.py.

## v8 — 2026-08-18: scene_mode (hard/soft) + the missing-prompt bug

### ⚠ THE MISSING-PROMPT BUG (root cause of "reference too strong, no source")
The WIN workflow's 1060 prompt input was DISCONNECTED (input link None while links array had 496; my WIN build pinned the generic prompt + muted the enhancer, then the re-enable never restored the link) → the model ran on REFS ONLY, no prompt. THAT caused the "reference very strongly, not getting source" symptom. FIXED: link 496 restored on 1060's prompt input (converter-verified). All harness batch results (mask.json template) were unaffected (prompt linked there).

### scene_mode widget (hard/soft) on H3PromptEnhancerPlus — appended LAST in widgets
- hard (default, proven): compact transfer directive — identity + style + layout only, NO scene/act description. Restored as _DETAILED_DESC_RI2V_HARD.
- soft: relaxed — also describes the explicit sexual act from the source video in the final prompt (_DETAILED_DESC_RI2V_SOFT + H3_RI2V_SCENE_SOFT_OVERRIDE rule). Compliance constraints kept (no [Shot N]/timestamps/<d>/dialogue/speaker IDs).
- RI2V-specific: only fires for task_type "ri2v"; other templates untouched. scene_mode=None → hard (safe).
- Files: h3_prompt_enhancer.py + h3_prompt_enhancer_plus.py (needs ComfyUI restart). WIN workflow set to hard explicitly.

## v9 — 2026-08-19: SUPER-DETAILED RESUME PROMPT (paste into a fresh session)

```
RI2V REFERENCE-IDENTITY PROJECT — continue here. Repo:
/media/mal/Crucible/AI-ART/ComfyUI/custom_nodes/ComfyUI-H3O (= /home/mal/AI/ComfyUI-H3O,
branch ri2v-ref-consistency). ComfyUI: 127.0.0.1:8188 (user runs it — check before
queueing; it's UP with all code changes loaded).

════════ 1. MISSION ════════
Make the user's RI2V workflow (video_minimax_h3_ri2v_VSR) consistently swap his
character (aliclo: dark-haired, septum ring, green eyes, cheek mole, tongue piercing,
mandala tattoos) into source NSFW scenes. "Kind of works but a lot of the time it doesn't."

════════ 2. THE CURRENT CONFIG (WIN workflow — user's daily driver) ════════
File: ComfyUI/user/default/workflows/video_minimax_h3_ri2v_VSR_mask_win.json
- SAM3 mask chain: 1301 LoadSAM3Model → 1302 SAM3Grounding("woman", conf 0.2, max -1)
  → 1204 MaskBlur+(12) → 1206 ImageCompositeMasked(dest=1205 Blur(25) of source,
  src=source, mask=feathered) → 1207 H3ImageToRefVideo(5) → 1060 ref_videos.ref_video_0.
  Source (1113) feeds the ENHANCER unmasked (scene for grok). Previews: 1303 = SAM3
  mask-overlay, 1304 = final masked composite, 1213 = mask-as-image.
- 1049 H3RefBoostChar: mode 4 (BYSSED/muted), enabled=False, source_scale 0.5.
- 1111 H3PromptEnhancerPlus: model x-ai/grok-4.3 (user's choice), task_type ri2v,
  scene_mode missing → defaults HARD (compact directive). User's base_prompt has the
  explicit act (anal/creampie/voices).
- 1060 prompt input RE-CONNECTED to 1111 (link 496 — was disconnected = the bug).
- 129 seed: user's own (random per run).

════════ 3. WHAT WAS BUILT (deliverables) ════════
- video_minimax_h3_ri2v_VSR_mask.json — in-graph GD-SAM1 person mask (fallback template,
  used by ALL harness batch runs — has the prompt link intact).
- ..._mask_face.json (face-only mask), ..._mask_faceref.json (face-crop refs),
  ..._mask_inject.json (custom_system_prompt face-transfer rules, UNTESTED),
  ..._mask_win.json (the user's live workflow — SAM3 version).
- Harness: /tmp/ri2v-source-mask/run_variants.py (RUNS/SCALE/no_boost/prompt_file envs),
  run_workflow.py, score_frames.py (OpenRouter grok-4.3 + rubric score_prompt.txt),
  fix_workflows.py (counter fixer), gdsam_mask.py, vscore.py.
- input/aliclo_face.png (768x932 face-crop ref from aliclo1), input/sweep/*_masked.png.

════════ 4. WHAT WAS FIXED (all verified) ════════
1. THE MISSING-PROMPT BUG: 1060's prompt input link was None (links array had 496) →
   model ran on refs only → "reference very strongly, not getting source". FIXED +
   converter-verified. THIS was the user's main recent symptom.
2. Workflow-edit counters: last_node_id/last_link_id were stale → random auto-wiring
   on user edits. Bumped on all mask*.json + rule documented (v7).
3. SAM3 isolation: COMFY_ENV_ISOLATE=0 in ~/.comfy-env/settings.env (pixi envs were
   incomplete; main venv torch 2.13.0+cu130 verified untouched). SAM3 nodes registered.
4. Local-enhancer OOM: h3o_shared.py unloads ComfyUI models before llama-server spawn.
5. Stale link 554 dual-ownership on 1112 cleared (phantom wire to 1060).

════════ 5. scene_mode (hard/soft) — NEW ════════
H3PromptEnhancerPlus widget (appended LAST): hard = compact transfer directive (no
scene/act in final prompt — PROVEN), soft = describes the explicit act from the video
(+ H3_RI2V_SCENE_SOFT_OVERRIDE rule). RI2V-specific. Defaults hard when absent.
User ran "perfectly" with hard + the prompt fix.

════════ 6. VERIFICATION / RESULTS (all grok-4.3-scored, 4 frames/run) ════════
- THE FULL WIN: xmen02 (comic) + mask + source_scale 0.5 + generic prompt
  (ri2v_prompt_gen.txt) + no boost → s202 IDENTITY 4A/3A/4A/4A + LAYOUT 4P×4.
  Standalone WIN-wf smoke: 3A/3A/3A/4A + 4P×4. Harness batch results stand (prompt linked).
- Mask fixed layout: ~16/16 frames 4P at source_scale 0.8 across mask strategies.
- Face ceiling: 5 strategies / ~22 runs — face markers (septum/mole/green eyes) never
  rendered reliably until the comic-source + generic-prompt + 0.5 combo.
- seed sensitivity: intrinsic flow-matching structure commitment (pinnable — lock seed;
  comic source stabilizes IDENTITY across seeds, not layout).

════════ 7. BATCH STATUS ════════
- Done: mask tests, face-mask, face-ref, no-boost × sources × prompts, scale sweep
  (0.5/0.6/0.7), generalization (enmafrost 2A/3A/4A/3A+4P, Xman 3A/3A/3A/4A+2-3P,
  xzcsarwq 2A-3A+4P), WIN-wf smoke (00153).
- Untested: mask_inject custom_system_prompt rules, SAM3.1 (Comfy-Org/sam3.1
  checkpoints, PR #13408), steps 20, higher output res, scene_mode=soft in anger.
- User's last runs: worked perfectly (prompt fix + hard directive + SAM3 mask).

════════ 8. KNOWN GAPS / OPEN ITEMS ════════
- Seed sensitivity: lock a good seed; comic sources stabilize identity not layout.
- scene_mode=soft untested — user's fallback if hard under-delivers the act.
- claude worker quota resets 8pm Melbourne (qc-dispatch --model claude).
- SAM3.1 upgrade path (16-object multiplex, per-object masks) — research recommends
  Comfy-Org/sam3.1 checkpoints.
- The user's in-UI edits beat the disk file — reload workflows after any scripted edit.

════════ 9. INFRA CHANGES ════════
- ~/.comfy-env/settings.env: COMFY_ENV_ISOLATE=0 (rollback: delete line + restart).
- comfyui-sam3 pack isolation env completed at ~/.ce/.pixi (torch 2.8 cu128 + flash-attn
  + cc_torch — SAM3 nodes now work; main venv untouched).
- h3_prompt_enhancer.py: _DETAILED_DESC_RI2V_HARD/_SOFT + H3_RI2V_SCENE_SOFT_OVERRIDE
  + scene_mode param. h3_prompt_enhancer_plus.py: scene_mode widget (LAST) + _un1.
  h3o_shared.py: VRAM unload before spawn. ALL need ComfyUI restart (already done — UP).
- Backups: /tmp/video_minimax_h3_ri2v_VSR.pre-src00.bak, .pre-speed.bak, .pre-mask.bak,
  .pre-sam3.bak, /tmp/h3_prompt_enhancer.pre-scene-desc.bak.
- Repo uncommitted (branch ri2v-ref-consistency): h3_prompt_enhancer.py (+scene_mode),
  h3_prompt_enhancer_plus.py, h3o_shared.py, ri2v_batch.py, 2 prompt txts, docs.
  Recommend committing once the user confirms the config; do NOT merge to main yet.

════════ 10. RULES ════════
- ⛔ NSFW HEADLESS ONLY: never open/render/view NSFW on Mal's PC (no Artifact/Read/
  viewers). Allowed: desktop-agent screenshot/analyze, visionproxy/API vision,
  numpy/PIL checks. Scores, never files. QUETZA.md/CLAUDE.md/Letta aa7727e2.
- Workflow-edit rule: bump last_node_id/last_link_id after scripted edits (fixer:
  /tmp/ri2v-source-mask/fix_workflows.py).
- Verify wiring in the CONVERTED prompt before queueing (docs discipline).
- Flash workers die 2-16 min ("running stop hook") — prefer claude/pro; STATE.md
  partial progress for flash slices.

════════ 11. WORKER FLEET + REPORTS ════════
Reports in ~/.quetza-data/conductor/reports/: ri2v-source-mask.md (verdicts+matrix),
ri2v-mask-research.md (SAM3 packs, workflow origins), ri2v-templates-research.md
(face-swap prompt structure), ri2v-seeds-research.md (seed sensitivity), 
ri2v-sam3-research.md (SAM3 fix + reference-too-strong diagnosis).
Packets: ~/.quetza-data/conductor/packets/. qc-dispatch <slice> --packet <file>.

════════ 12. FIRST STEPS FOR THE NEW SESSION ════════
1. Read: RESUME_PROMPT.md (this), HANDOFF_RI2V_ANALYSIS.md §11-13, the conductor
   reports above, ~/.quetza/QUETZA.md + ~/.claude/CLAUDE.md (rules).
2. Check ComfyUI up (127.0.0.1:8188); verify SAM3 nodes in object_info
   (LoadSAM3Model/SAM3Grounding) + 1060 prompt wired via ri2v_batch.convert_workflow.
3. User's workflow is their in-UI state — don't clobber; reload after edits.
4. If testing: run_variants.py + score_frames.py (see §6 configs). Verify the queued
   prompt (1113 source, 1302 woman mask, 1060 prompt <- 1111, scene_mode).
5. Open items per §8 — user decides. Amethyst: projects/ri2v-production-fix-2026-08-18.md.
```
