# ri2v-g46-validate — full-fix validation: SRC00 + grok-4.6 live enhancer

**Date:** 2026-08-18 · Production path (0112.jpg source + 4 aliclo refs, SRC00=0.0 in place, duration=5/124f, steps=4, node 129 set per seed + restored). Enhancer node 1111 ran LIVE with model `x-ai/grok-4.6` (widget idx 3 — already set pre-slice; verified in converted graph). ComfyUI verified up; queue empty before each run.

## Run matrix (identity = aliclo? layout = manga panels preserved?; both scorers via VisionProxy-MCP describe_image)
| seed | identity grok-4.3 | identity gemini-2.5 | layout grok-4.3 | layout gemini-2.5 | objective sat/gray |
|---|---|---|---|---|---|
| 101 | **aliclo** A (both passes) | manga B | PASS (hi conf) | FAIL* | 0.18 / 0.90 → B&W |
| 202 | manga B | manga B | PARTIAL (warm tint) | PASS | f1-3 color drift |
| 303 | manga B (hi-res) | manga B | FAIL (colorized 3D) | FAIL | heavy color |
| 404 | manga B | manga B | FAIL (photoreal) | PARTIAL | moderate color |

*gemini-101 layout FAIL contradicts objective B&W (sat 0.18, gray 0.9) — treated as misread; grok+objective carry 101.
Scoring: 9-cell montage per seed + high-res frames4.jpg second pass (identity only; hi-res authoritative). Scores: `/tmp/ri2v-g46v-validate/scores/`.

## Config verification (CONFIRMED, converted graph)
SRC00 node 1049 source_scale=0.0 ✓ · speed: duration=5 (node 132→131→124f), steps=4 (node 124) ✓ · node 1113=0112.jpg ✓ · enhancer 1111: model=x-ai/grok-4.6, same_subject=on, style_transfer=match_source, auto_describe=on(16128), editing_frame=on, ctx=120000 ✓ · node 129 restored to 949448543631237, workflow diff-vs-backup IDENTICAL ✓. Videos: MinimaxH3_00118..00121.mp4, all 5.17s/124f ✓.

## Prompt compliance (per-run, grok-4.6 live)
All 4 runs **byte-identical** (sha256 2bbd584028, 4187 B; temp 0.05 deterministic). PARTIAL vs g46-test bar: zero `[Shot N]`/MM:SS ✓; **`<d>[English] I'm cumming.</d>` + `(S1)/(S2)` shorthand + "camera" noun leaked into summary field** ✗; detailed_description = pure transfer directive, identity commitment + manga 5-panel layout anchor present ✓. Saved: `/tmp/ri2v-g46v/ENH_s{101..404}.prompt.txt`.

## Verdict: NOT the full fix (identity 1/4 under grok authority; 0/4 gemini)
Identity REGRESSED vs fix-stack baseline (generic prompt + SRC00 = 4/4 both models → 1/4). Layout IMPROVED structurally (panels preserved 4/4 vs fix-stack cinematic drift) but B&W manga fidelity only 1/4 clean (101). The grok-4.6 style anchor transfers panel STRUCTURE, not ink/screentone style or identity. "Stack them" hypothesis refuted.

## Recommended production config
Keep SRC00 (identity lever confirmed by fix-stack). No single prompt delivers both: generic prompt = identity 4/4, grok-4.6 = panels 4/4. Next levers: prompt merge (generic identity phrasing + grok layout anchor), or ref_scale↑. **Config-fidelity gap:** turbo LoRA NOT applied via API path — rgthree PowerLoraLoader slots live in widgets_values, convert_workflow passes only model/clip (verified in node source); 4-step scheduler DOES apply. All prior slices share this; A/B comparisons valid.

## Artifacts
Montages: `/tmp/ri2v-g46v-validate/ENH_s{101,202,303,404}/montage.jpg` (+frames4.jpg). Prompts: `/tmp/ri2v-g46v-validate/ENH_s*.prompt.txt`, `/tmp/ri2v-g46v/ENH_s*.prompt.txt`. Backup: `/tmp/ri2v-g46-validate.bak.json`. Repo: only this report added; no code touched.

## Cross-check (ri2v-g46-score relay)
Its avert_src contamination finding targets the OLD `/tmp/ri2v-g46v` session (MinimaxH3_00114..117) — NOT this slice. CONFIRMED clean: ComfyUI /history executed graphs for all 4 prompt_ids show node 1113=0112.jpg (per-seed noise, src_scale=0.0, steps=4); my montage cell5 = gray_frac 1.00/sat 0.00 (manga) vs old session's cell5 = sat 0.16 (color pin-up, RMSE 0.67 from manga); old-vs-my f0 RMSE 0.41 = different videos. Verdict stands on a clean 0112.jpg test.
