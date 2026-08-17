#!/usr/bin/env python3
"""RI2V reference-consistency experiment harness (video_minimax_h3_ri2v_VSR).

Rebuilds the user's RI2V workflow as a ComfyUI /prompt API graph with
per-variant overrides, so we can A/B the things that fight the reference
character:

  - video ref:  static 5-frame ref_video (H3ImageToRefVideo) vs NONE
  - ref set:    current 4-image set vs aliclo-only vs character sheet
  - refboost:   H3RefBoostChar slot/scale/repeat/protect/source_scale
  - ref_image_size: match vs max
  - prompt:     frozen string (captured from the enhancer baseline)

Every run writes a manifest row (variant, seed, prompt_id, status, files)
and extracts 4 evenly-spaced frames from the output mp4 for vision scoring.

Usage:
  python experiments/ri2v_batch.py --list
  python experiments/ri2v_batch.py --variant BASE --seed 123 --out-dir /tmp/ri2v
  python experiments/ri2v_batch.py --variant BASE --enhance  # enhancer wired
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "http://127.0.0.1:8188")
COMFYUI_INPUT = "/media/mal/Crucible/AI-ART/ComfyUI/input"
COMFYUI_OUTPUT = "/media/mal/Crucible/AI-ART/ComfyUI/output"
WORKFLOW = "/media/mal/Crucible/AI-ART/ComfyUI/user/default/workflows/video_minimax_h3_ri2v_VSR.json"

UNET = "minimax_h3_hybrid_fl2va_ref2va_b20-49-int8_NVME.safetensors"
CLIP = "MiniMax-H3/qwen3vl_32b_minimax_h3_ultra_uncensored_heretic_int8_convrot_NVME.safetensors"
VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

# Current active ref set in the workflow (1132 H3BatchImages -> enhancer)
REFS_CURRENT = ["aliclo2.png", "aliclo4.png", "Aliclo3.png",
                "2025-08-12_17-39_2.png"]
REFS_ALICLO3 = ["aliclo2.png", "aliclo4.png", "Aliclo3.png"]
REFS_SHEET = ["H3Gen_00001_.png"]  # 6-panel ri2i character sheet
SOURCE_IMG = "0112.jpg"            # the manga-page source -> static ref video

# Frozen prompt (filled from baseline capture; see --capture-prompt)
PROMPT_FILE = Path(__file__).parent / "ri2v_prompt_frozen.txt"
PROMPT_FILE_NOVID = Path(__file__).parent / "ri2v_prompt_frozen_novid.txt"


def _load_prompt(p: Path) -> str:
    if p.exists():
        return p.read_text().strip()
    raise SystemExit(f"Missing prompt file {p} — run the enhancer baseline first:\n"
                     "  python experiments/ri2v_batch.py --variant BASE --enhance")

REFBOOST_BASE = dict(enabled=True, ref_slots="all", ref_scale=3.0,
                     ref_repeat=3, protect_scene=False, source_noise=0.0,
                     schedule="linear", step_threshold=0.5, debug=False,
                     source_scale=0.8)


def snap_length(duration: float) -> int:
    """Match workflow ComfyMathExpression: n%17==5, n>=5."""
    n = max(5, round(duration * 24))
    return n + (5 - n % 17) % 17


# ═══════════════════════════════════════════════════════════════════════
# Graph construction
# ═══════════════════════════════════════════════════════════════════════

def _load_image(img_id: str, fname: str) -> dict:
    return {img_id: {"class_type": "LoadImage", "inputs": {"image": fname}}}


def _model_chain(model_src: str) -> dict:
    """The model chain, faithful to the workflow (LoRA loader skipped: all off)."""
    g = {
        "psage": {"class_type": "PathchSageAttentionKJ",
                  "inputs": {"model": [model_src, 0], "sage_attention": "auto",
                             "allow_compile": True}},
        "solattn": {"class_type": "SolAttnPatch",
                    "inputs": {"model": ["psage", 0], "tau": 1.3,
                               "start_percent": 0.2, "end_percent": 0.9,
                               "min_tokens": 4096, "int8_qk": True,
                               "sink_conditioning": "exact_kv_and_rows",
                               "morton": False, "morton_curve": "2d_frame",
                               "verbose": False, "use_tma": False}},
        "sigshift": {"class_type": "MiniMaxH3SigmaShift",
                     "inputs": {"model": ["solattn", 0], "shift_video": 12,
                                "shift_audio": 4}},
        "cache": {"class_type": "ApplyMiniMaxH3FirstBlockCache",
                  "inputs": {"model": ["sigshift", 0],
                             "mode": "H3 Fast — 0.10 / max 2",
                             "threshold": 0.1, "start_percent": 0.1,
                             "end_percent": 0.95, "max_consecutive_hits": 2,
                             "temporal_guard": False}},
    }
    return g


def _sampler(g: dict, main_node: str, seed: int) -> None:
    g["guider"] = {"class_type": "BasicGuider",
                   "inputs": {"model": ["spectrum", 0],
                              "conditioning": [main_node, 0]}}
    g["sched"] = {"class_type": "BasicScheduler",
                  "inputs": {"model": ["spectrum", 0], "scheduler": "simple",
                             "steps": 20, "denoise": 1}}
    g["ksamp"] = {"class_type": "KSamplerSelect",
                  "inputs": {"sampler_name": "res_multistep"}}
    g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    g["sampler"] = {"class_type": "SamplerCustomAdvanced",
                    "inputs": {"noise": ["noise", 0], "guider": ["guider", 0],
                               "sampler": ["ksamp", 0], "sigmas": ["sched", 0],
                               "latent_image": [main_node, 1]}}
    g["dec"] = {"class_type": "VAEDecode",
                "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    g["adec"] = {"class_type": "VAEDecodeAudio",
                 "inputs": {"samples": ["sampler", 0], "vae": ["avae", 0]}}


def build_variant(cfg: dict, seed: int, out_prefix: str) -> dict:
    """Controlled API graph from a variant config dict."""
    g = {}
    g["unet"] = {"class_type": "UNETLoader",
                 "inputs": {"unet_name": UNET, "weight_dtype": "default"}}
    g["clip"] = {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": CLIP, "type": "minimax",
                            "device": "default"}}
    g["vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": VAE}}
    g["avae"] = {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}}
    g.update(_model_chain("unet"))
    g["refboost"] = {"class_type": "H3RefBoostChar",
                     "inputs": {"model": ["cache", 0], **cfg["refboost"]}}
    boost_model = "refboost"
    if cfg.get("refboostv"):
        g["refboostv"] = {"class_type": "H3RefBoostV",
                          "inputs": {"model": ["refboost", 0],
                                     **cfg["refboostv"]}}
        boost_model = "refboostv"
    g["spectrum"] = {"class_type": "SpectrumApplyMiniMaxH3",
                     "inputs": {"model": [boost_model, 0], "enabled": True,
                                "blend_weight": 0.5, "degree": 4,
                                "ridge_lambda": 0.1, "window_size": 2,
                                "flex_window": 0.75, "warmup_steps": 5,
                                "tail_actual_steps": 1, "max_history": 8,
                                "debug": False}}

    # --- refs ---
    main_inputs = {
        "clip": ["clip", 0], "vae": ["vae", 0], "audio_vae": ["avae", 0],
        "prompt": cfg["prompt"],
        "length": snap_length(cfg.get("duration", 10.0)),
        "ref_image_size": cfg.get("ref_image_size", "max"),
    }
    if cfg.get("aspect_source"):
        # Reproduce the real workflow: aspect detector on the source image
        # drives ResolutionSelector (0.3MP, divisor 32) -> width/height.
        g.update(_load_image("asrc", cfg["aspect_source"]))
        g["adet"] = {"class_type": "H3AspectRatioDetector",
                     "inputs": {"image": ["asrc", 0], "aspect_ratio": ""}}
        g["rsel"] = {"class_type": "ResolutionSelector",
                     "inputs": {"aspect_ratio": ["adet", 0],
                                "megapixels": 0.3, "divisor": 32,
                                "multiple": 8}}
        main_inputs["width"] = ["rsel", 0]
        main_inputs["height"] = ["rsel", 1]
    else:
        main_inputs["width"] = cfg.get("width", 1344)
        main_inputs["height"] = cfg.get("height", 768)
    ref_offset = 0
    if cfg.get("scene_ref"):
        # SCENEIMG: the manga page becomes ref_image_0 (scene), character
        # refs shift to ref_image_1..N to match the scene-image prompt.
        g.update(_load_image("sref", cfg["scene_ref"]))
        main_inputs["ref_images.ref_image_0"] = ["sref", 0]
        ref_offset = 1
    for i, fname in enumerate(cfg["refs"]):
        rid = f"ref{i}"
        g.update(_load_image(rid, fname))
        main_inputs[f"ref_images.ref_image_{i + ref_offset}"] = [rid, 0]

    if cfg.get("video_ref"):
        vfile, vcount = cfg["video_ref"]
        g.update(_load_image("vsrc", vfile))
        g["refvid"] = {"class_type": "H3ImageToRefVideo",
                       "inputs": {"image": ["vsrc", 0], "frame_count": vcount}}
        main_inputs["ref_videos.ref_video_0"] = ["refvid", 0]

    g["main"] = {"class_type": "H3ReferenceToVideo", "inputs": main_inputs}

    _sampler(g, "main", seed)

    # VSR bypass for experiments (identity decided at base gen).
    # meta_batch/vae are VHS-specific object inputs — omit (empty strings crash).
    g["vhs"] = {"class_type": "VHS_VideoCombine",
                "inputs": {"images": ["dec", 0], "audio": ["adec", 0],
                           "frame_rate": 24, "loop_count": 0,
                           "filename_prefix": out_prefix,
                           "format": "video/h264-mp4", "pingpong": False,
                           "save_output": True, "pix_fmt": "yuv420p", "crf": 19,
                           "save_metadata": True, "trim_to_audio": False}}
    return g


OUTPUT_CLASSES = {"SaveImage", "SaveAnimatedWEBP", "PreviewImage",
                  "ShowText|pysssss", "VHS_VideoCombine", "SaveAudio",
                  "SaveVideo", "VHS_VideoCombine"}


CONTROL_AFTER = {"fixed", "increment", "decrement", "randomize"}


def _put(inputs: dict, name: str, value):
    """ComfyUI API prompt keys are FLAT — dict inputs use dotted names
    ('ref_videos.ref_video_0', 'resize_type.scale') and the executor groups
    them into the dict input at call time. No nesting here."""
    inputs[name] = value


def convert_workflow(wf_path: str) -> dict:
    """Faithful workflow-JSON -> API prompt conversion (frontend convention).

    widgets_values align with the widget-capable inputs (those carrying a
    "widget" key) in input order, linked or not. control_after_generate stores
    an extra control string ("randomize"...) right after an INT/FLOAT widget's
    value — skipped here. "upload" is a frontend-only LoadImage widget.
    Orphaned branches (no path to an output node) are pruned — the UI only
    executes output-reachable nodes, and some orphans (empty H3BatchImages)
    would crash if force-executed.
    """
    with open(wf_path) as f:
        wf = json.load(f)
    nodes = {n["id"]: n for n in wf["nodes"]}
    links = {l[0]: l for l in wf.get("links", [])}

    # reachability from output-class nodes
    consumed = {l[1] for l in links.values()}
    needed = {nid for nid, n in nodes.items()
              if n["type"] in OUTPUT_CLASSES}
    changed = True
    while changed:
        changed = False
        for l in links.values():
            _, src, _, dst, _, _ = l
            if dst in needed and src not in needed:
                needed.add(src)
                changed = True
    # a node with no outgoing links that is NOT an output is dead — prune it
    needed = {nid for nid in needed
              if nid in nodes and (nid in consumed or nodes[nid]["type"] in OUTPUT_CLASSES)}

    prompt = {}
    for nid, n in nodes.items():
        if nid not in needed:
            continue
        inputs = {}
        wv_raw = n.get("widgets_values")
        if isinstance(wv_raw, dict):
            # dict-widget node (VHS, rgthree, Bernini): values keyed by name
            wv, wi = None, 0
        else:
            wv, wi = list(wv_raw or []), 0
        for inp in n.get("inputs", []):
            name = inp["name"]
            is_widget = "widget" in inp
            linked = inp.get("link") is not None
            if is_widget:
                if wv is not None:
                    val = wv[wi] if wi < len(wv) else None
                    wi += 1
                else:
                    val = wv_raw.get(name)
                if linked:
                    l = links[inp["link"]]
                    _put(inputs, name, [str(l[1]), l[2]])
                elif name != "upload":
                    _put(inputs, name, val)
                # control_after_generate: skip the control string after INT/FLOAT
                if wv is not None and inp["type"] in ("INT", "FLOAT") \
                        and wi < len(wv) and isinstance(wv[wi], str) \
                        and wv[wi] in CONTROL_AFTER:
                    wi += 1
            elif linked:
                l = links[inp["link"]]
                _put(inputs, name, [str(l[1]), l[2]])
            # else: optional link-only input, unconnected -> omit
        prompt[str(nid)] = {"class_type": n["type"], "inputs": inputs}
    return prompt


# ═══════════════════════════════════════════════════════════════════════
# ComfyUI API
# ═══════════════════════════════════════════════════════════════════════

def queue_prompt(prompt: dict):
    data = json.dumps({"prompt": prompt}).encode()
    req = urllib.request.Request(f"{COMFYUI_HOST}/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()[:2000]}")
        return None


def wait_prompt(pid: str, timeout: int = 1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(
                    f"{COMFYUI_HOST}/history/{pid}", timeout=10) as r:
                h = json.loads(r.read().decode())
            if pid in h:
                st = h[pid].get("status", {})
                if st.get("completed"):
                    return h[pid], None
                if st.get("status_str") == "error" or st.get("error"):
                    return h[pid], st.get("error") or st.get("messages")
            time.sleep(3)
        except Exception as e:
            print(f"  poll err {e}")
            time.sleep(3)
    return None, "timeout"


def extract_frames(mp4: Path, outdir: Path, n: int = 4):
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                "format=duration", "-of", "json", str(mp4)],
                               capture_output=True, text=True, timeout=30)
        dur = float(json.loads(probe.stdout)["format"]["duration"])
    except Exception:
        dur = 10.0
    for i in range(n):
        t = dur * (i + 0.5) / n
        out = outdir / f"f{i}_{t:04.1f}s.jpg"
        subprocess.run(["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", str(mp4),
                        "-frames:v", "1", "-q:v", "3", str(out)],
                       capture_output=True, timeout=60)
    return sorted(outdir.glob("*.jpg"))


# ═══════════════════════════════════════════════════════════════════════
# Variants
# ═══════════════════════════════════════════════════════════════════════

VARIANTS = {
    "BASE": dict(refs=REFS_CURRENT, video_ref=(SOURCE_IMG, 5),
                 refboost=dict(REFBOOST_BASE),
                 ref_image_size="max", aspect_source=SOURCE_IMG,
                 prompt_file=PROMPT_FILE),
    "NOVID": dict(refs=REFS_CURRENT, video_ref=None,
                  refboost=dict(REFBOOST_BASE),
                  ref_image_size="max", aspect_source=SOURCE_IMG,
                  prompt_file=PROMPT_FILE_NOVID),
    "SRC00": dict(refs=REFS_CURRENT, video_ref=(SOURCE_IMG, 5),
                  refboost={**REFBOOST_BASE, "source_scale": 0.0},
                  ref_image_size="max", aspect_source=SOURCE_IMG,
                  prompt_file=PROMPT_FILE),
    "SRC04": dict(refs=REFS_CURRENT, video_ref=(SOURCE_IMG, 5),
                  refboost={**REFBOOST_BASE, "source_scale": 0.4},
                  ref_image_size="max", aspect_source=SOURCE_IMG,
                  prompt_file=PROMPT_FILE),
    "VBOOST": dict(refs=REFS_CURRENT, video_ref=(SOURCE_IMG, 5),
                   refboost=dict(REFBOOST_BASE),
                   refboostv=dict(enabled=True, ref_slots="all",
                                  ref_v_boost=1.3, source_v_scale=0.0,
                                  schedule="linear", step_threshold=0.5,
                                  debug=False),
                   ref_image_size="max", aspect_source=SOURCE_IMG,
                   prompt_file=PROMPT_FILE),
    "VBOOST08": dict(refs=REFS_CURRENT, video_ref=(SOURCE_IMG, 5),
                     refboost=dict(REFBOOST_BASE),
                     refboostv=dict(enabled=True, ref_slots="all",
                                    ref_v_boost=1.3, source_v_scale=0.8,
                                    schedule="linear", step_threshold=0.5,
                                    debug=False),
                     ref_image_size="max", aspect_source=SOURCE_IMG,
                     prompt_file=PROMPT_FILE),
    "VB_CHARONLY": dict(refs=REFS_CURRENT, video_ref=(SOURCE_IMG, 5),
                        refboost=dict(REFBOOST_BASE),
                        refboostv=dict(enabled=True, ref_slots="all",
                                       ref_v_boost=1.3, source_v_scale=1.0,
                                       schedule="linear", step_threshold=0.5,
                                       debug=False),
                        ref_image_size="max", aspect_source=SOURCE_IMG,
                        prompt_file=PROMPT_FILE),
    "VB_VIDEOZERO": dict(refs=REFS_CURRENT, video_ref=(SOURCE_IMG, 5),
                         refboost=dict(REFBOOST_BASE),
                         refboostv=dict(enabled=True, ref_slots="all",
                                        ref_v_boost=1.0, source_v_scale=0.0,
                                        schedule="linear", step_threshold=0.5,
                                        debug=False),
                         ref_image_size="max", aspect_source=SOURCE_IMG,
                         prompt_file=PROMPT_FILE),
    "SRC_NOISE": dict(refs=REFS_CURRENT, video_ref=(SOURCE_IMG, 5),
                      refboost={**REFBOOST_BASE, "source_noise": 0.8,
                                "source_scale": 1.0},
                      ref_image_size="max", aspect_source=SOURCE_IMG,
                      prompt_file=PROMPT_FILE),
    "VBOOST_CONST": dict(refs=REFS_CURRENT, video_ref=(SOURCE_IMG, 5),
                         refboost=dict(REFBOOST_BASE),
                         refboostv=dict(enabled=True, ref_slots="all",
                                        ref_v_boost=1.3, source_v_scale=0.3,
                                        schedule="constant",
                                        step_threshold=0.5, debug=False),
                         ref_image_size="max", aspect_source=SOURCE_IMG,
                         prompt_file=PROMPT_FILE),
    "VBOOST_EARLY": dict(refs=REFS_CURRENT, video_ref=(SOURCE_IMG, 5),
                         refboost=dict(REFBOOST_BASE),
                         refboostv=dict(enabled=True, ref_slots="all",
                                        ref_v_boost=1.3, source_v_scale=0.0,
                                        schedule="early_linear",
                                        step_threshold=0.5, debug=False),
                         ref_image_size="max", aspect_source=SOURCE_IMG,
                         prompt_file=PROMPT_FILE),
    "SCENEIMG": dict(refs=REFS_CURRENT, video_ref=None,
                     scene_ref=SOURCE_IMG,
                     refboost=dict(REFBOOST_BASE),
                     ref_image_size="max", aspect_source=SOURCE_IMG,
                     prompt_file=Path(__file__).parent / "ri2v_prompt_frozen_sceneimg.txt"),
    "ALICIO3": dict(refs=REFS_ALICLO3, video_ref=(SOURCE_IMG, 5),
                    refboost=dict(REFBOOST_BASE),
                    ref_image_size="max", aspect_source=SOURCE_IMG,
                    prompt_file=Path(__file__).parent / "ri2v_prompt_frozen_3ref.txt"),
    "SHEET1": dict(refs=REFS_SHEET, video_ref=(SOURCE_IMG, 5),
                   refboost=dict(REFBOOST_BASE),
                   ref_image_size="max", aspect_source=SOURCE_IMG,
                   prompt_file=Path(__file__).parent / "ri2v_prompt_frozen_sheet.txt"),
    "MATCH": dict(refs=REFS_CURRENT, video_ref=(SOURCE_IMG, 5),
                  refboost=dict(REFBOOST_BASE),
                  ref_image_size="match", aspect_source=SOURCE_IMG,
                  prompt_file=PROMPT_FILE),
    "PROTON": dict(refs=REFS_CURRENT, video_ref=(SOURCE_IMG, 5),
                   refboost={**REFBOOST_BASE, "protect_scene": True},
                   ref_image_size="max", aspect_source=SOURCE_IMG,
                   prompt_file=PROMPT_FILE),
    "NOBOOST": dict(refs=REFS_CURRENT, video_ref=(SOURCE_IMG, 5),
                    refboost={**REFBOOST_BASE, "enabled": False},
                    ref_image_size="max", aspect_source=SOURCE_IMG,
                    prompt_file=PROMPT_FILE),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="/tmp/ri2v")
    ap.add_argument("--enhance", action="store_true",
                    help="Run the faithful converted workflow (enhancer wired)")
    ap.add_argument("--workflow", default=WORKFLOW)
    args = ap.parse_args()

    if args.list:
        for k, v in VARIANTS.items():
            print(f"{k}: refs={[p.split('.')[0] for p in v['refs']]} "
                  f"video={v['video_ref'] and v['video_ref'][0]} "
                  f"boost={v['refboost']['source_scale'] if v['refboost'].get('enabled') else 'OFF'}")
        return

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = out_root / "manifest.jsonl"

    if args.enhance:
        prompt = convert_workflow(args.workflow)
        tag = "ENH"
    else:
        if args.variant not in VARIANTS:
            raise SystemExit(f"unknown variant {args.variant}")
        cfg = dict(VARIANTS[args.variant])
        cfg["prompt"] = _load_prompt(cfg["prompt_file"])
        tag = args.variant

    run_dir = out_root / f"{tag}_s{args.seed}"
    prefix = f"Ri2vExp/{tag}_{args.seed}"
    if not args.enhance:
        prompt = build_variant(cfg, args.seed, prefix)
    print(f"[{tag}] seed={args.seed} -> {run_dir}")
    if not args.enhance:
        print(f"  refs={cfg['refs']} video={cfg['video_ref']} "
              f"size={cfg['ref_image_size']} "
              f"boost_src={cfg['refboost'].get('source_scale')} "
              f"enabled={cfg['refboost'].get('enabled')}")

    t0 = time.time()
    res = queue_prompt(prompt)
    if not res or "prompt_id" not in res:
        print("QUEUE FAILED"); sys.exit(1)
    pid = res["prompt_id"]
    print(f"  queued {pid}")
    history, err = wait_prompt(pid)
    dur = time.time() - t0
    if err:
        print(f"  ERROR: {err}")
        with open(run_dir.with_suffix(".err"), "w") as f:
            json.dump(history, f, indent=2)
        sys.exit(1)
    h = history  # wait_prompt returns the history ENTRY for this prompt
    print(f"  done in {dur/60:.1f} min")

    # outputs from history — filenames carry a subfolder field
    outputs = h.get("outputs", {})
    files = []
    for nid, o in outputs.items():
        for key in ("images", "gifs", "audio", "videos"):
            for item in o.get(key, []):
                sub = item.get("subfolder") or ""
                files.append(f"{sub}/{item['filename']}" if sub
                             else item["filename"])
        if "text" in o:
            for t in o["text"]:
                print(f"  TEXT[{nid}]: {t[:400]}")
                (out_root / f"{tag}_s{args.seed}.prompt.txt").write_text(t)
    mp4s = [f for f in files if f.endswith((".mp4", ".webm"))]
    print(f"  files: {files}")
    manifest_row = {"tag": tag, "seed": args.seed, "prompt_id": pid,
                    "ok": True, "minutes": round(dur / 60, 1),
                    "files": files, "run_dir": str(run_dir)}
    if mp4s:
        mp4 = Path(COMFYUI_OUTPUT) / mp4s[0]
        frames = extract_frames(mp4, run_dir)
        manifest_row["frames"] = [str(f) for f in frames]
    with open(manifest, "a") as f:
        f.write(json.dumps(manifest_row) + "\n")


if __name__ == "__main__":
    main()
