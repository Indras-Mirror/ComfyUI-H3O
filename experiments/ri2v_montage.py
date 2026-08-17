#!/usr/bin/env python3
"""Build per-run comparison montages: ref character + source + output frames.

Each run dir (/tmp/ri2v/<TAG>_s<SEED>/) gets a montage.jpg:
  row 1: ref images (aliclo set / character sheet) — the intended character
  row 2: source (manga page) + the run's 4 extracted frames
Vision-scoring one montage scores a whole run in a single call.

Usage: python3 ri2v_montage.py /tmp/ri2v
"""

import subprocess
import sys
from pathlib import Path

COMFYUI_INPUT = Path("/media/mal/Crucible/AI-ART/ComfyUI/input")
REFS = ["aliclo2.png", "aliclo4.png", "Aliclo3.png", "2025-08-12_17-39_2.png"]
SHEET = [Path("/media/mal/Crucible/AI-ART/ComfyUI/output/CharacterSheets/H3Gen_00001_.png")]
SOURCE = "0112.jpg"


def montage(run_dir: Path, tag: str):
    frames = sorted(run_dir.glob("f*.jpg"))
    if not frames:
        return None
    cells = []
    # row 1: refs
    if tag.startswith("SHEET"):
        cells += [str(p) for p in SHEET]
    else:
        cells += [str(COMFYUI_INPUT / r) for r in REFS]
    # row 2: source + frames
    cells += [str(COMFYUI_INPUT / SOURCE)] + [str(f) for f in frames]
    out = run_dir / "montage.jpg"
    cmd = ["montage", "-label", "%f", "-geometry", "360x", "-tile",
           f"{len(cells)}x1", *cells, str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return out


def main():
    root = Path(sys.argv[1])
    made = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        m = montage(run_dir, run_dir.name.split("_")[0])
        if m:
            made.append(str(m))
    print("\n".join(made))


if __name__ == "__main__":
    main()
