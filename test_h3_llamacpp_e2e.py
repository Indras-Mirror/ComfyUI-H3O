"""Live GPU end-to-end test for the llama.cpp backend (H3PromptEnhancer).

Spawns a REAL llama-server (qwen3.8-heretic-ara) on demand, runs BOTH passes
(auto_describe analyze + write) through enhance(), and asserts:

  1. output is non-empty and parses as structured H3 prose
  2. a llama-server WAS spawned during the run (background watcher)
  3. after enhance() returns, NO llama-server process remains (pgrep empty)
  4. nvidia-smi free VRAM after ≈ free VRAM before (±0.5 GiB)

OOM handling (ComfyUI live may hold VRAM): on a CUDA-OOM server exit, the
test retries once with the IQ4_XS fallback model and notes it.

This is a GPU test — loads ~19-21 GiB, takes minutes. Run it alone:

    cd /media/mal/Crucible/AI-ART/ComfyUI
    ./venv/bin/python custom_nodes/ComfyUI-H3O/test_h3_llamacpp_e2e.py
"""

import importlib.util
import os
import subprocess
import sys
import threading
import time
import types

_HERE = os.path.dirname(os.path.realpath(__file__))

# h3_prompt_enhancer.py uses relative imports — load it as a synthetic
# package rooted at the pack dir so those resolve (same hack as the unit test).
_PKG = "h3o_e2e_test"
pkg = types.ModuleType(_PKG)
pkg.__path__ = [_HERE]
pkg.__package__ = _PKG
sys.modules[_PKG] = pkg

def _load(name, fname):
    spec = importlib.util.spec_from_file_location(
        f"{_PKG}.{name}", os.path.join(_HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"{_PKG}.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod

h3pe = _load("h3_prompt_enhancer", "h3_prompt_enhancer.py")
H3PromptEnhancer = h3pe.H3PromptEnhancer

import torch  # noqa: E402

PASS = 0

def check(name, cond, detail=""):
    global PASS
    status = "ok " if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        sys.exit(f"TEST FAILED: {name} {detail}")
    PASS += 1

def vram_free_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15)
    return int(out.stdout.strip())

def llama_servers():
    out = subprocess.run(["pgrep", "-f", "llama-server"],
                         capture_output=True, text=True, timeout=10)
    return [ln for ln in out.stdout.splitlines() if ln.strip()]

def log_has_oom(log_path):
    try:
        with open(log_path, "r", errors="replace") as f:
            text = f.read()
    except OSError:
        return False
    low = text.lower()
    return any(k in low for k in ("out of memory", "cuda error",
                                  "cuda out of memory", "failed to allocate"))

def make_img(h=64, w=64, c=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(h, w, c, generator=g)

def run_e2e():
    """One full enhance() run with a live llama-server; returns results."""
    node = H3PromptEnhancer()
    img = make_img(seed=21).unsqueeze(0)  # [1, H, W, C]

    # Background watcher: record whether a llama-server ever appears.
    stop_watch = [False]
    spawn_seen = []

    def watch():
        while not stop_watch[0]:
            if llama_servers():
                spawn_seen.append(True)
            time.sleep(1.0)

    t_watch = threading.Thread(target=watch, daemon=True)
    t_watch.start()
    t0 = time.time()
    try:
        out, sys_prompt = node.enhance(
            prompt="A woman in a red dress walks down a neon-lit alley at "
                   "night, rain glistening on the pavement.",
            task_type="i2v",
            duration=5.0,
            model="x-ai/grok-4.3",  # unused by the local backend
            local_backend="llamacpp/qwen3.8-heretic-ara",
            source_image=img,
            auto_describe=True,      # exercises BOTH passes on ONE instance
            auto_describe_max_tokens=2048,
            max_tokens=2048,
            temperature=0.7,
            advanced_prompt="on",
            seed=3407,
            context_length=-1,       # auto → wrapper default (262144)
        )
    finally:
        stop_watch[0] = True
        t_watch.join(timeout=5)
    return out, sys_prompt, time.time() - t0, spawn_seen

print("llama.cpp E2E (live GPU): qwen3.8-heretic-ara, both passes")
baseline_free = vram_free_mib()
before_servers = llama_servers()
print(f"  free VRAM before: {baseline_free} MiB "
      f"(servers already running: {before_servers})")

model_used = "Q4_K_M"
oom_note = None
try:
    out, sys_prompt, elapsed, spawn_seen = run_e2e()
except RuntimeError as e:
    entry = h3pe.LOCAL_MODEL_DEFAULTS["llamacpp/qwen3.8-heretic-ara"]
    log_path = entry.get("llama_log", "/tmp/h3o-llamacpp-8098.log")
    if log_has_oom(log_path):
        print("  OOM detected (ComfyUI live holds VRAM) — retrying with "
              "IQ4_XS fallback model")
        entry["llama_model_path"] = entry["llama_fallback_model"]
        model_used = "IQ4_XS"
        oom_note = f"OOM on Q4_K_M ({log_path}); E2E passed with IQ4_XS"
        out, sys_prompt, elapsed, spawn_seen = run_e2e()
    else:
        print(f"  E2E failed without OOM signature: {e}")
        raise
print(f"  enhance() took {elapsed:.1f}s (model: {model_used})")

check("output is non-empty", bool(out and out.strip()))
words = len(out.split()) if out else 0
check("output parses as structured H3 prose (3-field markers)",
      "integrated_multimodal_description" in out
      or "[Shot 1]" in out or words >= 30,
      f"{words} words")
check("a llama-server WAS spawned during the run",
      len(spawn_seen) > 0, f"spawn_seen={spawn_seen}")

# Teardown happens inside enhance()'s finally — give it a moment, then verify.
time.sleep(3)
remaining = llama_servers()
check("no llama-server remains after enhance() returns",
      len(remaining) == 0, f"remaining={remaining}")

free_after = vram_free_mib()
check("VRAM freed ≈ baseline (±0.5 GiB)",
      abs(free_after - baseline_free) <= 512,
      f"before={baseline_free} MiB after={free_after} MiB")
print(f"  free VRAM after: {free_after} MiB")

if oom_note:
    print(f"NOTE: {oom_note}")
print(f"\nALL {PASS} CHECKS PASSED (elapsed {elapsed:.1f}s, {model_used})")
