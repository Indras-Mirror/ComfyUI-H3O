"""Standalone mock-API tests for H3PromptEnhancer's rv2v batch path.

Verifies the same_subject=True path emits ONE unified <Subject 1> with
"ADDITIONAL VIEW of the SAME <Subject 1>" for pictures 2+, no literal "{i+1}"
artifacts, and the "SAME SUBJECT" merge rule. Also verifies same_subject=False
still emits per-image subject definitions.

IMPORTANT: reference images are anchored by their <Picture N> labels — the ONLY
image labels the H3 rendering model sees (comfy/text_encoders/minimax.py). Every
identity statement must name the exact <Picture i> that carries it; <Subject N>
is the speaking character name, never a bare unbindable anchor.

Usage:
    cd /media/mal/Crucible/AI-ART/ComfyUI
    ./venv/bin/python custom_nodes/ComfyUI-H3O/test_h3_prompt_enhancer.py
"""

import importlib.util
import os
import sys
import types

_HERE = os.path.dirname(os.path.realpath(__file__))

# h3_prompt_enhancer.py uses relative imports (from .h3o_shared import ...).
# Load it as a synthetic package rooted at the pack dir so those resolve.
_PKG = "h3o_prompt_test"
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

def make_img(h=64, w=64, c=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(h, w, c, generator=g)

def run_enhance(task_type="rv2v", same_subject=False, images_batch=None,
                source_video=None, source_image=None, auto_describe=False,
                prompt_override=None):
    node = H3PromptEnhancer()
    captured = {}

    def fake_caller(api_key, model, system_prompt, user_content, *args, **kwargs):
        captured["api_key"] = api_key
        captured["model"] = model
        captured["system_prompt"] = system_prompt
        captured["content"] = user_content
        # Last content item is the assembled user message text.
        captured["user_text"] = user_content[-1]["text"]
        return "MOCK OUTPUT"

    # Patch the module-level binding so enhance() hits the mock, not OpenRouter.
    h3pe._call_openrouter = fake_caller

    out, sys_prompt = node.enhance(
        prompt=(prompt_override
                if prompt_override is not None
                else "A woman walks into the scene."),
        task_type=task_type,
        duration=5.0,
        model="x-ai/grok-4.3",
        api_key="test-key",
        images_batch=images_batch,
        source_video=source_video,
        source_image=source_image,
        same_subject=same_subject,
        auto_describe=auto_describe,
        advanced_prompt="on",
    )
    captured["returned_h3_prompt"] = out
    captured["returned_system_prompt"] = sys_prompt
    return captured

# ── 1. rv2v + same_subject=True via images_batch ────────────────────────────
print("rv2v same_subject=True (images_batch)")
batch = torch.stack([make_img(seed=1), make_img(seed=2), make_img(seed=3)])
src_video = torch.stack([make_img(seed=10 + i) for i in range(5)])
cap = run_enhance(task_type="rv2v", same_subject=True,
                  images_batch=batch, source_video=src_video)
t = cap["user_text"]

check("api_key passed through", cap["api_key"] == "test-key")
check("model passed through", cap["model"] == "x-ai/grok-4.3")
check("mock output returned", cap["returned_h3_prompt"] == "MOCK OUTPUT")
check("advanced rv2v system prompt", "RV2V" in cap["system_prompt"]
      and "REFERENCE-IMAGE-AS-VIDEO" in cap["system_prompt"])

check("exactly one '<Subject 1> ... defines' line",
      t.count("defines <Subject 1>") == 1, f"count={t.count('defines <Subject 1>')}")
# The proven recipe does NOT name the original as its own subject — the original
# is described only as "the person being replaced" inside <Video 1>'s definition.
# <Subject 2> should NOT appear in the rv2v output.
check("original character NOT named as <Subject 2>",
      "<Subject 2>" not in t)
check("no <Subject 3>", "<Subject 3>" not in t)
check("no literal {{i+1}} artifact", "{i+1}" not in t)
check("no naked {{i}} interpolation", "{i}" not in t
      and "{i+1}" not in t and "{i}" not in t)
check("no 'Subject {i+1}' from non-f segment",
      "Subject {i+1}" not in t and "Subject {i}" not in t)

# image refs must be anchored with their <Picture N> labels (the ONLY image
# labels the H3 renderer emits — comfy/text_encoders/minimax.py)
check("same-subject: <Picture 1> anchors the unified subject",
      "<Picture 1>" in t and "defines <Subject 1>" in t
      and "full appearance" in t.lower())

# pictures 2+ must both be described as additional views of SAME <Subject 1>
add_views = t.count("ADDITIONAL VIEW of the SAME")
check("ADDITIONAL VIEW x2 (picture 2, 3)",
      add_views == 2, f"count={add_views}")
check("picture 2 line names <Subject 1>",
      "Reference image <Picture 2> is an ADDITIONAL VIEW of the SAME <Subject 1>" in t)
check("picture 3 line names <Subject 1>",
      "Reference image <Picture 3> is an ADDITIONAL VIEW of the SAME <Subject 1>" in t)
check("same-subject merge rule present", "SAME SUBJECT:" in t
      and "ONE unified subject" in t)
check("no SCENE/REPLACEMENT per-image rule when same_subject",
      "SCENE/REPLACEMENT RULE" not in t)
check("same-subject template: never split into per-image subjects",
      "NEVER split into" in t and "per-image subjects" in t
      and "every image is a view of the same single person" in t)
check("same-subject template: attribute_transfer replacement",
      "attribute_transfer" in t and "transfer face, body, hair" in t)
check("scene ref-video block present", "static reference video (<Video 1>) is attached" in t)

# ── 2. rv2v + same_subject=False must still per-image-subject ───────────────
print("\nrv2v same_subject=False (images_batch)")
cap2 = run_enhance(task_type="rv2v", same_subject=False,
                   images_batch=batch, source_video=src_video)
t2 = cap2["user_text"]

check("<Subject 1> anchored to <Picture 1>",
      "defines <Subject 1>" in t2 and "<Picture 1>" in t2
      and "from <Picture 1>" in t2)
check("<Subject 2> anchored to <Picture 2>",
      "defines <Subject 2>" in t2 and "<Picture 2>" in t2)
check("<Subject 3> anchored to <Picture 3>",
      "defines <Subject 3>" in t2 and "<Picture 3>" in t2)
check("no ADDITIONAL VIEW in separate mode",
      "ADDITIONAL VIEW of the SAME" not in t2)
check("SCENE/REPLACEMENT per-image rule present",
      "SCENE/REPLACEMENT RULE" in t2)
check("separate template: replacement character rule",
      "anchors a REPLACEMENT character" in t2 or "REPLACEMENT character" in t2)
check("separate template: per-image attribute_transfer",
      "attribute_transfer" in t2 and "transfer face, body, hair" in t2)
check("no literal {{i+1}} in separate mode", "{i+1}" not in t2)

# ── 3. no images_batch / individual slots (same_subject still unified) ──────
print("\nrv2v same_subject=True (individual slots)")
cap3 = run_enhance(task_type="rv2v", same_subject=True,
                   images_batch=None, source_video=src_video)
t3 = cap3["user_text"]
check("no images -> no subject defs", "defines <Subject 1>" not in t3)
check("no refs -> same-subject rule not injected", "SAME SUBJECT:" not in t3)

# ── 4. ri2v — replacement-preservation recipe, r2v template, no original ───
print("\nri2v (source video + images_batch)")
cap4 = run_enhance(task_type="ri2v", same_subject=True,
                   images_batch=batch, source_video=src_video)
t4 = cap4["user_text"]
check("ri2v uses the proven R2V template (reference-guided)",
      "reference-guided generation (R2V)" in t4
      and "RV2V" not in t4.split("Rules for reference mode")[0])
check("ri2v uses the FULL-REFERENCE system prompt (not RV2V)",
      "FULL-REFERENCE" in cap4["system_prompt"]
      and "REFERENCE-IMAGE-AS-VIDEO" not in cap4["system_prompt"])
check("ri2v source video labeled scene/structure, not a character subject",
      "<Video 1> supplies the" in t4
      and "do NOT define that original character as its own subject" in t4)
check("ri2v retention rule: replacement attribute_transfer, picture weak_reference",
      "attribute_transfer" in t4 and "weak_reference" in t4
      and "CHARACTER-SWAP RETENTION" in t4)
check("ri2v uses attribute_transfer for the replacement",
      "attribute_transfer" in t4 and "transfer" in t4.lower())
check("ri2v does NOT define the original as <Subject 2>",
      "<Subject 2>" not in t4 or "is the original" not in t4)

# ── 5. ri2v scene-as-image — source_image as scene, no source_video ────────
print("\nri2v scene-as-image (source_image + images_batch, same_subject=True)")
scene_img = make_img(seed=99).unsqueeze(0)  # [1, H, W, C] like ComfyUI IMAGE
cap5 = run_enhance(task_type="ri2v", same_subject=True,
                   images_batch=batch, source_video=None, source_image=scene_img)
t5 = cap5["user_text"]
n_images = sum(1 for c in cap5["content"] if c.get("type") == "image_url")
check("scene-as-image: 4 images sent to LLM (1 scene + 3 batch)",
      n_images == 4, f"got {n_images}")
check("scene-as-image: <Picture 1> is SCENE",
      "<Picture 1> provides the SCENE" in t5)
check("scene-as-image: no <Video 1>",
      "<Video 1>" not in t5)
check("scene-as-image: <Picture 2> defines <Subject 1> (primary view)",
      "<Picture 2>" in t5 and "defines" in t5
      and "<Subject 1>" in t5 and "REPLACEMENT" in t5)
check("scene-as-image: <Picture 3> is ADDITIONAL VIEW",
      "Reference image <Picture 3> is an ADDITIONAL VIEW" in t5)
check("scene-as-image: same-subject rule mentions <Picture 1> is scene",
      "<Picture 1> is the SCENE" in t5
      and "NOT a character" in t5)
check("scene-as-image: retention rule present",
      "CHARACTER-SWAP RETENTION" in t5)
check("scene-as-image: scene fully_preserved, character attribute_transfer",
      "fully_preserved" in t5 and "attribute_transfer" in t5)

# ── 6. ri2v scene-as-image, same_subject=False ────────────────────────────
print("\nri2v scene-as-image (source_image + images_batch, same_subject=False)")
cap6 = run_enhance(task_type="ri2v", same_subject=False,
                   images_batch=batch, source_video=None, source_image=scene_img)
t6 = cap6["user_text"]
check("scene-as-image separate: <Picture 1> is SCENE",
      "<Picture 1> provides the SCENE" in t6)
check("scene-as-image separate: <Picture 2> → <Subject 1>",
      "defines the" in t6 and "<Subject 1>" in t6 and "<Picture 2>" in t6)
check("scene-as-image separate: <Picture 3> → <Subject 2>",
      "<Subject 2>" in t6 and "<Picture 3>" in t6)
check("scene-as-image separate: subjects reference <Picture 1>'s scene",
      "<Picture 1>'s" in t6 and "scene" in t6)

# ── 7. H3PromptEnhancerPlus — image passthrough + enhanced rules ──────────
print("\nH3PromptEnhancerPlus (rv2v same_subject=True, enhanced_rules=True)")

plus_mod = _load("h3_prompt_enhancer_plus", "h3_prompt_enhancer_plus.py")
H3PromptEnhancerPlus = plus_mod.H3PromptEnhancerPlus

def run_enhance_plus(task_type="rv2v", same_subject=False, images_batch=None,
                     source_video=None, source_image=None, auto_describe=False,
                     enhanced_rules=True):
    node = H3PromptEnhancerPlus()
    captured = {}

    def fake_caller(api_key, model, system_prompt, user_content, *args, **kwargs):
        captured["api_key"] = api_key
        captured["model"] = model
        captured["system_prompt"] = system_prompt
        captured["content"] = user_content
        captured["user_text"] = user_content[-1]["text"]
        return "MOCK OUTPUT"

    h3pe._call_openrouter = fake_caller

    h3_prompt, sys_prompt, ref_images_out = node.enhance_plus(
        prompt="A woman walks into the scene.",
        task_type=task_type,
        duration=5.0,
        model="x-ai/grok-4.3",
        api_key="test-key",
        images_batch=images_batch,
        source_video=source_video,
        source_image=source_image,
        same_subject=same_subject,
        auto_describe=auto_describe,
        advanced_prompt="on",
        enhanced_rules=enhanced_rules,
    )
    captured["returned_h3_prompt"] = h3_prompt
    captured["returned_system_prompt"] = sys_prompt
    captured["ref_images_out"] = ref_images_out
    return captured


# 7a. Plus with enhanced rules ON
cap7 = run_enhance_plus(task_type="rv2v", same_subject=True,
                        images_batch=batch, source_video=src_video,
                        enhanced_rules=True)
check("plus: returns 3 outputs (h3_prompt, system_prompt, ref_images_out)",
      cap7["returned_h3_prompt"] == "MOCK OUTPUT"
      and isinstance(cap7["returned_system_prompt"], str)
      and cap7["ref_images_out"] is not None)
check("plus: enhanced rules in system prompt",
      "ENHANCED DIALOGUE RULES" in cap7["returned_system_prompt"]
      and "ENHANCED AUDIO RULES" in cap7["returned_system_prompt"]
      and "ENHANCED CONSISTENCY RULES" in cap7["returned_system_prompt"])
check("plus: ref_images_out has 3 images (batch of 3)",
      cap7["ref_images_out"].shape[0] == 3,
      f"shape={cap7['ref_images_out'].shape}")

# 7b. Plus with enhanced rules OFF
cap7b = run_enhance_plus(task_type="rv2v", same_subject=True,
                         images_batch=batch, source_video=src_video,
                         enhanced_rules=False)
check("plus: enhanced rules OFF - not in system prompt",
      "ENHANCED DIALOGUE RULES" not in cap7b["returned_system_prompt"])

# 7c. Plus ri2v scene-as-image — ref_images_out includes scene as first image
print("\nH3PromptEnhancerPlus ri2v scene-as-image")
cap7c = run_enhance_plus(task_type="ri2v", same_subject=True,
                         images_batch=batch, source_video=None,
                         source_image=scene_img, enhanced_rules=True)
check("plus scene-as-image: ref_images_out has 4 images (1 scene + 3 batch)",
      cap7c["ref_images_out"].shape[0] == 4,
      f"shape={cap7c['ref_images_out'].shape}")
check("plus scene-as-image: prompt still works (Picture 1 is SCENE)",
      "<Picture 1> provides the SCENE" in cap7c["user_text"])

# 7d. Plus with no ref images at all (t2v)
print("\nH3PromptEnhancerPlus t2v (no images)")
cap7d = run_enhance_plus(task_type="t2v", same_subject=False,
                         images_batch=None, source_video=None,
                         source_image=None, enhanced_rules=True)
check("plus t2v: ref_images_out is 1x1x1x3 dummy",
      cap7d["ref_images_out"].shape == (1, 1, 1, 3),
      f"shape={cap7d['ref_images_out'].shape}")

# ── 8. same_subject auto-detection from raw prompt phrasing ────────────────
# The widget is the hard switch, but "all same subject" in the user's prompt
# must also activate the merge rules.
print("\nsame_subject auto-detection (widget OFF, prompt says 'all same subject')")
cap8 = run_enhance(task_type="rv2v", same_subject=False,
                   images_batch=batch, source_video=src_video,
                   prompt_override="Replace the woman with the woman in the "
                                   "reference images (all same subject) including body.")
t8 = cap8["user_text"]
check("auto-detect: merge rule activated",
      "SAME SUBJECT:" in t8 and "ONE unified subject" in t8)
check("auto-detect: no per-image SCENE/REPLACEMENT rule",
      "SCENE/REPLACEMENT RULE" not in t8)
check("auto-detect: strong-references mandate present",
      "REFERENCES ARE STRONG" in t8 and "attribute_transfer" in t8
      and "never weak_reference" in t8)
check("auto-detect: mock output still returned",
      cap8["returned_h3_prompt"] == "MOCK OUTPUT")

print("\nsame_subject auto-detect NEGATIVE (no phrase in prompt)")
cap8b = run_enhance(task_type="rv2v", same_subject=False,
                    images_batch=batch, source_video=src_video,
                    prompt_override="A woman walks into the scene.")
check("no phrase: per-image rules stay (widget off)",
      "SCENE/REPLACEMENT RULE" in cap8b["user_text"]
      and "SAME SUBJECT:" not in cap8b["user_text"])

# ── 9. exact picture-count bound ───────────────────────────────────────────
# The LLM sometimes fabricates <Picture N> beyond the attached count (e.g.
# wrote <Picture 6>..<Picture 9> with only 5 images attached). The ref
# section now hard-bounds the valid <Picture N> labels.
print("\nexact picture-count bound (3 images attached)")
cap9 = run_enhance(task_type="rv2v", same_subject=True,
                   images_batch=batch, source_video=src_video)
t9 = cap9["user_text"]
check("bound: 'EXACTLY 3 reference images are attached'",
      "EXACTLY 3 reference image(s) are attached" in t9)
check("bound: lists all valid pictures in order",
      "<Picture 1>, <Picture 2>, <Picture 3>" in t9)
check("bound: forbids labels beyond the count",
      "beyond <Picture 3> does not exist" in t9
      and "MUST NOT appear" in t9)
check("bound: every picture listed must be in retention",
      "Every listed picture MUST appear in retention_analysis" in t9
      and "none is weak_reference" in t9)
check("bound: no fabricated label beyond the count",
      "beyond <Picture 3> does not exist" in t9
      and "MUST NOT appear" in t9)

print("bound with 5 images (images_batch)")
imgs5 = torch.stack([make_img(seed=s) for s in range(5)])
cap9b = run_enhance(task_type="rv2v", same_subject=True,
                    images_batch=imgs5, source_video=src_video)
t9b = cap9b["user_text"]
check("bound: 5-image case lists exactly 5 pictures",
      "EXACTLY 5 reference image(s) are attached" in t9b
      and "<Picture 1>, <Picture 2>, <Picture 3>, <Picture 4>, <Picture 5>" in t9b
      and "beyond <Picture 5> does not exist" in t9b)

print(f"\nALL {PASS} CHECKS PASSED")
