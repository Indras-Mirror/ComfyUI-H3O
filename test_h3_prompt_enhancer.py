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
                prompt_override=None, editing_frame="on",
                model="x-ai/grok-4.3", context_length=-1):
    node = H3PromptEnhancer()
    captured = {}

    def fake_caller(backend, url, model, system_prompt, user_content, **kwargs):
        captured["backend"] = backend
        captured["url"] = url
        captured["api_key"] = kwargs.get("api_key")
        captured["model"] = model
        captured["system_prompt"] = system_prompt
        captured["content"] = user_content
        # Last content item is the assembled user message text.
        captured["user_text"] = user_content[-1]["text"]
        captured["seed"] = kwargs.get("seed")
        captured["max_tokens"] = kwargs.get("max_tokens")
        captured["budgets"] = captured.get("budgets", []) + [
            kwargs.get("max_tokens")]
        captured["n_calls"] = captured.get("n_calls", 0) + 1
        return "MOCK OUTPUT"

    # Patch the single dispatch seam so enhance() hits the mock.
    h3pe._call_backend = fake_caller

    out, sys_prompt = node.enhance(
        prompt=(prompt_override
                if prompt_override is not None
                else "A woman walks into the scene."),
        task_type=task_type,
        duration=5.0,
        model=model,
        api_key="test-key",
        images_batch=images_batch,
        source_video=source_video,
        source_image=source_image,
        same_subject=same_subject,
        auto_describe=auto_describe,
        advanced_prompt="on",
        editing_frame=editing_frame,
        context_length=context_length,
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
                     enhanced_rules=True, editing_frame="on"):
    node = H3PromptEnhancerPlus()
    captured = {}

    def fake_caller(backend, url, model, system_prompt, user_content, **kwargs):
        captured["backend"] = backend
        captured["url"] = url
        captured["api_key"] = kwargs.get("api_key")
        captured["model"] = model
        captured["system_prompt"] = system_prompt
        captured["content"] = user_content
        captured["user_text"] = user_content[-1]["text"]
        captured["seed"] = kwargs.get("seed")
        captured["n_calls"] = captured.get("n_calls", 0) + 1
        return "MOCK OUTPUT"

    h3pe._call_backend = fake_caller

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
        editing_frame=editing_frame,
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
check("plus: ref_images_out is a LIST of 3 individual [1,H,W,C] images",
      isinstance(cap7["ref_images_out"], list)
      and len(cap7["ref_images_out"]) == 3
      and all(f.dim() == 4 and f.shape[0] == 1 for f in cap7["ref_images_out"]),
      f"type={type(cap7['ref_images_out'])}")
check("plus: ref_images_out frames are NOT stretched (native geometry kept)",
      all(tuple(cap7["ref_images_out"][i].shape[1:]) == tuple(batch[i].shape)
          for i in range(3)),
      [tuple(f.shape) for f in cap7["ref_images_out"]])

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
check("plus scene-as-image: ref_images_out is a LIST of 4 images (1 scene + 3 batch)",
      isinstance(cap7c["ref_images_out"], list)
      and len(cap7c["ref_images_out"]) == 4,
      f"type={type(cap7c['ref_images_out'])} len={len(cap7c['ref_images_out']) if isinstance(cap7c['ref_images_out'], list) else 'n/a'}")
check("plus scene-as-image: prompt still works (Picture 1 is SCENE)",
      "<Picture 1> provides the SCENE" in cap7c["user_text"])

# 7d. Plus with no ref images at all (t2v)
print("\nH3PromptEnhancerPlus t2v (no images)")
cap7d = run_enhance_plus(task_type="t2v", same_subject=False,
                         images_batch=None, source_video=None,
                         source_image=None, enhanced_rules=True)
check("plus t2v: ref_images_out is an empty list (no refs)",
      isinstance(cap7d["ref_images_out"], list)
      and len(cap7d["ref_images_out"]) == 0,
      f"type={type(cap7d['ref_images_out'])}")

# ── 8. same_subject tri-state (auto/on/off) ───────────────────────────────
# auto = honor prompt phrasing; on = force unified; off = force per-image.
print("\nsame_subject='auto' + prompt says 'all same subject'")
cap8 = run_enhance(task_type="rv2v", same_subject="auto",
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

print("\nsame_subject='auto' with NO phrase in prompt")
cap8b = run_enhance(task_type="rv2v", same_subject="auto",
                    images_batch=batch, source_video=src_video,
                    prompt_override="A woman walks into the scene.")
check("auto no phrase: per-image rules stay",
      "SCENE/REPLACEMENT RULE" in cap8b["user_text"]
      and "SAME SUBJECT:" not in cap8b["user_text"])

print("\nsame_subject='on' — force unified even with unrelated prompt")
cap8c = run_enhance(task_type="rv2v", same_subject="on",
                    images_batch=batch, source_video=src_video,
                    prompt_override="A woman walks into the scene.")
check("on: merge rule activated despite no phrase",
      "SAME SUBJECT:" in cap8c["user_text"]
      and "ONE unified subject" in cap8c["user_text"])
check("on: no per-image SCENE/REPLACEMENT rule",
      "SCENE/REPLACEMENT RULE" not in cap8c["user_text"])

print("\nsame_subject='off' — force per-image even with phrase in prompt")
cap8d = run_enhance(task_type="rv2v", same_subject="off",
                    images_batch=batch, source_video=src_video,
                    prompt_override="All three are the same person, I swear.")
check("off: per-image rules despite phrase",
      "SCENE/REPLACEMENT RULE" in cap8d["user_text"]
      and "SAME SUBJECT:" not in cap8d["user_text"])

# Legacy boolean still works: True == on
print("\nsame_subject=True (legacy boolean) still forces merge")
cap8e = run_enhance(task_type="rv2v", same_subject=True,
                    images_batch=batch, source_video=src_video,
                    prompt_override="A woman walks into the scene.")
check("legacy True: merge rule activated",
      "SAME SUBJECT:" in cap8e["user_text"]
      and "ONE unified subject" in cap8e["user_text"])
# Legacy False == off
print("\nsame_subject=False (legacy boolean) forces per-image")
cap8f = run_enhance(task_type="rv2v", same_subject=False,
                    images_batch=batch, source_video=src_video,
                    prompt_override="All same person here.")
check("legacy False: per-image rules despite phrase",
      "SCENE/REPLACEMENT RULE" in cap8f["user_text"]
      and "SAME SUBJECT:" not in cap8f["user_text"])

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

# ── 10. character-replacement editing frame (folded from RefPack register) ─
# Applies to ANY job with a plate video + identity refs — rv2v AND ri2v.
print("\nediting frame: rv2v with plate video (H3ImageToRefVideo)")
cap10 = run_enhance(task_type="rv2v", same_subject=True,
                    images_batch=batch, source_video=src_video)
t10 = cap10["user_text"]
check("rv2v: editing frame injected",
      "CHARACTER-REPLACEMENT EDITING FRAME" in t10)
check("rv2v: motion inheritance demanded",
      "MOTION INHERITANCE" in t10 and "screen position" in t10)
check("rv2v: integration/optics/lighting carried",
      "INTEGRATION" in t10 and "OPTICS" in t10 and "LIGHTING" in t10)
check("rv2v: no new scene rule",
      "Do NOT write a new scene" in t10)

print("\nediting frame: ri2v with source video")
cap10b = run_enhance(task_type="ri2v", same_subject=True,
                     images_batch=batch, source_video=src_video)
t10b = cap10b["user_text"]
check("ri2v: editing frame injected",
      "CHARACTER-REPLACEMENT EDITING FRAME" in t10b)
check("ri2v: plate partially_preserved demanded",
      "partially_preserved" in t10b)

print("\nediting frame: ri2v scene-as-image (no plate video) — NOT injected")
cap10c = run_enhance(task_type="ri2v", same_subject=True,
                     images_batch=batch, source_video=None,
                     source_image=scene_img)
t10c = cap10c["user_text"]
check("scene-as-image: editing frame absent (no <Video 1> plate)",
      "CHARACTER-REPLACEMENT EDITING FRAME" not in t10c)

print("\nediting frame toggle OFF: rv2v with plate video")
cap10d = run_enhance(task_type="rv2v", same_subject=True,
                     images_batch=batch, source_video=src_video,
                     editing_frame="off")
t10d = cap10d["user_text"]
check("off: no CHARACTER-REPLACEMENT EDITING FRAME header",
      "CHARACTER-REPLACEMENT EDITING FRAME" not in t10d)
check("off: no 'EDITED VERSION of <Video' framing sentence",
      "EDITED VERSION of <Video 1>" not in t10d)
check("off: no MOTION INHERITANCE demands",
      "MOTION INHERITANCE" not in t10d)
check("off: no integration/optics/lighting block",
      "INTEGRATION:" not in t10d and "OPTICS + LIGHTING" not in t10d)
check("off: no 'Do NOT write a new scene' rule",
      "Do NOT write a new scene" not in t10d)
check("off: base rv2v ref section still present",
      "static reference video (<Video 1>) is attached" in t10d)

print("\nediting frame toggle OFF: ri2v with source video")
cap10e = run_enhance(task_type="ri2v", same_subject=True,
                     images_batch=batch, source_video=src_video,
                     editing_frame="off")
t10e = cap10e["user_text"]
check("off ri2v: no editing frame header",
      "CHARACTER-REPLACEMENT EDITING FRAME" not in t10e)
check("off ri2v: no 'EDITED VERSION of <Video' framing sentence",
      "EDITED VERSION of <Video 1>" not in t10e)
check("off ri2v: no MOTION INHERITANCE demands",
      "MOTION INHERITANCE" not in t10e)
check("off ri2v: base ri2v scene/structure block still present",
      "<Video 1> supplies the" in t10e
      and "partially_preserved" in t10e)

print("\nediting frame ON (default): rv2v with plate video — frame present")
cap10f = run_enhance(task_type="rv2v", same_subject=True,
                     images_batch=batch, source_video=src_video,
                     editing_frame="on")
t10f = cap10f["user_text"]
check("on rv2v: editing frame injected",
      "CHARACTER-REPLACEMENT EDITING FRAME" in t10f
      and "MOTION INHERITANCE" in t10f)
check("on rv2v: framing sentence present",
      "EDITED VERSION of <Video 1>" in t10f)

print("\nediting frame toggle OFF: ri2v scene-as-image — still absent")
cap10g = run_enhance(task_type="ri2v", same_subject=True,
                     images_batch=batch, source_video=None,
                     source_image=scene_img, editing_frame="off")
t10g = cap10g["user_text"]
check("scene-as-image off: frame absent regardless of toggle",
      "CHARACTER-REPLACEMENT EDITING FRAME" not in t10g
      and "MOTION INHERITANCE" not in t10g)

print("\nediting frame toggle: Plus node rv2v with plate video")
cap10h = run_enhance_plus(task_type="rv2v", same_subject=True,
                          images_batch=batch, source_video=src_video,
                          enhanced_rules=True, editing_frame="on")
t10h = cap10h["user_text"]
check("plus on: editing frame injected",
      "CHARACTER-REPLACEMENT EDITING FRAME" in t10h
      and "MOTION INHERITANCE" in t10h)
cap10i = run_enhance_plus(task_type="rv2v", same_subject=True,
                          images_batch=batch, source_video=src_video,
                          enhanced_rules=True, editing_frame="off")
t10i = cap10i["user_text"]
check("plus off: editing frame suppressed",
      "CHARACTER-REPLACEMENT EDITING FRAME" not in t10i
      and "MOTION INHERITANCE" not in t10i
      and "EDITED VERSION of <Video 1>" not in t10i)
check("plus off: enhanced rules still injected",
      "ENHANCED DIALOGUE RULES" in cap10i["returned_system_prompt"])
check("plus off: ref_images_out still passes through (list of 3)",
      isinstance(cap10i["ref_images_out"], list)
      and len(cap10i["ref_images_out"]) == 3,
      f"type={type(cap10i['ref_images_out'])}")

# ── 11. llamacpp backend routing (registry url/model, spawn/kill wiring) ───
# The on-demand llama-server lifecycle is mocked here (no GPU); the live
# spawn→serve→shutdown is covered by test_h3_llamacpp_e2e.py.
print("\nllamacpp routing: qwen3.8-heretic-ara (spawned, both passes)")
img = make_img(seed=7).unsqueeze(0)  # [1, H, W, C]

def run_enhance_local(local_backend="llamacpp/qwen3.8-heretic-ara",
                      task_type="i2v", auto_describe=True, source_image=img,
                      seed=-1, context_length=-1,
                      spawn_result=("spawned", 424242),
                      prompt_override=None):
    node = H3PromptEnhancer()
    captured = {}
    spawn_calls = []
    kill_calls = []

    def fake_caller(backend, url, model, system_prompt, user_content, **kwargs):
        captured["backend"] = backend
        captured["url"] = url
        captured["model"] = model
        captured["system_prompt"] = system_prompt
        captured["user_text"] = user_content[-1]["text"]
        captured["seed"] = kwargs.get("seed")
        captured["max_tokens"] = kwargs.get("max_tokens")
        captured["budgets"] = captured.get("budgets", []) + [
            kwargs.get("max_tokens")]
        captured["n_calls"] = captured.get("n_calls", 0) + 1
        return "MOCK OUTPUT"

    def fake_spawn(defaults, **kw):
        spawn_calls.append((defaults.get("label"), kw.get("context_length")))
        return spawn_result

    def fake_kill(pid):
        kill_calls.append(pid)

    h3pe._call_backend = fake_caller
    h3pe._spawn_llama_server = fake_spawn
    h3pe._kill_llama_server = fake_kill
    h3pe._check_ollama = lambda url: True   # no network in unit tests
    h3pe._unload_ollama_model = lambda *a, **k: None  # no real ollama unload

    out, sys_prompt = node.enhance(
        prompt=(prompt_override if prompt_override is not None
                else "A woman walks into the scene."),
        task_type=task_type,
        duration=5.0,
        model="x-ai/grok-4.3",
        local_backend=local_backend,
        source_image=source_image,
        auto_describe=auto_describe,
        advanced_prompt="on",
        seed=seed,
        context_length=context_length,
    )
    captured["returned_h3_prompt"] = out
    captured["spawn_calls"] = spawn_calls
    captured["kill_calls"] = kill_calls
    return captured

cap11 = run_enhance_local(local_backend="llamacpp/qwen3.8-heretic-ara",
                          seed=1234, context_length=131072)
check("llamacpp: routed to llamacpp backend", cap11["backend"] == "llamacpp")
check("llamacpp: url from registry (8098)",
      cap11["url"] == "http://127.0.0.1:8098", f"url={cap11['url']}")
check("llamacpp: model id from registry",
      cap11["model"] == "Qwen3.8-27B-heretic-ara.i1-Q4_K_M.gguf",
      f"model={cap11['model']}")
check("llamacpp: spawn called once with node context_length",
      cap11["spawn_calls"] ==
      [("Qwen3.8-27B heretic-ara (llama.cpp) 16.8GB", 131072)],
      f"spawn_calls={cap11['spawn_calls']}")
check("llamacpp: kill called with the spawned pid",
      cap11["kill_calls"] == [424242], f"kill_calls={cap11['kill_calls']}")
check("llamacpp: both passes share one instance (analyze + write)",
      cap11["n_calls"] == 2, f"n_calls={cap11['n_calls']}")
check("llamacpp: seed threaded through", cap11["seed"] == 1234)
check("llamacpp: auto ctx (-1) falls back to backend default",
      run_enhance_local(context_length=-1)["spawn_calls"][0][1] == -1)
check("llamacpp: LLAMA style prefix applied (not JoyCaption)",
      "Narrate the scene directly" in cap11["system_prompt"]
      and "NOT captions or lists" not in cap11["system_prompt"])
check("llamacpp: NSFW permission still applied",
      "anatomically accurate descriptions" in cap11["system_prompt"])

print("\nllamacpp routing: muse-glimmer (8097, YaRN entry)")
cap11b = run_enhance_local(local_backend="llamacpp/muse-glimmer", seed=-1)
check("glimmer: url from registry (8097)",
      cap11b["url"] == "http://127.0.0.1:8097", f"url={cap11b['url']}")
check("glimmer: model id from registry",
      cap11b["model"] == "darkc0de_Muse-Glimmer-30B-heretic-IQ4_NL",
      f"model={cap11b['model']}")
check("glimmer: spawn/kill wired",
      cap11b["spawn_calls"][0][0] == "Muse-Glimmer-30B heretic (llama.cpp) 16.2GB"
      and cap11b["kill_calls"] == [424242])

print("\nllamacpp routing: reuse path (no kill)")
cap11c = run_enhance_local(local_backend="llamacpp/qwen3.8-heretic-ara",
                           spawn_result=("reused", None))
check("reuse: no kill when server was reused", cap11c["kill_calls"] == [])
check("reuse: dispatch still routed to llamacpp", cap11c["backend"] == "llamacpp")

print("\nollama/joycaption: unchanged backend + JoyCaption style prompt")
cap11d = run_enhance_local(local_backend="ollama/joycaption", seed=7)
check("ollama: routed to ollama backend", cap11d["backend"] == "ollama")
check("ollama: url from registry",
      cap11d["url"] == "http://localhost:11434", f"url={cap11d['url']}")
check("ollama: JoyCaption style prompt applied",
      "NOT captions or lists" in cap11d["system_prompt"])
check("ollama: LLAMA style prompt NOT applied",
      "Narrate the scene directly" not in cap11d["system_prompt"])

# ── 12. generation budget ceiling (refinement #2: context_length) ───────────
# context_length is the CEILING for both passes on remote + llamacpp:
# per-call max_tokens = max(existing widget budget, min(context_length, cap)).
print("\ngeneration budget ceiling (remote)")
cap12 = run_enhance(task_type="i2v", source_image=img, auto_describe=True,
                    context_length=-1)  # auto → grok-4.3 cap 131072 (verified)
check("remote auto (grok): write budget raised to 131072",
      cap12["max_tokens"] == 131072,
      f"write={cap12['max_tokens']} budgets={cap12['budgets']}")
check("remote auto: analyze budget also 131072",
      cap12["budgets"] == [131072, 131072], f"budgets={cap12['budgets']}")
cap12b = run_enhance(task_type="t2v", model="google/gemini-2.5-flash")
check("remote gemini: verified cap 65536 applies",
      cap12b["max_tokens"] == 65536, f"write={cap12b['max_tokens']}")
cap12c = run_enhance(task_type="t2v", context_length=8192)
check("remote explicit ctx 8192: ceiling = min(8192, 16384)",
      cap12c["max_tokens"] == 8192, f"write={cap12c['max_tokens']}")
cap12d = run_enhance(task_type="t2v", context_length=4096)
check("remote explicit ctx 4096: widget floor holds (4096)",
      cap12d["max_tokens"] == 4096, f"write={cap12d['max_tokens']}")

print("\ngeneration budget ceiling (llamacpp + ollama)")
cap12e = run_enhance_local(local_backend="llamacpp/qwen3.8-heretic-ara",
                           task_type="i2v", source_image=img,
                           auto_describe=True, context_length=-1)
check("llamacpp auto: write budget 16384 (min(ctx, 16384))",
      cap12e["max_tokens"] == 16384, f"write={cap12e['max_tokens']}")
check("llamacpp auto: analyze budget also 16384",
      cap12e["budgets"] == [16384, 16384], f"budgets={cap12e['budgets']}")
cap12f = run_enhance_local(local_backend="ollama/joycaption", task_type="i2v",
                           source_image=img, auto_describe=True, seed=7)
check("ollama: budget unchanged (4096 widget, no ceiling)",
      cap12f["max_tokens"] == 4096, f"write={cap12f['max_tokens']}")

# ── 13. chain_conversation toggle (two-pass conversation chaining) ─────────
# A/B: OFF = the write call is independent (payload byte-identical); ON = the
# write request carries pass 1's messages + raw assistant reply as history.
import json  # noqa: E402

ANALYZE_JSON = ('[{"image_id":"image0","subject":"a woman in red",'
                '"scene":"a bar","current_state":"walking"}]')

def run_enhance_chain(chain_conversation="off", local_backend=None,
                      task_type="i2v", source_image=img, auto_describe=True,
                      editing_frame="on", spawn_result=("spawned", 424242)):
    node = H3PromptEnhancer()
    calls = []

    def fake_caller(backend, url, model, system_prompt, user_content, **kwargs):
        calls.append({
            "backend": backend, "url": url, "model": model,
            "system_prompt": system_prompt, "content": user_content,
            "user_text": user_content[-1]["text"],
            "kwargs": kwargs,
        })
        if kwargs.get("return_raw"):
            # Mirror what the real leaves return in return_raw mode.
            messages_sent = [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": user_content}]
            return (ANALYZE_JSON, "RAW ANALYZE REPLY", messages_sent)
        return "MOCK OUTPUT"

    h3pe._call_backend = fake_caller
    if local_backend:
        h3pe._spawn_llama_server = lambda defaults, **kw: spawn_result
        h3pe._kill_llama_server = lambda pid: None
        h3pe._check_ollama = lambda url: True
        h3pe._unload_ollama_model = lambda *a, **k: None

    kw = dict(
        prompt="A woman walks into the scene.",
        task_type=task_type,
        duration=5.0,
        model="x-ai/grok-4.3",
        source_image=source_image,
        auto_describe=auto_describe,
        advanced_prompt="on",
        editing_frame=editing_frame,
        chain_conversation=chain_conversation,
    )
    if local_backend:
        kw["local_backend"] = local_backend
    else:
        kw["api_key"] = "test-key"
    out, sys_prompt = node.enhance(**kw)
    return calls, out, sys_prompt

print("\nchain_conversation OFF (default) — write call independent")
calls_off, out_off, _ = run_enhance_chain("off")
check("chain off: 2 calls (analyze + write)",
      len(calls_off) == 2, f"n={len(calls_off)}")
check("chain off: analyze call not in return_raw mode",
      not calls_off[0]["kwargs"].get("return_raw"))
check("chain off: write call has NO history kwarg",
      "history" not in calls_off[1]["kwargs"],
      f"kwargs={calls_off[1]['kwargs']}")
check("chain off: write system+user still passed independently",
      "Context Analysis" in calls_off[1]["user_text"]
      and calls_off[1]["system_prompt"] != calls_off[0]["system_prompt"])
check("chain off: mock output returned", out_off == "MOCK OUTPUT")

print("\nchain_conversation ON — write carries pass 1 conversation")
calls_on, out_on, _ = run_enhance_chain("on")
check("chain on: 2 calls (analyze + write)",
      len(calls_on) == 2, f"n={len(calls_on)}")
analyze_on, write_on = calls_on
check("chain on: analyze call uses return_raw",
      analyze_on["kwargs"].get("return_raw") is True)
hist = write_on["kwargs"].get("history")
check("chain on: write call received history (3 msgs)",
      hist is not None and len(hist) == 3,
      f"len={len(hist) if hist else None}")
check("chain on: history[0] = pass1 system message",
      hist[0] == {"role": "system", "content": analyze_on["system_prompt"]})
check("chain on: history[1] = pass1 user message",
      hist[1] == {"role": "user", "content": analyze_on["content"]})
check("chain on: history[2] = pass1 raw assistant reply",
      hist[2] == {"role": "assistant", "content": "RAW ANALYZE REPLY"})
full_msgs = ([{"role": "system", "content": write_on["system_prompt"]}]
             + hist + [{"role": "user", "content": write_on["content"]}])
check("chain on: write msgs = [system]+pass1 msgs+assistant raw+user",
      [m["role"] for m in full_msgs] ==
      ["system", "system", "user", "assistant", "user"])
check("chain on: Context Analysis text still in write template",
      "--- Context Analysis" in write_on["user_text"])
check("chain on: mock output returned", out_on == "MOCK OUTPUT")

print("\nchain_conversation ON — routing (openrouter / llamacpp / ollama)")
calls_or, _, _ = run_enhance_chain("on")
check("chain on: openrouter routed for both passes",
      calls_or[0]["backend"] == "openrouter"
      and calls_or[1]["backend"] == "openrouter")
calls_ll, _, _ = run_enhance_chain("on",
                                   local_backend="llamacpp/qwen3.8-heretic-ara")
check("chain on: llamacpp routed, history flows",
      calls_ll[0]["backend"] == "llamacpp"
      and calls_ll[1]["backend"] == "llamacpp"
      and calls_ll[1]["kwargs"].get("history") is not None)
calls_ol, _, _ = run_enhance_chain("on", local_backend="ollama/joycaption")
check("chain on: ollama routed, history flows",
      calls_ol[0]["backend"] == "ollama"
      and calls_ol[1]["backend"] == "ollama"
      and calls_ol[1]["kwargs"].get("history") is not None)

print("\nchain_conversation — widget order + Plus passthrough")
base_keys = list(H3PromptEnhancer.INPUT_TYPES()["optional"].keys())
check("base: chain_conversation is LAST optional widget",
      base_keys[-1] == "chain_conversation", f"last={base_keys[-1]}")
plus_keys = list(H3PromptEnhancerPlus.INPUT_TYPES()["optional"].keys())
check("plus: chain_conversation is LAST optional widget (after enhanced_rules)",
      plus_keys[-1] == "chain_conversation", f"last={plus_keys[-1]}")

def run_enhance_plus_chain(chain_conversation="on"):
    node = H3PromptEnhancerPlus()
    calls = []

    def fake_caller(backend, url, model, system_prompt, user_content, **kwargs):
        calls.append({"system_prompt": system_prompt, "content": user_content,
                      "kwargs": kwargs})
        if kwargs.get("return_raw"):
            return (ANALYZE_JSON, "RAW ANALYZE REPLY",
                    [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_content}])
        return "MOCK OUTPUT"

    h3pe._call_backend = fake_caller
    h3_prompt, sys_prompt, ref_batch = node.enhance_plus(
        prompt="A woman walks into the scene.",
        task_type="i2v",
        duration=5.0,
        model="x-ai/grok-4.3",
        api_key="test-key",
        source_image=img,
        auto_describe=True,
        advanced_prompt="on",
        enhanced_rules=True,
        chain_conversation=chain_conversation,
    )
    return calls, h3_prompt, ref_batch

calls_p, out_p, ref_p = run_enhance_plus_chain("on")
check("plus: chain_conversation passes through (write history present)",
      len(calls_p) == 2 and calls_p[1]["kwargs"].get("history") is not None)
check("plus: enhanced rules still applied",
      "ENHANCED DIALOGUE RULES" in calls_p[1]["system_prompt"])
check("plus: ref_images_out still passes through (list)",
      isinstance(ref_p, list), f"type={type(ref_p)}")

# ── 14. seam: real leaf message construction (history splice + return_raw) ──
# The node-level tests above mock _call_backend; here the REAL leaf functions
# run against a mocked urlopen to prove the OFF payload is byte-identical and
# the history splice produces [system]+history+[user].
print("\nseam: real _call_backend leaf construction (mocked urlopen)")
h3o_mod = sys.modules["h3o_prompt_test.h3o_shared"]
SEAM_RESP = (b'{"choices":[{"message":{"content":"{\\"rewritten_text\\": '
             b'\\"HELLO\\"}"}}]}')

seam_cap = {}

def fake_urlopen(req, timeout=None):
    seam_cap["payload"] = json.loads(req.data.decode())
    seam_cap["url"] = req.full_url

    class R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return SEAM_RESP

    return R()

_old_urlopen = h3o_mod.urllib.request.urlopen
h3o_mod.urllib.request.urlopen = fake_urlopen
try:
    parsed_off = h3o_mod._call_llama_chat(
        "http://127.0.0.1:8098", "m", "USER TEXT", system_prompt="SYS")
    msgs_off = seam_cap["payload"]["messages"]
    check("seam off: parsed text extracted", parsed_off == "HELLO")
    check("seam off: messages == [system, user] (byte-identical shape)",
          msgs_off == [{"role": "system", "content": "SYS"},
                       {"role": "user", "content": "USER TEXT"}],
          f"msgs={msgs_off}")

    parsed_r, raw_r, sent_r = h3o_mod._call_llama_chat(
        "http://127.0.0.1:8098", "m", "USER TEXT", system_prompt="SYS",
        return_raw=True)
    check("seam raw: returns (parsed, raw_content, messages_sent)",
          parsed_r == "HELLO" and raw_r == '{"rewritten_text": "HELLO"}'
          and sent_r == msgs_off)

    hist = [{"role": "assistant", "content": "A1"}]
    parsed_h, raw_h, sent_h = h3o_mod._call_llama_chat(
        "http://127.0.0.1:8098", "m", "USER TEXT", system_prompt="SYS",
        history=hist, return_raw=True)
    check("seam history: [system]+history+[user]",
          sent_h == [{"role": "system", "content": "SYS"},
                     {"role": "assistant", "content": "A1"},
                     {"role": "user", "content": "USER TEXT"}])

    parsed_b, raw_b, sent_b = h3o_mod._call_backend(
        "llamacpp", "http://127.0.0.1:8098", "m", "SYS", "USER TEXT",
        history=hist, return_raw=True)
    check("seam backend: history+return_raw threaded through dispatcher",
          sent_b == [{"role": "system", "content": "SYS"},
                     {"role": "assistant", "content": "A1"},
                     {"role": "user", "content": "USER TEXT"}])

    parsed_or, raw_or, sent_or = h3o_mod._call_openrouter(
        "KEY", "m", "SYS", "USER TEXT", return_raw=True)
    check("seam openrouter: tuple contract holds",
          parsed_or == "HELLO" and raw_or == '{"rewritten_text": "HELLO"}'
          and sent_or == [{"role": "system", "content": "SYS"},
                          {"role": "user", "content": "USER TEXT"}])
finally:
    h3o_mod.urllib.request.urlopen = _old_urlopen

print(f"\nALL {PASS} CHECKS PASSED")
