"""Shared LLM client + helpers for ComfyUI-H3O.

Extracted verbatim from ComfyUI-BerniniPromptEnhancer/nodes.py so the H3
nodes work standalone. API keys are NEVER hardcoded — they come from node
inputs or the OPENROUTER_API_KEY environment variable.
"""

import base64
import json
import logging
import os
import time
import urllib.error
import urllib.request
from io import BytesIO

import numpy as np
from PIL import Image

_time = time

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Local llama.cpp backend defaults ─────────────────────────────────────

LOCAL_MODEL_DEFAULTS = {
    "ollama/joycaption": {
        "label": "JoyCaption Beta One (Ollama) 7.5GB",
        "backend": "ollama",
        "ollama_model": "aha2025/llama-joycaption-beta-one-hf-llava:Q6_K",
        "ollama_url": "http://localhost:11434",
    },
}

# Ollama default URL
OLLAMA_DEFAULT_URL = "http://localhost:11434"

def _check_ollama(url=None):
    """Return True if Ollama is reachable at the given URL."""
    try:
        u = url or OLLAMA_DEFAULT_URL
        req = urllib.request.Request(f"{u}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def _unload_ollama_model(ollama_url, model_name):
    """Unload a specific model from Ollama VRAM via the generate API.

    Ollama keeps the last used model in VRAM indefinitely. Sending a tiny
    request with keep_alive=0 forces it to unload, freeing VRAM for ComfyUI.
    """
    try:
        payload = json.dumps({
            "model": model_name,
            "prompt": ".", "stream": False,
            "keep_alive": 0,
            "max_tokens": 1,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{ollama_url}/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            pass  # we don't care about the response
        logging.info(f"[PromptEnhancer] Ollama model '{model_name}' unloaded — VRAM freed")
    except Exception as e:
        logging.warning(f"[PromptEnhancer] Failed to unload Ollama model: {e}")


LOCAL_NSFW_PERMISSION = (
    "This workflow handles adult/NSFW video content. "
    "Describe all nudity, sexual acts, body parts, and fluids with explicit "
    "anatomical precision. Use clinical terms: penis, vagina, breasts, anus, "
    "semen, saliva. Never use euphemisms or self-censor. "
    "The user expects unfiltered, anatomically accurate descriptions."
)

# JoyCaption was trained on caption datasets (list-like structure).
# This style prompt forces flowing single-paragraph prose matching what
# Bernini's T5 text encoder expects from its training data.
JOYCAPTION_STYLE_PROMPT = (
    "You write flowing single-paragraph prose for AI video generation prompts, "
    "NOT captions or lists. No bullet points, no section headers, no label-like "
    "structure. One continuous natural paragraph. Describe the scene directly "
    "as if narrating it — subjects, actions, appearance, setting, lighting, "
    "camera all woven into fluid prose. Never start sentences with labels like "
    "'Subjects:' or 'Setting:' or 'Action:'. Just write."
)



def _tensor_to_base64(tensor, max_size=848):
    """Convert a ComfyUI IMAGE tensor [H,W,C] or [B,H,W,C] to base64 PNG."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    w, h = img.size
    scale = min(max_size / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _call_openrouter(api_key, model, system_prompt, user_content,
                     timeout=120, temperature=0.7, max_tokens=2048, retries=2,
                     seed=None):
    """Call OpenRouter chat completions API with retry on transient errors.

    `seed` (optional): pass an int >= 0 for reproducible sampling, or -1 for a
    fresh random seed each call. None / omitted leaves the payload unchanged
    (provider-side non-deterministic sampling) for backward compatibility.
    """
    _time = time

    messages = [{"role": "user", "content": user_content}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        if seed < 0:
            seed = int.from_bytes(os.urandom(4), "big")
        payload["seed"] = seed

    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/comfyui",
        "X-Title": "BerniniPromptEnhancer",
    }

    last_err = None
    content = None
    for attempt in range(1 + retries):
        req = urllib.request.Request(OPENROUTER_URL, data=data,
                                    headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            last_err = f"OpenRouter API error {e.code}: {body}"
            if e.code in (429, 502, 503, 504) and attempt < retries:
                _time.sleep(2 ** attempt)
                continue
            raise RuntimeError(last_err) from e
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = str(e)
            if attempt < retries:
                _time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"OpenRouter request failed after {retries + 1} attempts: {last_err}") from e

        # Some models return `content: null` (empty completion, refusal, or
        # reasoning-only output that never materialized into text). Treat that
        # as a retryable error with a clear final message.
        try:
            choice = result["choices"][0]
            msg = choice.get("message") if isinstance(choice, dict) else None
            content = msg.get("content") if isinstance(msg, dict) else None
            finish_reason = choice.get("finish_reason", "unknown") if isinstance(choice, dict) else "unknown"
        except (KeyError, IndexError, TypeError, AttributeError):
            content = None
            finish_reason = "unknown"
        if content is not None:
            break
        last_err = (
            f"OpenRouter returned empty completion (finish_reason={finish_reason}) "
            f"for model '{model}': message content is null."
        )
        if attempt < retries:
            _time.sleep(2 ** attempt)
            continue
        raise RuntimeError(f"OpenRouter returned empty content after {retries + 1} attempts: {last_err}") from None

    text = content.strip()

    # Try parsing: clean JSON, fenced JSON anywhere, or embedded JSON object.
    try:
        parsed = json.loads(text)
        return parsed.get("rewritten_text", text)
    except (json.JSONDecodeError, AttributeError):
        pass

    # Try extracting from ``` fences (anywhere in the response).
    if "```" in text:
        blocks = text.split("```")
        for i, block in enumerate(blocks):
            if i % 2 == 1:  # odd-indexed = inside fences
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                if block.startswith("{"):
                    try:
                        parsed = json.loads(block)
                        return parsed.get("rewritten_text", text)
                    except (json.JSONDecodeError, AttributeError):
                        continue

    # Fallback: find {"rewritten_text" anywhere and extract the JSON object.
    idx = text.find('{"rewritten_text"')
    if idx != -1:
        depth = 0
        for j in range(idx, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[idx:j + 1])
                        return parsed.get("rewritten_text", text)
                    except (json.JSONDecodeError, AttributeError):
                        break

    # Last resort: strip any trailing ``` block and return the plain text.
    if "```" in text:
        text = text.split("```")[0].strip()

    return text


def _call_ollama_chat(ollama_url, model_name, user_content, system_prompt="",
                      temperature=0.7, max_tokens=2048, timeout=180,
                      seed=None):
    """Send a chat completion to Ollama's OpenAI-compatible endpoint.

    Same interface as _call_local_chat. Ollama handles model lifecycle
    internally — no server spawn/kill needed.
    """
    url = f"{ollama_url}/v1/chat/completions"

    messages = [{"role": "user", "content": user_content}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        if seed < 0:
            seed = int.from_bytes(os.urandom(4), "big")
        payload["seed"] = seed

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    last_err = None
    for attempt in range(3):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            last_err = f"Ollama error {e.code}: {body[:500]}"
            if e.code in (429, 502, 503, 504) and attempt < 2:
                _time.sleep(2 ** attempt)
                continue
            raise RuntimeError(last_err) from e
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = str(e)
            if attempt < 2:
                _time.sleep(2 ** attempt)
                continue
            raise RuntimeError(
                f"Ollama request failed after 3 attempts: {last_err}") from e

    text = result["choices"][0]["message"]["content"]
    if text is None:
        text = ""
    text = text.strip()

    if not text:
        raise RuntimeError("Ollama returned empty response")

    # Parse JSON response (same extraction logic as _call_openrouter)
    try:
        parsed = json.loads(text)
        return parsed.get("rewritten_text", text)
    except (json.JSONDecodeError, AttributeError):
        pass

    # Try extracting from ``` fences
    if "```" in text:
        blocks = text.split("```")
        for i, block in enumerate(blocks):
            if i % 2 == 1:
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                if block.startswith("{"):
                    try:
                        parsed = json.loads(block)
                        return parsed.get("rewritten_text", text)
                    except (json.JSONDecodeError, AttributeError):
                        continue

    # Fallback: find {"rewritten_text" anywhere
    idx = text.find('{"rewritten_text"')
    if idx != -1:
        depth = 0
        for j in range(idx, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[idx:j + 1])
                        return parsed.get("rewritten_text", text)
                    except (json.JSONDecodeError, AttributeError):
                        break

    if "```" in text:
        text = text.split("```")[0].strip()

    return text
