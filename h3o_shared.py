"""Shared LLM client + helpers for ComfyUI-H3O.

Extracted verbatim from ComfyUI-BerniniPromptEnhancer/nodes.py so the H3
nodes work standalone. API keys are NEVER hardcoded — they come from node
inputs or the OPENROUTER_API_KEY environment variable.
"""

import base64
import json
import logging
import os
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from io import BytesIO

import numpy as np
from PIL import Image

_time = time

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Per-model maximum OUTPUT tokens for the OpenRouter models in the node's
# dropdown. OpenRouter's /v1/models API does NOT publish max_completion_tokens
# for these (all None as of 2026-08) — the entries below were verified
# EMPIRICALLY with 1-token probe requests at max_tokens=131072 (accepted,
# 2026-08-16): every x-ai/grok-4* ID on OpenRouter (4.3, 4.5, 4.6, 4.20)
# accepts 131072, so that is the grok-family ceiling. Provider-documented
# entries: Google Gemini 2.5 Flash 65536, DeepSeek V3 chat 8192, OpenAI
# gpt-4o 16384. Custom model strings fall back to the 16384 default.
OPENROUTER_MODEL_OUTPUT_CAPS = {
    "x-ai/grok-4.3": 131072,
    "x-ai/grok-4.5": 131072,
    "x-ai/grok-4.6": 131072,
    "x-ai/grok-4.20": 131072,
    "google/gemini-2.5-flash": 65536,
    "deepseek/deepseek-chat-v3-0324": 8192,
    "openai/gpt-4o": 16384,
}


def _resolve_generation_budget(context_length, output_cap=32768):
    """context_length as the generation-budget CEILING (refinement #2).

    Returns the ceiling max_tokens for a single LLM call: auto (-1/0) →
    `output_cap`; explicit → min(context_length, output_cap). Callers apply it
    as max(existing widget budget, ceiling) so saved workflows keep at least
    their old per-pass budget (the ceiling only ever raises it).
    """
    try:
        ctx = int(context_length or 0)
    except (TypeError, ValueError):
        ctx = 0
    if ctx <= 0:
        return output_cap
    return min(ctx, output_cap)

# ── Local llama.cpp backend defaults ─────────────────────────────────────

LOCAL_MODEL_DEFAULTS = {
    "ollama/joycaption": {
        "label": "JoyCaption Beta One (Ollama) 7.5GB",
        "backend": "ollama",
        "ollama_model": "aha2025/llama-joycaption-beta-one-hf-llava:Q6_K",
        "ollama_url": "http://localhost:11434",
    },
    # Generic llama.cpp entry: connect to ANY llama.cpp-compatible server.
    # Set the node's llamacpp_url (or H3O_LLAMACPP_URL) and pick the model
    # name via custom_model — the server is assumed to be already running,
    # nothing is spawned or killed. Named on-demand models (llama_bin +
    # model paths, spawn/teardown) can be added via the user config file
    # ($H3O_LLAMACPP_CONFIG or ~/.config/h3o/llamacpp.json) — see
    # llamacpp.example.json.
    "llamacpp": {
        "label": "llamacpp (external server)",
        "backend": "llamacpp",
        "llama_url": "",
        "llama_model": "",
    },
}


# Ollama default URL
OLLAMA_DEFAULT_URL = "http://localhost:11434"


def _load_user_llamacpp_config():
    """Merge user-defined local backend entries from a JSON config file.

    Path: $H3O_LLAMACPP_CONFIG or ~/.config/h3o/llamacpp.json. Entries follow
    the same schema as LOCAL_MODEL_DEFAULTS (see llamacpp.example.json) and
    are merged by key — a user entry with the same key REPLACES the built-in.
    """
    import os as _os
    path = (_os.environ.get("H3O_LLAMACPP_CONFIG")
            or _os.path.expanduser("~/.config/h3o/llamacpp.json"))
    if not _os.path.exists(path):
        return
    try:
        with open(path) as f:
            cfg = json.load(f)
    except Exception as e:
        logging.warning(f"[H3O] failed to read llamacpp config {path}: {e}")
        return
    if not isinstance(cfg, dict):
        logging.warning(f"[H3O] llamacpp config {path} must be a JSON object")
        return
    for key, entry in cfg.items():
        if not isinstance(entry, dict) or "backend" not in entry:
            logging.warning(f"[H3O] skipping invalid llamacpp config entry '{key}'")
            continue
        entry.setdefault("label", key)
        LOCAL_MODEL_DEFAULTS[key] = entry
    logging.info(f"[H3O] loaded {len(cfg)} local backend(s) from {path}")


_load_user_llamacpp_config()

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

# The llama.cpp fleet models (Qwen3.8 heretic-ara / Muse-Glimmer) are NOT
# JoyCaption-trained — the caption-training style prompt does not apply. They
# are H3-format prompt writers; give them the same prose goal in a brief,
# model-agnostic prefix.
LLAMA_STYLE_PROMPT = (
    "You write flowing single-paragraph prose for AI video generation prompts — "
    "no bullet points, no section headers, no label-like structure. Narrate the "
    "scene directly: subjects, actions, appearance, setting, lighting, and "
    "camera woven into fluid prose. Never start sentences with labels like "
    "'Subjects:' or 'Setting:'. Just write."
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
                     seed=None, return_raw=False, history=None):
    """Call OpenRouter chat completions API with retry on transient errors.

    `seed` (optional): pass an int >= 0 for reproducible sampling, or -1 for a
    fresh random seed each call. None / omitted leaves the payload unchanged
    (provider-side non-deterministic sampling) for backward compatibility.

    `history` (optional): list of extra messages spliced between the system
    prompt and the user message (multi-turn chaining). None → the exact
    [system?, user] payload of the pre-chain call, byte-identical.
    `return_raw` (optional): when True, return (parsed_text, raw_content,
    messages_sent) instead of parsed_text — raw_content is the backend's
    message.content, messages_sent the exact list this call sent.
    """
    _time = time

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_content})

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

    def _finish(parsed_text):
        """Return parsed_text, or the (parsed, raw, messages) tuple in
        return_raw mode. `content` is the raw backend reply captured above."""
        if return_raw:
            return (parsed_text, content, messages)
        return parsed_text

    # Try parsing: clean JSON, fenced JSON anywhere, or embedded JSON object.
    try:
        parsed = json.loads(text)
        return _finish(parsed.get("rewritten_text", text))
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
                        return _finish(parsed.get("rewritten_text", text))
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
                        return _finish(parsed.get("rewritten_text", text))
                    except (json.JSONDecodeError, AttributeError):
                        break

    # Last resort: strip any trailing ``` block and return the plain text.
    if "```" in text:
        text = text.split("```")[0].strip()

    return _finish(text)


def _call_ollama_chat(ollama_url, model_name, user_content, system_prompt="",
                      temperature=0.7, max_tokens=2048, timeout=180,
                      seed=None, return_raw=False, history=None):
    """Send a chat completion to Ollama's OpenAI-compatible endpoint.

    Same interface as _call_local_chat. Ollama handles model lifecycle
    internally — no server spawn/kill needed.

    `history` (optional): extra messages spliced between system and user
    (multi-turn chaining); None → the pre-chain [system?, user] payload.
    `return_raw` (optional): when True, return (parsed_text, raw_content,
    messages_sent) instead of parsed_text.
    """
    url = f"{ollama_url}/v1/chat/completions"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_content})

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

    raw_content = result["choices"][0]["message"]["content"]
    if raw_content is None:
        raw_content = ""
    text = raw_content.strip()

    if not text:
        raise RuntimeError("Ollama returned empty response")

    parsed = _parse_rewritten_text(text)
    if return_raw:
        return (parsed, raw_content, messages)
    return parsed


def _parse_rewritten_text(text):
    """4-stage response parser shared by the local backends (ollama/llamacpp).

    Same extraction logic as _call_openrouter's inline parser:
      1. json.loads → parsed.get("rewritten_text")
      2. fenced-JSON extraction (```json { ... } ``` anywhere)
      3. bare {"rewritten_text" ...} object scan
      4. prose strip (drop a trailing ``` fence)
    """
    if not text:
        return text
    # Strip reasoning/thinking blocks BEFORE extraction — thinking models
    # (glimmer --reasoning, qwen enable_thinking) can leak reasoning into the
    # content; it must never reach the final prompt. Covers llama.cpp's
    # <reasoning> markers and qwen's <think> (incl. 干-prefixed) markers.
    stripped = re.sub(
        r"<reasoning>.*?</reasoning>|<think>.*?</think>|干think.*?干/think>|干reasoning.*?干/reasoning>",
        "", text, flags=re.DOTALL)
    text = stripped
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


def _call_llama_chat(llama_url, model_name, user_content, system_prompt="",
                     temperature=0.7, max_tokens=2048, timeout=180,
                     seed=None, retries=2, return_raw=False, history=None):
    """Send a chat completion to llama.cpp's llama-server OpenAI endpoint.

    llama-server speaks the same /v1/chat/completions contract as Ollama —
    identical messages payload (multimodal content parts, base64 data-URL
    images), same 429/502/503/504 + URLError/TimeoutError retry policy with
    2^attempt backoff, same 4-stage response parser. `seed` is honored
    per-request by llama-server.

    `history` (optional): extra messages spliced between system and user
    (multi-turn chaining); None → the pre-chain [system?, user] payload.
    `return_raw` (optional): when True, return (parsed_text, raw_content,
    messages_sent) instead of parsed_text.
    """
    url = f"{llama_url}/v1/chat/completions"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_content})

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
    for attempt in range(1 + retries):
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            last_err = f"llama.cpp error {e.code}: {body[:500]}"
            if e.code in (429, 502, 503, 504) and attempt < retries:
                _time.sleep(2 ** attempt)
                continue
            raise RuntimeError(last_err) from e
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = str(e)
            if attempt < retries:
                _time.sleep(2 ** attempt)
                continue
            raise RuntimeError(
                f"llama.cpp request failed after {retries + 1} attempts: "
                f"{last_err}") from e

    raw_content = result["choices"][0]["message"]["content"]
    if raw_content is None:
        raw_content = ""
    text = raw_content.strip()

    if not text:
        raise RuntimeError("llama.cpp returned empty response")

    parsed = _parse_rewritten_text(text)
    if return_raw:
        return (parsed, raw_content, messages)
    return parsed


def _call_backend(backend, url, model, system_prompt, user_content,
                  api_key=None, temperature=0.7, max_tokens=2048,
                  timeout=180, retries=2, seed=None,
                  return_raw=False, history=None):
    """Unified LLM dispatch — the ONE seam H3 nodes call.

    Routes to ollama / llamacpp / openrouter. Tests monkeypatch this single
    symbol to record payloads and assert the routing choice.

    `return_raw` / `history` are threaded through to the leaf callers:
    history splices extra messages between system and user (chain mode),
    return_raw returns (parsed_text, raw_content, messages_sent) instead of
    parsed_text. Both default to the pre-chain behavior, byte-identical.
    """
    if backend == "openrouter":
        if not api_key:
            raise ValueError(
                "No API key provided. Set OPENROUTER_API_KEY env var "
                "or connect a key to the api_key input.")
        return _call_openrouter(
            api_key, model, system_prompt, user_content,
            timeout=timeout, temperature=temperature,
            max_tokens=max_tokens, retries=retries, seed=seed,
            return_raw=return_raw, history=history)
    if backend == "ollama":
        return _call_ollama_chat(
            url, model, user_content, system_prompt,
            temperature=temperature, max_tokens=max_tokens,
            timeout=timeout, seed=seed,
            return_raw=return_raw, history=history)
    if backend == "llamacpp":
        return _call_llama_chat(
            url, model, user_content, system_prompt,
            temperature=temperature, max_tokens=max_tokens,
            timeout=timeout, seed=seed, retries=retries,
            return_raw=return_raw, history=history)
    raise ValueError(f"Unknown backend: {backend}")


# ════════════════════════════════════════════════════════════════════════════
# llama.cpp lifecycle — on-demand spawn → serve → shutdown
# (lifted from the fleet wrappers ~/.local/bin/qwen3.8-quetza / glimmer-quetza)
# ════════════════════════════════════════════════════════════════════════════

_LLAMA_READY_TIMEOUT = 300   # seconds to wait for /v1/models after spawn
_LLAMA_POLL_INTERVAL = 2.0
_LLAMA_KILL_GRACE = 5.0      # SIGTERM → SIGKILL grace
_LLAMA_MIN_CTX = 16384       # below this the per-backend default is used


def _resolve_llama_ctx(context_length, defaults):
    """Resolve the node-selected context_length against a backend entry.

    context_length <= 0 (the widget's "auto") → the backend's wrapper default
    (llama_ctx). Explicit values are clamped to the model max (llama_ctx_max).
    """
    default_ctx = defaults.get("llama_ctx", 262144)
    model_max = defaults.get("llama_ctx_max", 262144)
    try:
        ctx = int(context_length or 0)
    except (TypeError, ValueError):
        ctx = 0
    if ctx <= 0:
        return default_ctx
    if ctx < _LLAMA_MIN_CTX:
        logging.warning(
            f"[PromptEnhancer] context_length {ctx} below floor "
            f"{_LLAMA_MIN_CTX} — using backend default {default_ctx}")
        return default_ctx
    if ctx > model_max:
        logging.warning(
            f"[PromptEnhancer] context_length {ctx} exceeds model max "
            f"{model_max} — clamping")
        return model_max
    return ctx


def _probe_llama_server(url, timeout=3):
    """Return True if a HEALTHY llama-server answers GET {url}/v1/models."""
    try:
        req = urllib.request.Request(f"{url}/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _port_from_url(url):
    """Extract the TCP port from an http://host:port URL (default 80)."""
    m = re.search(r":(\d+)$", url)
    return int(m.group(1)) if m else 80


def _port_owner_pid(port):
    """Return the PID owning `port`, or None if free (lsof; socket fallback)."""
    try:
        out = subprocess.run(
            ["lsof", "-ti", f":{port}"], capture_output=True, text=True,
            timeout=5)
        lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        return int(lines[0]) if lines else None
    except Exception:
        pass
    try:  # Fallback: binding the socket succeeds ⇒ port is free.
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", port))
        s.close()
        return None
    except OSError:
        return -1  # busy but owner unknown (lsof missing)


def _proc_name(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(
                errors="replace").strip()
    except Exception:
        return f"PID {pid}"


def _is_llama_server_proc(pid):
    return "llama-server" in _proc_name(pid)


def _log_tail(path, n=30):
    try:
        with open(path, "r", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except Exception as e:
        return f"(cannot read {path}: {e})"


def _clean_spawn_env():
    """Env for the spawned llama-server — full environment minus the
    Quetza-cli/model-endpoint vars that would confuse a bare server."""
    skip = ("QUETZA_", "ANTHROPIC_", "MODEL_NAME", "LOCAL_MODEL_NAME")
    return {k: v for k, v in os.environ.items()
            if not (k.startswith("QUETZA_") or k.startswith("ANTHROPIC_")
                    or k in skip)}


def _build_llama_cmd(defaults, context_length=-1):
    """Assemble the llama-server argv from a LOCAL_MODEL_DEFAULTS entry.

    Flags come verbatim from the wrapper (llama_flags); only the context is
    resolved per-run: qwen → `-c N`; glimmer (llama_yarn) → `-c N` +
    --override-kv context_length=int:N + YaRN --rope-scale per wrapper pattern.
    """
    bin_path = defaults.get("llama_bin")
    model_path = defaults.get("llama_model_path")
    if not bin_path or not model_path:
        raise ValueError(
            f"llamacpp entry missing llama_bin/llama_model_path: "
            f"{defaults.get('label')}")
    ctx = _resolve_llama_ctx(context_length, defaults)
    port = _port_from_url(defaults.get("llama_url", ""))
    cmd = [bin_path, "-m", model_path,
           "--mmproj", defaults.get("llama_mmproj_path", ""),
           "--port", str(port), "-c", str(ctx)]
    if defaults.get("llama_yarn"):
        orig = int(defaults.get("llama_yarn_orig_ctx", 131072))
        rope_scale = max(ctx // orig, 1)
        override_key = defaults.get("llama_override_kv",
                                    "muse-glimmer.context_length")
        cmd += ["--override-kv", f"{override_key}=int:{ctx}",
                "--rope-scaling", "yarn",
                "--yarn-orig-ctx", str(orig),
                "--rope-scale", str(rope_scale)]
    for flag in defaults.get("llama_flags", []):
        cmd.append(str(flag))
    return cmd


def _spawn_llama_server(defaults, context_length=-1,
                        ready_timeout=_LLAMA_READY_TIMEOUT):
    """Spawn (or reuse) a llama-server for a llamacpp backend entry.

    Returns (ownership, pid):
      "reused"  — a HEALTHY server already answers on the port (a fleet wrapper
                  the user started — never killed).
      "spawned" — this call started it; teardown MUST kill pid to free VRAM.

    Raises RuntimeError on port conflict with a non-llama-server process, an
    early server exit (with log tail), or ready-timeout (server killed).
    """
    url = defaults.get("llama_url") or defaults.get("ollama_url")
    if not url:
        raise ValueError(
            f"llamacpp entry has no llama_url: {defaults.get('label')}")
    port = _port_from_url(url)
    log_path = defaults.get("llama_log", f"/tmp/h3o-llamacpp-{port}.log")

    # 1. Healthy server already up → reuse.
    if _probe_llama_server(url):
        logging.info(f"[PromptEnhancer] Reusing llama-server at {url}")
        return "reused", None

    # 2. Port busy — never kill what we didn't start.
    owner = _port_owner_pid(port)
    if owner is not None and owner != 0:
        if owner > 0 and _is_llama_server_proc(owner):
            deadline = time.time() + ready_timeout
            while time.time() < deadline:
                if _probe_llama_server(url):
                    return "reused", None
                time.sleep(_LLAMA_POLL_INTERVAL)
            raise RuntimeError(
                f"llama-server on port {port} (PID {owner}) did not become "
                f"ready within {ready_timeout}s. Check its log: {log_path}")
        raise RuntimeError(
            f"Port {port} is taken by PID {owner} "
            f"({_proc_name(owner)[:80]}) — not a llama-server. A fleet "
            "wrapper (qwen3.8-quetza=8098, glimmer-quetza=8097) or another "
            f"service may be holding it; close it or use a different backend.")

    # 3. Port free → spawn detached (own session, cwd /tmp, log file).
    # Free ComfyUI's VRAM first: llama-server is a separate process ComfyUI
    # cannot see, so its allocation would otherwise collide with the resident
    # video model (cudaMalloc OOM — Glimmer 30B after an H3 run).
    try:
        import comfy.model_management as _mm
        _mm.unload_all_models()
        _mm.soft_empty_cache()
        logging.info("[PromptEnhancer] Unloaded ComfyUI models for llama-server spawn")
    except Exception as _e:  # never block the spawn on cleanup failures
        logging.warning(f"[PromptEnhancer] VRAM cleanup skipped: {_e}")
    cmd = _build_llama_cmd(defaults, context_length)
    logging.info(
        f"[PromptEnhancer] Spawning llama-server on port {port}: "
        f"{os.path.basename(defaults.get('llama_model_path', ''))} "
        f"ctx={_resolve_llama_ctx(context_length, defaults)}")
    try:
        log_file = open(log_path, "wb")
    except OSError as e:
        raise RuntimeError(
            f"Cannot open llama-server log {log_path}: {e}") from e
    proc = subprocess.Popen(
        cmd, cwd="/tmp", env=_clean_spawn_env(),
        stdout=log_file, stderr=subprocess.STDOUT,
        start_new_session=True)
    log_file.close()

    deadline = time.time() + ready_timeout
    while time.time() < deadline:
        if _probe_llama_server(url):
            logging.info(
                f"[PromptEnhancer] llama-server ready (PID {proc.pid})")
            return "spawned", proc.pid
        if proc.poll() is not None:
            tail = _log_tail(log_path)
            raise RuntimeError(
                f"llama-server exited early (code {proc.returncode}) on port "
                f"{port}. Log tail ({log_path}):\n{tail}")
        time.sleep(_LLAMA_POLL_INTERVAL)

    _kill_llama_server(proc.pid)
    raise RuntimeError(
        f"llama-server on port {port} failed to become ready within "
        f"{ready_timeout}s. Log tail ({log_path}):\n{_log_tail(log_path)}")


def _kill_llama_server(pid):
    """Terminate a llama-server we spawned: SIGTERM → 5s grace → SIGKILL.

    Verifies the process is gone afterwards and logs the lifecycle. Safe to
    call with a stale/None pid.
    """
    if not pid:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        logging.info(f"[PromptEnhancer] llama-server PID {pid} already gone")
        return
    except OSError as e:
        logging.warning(
            f"[PromptEnhancer] SIGTERM to llama-server PID {pid} failed: {e}")
        return
    try:
        os.waitpid(pid, 0)
    except (ChildProcessError, OSError):
        pass  # not our child (reused) or already reaped — poll below
    deadline = time.time() + _LLAMA_KILL_GRACE
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            logging.info(
                f"[PromptEnhancer] llama-server PID {pid} stopped — VRAM freed")
            return
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
        logging.warning(
            f"[PromptEnhancer] llama-server PID {pid} SIGKILL after grace")
    except OSError as e:
        logging.warning(
            f"[PromptEnhancer] SIGKILL llama-server PID {pid} failed: {e}")
