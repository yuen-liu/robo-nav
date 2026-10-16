"""
Shared VLM calls for robo-nav: images in, schema-validated JSON out.

Backend is picked by ROBO_NAV_VLM: "claude" (default), "gemini" or "qwen". Callers always build
Anthropic-style content blocks; the Gemini and Qwen paths translate them.

  claude  ANTHROPIC_API_KEY or an `ant auth login` profile. Requests opt into server-side
          fallback, so a safety decline is retried on Anthropic's recommended fallback model
          instead of failing the experiment.
  gemini  GEMINI_API_KEY (or GOOGLE_API_KEY). Model from ROBO_NAV_GEMINI_MODEL.
  qwen    a local vLLM OpenAI-compatible server (no API key, no API bill), e.g.
            vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000 --max-model-len 65536 \
                --limit-mm-per-prompt '{"image": 40}'
          URL from ROBO_NAV_QWEN_URL, model from ROBO_NAV_QWEN_MODEL (must match what is served).
          JSON is enforced with vLLM structured outputs; --effort is ignored.
"""

import base64
import datetime
import io
import json
import os
import threading
from typing import List, Optional, Union

import numpy as np
import requests

MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
GEMINI_MODEL = os.environ.get("ROBO_NAV_GEMINI_MODEL", "gemini-4-argon")
# Gemini thinking levels are low/medium/high; Claude's xhigh/max map to high.
GEMINI_THINKING = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}
QWEN_URL = os.environ.get("ROBO_NAV_QWEN_URL", "http://127.0.0.1:8000/v1")
QWEN_MODEL = os.environ.get("ROBO_NAV_QWEN_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
# Prompt + max_tokens must fit the server's --max-model-len; callers' 32k/64k caps are for Claude.
QWEN_MAX_TOKENS = int(os.environ.get("ROBO_NAV_QWEN_MAX_TOKENS", "16384"))


# List prices, $ per 1M tokens (as of 2026-10): input, output, cache read, cache write. Output
# includes thinking tokens. A model missing here gets tokens logged but no cost.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "gemini-4-argon": (4.00, 20.00, None, None),  # $2 / $10 introductory through 2026-11-30
}
INTRO_PRICES = {"gemini-4-argon": ((2.00, 10.00, None, None), datetime.date(2026, 11, 30))}

USAGE: List[dict] = []  # every call this process made (see usage_summary)
_usage_lock = threading.Lock()


def _record(usage: Optional[list], model: str, inp: int, out: int, cache_read: int = 0, cache_write: int = 0) -> dict:
    u = {"model": model, "input": int(inp or 0), "output": int(out or 0),
         "cache_read": int(cache_read or 0), "cache_write": int(cache_write or 0)}
    u["cost_usd"] = call_cost(u)
    with _usage_lock:
        USAGE.append(u)
    if usage is not None:
        usage.append(u)
    return u


def call_cost(u: dict) -> Optional[float]:
    if backend() == "qwen":
        return 0.0  # local vLLM: no API bill
    price = PRICES.get(u["model"])
    if u["model"] in INTRO_PRICES and datetime.date.today() <= INTRO_PRICES[u["model"]][1]:
        price = INTRO_PRICES[u["model"]][0]
    if price is None:
        return None
    p_in, p_out, p_read, p_write = price
    cost = u["input"] * p_in + u["output"] * p_out
    cost += u["cache_read"] * (p_read if p_read is not None else p_in)
    cost += u["cache_write"] * (p_write if p_write is not None else p_in)
    return cost / 1e6


def usage_summary(calls: Optional[List[dict]] = None) -> dict:
    """Totals over `calls` (default: every call this process made)."""
    calls = USAGE if calls is None else calls
    out = {"calls": len(calls)}
    for k in ("input", "output", "cache_read", "cache_write"):
        out[k] = sum(c[k] for c in calls)
    costs = [c["cost_usd"] for c in calls]
    out["cost_usd"] = round(sum(costs), 4) if costs and None not in costs else None
    return out


def print_cost(eps: List[dict]):
    """Planner token usage and list-price cost over the episodes that logged it (incl. outages)."""
    used = [e["usage"] for e in eps if e.get("usage", {}).get("calls")]
    if not used:
        return
    calls = sum(u["calls"] for u in used)
    tok = {k: sum(u[k] for u in used) for k in ("input", "output", "cache_read", "cache_write")}
    costs = [u["cost_usd"] for u in used]
    cost = f"${sum(costs):.2f} total, ${sum(costs) / len(used):.3f}/episode, ${sum(costs) / calls:.4f}/call" \
        if None not in costs else "cost unknown (model not in PRICES)"
    print(f"\nplanner usage over {len(used)} episodes, {calls} calls: {tok['input']:,} in, {tok['output']:,} out, "
          f"{tok['cache_read']:,} cache read, {tok['cache_write']:,} cache write -> {cost}")


def backend() -> str:
    b = os.environ.get("ROBO_NAV_VLM", "claude").lower()
    if b not in ("claude", "gemini", "qwen"):
        raise ValueError(f"ROBO_NAV_VLM must be claude, gemini or qwen, got {b!r}")
    return b


def model_name() -> str:
    return {"claude": MODEL, "gemini": GEMINI_MODEL, "qwen": QWEN_MODEL}[backend()]


def client():
    if backend() == "qwen":
        return requests.Session()
    if backend() == "gemini":
        from google import genai
        from google.genai import types
        # Gemini sheds load with 503s; retry with backoff instead of failing the episode.
        retry = types.HttpRetryOptions(attempts=10, initial_delay=2.0, max_delay=30.0,
                                       http_status_codes=[429, 500, 503, 504])
        return genai.Client(http_options=types.HttpOptions(retry_options=retry))
    import anthropic
    return anthropic.Anthropic()


def image_block(src: Union[str, bytes, np.ndarray], max_width: int = 768) -> dict:
    """JPEG image block from a path, encoded bytes, or an RGB array; downscaled to max_width."""
    from PIL import Image
    if isinstance(src, np.ndarray):
        img = Image.fromarray(src)
    else:
        img = Image.open(src if isinstance(src, str) else io.BytesIO(src))
    img = img.convert("RGB")
    if img.width > max_width:
        img = img.resize((max_width, round(img.height * max_width / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.standard_b64encode(buf.getvalue()).decode("utf-8")}}


def ask_json(cl, content: List[dict], schema: dict, system: Optional[Union[str, List[dict]]] = None,
             max_tokens: int = 16000, effort: str = "medium", usage: Optional[list] = None) -> dict:
    """One user turn -> JSON matching `schema`. `system` may be a list of blocks (for caching).
    The call's token usage (and list-price cost) is appended to `usage` if given, and to USAGE."""
    if backend() == "gemini":
        return _ask_json_gemini(cl, content, schema, system, max_tokens, effort, usage)
    if backend() == "qwen":
        return _ask_json_qwen(cl, content, schema, system, max_tokens, usage)
    kwargs = {"system": system} if system else {}
    with cl.beta.messages.stream(
        model=MODEL,
        max_tokens=max_tokens,
        betas=[FALLBACK_BETA],
        fallbacks="default",
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": content}],
        **kwargs,
    ) as stream:
        response = stream.get_final_message()
    u = response.usage
    _record(usage, response.model, u.input_tokens, u.output_tokens,
            getattr(u, "cache_read_input_tokens", 0), getattr(u, "cache_creation_input_tokens", 0))
    if response.stop_reason == "refusal":
        raise RuntimeError(f"Claude declined: {response.stop_details}")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Claude hit max_tokens before finishing the JSON answer")
    return json.loads(next(b.text for b in response.content if b.type == "text"))


def _gemini_parts(blocks: List[dict]) -> list:
    """Anthropic text/image blocks -> genai Parts (cache_control is dropped; Gemini caches implicitly)."""
    from google.genai import types
    parts = []
    for b in blocks:
        if b["type"] == "text":
            parts.append(types.Part.from_text(text=b["text"]))
        elif b["type"] == "image":
            src = b["source"]
            parts.append(types.Part.from_bytes(data=base64.standard_b64decode(src["data"]),
                                               mime_type=src["media_type"]))
        else:
            raise ValueError(f"unsupported content block for Gemini: {b['type']}")
    return parts


def _ask_json_gemini(cl, content: List[dict], schema: dict, system, max_tokens: int, effort: str,
                     usage: Optional[list] = None) -> dict:
    from google.genai import types
    if isinstance(system, list):
        system = "\n\n".join(b["text"] for b in system)
    config = types.GenerateContentConfig(
        system_instruction=system or None,
        max_output_tokens=max_tokens,
        response_mime_type="application/json",
        response_json_schema=schema,
        thinking_config=types.ThinkingConfig(thinking_level=GEMINI_THINKING.get(effort, effort)),
    )
    response = cl.models.generate_content(model=GEMINI_MODEL, contents=_gemini_parts(content), config=config)
    m = response.usage_metadata
    if m is not None:  # thinking is billed as output; prompt_token_count includes the cached part
        cached = m.cached_content_token_count or 0
        _record(usage, GEMINI_MODEL, (m.prompt_token_count or 0) - cached,
                (m.candidates_token_count or 0) + (m.thoughts_token_count or 0), cached)
    cand = response.candidates[0] if response.candidates else None
    reason = cand.finish_reason if cand else None
    if reason == types.FinishReason.MAX_TOKENS:
        raise RuntimeError("Gemini hit max_output_tokens before finishing the JSON answer")
    if cand is None or reason != types.FinishReason.STOP:
        raise RuntimeError(f"Gemini declined: finish_reason={reason} feedback={response.prompt_feedback}")
    return json.loads(response.text)


def _openai_parts(blocks: List[dict]) -> list:
    """Anthropic text/image blocks -> OpenAI chat content parts (cache_control dropped)."""
    parts = []
    for b in blocks:
        if b["type"] == "text":
            parts.append({"type": "text", "text": b["text"]})
        elif b["type"] == "image":
            src = b["source"]
            parts.append({"type": "image_url", "image_url": {"url": f"data:{src['media_type']};base64,{src['data']}"}})
        else:
            raise ValueError(f"unsupported content block for Qwen: {b['type']}")
    return parts


def _ask_json_qwen(cl, content: List[dict], schema: dict, system, max_tokens: int,
                   usage: Optional[list] = None, retries: int = 5) -> dict:
    import time
    if isinstance(system, list):
        system = "\n\n".join(b["text"] for b in system)
    messages = ([{"role": "system", "content": system}] if system else []) + \
        [{"role": "user", "content": _openai_parts(content)}]
    body = {"model": QWEN_MODEL, "messages": messages, "max_tokens": min(max_tokens, QWEN_MAX_TOKENS),
            "temperature": 0.0,
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema, "strict": True}}}
    for attempt in range(retries):
        try:
            r = cl.post(f"{QWEN_URL}/chat/completions", json=body, timeout=600)
            if r.status_code < 500:
                break
        except requests.ConnectionError:
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)  # server still loading, or a transient 5xx
    if r.status_code != 200:
        raise RuntimeError(f"Qwen server {r.status_code}: {r.text[:500]}")
    reply = r.json()
    u = reply.get("usage") or {}
    _record(usage, QWEN_MODEL, u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
    choice = reply["choices"][0]
    if choice["finish_reason"] == "length":
        raise RuntimeError("Qwen hit max_tokens before finishing the JSON answer")
    return json.loads(choice["message"]["content"])


def obj_schema(properties: dict, required: Optional[list] = None) -> dict:
    return {"type": "object", "properties": properties,
            "required": required if required is not None else list(properties),
            "additionalProperties": False}
