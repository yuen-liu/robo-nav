"""
Shared VLM calls for robo-nav: images in, schema-validated JSON out.

Backend is picked by ROBO_NAV_VLM: "claude" (default) or "gemini". Callers always build
Anthropic-style content blocks; the Gemini path translates them.

  claude  ANTHROPIC_API_KEY or an `ant auth login` profile. Requests opt into server-side
          fallback, so a safety decline is retried on Anthropic's recommended fallback model
          instead of failing the experiment.
  gemini  GEMINI_API_KEY (or GOOGLE_API_KEY). Model from ROBO_NAV_GEMINI_MODEL.
"""

import base64
import io
import json
import os
from typing import List, Optional, Union

import numpy as np

MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
GEMINI_MODEL = os.environ.get("ROBO_NAV_GEMINI_MODEL", "gemini-4-argon")
# Gemini thinking levels are low/medium/high; Claude's xhigh/max map to high.
GEMINI_THINKING = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}


def backend() -> str:
    b = os.environ.get("ROBO_NAV_VLM", "claude").lower()
    if b not in ("claude", "gemini"):
        raise ValueError(f"ROBO_NAV_VLM must be claude or gemini, got {b!r}")
    return b


def client():
    if backend() == "gemini":
        from google import genai
        return genai.Client()
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
             max_tokens: int = 16000, effort: str = "medium") -> dict:
    """One user turn -> JSON matching `schema`. `system` may be a list of blocks (for caching)."""
    if backend() == "gemini":
        return _ask_json_gemini(cl, content, schema, system, max_tokens, effort)
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


def _ask_json_gemini(cl, content: List[dict], schema: dict, system, max_tokens: int, effort: str) -> dict:
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
    cand = response.candidates[0] if response.candidates else None
    reason = cand.finish_reason if cand else None
    if reason == types.FinishReason.MAX_TOKENS:
        raise RuntimeError("Gemini hit max_output_tokens before finishing the JSON answer")
    if cand is None or reason != types.FinishReason.STOP:
        raise RuntimeError(f"Gemini declined: finish_reason={reason} feedback={response.prompt_feedback}")
    return json.loads(response.text)


def obj_schema(properties: dict, required: Optional[list] = None) -> dict:
    return {"type": "object", "properties": properties,
            "required": required if required is not None else list(properties),
            "additionalProperties": False}
