"""
Shared Claude calls for robo-nav: images in, schema-validated JSON out.

Credentials come from the environment (ANTHROPIC_API_KEY or an `ant auth login` profile).
Requests opt into server-side fallback, so a safety decline is retried on Anthropic's
recommended fallback model instead of failing the experiment.
"""

import base64
import io
import json
from typing import List, Optional, Union

import numpy as np

MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def client():
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


def obj_schema(properties: dict, required: Optional[list] = None) -> dict:
    return {"type": "object", "properties": properties,
            "required": required if required is not None else list(properties),
            "additionalProperties": False}
