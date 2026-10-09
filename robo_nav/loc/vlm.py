"""
Vision-language localization for `robo_nav.loc.benchmark`: Claude picks the room from the robot's camera view.

  vlm_text    the map is only text: Claude first writes one distinguishing description per room
              from a few map frames (cached to <dataset>/cache/vlm_room_descriptions.json), then
              each query is matched against those descriptions
  vlm_images  the map is a few exemplar frames per room, shown with every query (prompt-cached)
  *_look      the query is the 8-view look-around instead of a single frame

Needs Anthropic credentials (ANTHROPIC_API_KEY or an `ant auth login` profile). Answers are
cached per query in <dataset>/cache/vlm_<method>.jsonl, so reruns don't re-spend.
"""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np

from robo_nav.claude import ask_json as ask, client as make_client, image_block

def exemplars(ds: dict, per_room: int) -> Dict[str, List[str]]:
    """Evenly spaced map frames per room (the tour order spreads them over the room)."""
    out = {}
    for room in sorted(set(ds["map_room"]) - {""}):
        idx = [i for i, r in enumerate(ds["map_room"]) if r == room]
        pick = np.linspace(0, len(idx) - 1, per_room).round().astype(int)
        out[room] = [ds["map_paths"][idx[k]] for k in pick]
    return out


def room_descriptions(client, ds: dict, cache_dir: str, per_room: int = 8) -> Dict[str, str]:
    path = os.path.join(cache_dir, "vlm_room_descriptions.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    ex = exemplars(ds, per_room)
    content = [{"type": "text", "text":
                "These are camera frames from a small wheeled robot (camera ~20 cm above the floor) "
                "driving through one house. Each group of frames is labeled with its room."}]
    for room, paths in ex.items():
        content.append({"type": "text", "text": f"Room: {room}"})
        content.extend(image_block(p) for p in paths)
    content.append({"type": "text", "text":
                    "For every room, write a description (3-5 sentences) that would let someone identify "
                    "that room from a single low camera view facing any direction. Focus on distinctive, "
                    "view-independent cues: furniture and objects, their colors and materials, floor and "
                    "wall finish, windows and doors. Several rooms share a type (e.g. three bedrooms), so "
                    "say explicitly what tells each one apart from the others of its type."})
    schema = {"type": "object", "properties": {r: {"type": "string"} for r in ex},
              "required": list(ex), "additionalProperties": False}
    desc = ask(client, content, schema, max_tokens=16000)
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(desc, f, indent=1)
    return desc


def evaluate_vlm(ds: dict, method: str, workers: int = 8) -> List[dict]:
    client = make_client()
    look = method.endswith("_look")
    rooms = sorted(set(ds["map_room"]) - {""})
    cache_dir = os.path.join(ds["root"], "cache")
    os.makedirs(cache_dir, exist_ok=True)
    schema = {"type": "object",
              "properties": {"room": {"type": "string", "enum": rooms},
                             "confidence": {"type": "number", "description": "0 to 1"}},
              "required": ["room", "confidence"], "additionalProperties": False}
    question = ("Which room is the robot in? " +
                ("The images are the same spot, turning 45 degrees left between each. " if look else "") +
                "Answer with the room name and your confidence from 0 to 1.")

    if method.startswith("vlm_text"):
        desc = room_descriptions(client, ds, cache_dir)
        system = ("You localize a small indoor robot (camera ~20 cm above the floor) from its camera. "
                  "The house's rooms are described below; several rooms share a type, so rely on the "
                  "distinguishing details.\n\n" + "\n\n".join(f"{r}: {d}" for r, d in desc.items()))
        prefix = []
    else:
        system = ("You localize a small indoor robot (camera ~20 cm above the floor) from its camera. "
                  "Reference frames from every room of the house are given first, labeled by room.")
        prefix = []
        for room, paths in exemplars(ds, 6).items():
            prefix.append({"type": "text", "text": f"Reference room: {room}"})
            prefix.extend(image_block(p) for p in paths)
        prefix[-1] = {**prefix[-1], "cache_control": {"type": "ephemeral"}}  # reuse across queries

    cache_path = os.path.join(cache_dir, f"vlm_{method}.jsonl")
    done: Dict[int, dict] = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                r = json.loads(line)
                done[r["i"]] = r
    lock = threading.Lock()

    def one(i: int) -> dict:
        if i in done:
            return done[i]
        views = ds["q_paths"][i] if look else ds["q_paths"][i][:1]
        content = prefix + [{"type": "text", "text": "Robot camera now:"}] + \
            [image_block(p) for p in views] + [{"type": "text", "text": question}]
        ans = ask(client, content, schema, system=system)
        rec = {"i": i, "room": ans["room"], "conf": float(ans["confidence"])}
        with lock, open(cache_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    n = len(ds["q_room"])
    if prefix:  # write the cache with one request before fanning out
        one(0)
    with ThreadPoolExecutor(workers) as pool:
        recs = list(pool.map(one, range(n)))
    return [{"room": r["room"], "xy": None, "yaw": None, "conf": r["conf"]} for r in recs]
