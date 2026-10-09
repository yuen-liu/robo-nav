"""
Build a landmark map from a walkthrough video: Claude watches the frames in order and annotates
the landmarks itself -- objects, furniture, doors, wall corners -- and how they connect along the
walk. No poses, no room names: the result is purely relative ("past the grey couch, the wooden
double doors are ahead").

Works on the sim's rendered mapping tour or on a real phone recording:
  python -m robo_nav.landmark.annotate --frames sim_data/loc/val2/map/frames --every 10 --out sim_data/landmark/val2
  python -m robo_nav.landmark.annotate --video tm4_data/part1.mp4 --fps 1 --out sim_data/landmark/part1

Output (<out>/):
  keyframes/k0000.jpg ...   the frames Claude saw (k-index = order in the walk)
  chunks.json               per-chunk raw annotations (cached; reruns skip finished chunks)
  landmark_map.json         consolidated map: landmarks, walk timeline, connections
  landmark_map.md           the same, readable (this text is what the navigator is given)
"""

import argparse
import json
import os
from glob import glob
from typing import List

from robo_nav.claude import ask_json, client as make_client, image_block, obj_schema

NO_ROOMS = ("Never name rooms or room types (no 'kitchen', 'bedroom', 'bathroom', 'living room', "
            "'office', 'hallway'): describe only physical things -- furniture, appliances, objects, "
            "doors and doorways, wall corners, windows, floor and wall finishes, colors, sizes.")


def extract_keyframes(args) -> List[str]:
    out_dir = os.path.join(args.out, "keyframes")
    existing = sorted(glob(os.path.join(out_dir, "k*.jpg")))
    if existing:
        return existing
    os.makedirs(out_dir, exist_ok=True)
    from PIL import Image
    paths = []
    if args.frames:
        src = sorted(p for p in glob(os.path.join(args.frames, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png")))
        for k, p in enumerate(src[::args.every]):
            dst = os.path.join(out_dir, f"k{k:04d}.jpg")
            Image.open(p).convert("RGB").save(dst, quality=90)
            paths.append(dst)
    else:
        import cv2
        cap = cv2.VideoCapture(args.video)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(1, round(src_fps / args.fps))
        i = k = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                if args.rotate_cw:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                h, w = frame.shape[:2]
                if w > args.max_width:
                    frame = cv2.resize(frame, (args.max_width, round(h * args.max_width / w)), interpolation=cv2.INTER_AREA)
                dst = os.path.join(out_dir, f"k{k:04d}.jpg")
                cv2.imwrite(dst, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                paths.append(dst)
                k += 1
            i += 1
        cap.release()
    return paths


CHUNK_SCHEMA = obj_schema({
    "segments": {"type": "array", "items": obj_schema({
        "start": {"type": "integer", "description": "first k-index of the segment"},
        "end": {"type": "integer", "description": "last k-index of the segment"},
        "motion": {"type": "string", "description": "how the camera moves, e.g. 'forward', 'turning left', 'turning right in place', 'through a doorway'"},
        "landmarks": {"type": "array", "items": obj_schema({
            "name": {"type": "string", "description": "short unique name, reused across the walk for the same thing"},
            "where": {"type": "string", "description": "relative to the camera: ahead / left / right / passing on the left / behind ..."},
        })},
        "note": {"type": "string", "description": "anything that would help someone retrace this stretch"},
    })},
    "new_landmarks": {"type": "array", "items": obj_schema({
        "name": {"type": "string"},
        "description": {"type": "string", "description": "what it looks like: color, material, size, shape, what is near it"},
    })},
})


def annotate_chunks(cl, keyframes: List[str], args) -> list:
    cache = os.path.join(args.out, "chunks.json")
    chunks = json.load(open(cache)) if os.path.exists(cache) else []
    known = {lm["name"]: lm["description"] for c in chunks for lm in c["new_landmarks"]}
    step = args.chunk - args.overlap
    starts = list(range(0, max(1, len(keyframes) - args.overlap), step))
    for ci, start in enumerate(starts):
        if ci < len(chunks):
            continue
        ks = list(range(start, min(start + args.chunk, len(keyframes))))
        content = [{"type": "text", "text":
                    f"Frames {ks[0]}-{ks[-1]} of a {len(keyframes)}-frame walk through one building, in order. "
                    f"The camera is {args.camera}. Each frame is labeled with its k-index."}]
        for k in ks:
            content.append({"type": "text", "text": f"k{k:04d}"})
            content.append(image_block(keyframes[k], args.max_width))
        content.append({"type": "text", "text":
                        "Split these frames into segments of continuous motion and list the landmarks visible in "
                        "each, with where they are relative to the camera. Landmarks are things someone could "
                        "navigate by: furniture, appliances, large objects, doors and doorways, wall corners, "
                        "windows, distinctive floor or wall areas. Skip small clutter. " + NO_ROOMS + "\n\n"
                        "Landmarks already named earlier in the walk (reuse these exact names when it is the same "
                        "physical thing; only add genuinely new ones to new_landmarks):\n" +
                        ("\n".join(f"- {n}: {d}" for n, d in known.items()) or "(none yet)")})
        res = ask_json(cl, content, CHUNK_SCHEMA, max_tokens=32000, effort=args.effort)
        res["frames"] = [ks[0], ks[-1]]
        chunks.append(res)
        known.update({lm["name"]: lm["description"] for lm in res["new_landmarks"]})
        with open(cache, "w") as f:
            json.dump(chunks, f, indent=1)
        print(f"  chunk {ci + 1}/{len(starts)} (k{ks[0]}-k{ks[-1]}): "
              f"{len(res['segments'])} segments, {len(res['new_landmarks'])} new landmarks", flush=True)
    return chunks


MAP_SCHEMA = obj_schema({
    "landmarks": {"type": "array", "items": obj_schema({
        "name": {"type": "string"},
        "description": {"type": "string"},
        "seen_at": {"type": "array", "items": {"type": "integer"}, "description": "k-indices where it is clearly visible"},
    })},
    "connections": {"type": "array", "items": obj_schema({
        "from": {"type": "string"},
        "to": {"type": "string"},
        "how": {"type": "string", "description": "how to get from one to the other using only landmarks and turns"},
    })},
    "walk_summary": {"type": "array", "items": {"type": "string"},
                     "description": "the whole walk retold as ordered landmark-to-landmark steps"},
})


def consolidate(cl, chunks: list, n_frames: int, args) -> dict:
    content = [{"type": "text", "text":
                f"Below are frame-by-frame annotations of one {n_frames}-frame walk through a building "
                f"(camera {args.camera}), written in chunks. Merge them into a single landmark map:\n"
                "1. landmarks: one entry per physical landmark (merge duplicates that are clearly the same "
                "thing under different names; keep same-looking but different things separate).\n"
                "2. connections: for landmarks that follow each other along the walk, how to get from one to the "
                "next (both directions where the walk covers both), e.g. 'with the grey couch on your left, go "
                "straight to the wooden double doors'.\n"
                "3. walk_summary: the walk as ordered landmark-to-landmark steps.\n" + NO_ROOMS + "\n\n" +
                json.dumps(chunks)}]
    return ask_json(cl, content, MAP_SCHEMA, max_tokens=64000, effort=args.effort)


def to_markdown(m: dict) -> str:
    lines = ["# Landmarks"] + [f"- **{lm['name']}**: {lm['description']}" for lm in m["landmarks"]]
    lines += ["", "# How landmarks connect"] + [f"- {c['from']} -> {c['to']}: {c['how']}" for c in m["connections"]]
    lines += ["", "# The recorded walk"] + [f"{i + 1}. {s}" for i, s in enumerate(m["walk_summary"])]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Claude-annotated landmark map from a walkthrough video")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=str, help="Recording (.mp4)")
    src.add_argument("--frames", type=str, help="Folder of ordered frames (e.g. the sim mapping tour)")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--fps", type=float, default=1.0, help="Keyframes per second of video")
    parser.add_argument("--every", type=int, default=10, help="Keep every Nth frame of a frame folder")
    parser.add_argument("--rotate_cw", action="store_true", help="Rotate video frames 90 deg clockwise")
    parser.add_argument("--max_width", type=int, default=768)
    parser.add_argument("--chunk", type=int, default=30, help="Frames per annotation request")
    parser.add_argument("--overlap", type=int, default=3)
    parser.add_argument("--camera", type=str, default="held at chest height by a person walking",
                        help="How the camera is mounted (said to Claude); sim: 'on a small robot, ~20 cm above the floor'")
    parser.add_argument("--effort", type=str, default="medium")
    args = parser.parse_args()

    keyframes = extract_keyframes(args)
    print(f"{len(keyframes)} keyframes in {args.out}/keyframes")
    cl = make_client()
    chunks = annotate_chunks(cl, keyframes, args)
    m = consolidate(cl, chunks, len(keyframes), args)
    m["source"] = {"video": args.video, "frames": args.frames, "n_keyframes": len(keyframes), "camera": args.camera}
    with open(os.path.join(args.out, "landmark_map.json"), "w") as f:
        json.dump(m, f, indent=1)
    with open(os.path.join(args.out, "landmark_map.md"), "w") as f:
        f.write(to_markdown(m))
    print(f"{len(m['landmarks'])} landmarks, {len(m['connections'])} connections -> {args.out}/landmark_map.md")


if __name__ == "__main__":
    main()
