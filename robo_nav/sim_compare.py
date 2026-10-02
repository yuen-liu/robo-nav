"""
Side-by-side, sped-up comparison of two `sim_nav --video` episodes. Both play at the same speed
from t=0, so the shorter episode finishes first and holds its final frame (outcome banner) while
the other keeps going: the time difference stays visible.

  python -m robo_nav.sim_compare sim_data/runs/batch2/kitchen_direction_r1.mp4 \
      sim_data/runs/batch2/kitchen_none_r1.mp4 -o kitchen_compare.mp4 \
      --left_title "With room-graph directions" --right_title "Plain instruction"
"""

import argparse

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = (32, 28, 26)


def read_all(path: str):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit(f"No frames in {path}")
    return frames, fps


def even(n: int) -> int:
    return n - n % 2


def main():
    parser = argparse.ArgumentParser(description="Sped-up side-by-side of two episode videos")
    parser.add_argument("left", type=str)
    parser.add_argument("right", type=str)
    parser.add_argument("-o", "--out", type=str, required=True)
    parser.add_argument("--left_title", type=str, default="")
    parser.add_argument("--right_title", type=str, default="")
    parser.add_argument("--speed", type=float, default=8.0)
    parser.add_argument("--scale", type=float, default=0.8, help="Resize each panel by this factor")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--hold", type=float, default=2.0, help="Seconds to hold the final frame")
    args = parser.parse_args()

    (lf, lfps), (rf, rfps) = read_all(args.left), read_all(args.right)
    h, w = lf[0].shape[:2]
    pw, ph = even(int(w * args.scale)), even(int(h * args.scale))
    header = 56
    gap = 8
    size = (2 * pw + gap, ph + header)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"avc1"), args.fps, size)

    duration = max(len(lf) / lfps, len(rf) / rfps) / args.speed + args.hold
    for i in range(int(duration * args.fps)):
        src_t = i / args.fps * args.speed
        canvas = np.full((size[1], size[0], 3), BG, np.uint8)
        for k, (frames, fps, title) in enumerate(((lf, lfps, args.left_title), (rf, rfps, args.right_title))):
            frame = frames[min(int(src_t * fps), len(frames) - 1)]
            x0 = k * (pw + gap)
            canvas[header:, x0:x0 + pw] = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
            (tw, _), _ = cv2.getTextSize(title, FONT, 0.8, 2)
            cv2.putText(canvas, title, (x0 + (pw - tw) // 2, 38), FONT, 0.8, (240, 240, 240), 2, cv2.LINE_AA)
        badge = f"{args.speed:g}x speed"
        (bw, _), _ = cv2.getTextSize(badge, FONT, 0.55, 1)
        cv2.putText(canvas, badge, (size[0] - bw - 10, 20), FONT, 0.55, (170, 170, 170), 1, cv2.LINE_AA)
        writer.write(canvas)
    writer.release()
    print(f"Wrote {args.out} ({duration:.1f}s, {size[0]}x{size[1]})")


if __name__ == "__main__":
    main()
