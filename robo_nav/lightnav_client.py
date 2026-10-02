"""
Minimal client for a LightNav-0 `lightnav-serve` WebSocket server.

Protocol (docs/PROTOCOL.md in lightorigins/LightNav-0):
  -> {"action": "login", "data": {"clientId": ...}}
  -> {"action": "reset", "data": {}}                       # every episode boundary
  -> {"action": "next",  "data": {"seq", "image": b64 JPEG, "instruction"}}
     instruction None/"" = buffer-only (frame stored, no inference)
  <- data.actions.actions: [[forward_m, lateral_m(+left), yaw_rad(+ccw)], ...]
     cumulative, robot-local relative to the pose at the sent frame.

Usage (server on a GPU box, tunneled to localhost:8050):
  python -m robo_nav.lightnav_client tm4_data/part1.mp4 \
      --instruction "go to the microkitchen" --fps 4 --show
"""

import argparse
import asyncio
import base64
import json
import os
from typing import Iterator, List, Optional

import cv2
import numpy as np
import websockets


class LightNavClient:
    def __init__(self, url: str = "ws://127.0.0.1:8050", client_id: str = "robo-nav"):
        self.url = url
        self.client_id = client_id
        self.ws = None
        self.seq = 0

    async def __aenter__(self):
        self.ws = await websockets.connect(self.url, max_size=64 * 1024 * 1024)
        await self._call("login", {"clientId": self.client_id})
        return self

    async def __aexit__(self, *exc):
        await self.ws.close()

    async def _call(self, action: str, data: dict) -> dict:
        await self.ws.send(json.dumps({"action": action, "data": data}))
        resp = json.loads(await self.ws.recv())
        if resp["data"].get("rc", 0) != 0:
            raise RuntimeError(f"LightNav {action} failed: {resp['data']}")
        return resp["data"]

    async def reset(self) -> None:
        self.seq = 0
        await self._call("reset", {})

    async def step(self, frame_bgr: np.ndarray, instruction: Optional[str]) -> dict:
        """Send one frame. Returns the raw response data; `waypoints` is added as an
        (H, 3) array when the server ran inference (None for buffer-only frames)."""
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        assert ok, "JPEG encode failed"
        data = await self._call("next", {
            "seq": self.seq,
            "image": base64.b64encode(buf.tobytes()).decode("ascii"),
            "instruction": instruction,
        })
        self.seq += 1
        acts = data.get("actions")
        data["waypoints"] = np.asarray(acts["actions"], dtype=np.float32) if acts else None
        return data


def iter_frames(source: str, fps: float) -> Iterator[np.ndarray]:
    """Yield BGR frames from a video (subsampled to ~fps) or a folder of images (sorted)."""
    if os.path.isdir(source):
        names = sorted(f for f in os.listdir(source) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        for name in names:
            yield cv2.imread(os.path.join(source, name))
        return
    cap = cv2.VideoCapture(source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    stride = max(1, round(src_fps / fps))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            yield frame
        i += 1
    cap.release()


def draw_prediction(frame_bgr: np.ndarray, data: dict, instruction: str, t: int,
                    display_width: int = 960, plot_range_m: float = 2.0) -> np.ndarray:
    """Overlay one prediction on the frame: `apos` pointing target (assumed to be in
    source-frame pixels) plus a top-down inset of the waypoints (forward = up, +left = left)."""
    h, w = frame_bgr.shape[:2]
    s = display_width / w
    img = cv2.resize(frame_bgr, (display_width, round(h * s)))
    wp = data.get("waypoints")

    apos = (data.get("pointing") or {}).get("apos_px")
    if apos is not None:
        cv2.drawMarker(img, (round(apos[0] * s), round(apos[1] * s)), (0, 0, 255),
                       cv2.MARKER_CROSS, 24, 3)

    # Top-down inset, robot at bottom-center facing up
    size = 220
    inset = np.full((size, size, 3), 30, np.uint8)
    px_per_m = (size - 20) / plot_range_m
    origin = np.array([size // 2, size - 10])
    for r in np.arange(0.5, plot_range_m + 1e-6, 0.5):
        cv2.circle(inset, tuple(int(v) for v in origin), int(r * px_per_m), (70, 70, 70), 1)
    cv2.circle(inset, tuple(int(v) for v in origin), 5, (255, 255, 255), -1)
    if wp is not None:
        pts = [origin] + [origin + np.array([-y, -x]) * px_per_m for x, y, _ in wp]
        pts = np.round(pts).astype(np.int32)
        color = (0, 0, 255) if data.get("stop") else (0, 220, 0)
        cv2.polylines(inset, [pts], False, color, 2)
        for p in pts[1:]:
            cv2.circle(inset, tuple(int(v) for v in p), 3, color, -1)
    img[10:10 + size, img.shape[1] - size - 10:img.shape[1] - 10] = inset

    status = "buffering" if wp is None else (
        f"stop={data.get('stop')}  {data.get('latency_ms', 0):.0f}ms")
    for i, line in enumerate([f"[{t:04d}] {status}", instruction]):
        org = (10, 28 + 26 * i)
        cv2.putText(img, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return img


async def run_open_loop(source: str, instruction: str, url: str, fps: float, warmup: int,
                        show: bool = False) -> List[dict]:
    """Stream a recorded clip through the model open-loop and log each prediction.
    The first `warmup` frames are buffer-only to fill the history window.
    With `show`, a live window draws each prediction as it arrives (q / Esc to stop)."""
    results = []
    async with LightNavClient(url) as client:
        await client.reset()
        for t, frame in enumerate(iter_frames(source, fps)):
            data = await client.step(frame, None if t < warmup else instruction)
            if show:
                cv2.imshow("LightNav", draw_prediction(frame, data, instruction, t))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if data["waypoints"] is None:
                continue
            wp = data["waypoints"]
            print(f"[{t:04d}] stop={data['stop']} lat={data['latency_ms']:.0f}ms "
                  f"wp0=({wp[0, 0]:+.2f}, {wp[0, 1]:+.2f}, {wp[0, 2]:+.2f}) "
                  f"end=({wp[-1, 0]:+.2f}, {wp[-1, 1]:+.2f}) "
                  f"apos={(data.get('pointing') or {}).get('apos_px')}")
            results.append({"t": t, "stop": data["stop"], "waypoints": wp.tolist(),
                            "pointing": data.get("pointing"), "raw_text": data.get("raw_text")})
    if show:
        cv2.destroyAllWindows()
    return results


def main():
    parser = argparse.ArgumentParser(description="Stream a clip through a LightNav-0 server (open-loop)")
    parser.add_argument("source", type=str, help="Video file or folder of frames")
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--url", type=str, default="ws://127.0.0.1:8050")
    parser.add_argument("--fps", type=float, default=4.0, help="Subsample video to this rate (LightNav examples use 4)")
    parser.add_argument("--warmup", type=int, default=0, help="Buffer-only frames before the first prediction")
    parser.add_argument("--out", type=str, default=None, help="Optional JSON path to dump predictions")
    parser.add_argument("--show", action="store_true", help="Live window with waypoints + pointing target (q to quit)")
    args = parser.parse_args()

    results = asyncio.run(run_open_loop(args.source, args.instruction, args.url, args.fps, args.warmup,
                                        show=args.show))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"Wrote {len(results)} predictions to {args.out}")


if __name__ == "__main__":
    main()
