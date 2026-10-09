"""
Episode video for `robo_nav.sim.nav`: robot camera + chase camera on the left, a top-down floor plan on
the right (room polygons, doorways, planned route, driven trajectory), and a caption with the
current instruction. Frames come from the sim's /api/camera.jpg and /api/third-person.jpg.
"""

import asyncio
import math
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import requests

CAM_W, CAM_H = 480, 270
MAP_SIZE = 2 * CAM_H
CAPTION_H = 76
FONT = cv2.FONT_HERSHEY_SIMPLEX

BG = (32, 28, 26)
ROOM_FILL = (70, 62, 58)
ROOM_EDGE = (150, 140, 135)
GOAL_FILL = (60, 110, 60)
ROUTE = (0, 200, 255)
TRAIL = (255, 190, 60)
ROBOT = (255, 255, 255)
OUTCOME_COLORS = {"success": (90, 210, 90)}


def open_video_writer(path: str, fps: float, size: tuple) -> "cv2.VideoWriter":
    """H.264 where OpenCV can encode it (plays everywhere), else MPEG-4 Part 2. Some Linux OpenCV
    builds map 'avc1' to a hardware encoder (h264_v4l2m2m) that has no device and fails to open."""
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"no working video encoder for {path}")


def text(img, s, org, scale=0.55, color=(235, 235, 235), thick=1):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


class FloorPlan:
    def __init__(self, rooms: List[dict], doors: List[dict], size: int = MAP_SIZE, pad: int = 24):
        pts = np.array([p for r in rooms for p in r["polygon"]], dtype=float)
        self.lo, hi = pts.min(0), pts.max(0)
        self.scale = (size - 2 * pad) / (hi - self.lo).max()
        self.size, self.pad = size, pad
        self.rooms, self.doors = rooms, doors

    def px(self, x: float, y: float) -> tuple:
        """World (x, y) -> image (col, row); world +y is up on screen."""
        return (int(self.pad + (x - self.lo[0]) * self.scale),
                int(self.size - self.pad - (y - self.lo[1]) * self.scale))

    def draw(self, goal: str, route: List[tuple], trail: List[tuple], pose: Optional[dict]) -> np.ndarray:
        img = np.full((self.size, self.size, 3), BG, np.uint8)
        for r in self.rooms:
            poly = np.array([self.px(*p) for p in r["polygon"]], np.int32)
            cv2.fillPoly(img, [poly], GOAL_FILL if r["name"] == goal else ROOM_FILL)
            cv2.polylines(img, [poly], True, ROOM_EDGE, 2, cv2.LINE_AA)
        for d in self.doors:
            cv2.circle(img, self.px(*d["center"]), 5, (200, 200, 200), -1, cv2.LINE_AA)
        for r in self.rooms:
            c = np.array(r["polygon"]).mean(0)
            col, row = self.px(*c)
            label = r["name"]
            (w, _), _ = cv2.getTextSize(label, FONT, 0.4, 1)
            text(img, label, (col - w // 2, row), 0.4, (220, 220, 220))
        if len(route) > 1:
            cv2.polylines(img, [np.array([self.px(*p) for p in route], np.int32)], False, ROUTE, 2, cv2.LINE_AA)
        if len(trail) > 1:
            cv2.polylines(img, [np.array([self.px(*p) for p in trail], np.int32)], False, TRAIL, 2, cv2.LINE_AA)
        if pose is not None:
            c = self.px(pose["x"], pose["y"])
            tip = self.px(pose["x"] + 0.6 * math.cos(pose["yaw"]), pose["y"] + 0.6 * math.sin(pose["yaw"]))
            cv2.circle(img, c, 7, ROBOT, -1, cv2.LINE_AA)
            cv2.arrowedLine(img, c, tip, ROBOT, 2, cv2.LINE_AA, tipLength=0.4)
        return img


class EpisodeRecorder:
    """Polls the sim cameras at `fps` and writes a composited H.264 MP4.

    `status` is a dict the navigation loop keeps updating: goal, room, instruction, route
    (list of world points), to_go (rooms left on the plan), outcome."""

    def __init__(self, path: str, sim_http: str, plan: FloorPlan, status: Dict,
                 pose: Callable[[], dict], fps: float = 10.0):
        self.path, self.sim_http, self.plan, self.status, self.pose = path, sim_http, plan, status, pose
        self.fps = fps
        self.trail: List[tuple] = []
        self.size = (CAM_W + MAP_SIZE, MAP_SIZE + CAPTION_H)
        self.writer = open_video_writer(path, fps, self.size)
        self.session = requests.Session()
        self.last = None
        self.t0 = None
        self._task = None

    def _grab(self, name: str) -> np.ndarray:
        try:
            buf = self.session.get(f"{self.sim_http}/api/{name}", timeout=1).content
            img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                return cv2.resize(img, (CAM_W, CAM_H))
        except requests.RequestException:
            pass
        return np.zeros((CAM_H, CAM_W, 3), np.uint8)

    def compose(self, first: np.ndarray, third: np.ndarray, elapsed: float) -> np.ndarray:
        st = self.status
        pose = self.pose()
        if pose is not None:
            xy = (pose["x"], pose["y"])
            if not self.trail or math.hypot(xy[0] - self.trail[-1][0], xy[1] - self.trail[-1][1]) > 0.03:
                self.trail.append(xy)
        frame = np.full((self.size[1], self.size[0], 3), BG, np.uint8)
        frame[:CAM_H, :CAM_W] = first
        frame[CAM_H:2 * CAM_H, :CAM_W] = third
        plan_img = self.plan.draw(st.get("goal"), st.get("route", []), self.trail, pose)
        if st.get("goal_xy"):  # goal object (landmark runs): star on the floor plan
            cv2.drawMarker(plan_img, self.plan.px(*st["goal_xy"]), (80, 220, 255), cv2.MARKER_STAR, 22, 2, cv2.LINE_AA)
        frame[:MAP_SIZE, CAM_W:] = plan_img
        text(frame, "robot camera", (8, 20), 0.5)
        text(frame, "chase camera", (8, CAM_H + 20), 0.5)

        y0 = MAP_SIZE
        to_go = f"rooms to go: {st['to_go']}" if st.get("to_go") is not None else ""
        header = f"goal: {st.get('goal_label') or st.get('goal', '')}   in: {st.get('room') or '?'}   {to_go}   t={elapsed:4.1f}s"
        text(frame, header, (10, y0 + 26), 0.55, (200, 200, 200))
        instr = st.get("instruction", "")
        if len(instr) > 95:
            instr = instr[:92] + "..."
        text(frame, f"LightNav-0: \"{instr}\"", (10, y0 + 58), 0.6, ROUTE)

        if st.get("outcome"):
            label = st["outcome"].upper().replace("_", " ")
            color = OUTCOME_COLORS.get(st["outcome"], (80, 80, 240))
            (w, h), _ = cv2.getTextSize(label, FONT, 1.4, 3)
            x0, y0 = CAM_W + (MAP_SIZE - w) // 2, MAP_SIZE // 2 + h // 2
            cv2.rectangle(frame, (x0 - 16, y0 - h - 16), (x0 + w + 16, y0 + 16), BG, -1)
            cv2.rectangle(frame, (x0 - 16, y0 - h - 16), (x0 + w + 16, y0 + 16), color, 2)
            text(frame, label, (x0, y0), 1.4, color, 3)
        return frame

    async def _run(self):
        loop = asyncio.get_running_loop()
        self.t0 = loop.time()
        period = 1.0 / self.fps
        next_t = self.t0
        while True:
            first, third = await asyncio.gather(asyncio.to_thread(self._grab, "camera.jpg"),
                                                asyncio.to_thread(self._grab, "third-person.jpg"))
            self.last = self.compose(first, third, loop.time() - self.t0)
            self.writer.write(self.last)
            next_t += period
            await asyncio.sleep(max(0.0, next_t - loop.time()))
            while loop.time() - next_t > period:  # fell behind: duplicate to keep real-time pacing
                self.writer.write(self.last)
                next_t += period

    def start(self):
        self._task = asyncio.create_task(self._run())

    async def close(self, hold_s: float = 2.5):
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self.last is not None:  # re-render the final frame with the outcome banner and hold it
            first = self.last[:CAM_H, :CAM_W].copy()
            third = self.last[CAM_H:2 * CAM_H, :CAM_W].copy()
            final = self.compose(first, third, asyncio.get_running_loop().time() - self.t0)
            for _ in range(int(hold_s * self.fps)):
                self.writer.write(final)
        self.writer.release()
