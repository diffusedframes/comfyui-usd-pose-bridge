"""Small Pillow renderer for OpenPose/ControlNet pose dictionaries.

This implementation draws only supplied, visible landmarks. Colors are assigned
by COCO semantics so adding BODY_25's mid-hip cannot shift leg or face colors.
"""

from __future__ import annotations

import colorsys
import math
from collections.abc import Mapping
from numbers import Real

import numpy as np
from PIL import Image, ImageDraw

from .mapping import BODY_PARTS, COCO_18


# Familiar 18-color COCO rainbow, indexed by COCO landmark semantics.
_PALETTE = (
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85),
)
_JOINT_COLORS = dict(zip(COCO_18, _PALETTE, strict=True))
_JOINT_COLORS.update({"mid_hip": _PALETTE[1], "left_big_toe": _PALETTE[13],
                      "left_small_toe": _PALETTE[13], "left_heel": _PALETTE[13],
                      "right_big_toe": _PALETTE[10], "right_small_toe": _PALETTE[10],
                      "right_heel": _PALETTE[10]})
_COCO_LINKS = (
    ("neck", "right_shoulder"), ("neck", "left_shoulder"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("neck", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("neck", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("neck", "nose"), ("nose", "right_eye"), ("right_eye", "right_ear"),
    ("nose", "left_eye"), ("left_eye", "left_ear"),
)
_LINK_COLORS = {link: _PALETTE[i] for i, link in enumerate(_COCO_LINKS)}


def _dimension(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite positive integer.")
    if not math.isfinite(value) or value <= 0 or int(value) != value:
        raise ValueError(f"{label} must be a finite positive integer.")
    return int(value)


def _points(value: object, label: str, allowed_counts: tuple[int, ...]) -> np.ndarray:
    if value is None:
        return np.empty((0, 3), dtype=np.float64)
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a flat numeric array of x, y, confidence triples.") from exc
    if result.ndim != 1 or result.size % 3:
        raise ValueError(f"{label} must be flat and its length must be a multiple of 3.")
    count = result.size // 3
    if count and count not in allowed_counts:
        expected = " or ".join(map(str, allowed_counts))
        raise ValueError(f"{label} must have {expected} landmarks, or be empty.")
    result = result.reshape((-1, 3))
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains non-finite values.")
    if ((result[:, 2] < 0) | (result[:, 2] > 1)).any():
        raise ValueError(f"{label} confidence must be in [0, 1].")
    return result


def _circle(draw: ImageDraw.ImageDraw, point: np.ndarray, radius: int, color: tuple[int, int, int]) -> None:
    if point[2] <= 0:
        return
    x, y = (float(value) for value in point[:2])
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def _line(draw: ImageDraw.ImageDraw, first: np.ndarray, second: np.ndarray,
          width: int, color: tuple[int, int, int]) -> None:
    if first[2] > 0 and second[2] > 0:
        draw.line((float(first[0]), float(first[1]), float(second[0]), float(second[1])),
                  fill=color, width=width)


def _render_body(draw: ImageDraw.ImageDraw, points: np.ndarray, pose_format: str, width: int) -> None:
    if not len(points):
        return
    named = dict(zip(BODY_PARTS[pose_format], points, strict=True))
    for link in _COCO_LINKS:
        if pose_format == "BODY_25" and link in (("neck", "right_hip"), ("neck", "left_hip")):
            continue
        _line(draw, named[link[0]], named[link[1]], width, _LINK_COLORS[link])
    if pose_format == "BODY_25":
        for first, second, color in (
            ("neck", "mid_hip", _JOINT_COLORS["neck"]),
            ("mid_hip", "right_hip", _LINK_COLORS[("neck", "right_hip")]),
            ("mid_hip", "left_hip", _LINK_COLORS[("neck", "left_hip")]),
            ("left_ankle", "left_big_toe", _JOINT_COLORS["left_ankle"]),
            ("left_big_toe", "left_small_toe", _JOINT_COLORS["left_ankle"]),
            ("left_ankle", "left_heel", _JOINT_COLORS["left_ankle"]),
            ("right_ankle", "right_big_toe", _JOINT_COLORS["right_ankle"]),
            ("right_big_toe", "right_small_toe", _JOINT_COLORS["right_ankle"]),
            ("right_ankle", "right_heel", _JOINT_COLORS["right_ankle"]),
        ):
            _line(draw, named[first], named[second], width, color)
    for semantic, point in named.items():
        _circle(draw, point, max(1, width), _JOINT_COLORS[semantic])


def _render_hand(draw: ImageDraw.ImageDraw, points: np.ndarray, width: int) -> None:
    if not len(points):
        return
    edge = 0
    for finger in range(5):
        chain = (0,) + tuple(range(1 + finger * 4, 5 + finger * 4))
        for first, second in zip(chain, chain[1:], strict=False):
            color = tuple(round(255 * v) for v in colorsys.hsv_to_rgb(edge / 20, 1, 1))
            _line(draw, points[first], points[second], max(1, width // 2), color)
            edge += 1
    for point in points:
        _circle(draw, point, max(1, width // 2), (255, 0, 0))


def render_frames(frames: list[dict], stick_width: int = 4, draw_hands: bool = True,
                  draw_face: bool = True) -> np.ndarray:
    """Return a ComfyUI-compatible float32 RGB batch in [0, 1], without torch.

    Explicit coordinate_space is recommended. For imported auxiliary-node poses
    without it, visible coordinates entirely within [0, 1] mean normalized;
    otherwise they mean pixels. Confidence-zero entries do not affect inference.
    All frames in an image batch must have the same canvas dimensions.
    """
    if not isinstance(frames, (list, tuple)) or not frames:
        raise ValueError("frames must contain at least one pose frame.")
    stick_width = _dimension(stick_width, "stick_width")
    rendered = []
    batch_dimensions = None
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise ValueError(f"Frame {frame_index} must be an object.")
        width = _dimension(frame.get("canvas_width"), f"Frame {frame_index} canvas_width")
        height = _dimension(frame.get("canvas_height"), f"Frame {frame_index} canvas_height")
        dimensions = (width, height)
        if batch_dimensions is not None and dimensions != batch_dimensions:
            raise ValueError("All frames must have the same canvas dimensions to form an IMAGE batch.")
        batch_dimensions = dimensions
        people = frame.get("people", [])
        if not isinstance(people, (list, tuple)) or any(not isinstance(p, Mapping) for p in people):
            raise ValueError(f"Frame {frame_index} people must be a list of person objects.")
        pose_format = frame.get("pose_format")
        if pose_format is None:
            counts = {len(p.get("pose_keypoints_2d", []) or []) for p in people}
            pose_format = "BODY_25" if 75 in counts else "COCO_18"
        if pose_format not in BODY_PARTS:
            raise ValueError(f"Frame {frame_index} pose_format must be COCO_18 or BODY_25.")
        parsed = []
        for person_index, person in enumerate(people):
            prefix = f"Frame {frame_index} person {person_index}"
            parsed.append({
                "body": _points(person.get("pose_keypoints_2d"), f"{prefix} body", (len(BODY_PARTS[pose_format]),)),
                "left_hand": _points(person.get("hand_left_keypoints_2d"), f"{prefix} left hand", (21,)),
                "right_hand": _points(person.get("hand_right_keypoints_2d"), f"{prefix} right hand", (21,)),
                "face": _points(person.get("face_keypoints_2d"), f"{prefix} face", (68, 70)),
            })
        coordinate_space = frame.get("coordinate_space")
        if coordinate_space is None:
            visible = [points[points[:, 2] > 0, :2] for person in parsed for points in person.values() if len(points)]
            coordinate_space = "normalized" if all(((p >= 0) & (p <= 1)).all() for p in visible) else "pixel"
        if coordinate_space not in ("normalized", "pixel"):
            raise ValueError(f"Frame {frame_index} coordinate_space must be normalized or pixel.")
        canvas = Image.new("RGB", dimensions, (0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        for person in parsed:
            if coordinate_space == "normalized":
                person = {key: points * np.array((width, height, 1)) for key, points in person.items()}
            _render_body(draw, person["body"], pose_format, stick_width)
            if draw_hands:
                _render_hand(draw, person["left_hand"], stick_width)
                _render_hand(draw, person["right_hand"], stick_width)
            if draw_face:
                for point in person["face"]:
                    _circle(draw, point, max(1, stick_width // 3), (255, 255, 255))
        rendered.append(np.asarray(canvas, dtype=np.float32) / np.float32(255))
    return np.stack(rendered, axis=0)
