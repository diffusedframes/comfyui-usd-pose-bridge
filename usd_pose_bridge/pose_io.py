"""OpenPose JSON interchange and confined, non-overwriting output files."""

from __future__ import annotations

import copy
import json
import math
from numbers import Real
from pathlib import Path, PurePosixPath


def resolve_input_path(value: str, pack_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Enter a local file path.")
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        # A deterministic bundled example remains portable when the pack is moved.
        if path.parts and path.parts[0] == "examples":
            path = pack_root / path
        else:
            try:
                import folder_paths
            except ImportError:
                path = pack_root / path
            else:
                path = Path(folder_paths.get_input_directory()) / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"File does not exist: {path}")
    return path


def load_pose_json(path: Path, width: int = 512, height: int = 512, coordinate_space: str = "auto") -> list[dict]:
    if coordinate_space not in {"auto", "pixel", "normalized"}:
        raise ValueError("coordinate_space must be auto, pixel, or normalized.")
    try:
        with path.open(encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read OpenPose JSON {path}: {exc}") from exc
    frames = data if isinstance(data, list) else [data]
    if not frames or any(not isinstance(frame, dict) for frame in frames):
        raise ValueError("OpenPose JSON must be a frame object or a nonempty list of frame objects.")
    result = copy.deepcopy(frames)
    for frame in result:
        if "people" not in frame or not isinstance(frame["people"], list):
            raise ValueError("Each OpenPose frame must contain a people list.")
        frame.setdefault("canvas_width", width)
        frame.setdefault("canvas_height", height)
        if coordinate_space != "auto":
            frame["coordinate_space"] = coordinate_space
        if frame.get("coordinate_space") not in (None, "pixel", "normalized"):
            raise ValueError("Frame coordinate_space must be pixel or normalized.")
        lengths = set()
        for person in frame["people"]:
            if not isinstance(person, dict):
                raise ValueError("Each person must be an OpenPose object.")
            body = person.get("pose_keypoints_2d")
            if not isinstance(body, list) or len(body) not in (54, 75):
                raise ValueError("Expected COCO_18 (54 values) or BODY_25 (75 values) body keypoints.")
            for key, counts in (("pose_keypoints_2d", (54, 75)), ("hand_left_keypoints_2d", (0, 63)),
                                ("hand_right_keypoints_2d", (0, 63)), ("face_keypoints_2d", (0, 204, 210))):
                values = person.get(key)
                if values is None and key != "pose_keypoints_2d":
                    continue
                if not isinstance(values, list) or len(values) not in counts:
                    raise ValueError(f"Invalid number of values in {key}.")
                if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
                       for value in values):
                    raise ValueError(f"{key} must contain finite numeric x, y, confidence triples.")
                if any(not 0 <= confidence <= 1 for confidence in values[2::3]):
                    raise ValueError(f"{key} confidence must be in [0, 1].")
            lengths.add(len(body))
        if len(lengths) > 1:
            raise ValueError("All people in one frame must use the same body format.")
        if lengths:
            inferred = "BODY_25" if 75 in lengths else "COCO_18"
            if frame.get("pose_format", inferred) != inferred:
                raise ValueError("pose_format does not match the number of body keypoints.")
            frame["pose_format"] = inferred
    return result


def save_pose_json(frames: list[dict], output_dir: Path, filename_prefix: str, output_format: str) -> list[str]:
    if not frames or not isinstance(frames, list):
        raise ValueError("Cannot save an empty pose sequence.")
    if output_format not in {"sequence", "per_frame"}:
        raise ValueError("format must be sequence or per_frame.")
    if not isinstance(filename_prefix, str) or not filename_prefix.strip():
        raise ValueError("filename_prefix cannot be empty.")
    prefix = filename_prefix.strip().replace("\\", "/")
    parts = PurePosixPath(prefix)
    if parts.is_absolute() or ":" in prefix or any(p in {"..", "."} for p in prefix.split("/")):
        raise ValueError("filename_prefix must be a relative path inside ComfyUI's output folder.")
    root = Path(output_dir).resolve()
    target = (root / str(parts)).resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError("filename_prefix must stay inside ComfyUI's output folder.")
    # Serialize before creating any files, including validation of nonfinite numbers.
    payloads = [json.dumps(frame, indent=2, allow_nan=False) for frame in frames]
    sequence = json.dumps(frames, indent=2, allow_nan=False) if output_format == "sequence" else None
    target.parent.mkdir(parents=True, exist_ok=True)
    for counter in range(1, 1_000_000):
        candidate = target.with_name(f"{target.name}_{counter:05d}")
        if output_format == "sequence":
            candidate = candidate.with_name(candidate.name + ".json")
            try:
                with candidate.open("x", encoding="utf-8") as handle:
                    handle.write(sequence)
                    handle.write("\n")
            except FileExistsError:
                continue
            return [str(candidate)]
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        paths = []
        for index, payload in enumerate(payloads):
            frame_path = candidate / f"{index:06d}_keypoints.json"
            with frame_path.open("x", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
            paths.append(str(frame_path))
        return paths
    raise ValueError("No unused output filename available for this prefix.")
