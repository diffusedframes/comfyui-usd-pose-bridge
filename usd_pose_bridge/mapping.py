"""Conservative rig-name mapping to the public OpenPose landmark layouts.

Only the shoulder and hip midpoints are synthesized. A rig's head, clavicles,
and finger bones are deliberately not substitutes for unobserved landmarks.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence


COCO_18 = (
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
    "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
    "left_eye", "right_ear", "left_ear",
)
BODY_25 = COCO_18[:8] + ("mid_hip",) + COCO_18[8:] + (
    "left_big_toe", "left_small_toe", "left_heel", "right_big_toe",
    "right_small_toe", "right_heel",
)
BODY_PARTS = {"COCO_18": COCO_18, "BODY_25": BODY_25}
_MISSING = (0.0, 0.0, 0.0)
_MAPPING_KEYS = {"body", "left_hand", "right_hand", "face"}


def _leaf(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def _short_name(name: str) -> str:
    return _leaf(name).rsplit(":", 1)[-1]


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _short_name(name).lower())


def _validate_mapping(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Joint mapping must be a JSON object.")
    unknown = set(value) - _MAPPING_KEYS
    if unknown:
        raise ValueError(f"Unknown mapping key(s): {', '.join(sorted(map(str, unknown)))}. "
                         "Use body, left_hand, right_hand, or face.")
    body = value.get("body", {})
    if not isinstance(body, dict):
        raise ValueError("Mapping 'body' must be an object of semantic names to joint paths or null.")
    unknown = set(body) - set(BODY_25)
    if unknown:
        raise ValueError(f"Unknown body landmark(s): {', '.join(sorted(map(str, unknown)))}. "
                         f"Valid names: {', '.join(BODY_25)}.")
    for semantic, ref in body.items():
        _validate_ref(ref, f"body.{semantic}")
    for key, lengths in (("left_hand", (21,)), ("right_hand", (21,)), ("face", (68, 70))):
        if key not in value or value[key] is None:
            continue
        refs = value[key]
        if not isinstance(refs, list) or len(refs) not in lengths:
            expected = " or ".join(map(str, lengths))
            raise ValueError(f"Mapping '{key}' must be null or a list of {expected} joint paths/null entries.")
        for index, ref in enumerate(refs):
            _validate_ref(ref, f"{key}[{index}]")
    return value


def _validate_ref(ref: object, label: str) -> None:
    if ref is not None and (not isinstance(ref, str) or not ref.strip()):
        raise ValueError(f"Mapping '{label}' must be a nonempty joint path/name or null.")


def parse_mapping(text: str) -> dict:
    """Parse a JSON mapping, rejecting misspelled fields and malformed arrays."""
    if not isinstance(text, str):
        raise ValueError("Joint mapping must be JSON text.")
    if not text.strip():
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid mapping JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}.") from exc
    return _validate_mapping(value)


def _aliases(semantic: str) -> tuple[str, ...]:
    """Aliases exclude clavicles, generic heads, and toes with unknown landmarks."""
    direct = {"nose": ("nose",), "neck": ("neck", "neck1", "neck01"),
              "mid_hip": ("midhip", "hips", "pelvis")}
    if semantic in direct:
        return direct[semantic]
    side, part = semantic.split("_", 1)
    short = side[0]
    suffix = short
    variants = {
        "shoulder": (f"{side}arm", f"{side}upperarm", f"upperarm{suffix}",
                     f"upperarm{side}", f"{short}shoulder"),
        "elbow": (f"{side}elbow", f"{short}elbow", f"{side}forearm", f"{side}lowerarm",
                  f"lowerarm{suffix}", f"forearm{suffix}", f"lowerarm{side}"),
        "wrist": (f"{side}wrist", f"{short}wrist", f"{side}hand", f"hand{suffix}", f"hand{side}"),
        "hip": (f"{side}hip", f"{short}hip", f"{side}upleg", f"{side}thigh", f"thigh{suffix}",
                f"upperleg{suffix}", f"thigh{side}"),
        "knee": (f"{side}knee", f"{short}knee", f"{side}leg", f"{side}lowerleg", f"{side}calf",
                 f"calf{suffix}", f"shin{suffix}", f"lowerleg{suffix}"),
        "ankle": (f"{side}ankle", f"{short}ankle", f"{side}foot", f"foot{suffix}", f"foot{side}"),
        "eye": (f"{side}eye", f"{short}eye", f"eye{suffix}"),
        "ear": (f"{side}ear", f"{short}ear", f"ear{suffix}"),
        "big_toe": (f"{side}bigtoe", f"{short}bigtoe", f"bigtoe{suffix}"),
        "small_toe": (f"{side}smalltoe", f"{short}smalltoe", f"smalltoe{suffix}"),
        "heel": (f"{side}heel", f"{short}heel", f"heel{suffix}"),
    }
    return variants.get(part, ())


def _finger_aliases(side: str, finger: str, segment: int) -> tuple[str, ...]:
    short = side[0]
    fingers = (finger, "little") if finger == "pinky" else (finger,)
    names = []
    for name in fingers:
        for number in (str(segment), f"{segment:02d}"):
            names.extend((f"{side}hand{name}{number}", f"{side}{name}{number}",
                          f"{name}{number}{short}", f"{name}{number}{side}"))
        if segment == 4:
            for ending in ("tip", "end", "3end", "03end"):
                names.extend((f"{side}hand{name}{ending}", f"{side}{name}{ending}",
                              f"{name}{ending}{short}", f"{name}{ending}{side}"))
    return tuple(names)


def build_person(joint_names: list[str], projected_points: list[tuple[float, float, float]],
                 pose_format: str = "COCO_18", overrides: dict | None = None) -> tuple[dict, dict]:
    """Map projected joints without changing their coordinate convention.

    References in overrides are exact full joint paths or unique leaf names;
    namespace-free leaf names are also accepted when unique. Unresolved automatic
    aliases remain missing, while unresolved explicit references are errors.
    """
    if pose_format not in BODY_PARTS:
        raise ValueError("pose_format must be COCO_18 or BODY_25.")
    overrides = _validate_mapping({} if overrides is None else overrides)
    if len(joint_names) != len(projected_points):
        raise ValueError("joint_names and projected_points must have the same length.")
    if any(not isinstance(name, str) or not name for name in joint_names):
        raise ValueError("Joint names must be nonempty strings.")
    points = []
    for index, point in enumerate(projected_points):
        try:
            if len(point) != 3:
                raise ValueError
            converted = tuple(float(v) for v in point)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Projected joint {index} must contain numeric x, y, confidence.") from exc
        if not all(math.isfinite(v) for v in converted) or not 0 <= converted[2] <= 1:
            raise ValueError(f"Projected joint {index} requires finite coordinates and confidence in [0, 1].")
        points.append(converted)
    report = {"mapped": {}, "missing": [], "derived": {}, "ambiguous": {}, "unmapped_joints": []}
    used = set()
    canonical = {}
    normalized = {}
    for index, name in enumerate(joint_names):
        canonical.setdefault(_short_name(name).lower(), []).append(index)
        normalized.setdefault(_normalize(name), []).append(index)

    def resolve_ref(ref: str, label: str) -> int:
        found = [i for i, name in enumerate(joint_names) if name == ref]
        if not found:
            found = [i for i, name in enumerate(joint_names) if _leaf(name) == ref]
        if not found:
            found = [i for i, name in enumerate(joint_names) if _short_name(name) == ref]
        if len(found) != 1:
            if found:
                raise ValueError(f"Ambiguous joint reference '{ref}' for {label}: "
                                 f"{', '.join(joint_names[i] for i in found)}. Use a full path.")
            raise ValueError(f"Unknown joint reference '{ref}' for {label}.")
        return found[0]

    def auto(label: str, aliases: Sequence[str], semantic: str | None = None) -> int | None:
        # Preserve literal semantic names such as left_shoulder. Mixamo's
        # LeftShoulder is a clavicle and intentionally is not this literal match.
        candidates = canonical.get(semantic, []) if semantic else []
        if not candidates:
            candidates = sorted({i for alias in aliases for i in normalized.get(alias, [])})
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            report["ambiguous"][label] = [joint_names[i] for i in candidates]
        return None

    def assign(label: str, index: int | None) -> tuple[float, float, float]:
        if index is None:
            report["missing"].append(label)
            return _MISSING
        report["mapped"][label] = joint_names[index]
        used.add(index)
        return points[index]

    resolved = {}
    body_points = {}
    body_overrides = overrides.get("body", {})
    # Validate even overrides for landmarks absent from the selected body format.
    explicit_indices = {name: (None if ref is None else resolve_ref(ref, f"body.{name}"))
                        for name, ref in body_overrides.items()}
    for semantic in BODY_PARTS[pose_format]:
        index = explicit_indices.get(semantic) if semantic in body_overrides else auto(
            semantic, _aliases(semantic), semantic)
        resolved[semantic] = index
        body_points[semantic] = assign(semantic, index)
    for semantic, sources in (("neck", ("left_shoulder", "right_shoulder")),
                              ("mid_hip", ("left_hip", "right_hip"))):
        if (semantic not in body_points or resolved[semantic] is not None or semantic in body_overrides
                or semantic in report["ambiguous"]):
            continue
        a, b = (body_points[name] for name in sources)
        if a[2] > 0 and b[2] > 0:
            body_points[semantic] = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, min(a[2], b[2]))
            report["derived"][semantic] = {"method": "midpoint", "sources": list(sources)}
            report["missing"].remove(semantic)

    def hand(side: str) -> list[float] | None:
        key = f"{side}_hand"
        if key in overrides:
            if overrides[key] is None:
                return None
            indices = [None if ref is None else resolve_ref(ref, f"{key}[{i}]")
                       for i, ref in enumerate(overrides[key])]
        else:
            indices = [resolved.get(f"{side}_wrist")]
            for finger in ("thumb", "index", "middle", "ring", "pinky"):
                for segment in range(1, 5):
                    label = f"{key}[{len(indices)}]"
                    indices.append(auto(label, _finger_aliases(side, finger, segment)))
            if not any(index is not None for index in indices[1:]):
                return None
        flattened = [value for i, index in enumerate(indices) for value in assign(f"{key}[{i}]", index)]
        return flattened if any(index is not None for index in indices) else None

    face = None
    if overrides.get("face") is not None:
        face = [value for i, ref in enumerate(overrides["face"])
                for value in assign(f"face[{i}]", None if ref is None else resolve_ref(ref, f"face[{i}]"))]
    person = {
        "pose_keypoints_2d": [value for semantic in BODY_PARTS[pose_format] for value in body_points[semantic]],
        "hand_left_keypoints_2d": hand("left"),
        "hand_right_keypoints_2d": hand("right"),
        "face_keypoints_2d": face,
    }
    report["unmapped_joints"] = [name for i, name in enumerate(joint_names) if i not in used]
    return person, report
