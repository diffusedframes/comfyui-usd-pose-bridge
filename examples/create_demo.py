"""Build the small, animated OpenUSD scene used by the example workflows.

Run with the pack's Python environment, from any working directory::

    python path/to/examples/create_demo.py

Requires ``usd-core`` only. No DCC application, geometry, or external assets
are needed: the scene contains two UsdSkel skeletons and a USD camera.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdSkel, Vt


def _face_points():
    """Return the conventional 68 facial landmarks, followed by two pupils."""
    result = []
    # Jaw: image-left temple, chin, then image-right temple.
    for index in range(17):
        angle = math.pi + math.pi * index / 16
        result.append((0.086 * math.cos(angle), 1.724 + 0.103 * math.sin(angle), 0.093))
    for center in (-0.044, 0.044):
        for index in range(5):
            x = center - 0.030 + 0.015 * index
            y = 1.765 + 0.011 * math.sin(math.pi * index / 4)
            result.append((x, y, 0.100))
    result.extend((0.0, y, 0.108 + 0.008 * index) for index, y in enumerate((1.754, 1.736, 1.718, 1.700)))
    result.extend((x, 1.687 - 0.006 * math.cos(math.pi * index / 4), 0.128) for index, x in enumerate((-0.029, -0.015, 0.0, 0.015, 0.029)))
    for center in (-0.044, 0.044):
        for index in range(6):
            angle = math.pi - 2 * math.pi * index / 6
            result.append((center + 0.025 * math.cos(angle), 1.738 + 0.012 * math.sin(angle), 0.111))
    for index in range(12):
        angle = math.pi - 2 * math.pi * index / 12
        result.append((0.037 * math.cos(angle), 1.651 + 0.016 * math.sin(angle), 0.108))
    for index in range(8):
        angle = math.pi - 2 * math.pi * index / 8
        result.append((0.023 * math.cos(angle), 1.651 + 0.007 * math.sin(angle), 0.111))
    result.extend((x, 1.738, 0.114) for x in (-0.044, 0.044))
    assert len(result) == 70
    return result


def _joint_data():
    paths, positions, local_offsets, parents, by_name = [], [], [], [], {}

    def add(name, parent, position):
        parent_index = by_name.get(parent, -1)
        path = f"{paths[parent_index]}/{name}" if parent_index >= 0 else name
        point = Gf.Vec3f(*position)
        offset = point - positions[parent_index] if parent_index >= 0 else point
        by_name[name] = len(paths)
        paths.append(path)
        positions.append(point)
        local_offsets.append(offset)
        parents.append(parent_index)

    add("pelvis", None, (0.0, 0.95, 0.0))
    add("spine", "pelvis", (0.0, 1.15, 0.0))
    add("chest", "spine", (0.0, 1.41, 0.0))
    add("neck", "chest", (0.0, 1.54, 0.0))
    add("head", "neck", (0.0, 1.70, 0.0))
    add("nose", "head", (0.0, 1.70, 0.132))
    add("left_eye", "head", (0.044, 1.738, 0.111))
    add("right_eye", "head", (-0.044, 1.738, 0.111))
    add("left_ear", "head", (0.094, 1.716, 0.0))
    add("right_ear", "head", (-0.094, 1.716, 0.0))

    for side, sign in (("left", 1), ("right", -1)):
        add(f"{side}_shoulder", "chest", (sign * 0.19, 1.45, 0.0))
        add(f"{side}_elbow", f"{side}_shoulder", (sign * 0.41, 1.22, 0.0))
        add(f"{side}_wrist", f"{side}_elbow", (sign * 0.58, 1.01, 0.012))
        wrist = positions[by_name[f"{side}_wrist"]]
        for finger, spread, length in (
            ("thumb", 0.043, 0.026),
            ("index", 0.027, 0.028),
            ("middle", 0.009, 0.031),
            ("ring", -0.009, 0.029),
            ("pinky", -0.027, 0.024),
        ):
            parent = f"{side}_wrist"
            for number in range(1, 5):
                if finger == "thumb":
                    point = wrist + Gf.Vec3f(sign * (spread + number * 0.013), -0.018 - number * length * 0.65, 0.017)
                else:
                    point = wrist + Gf.Vec3f(sign * spread, -0.029 - number * length, 0.008)
                name = f"{side}_{finger}{number}"
                add(name, parent, point)
                parent = name

        add(f"{side}_hip", "pelvis", (sign * 0.105, 0.92, 0.0))
        add(f"{side}_knee", f"{side}_hip", (sign * 0.13, 0.52, 0.02))
        add(f"{side}_ankle", f"{side}_knee", (sign * 0.14, 0.095, 0.0))
        add(f"{side}_big_toe", f"{side}_ankle", (sign * 0.12, 0.045, 0.16))
        add(f"{side}_small_toe", f"{side}_ankle", (sign * 0.18, 0.045, 0.145))
        add(f"{side}_heel", f"{side}_ankle", (sign * 0.14, 0.04, -0.055))

    for index, point in enumerate(_face_points()):
        add(f"face_{index:02d}", "head", point)
    return paths, positions, local_offsets, by_name


def _quat(axis, degrees):
    value = Gf.Rotation(Gf.Vec3d(*axis), degrees).GetQuat()
    return Gf.Quatf(value.GetReal(), Gf.Vec3f(value.GetImaginary()))


def _create_character(stage, name, translation, partner=False):
    root_path = f"/World/Characters/{name}"
    root = UsdSkel.Root.Define(stage, root_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*translation))
    paths, positions, local_offsets, indices = _joint_data()
    skeleton = UsdSkel.Skeleton.Define(stage, root_path + "/Skeleton")
    skeleton.CreateJointsAttr().Set(Vt.TokenArray(paths))
    skeleton.CreateJointNamesAttr().Set(Vt.TokenArray([path.rsplit("/", 1)[-1] for path in paths]))
    skeleton.CreateRestTransformsAttr().Set(Vt.Matrix4dArray([Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(point)) for point in local_offsets]))
    skeleton.CreateBindTransformsAttr().Set(Vt.Matrix4dArray([Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(point)) for point in positions]))
    skeleton.GetPrim().SetCustomDataByKey("description", "Canonical body, two 21-point hands, and 70 facial landmarks.")

    animation = UsdSkel.Animation.Define(stage, root_path + "/Animation")
    animation.CreateJointsAttr().Set(Vt.TokenArray(paths))
    translation_attr = animation.CreateTranslationsAttr()
    rotation_attr = animation.CreateRotationsAttr()
    scale_attr = animation.CreateScalesAttr()
    scale_attr.Set(Vt.Vec3hArray([Gf.Vec3h(1)] * len(paths)))
    for frame in range(1, 25):
        phase = 2 * math.pi * (frame - 1) / 24
        offsets = Vt.Vec3fArray(local_offsets)
        offsets[indices["pelvis"]] += Gf.Vec3f(0.020 * math.sin(phase), 0.012 * math.sin(2 * phase), 0)
        rotations = [Gf.Quatf(1)] * len(paths)

        def rotate(joint, axis, degrees, rotations=rotations):
            rotations[indices[joint]] = _quat(axis, degrees)

        rotate("spine", (0, 0, 1), (3 if partner else -3) * math.sin(phase))
        rotate("chest", (0, 1, 0), (7 if partner else -7) + 5 * math.sin(phase))
        rotate("neck", (0, 0, 1), (5 if partner else -5) + 3 * math.cos(phase))
        if partner:
            rotate("left_shoulder", (0, 0, 1), 110 + 10 * math.sin(phase))
            rotate("left_elbow", (0, 0, 1), 25 + 22 * math.sin(phase + 0.7))
            rotate("left_wrist", (0, 0, 1), 16 * math.sin(phase * 2))
            rotate("right_shoulder", (0, 0, 1), -12)
            rotate("right_elbow", (0, 0, 1), -60)
        else:
            rotate("right_shoulder", (0, 0, 1), -103 + 12 * math.sin(phase))
            rotate("right_elbow", (0, 0, 1), -42 + 14 * math.sin(phase + 0.5))
            rotate("right_wrist", (0, 0, 1), 13 * math.sin(phase * 2))
            rotate("left_shoulder", (0, 0, 1), 12)
            rotate("left_elbow", (0, 0, 1), 65)
        for side, sign in (("left", 1), ("right", -1)):
            rotate(f"{side}_hip", (0, 0, 1), sign * (3 + math.sin(phase)))
            for finger in ("thumb", "index", "middle", "ring", "pinky"):
                for segment in (2, 3, 4):
                    rotate(f"{side}_{finger}{segment}", (1, 0, 0), 8 + 4 * math.sin(phase))
        translation_attr.Set(offsets, frame)
        rotation_attr.Set(Vt.QuatfArray(rotations), frame)
        if frame == 1:
            translation_attr.Set(offsets)
            rotation_attr.Set(Vt.QuatfArray(rotations))
    UsdSkel.BindingAPI.Apply(skeleton.GetPrim()).CreateAnimationSourceRel().SetTargets([animation.GetPath()])


def create_demo(destination):
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(destination))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(24)
    stage.SetFramesPerSecond(24)
    stage.SetStartTimeCode(1)
    stage.SetEndTimeCode(24)
    stage.GetRootLayer().documentation = "USD Pose Bridge: two fully animated anatomical skeletons; 24 frames at 24 fps."
    UsdGeom.Xform.Define(stage, "/World/Characters")
    _create_character(stage, "Lead", (-0.62, 0, 0))
    _create_character(stage, "Partner", (0.62, 0, -0.12), partner=True)
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateProjectionAttr().Set(UsdGeom.Tokens.perspective)
    camera.CreateFocalLengthAttr().Set(50.0)
    camera.CreateHorizontalApertureAttr().Set(36.0)
    camera.CreateVerticalApertureAttr().Set(27.0)
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.1, 100.0))
    camera.AddTranslateOp().Set(Gf.Vec3d(0, 1.05, 4.8))
    camera.AddRotateXOp().Set(-1.0)
    camera.GetPrim().SetCustomDataByKey("description", "Perspective camera, composed translation and rotation; target framing is 1024 x 768.")
    stage.GetRootLayer().Save()
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("demo.usda"))
    args = parser.parse_args()
    print(create_demo(args.output))
