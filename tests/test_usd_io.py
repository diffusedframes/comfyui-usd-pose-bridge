"""Numerical tests against actual OpenUSD, rather than mocked USD objects."""

import json
import math
from pathlib import Path

import pytest

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel, UsdUtils
except (ImportError, OSError):
    pytest.skip("OpenUSD runtime is optional; install usd-core to run integration tests", allow_module_level=True)

from usd_pose_bridge.usd_io import inspect_stage, open_stage, sample_stage


def approx_points(values):
    return [pytest.approx(value) for value in values]


def matrix(x=0, y=0, z=0):
    return Gf.Matrix4d(1).SetTranslate((x, y, z))


def scene(points=((0, 0, -10),)):
    stage = Usd.Stage.CreateInMemory()
    root = UsdSkel.Root.Define(stage, "/Character")
    skeleton = UsdSkel.Skeleton.Define(stage, "/Character/Rig")
    skeleton.CreateJointsAttr([f"joint{i}" for i in range(len(points))])
    skeleton.CreateRestTransformsAttr([matrix(*point) for point in points])
    skeleton.CreateBindTransformsAttr([matrix(*point) for point in points])
    camera = UsdGeom.Camera.Define(stage, "/Camera")
    camera.CreateHorizontalApertureAttr(20)
    camera.CreateVerticalApertureAttr(20)
    camera.CreateFocalLengthAttr(10)
    camera.CreateClippingRangeAttr((1, 100))
    return stage, root, skeleton, camera


def points(stage, **kwargs):
    return sample_stage(stage, width=101, height=101, **kwargs)[0]["skeletons"][0]["points"]


def animate(root, skeleton, translations, order=None):
    animation = UsdSkel.Animation.Define(skeleton.GetPrim().GetStage(), root.GetPath().AppendChild("Animation"))
    joints = list(order or skeleton.GetJointsAttr().Get())
    animation.CreateJointsAttr(joints)
    for frame, values in translations.items():
        animation.CreateTranslationsAttr().Set(values, frame)
        animation.CreateRotationsAttr().Set([Gf.Quatf(1)] * len(joints), frame)
        animation.CreateScalesAttr().Set([Gf.Vec3h(1)] * len(joints), frame)
    UsdSkel.BindingAPI.Apply(root.GetPrim()).CreateAnimationSourceRel().SetTargets([animation.GetPath()])
    return animation


def test_inspection_is_serializable_and_reports_authored_metadata():
    stage, root, skeleton, _ = scene()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetStartTimeCode(10)
    stage.SetEndTimeCode(20)
    stage.SetTimeCodesPerSecond(30)
    animate(root, skeleton, {10: [(0, 0, -10)], 20: [(1, 0, -10)]})
    data = inspect_stage(stage)
    json.dumps(data, allow_nan=False)
    assert data["up_axis"] == "Z"
    assert data["meters_per_unit"] == 1.0
    assert (data["time_start"], data["time_end"], data["time_codes_per_second"]) == (10, 20, 30)
    assert data["cameras"] == ["/Camera"]
    assert data["skeletons"] == [{"path": "/Character/Rig", "joints": ["joint0"], "animated": True}]


def test_perspective_and_top_left_pixel_coordinates():
    stage, *_ = scene(((0, 0, -10), (5, 5, -10), (-10, -10, -10)))
    assert points(stage) == approx_points([(50, 50, 1), (75, 25, 1), (0, 100, 1)])


def test_local_joint_hierarchy_and_world_ancestors_are_composed():
    stage, root, skeleton, _ = scene()
    skeleton.CreateJointsAttr(["Root", "Root/Child"])
    skeleton.CreateRestTransformsAttr([matrix(1, 0, -10), matrix(1, 0, 0)])
    skeleton.CreateBindTransformsAttr([matrix(), matrix()])
    UsdGeom.Xformable(root).AddTranslateOp().Set((2, 0, 0))
    UsdGeom.Xformable(skeleton).AddTranslateOp().Set((1, 0, 0))
    assert points(stage) == approx_points([(70, 50, 1), (75, 50, 1)])


def test_bound_animation_remaps_joint_order_and_rest_fallback():
    stage, root, skeleton, _ = scene()
    skeleton.CreateJointsAttr(["Root", "Root/Head", "Root/Other"])
    skeleton.CreateRestTransformsAttr([matrix(0, 0, -10), matrix(0, 2, 0), matrix(-2, 0, 0)])
    skeleton.CreateBindTransformsAttr([matrix(), matrix(), matrix()])
    animate(root, skeleton, {1: [(0, 2, 0), (0, 0, -10)],
                             3: [(0, 4, 0), (2, 0, -10)]}, order=["Root/Head", "Root"])
    frames = sample_stage(stage, frames=[1, 2, 3], width=101, height=101)
    assert frames[0]["skeletons"][0]["points"] == approx_points([(50, 50, 1), (50, 40, 1), (40, 50, 1)])
    assert frames[1]["skeletons"][0]["points"] == approx_points([(55, 50, 1), (55, 35, 1), (45, 50, 1)])
    assert frames[2]["skeletons"][0]["points"] == approx_points([(60, 50, 1), (60, 30, 1), (50, 50, 1)])


def test_animated_skeleton_and_camera_transforms_and_lens_are_sampled():
    stage, root, _, camera = scene(((2, 0, -10),))
    shift = UsdGeom.Xformable(root).AddTranslateOp()
    shift.Set((0, 0, 0), 1)
    shift.Set((2, 0, 0), 3)
    camera_shift = camera.AddTranslateOp()
    camera_shift.Set((0, 0, 0), 1)
    camera_shift.Set((1, 0, 0), 3)
    camera.GetFocalLengthAttr().Set(10, 1)
    camera.GetFocalLengthAttr().Set(20, 3)
    frames = sample_stage(stage, frames=[1, 3], width=101, height=101)
    assert frames[0]["skeletons"][0]["points"] == approx_points([(60, 50, 1)])
    assert frames[1]["skeletons"][0]["points"] == approx_points([(80, 50, 1)])
    assert inspect_stage(stage)["skeletons"][0]["animated"] is True


def test_orthographic_projection_does_not_scale_with_depth():
    stage, _, _, camera = scene(((0.5, 0.5, -2), (0.5, 0.5, -20)))
    camera.CreateProjectionAttr(UsdGeom.Tokens.orthographic)
    # 20 tenths of one scene unit gives a 2x2 orthographic window.
    assert points(stage) == approx_points([(75, 25, 1), (75, 25, 1)])


def test_aperture_offsets_are_preserved():
    stage, _, _, camera = scene()
    camera.CreateHorizontalApertureOffsetAttr(5)
    camera.CreateVerticalApertureOffsetAttr(-5)
    assert points(stage) == approx_points([(25, 25, 1)])


def test_fit_and_crop_conform_camera_window_to_output_aspect():
    stage, *_ = scene(((5, 5, -10),))
    fit = sample_stage(stage, width=201, height=101, camera_fit="fit")[0]["skeletons"][0]["points"]
    crop = sample_stage(stage, width=201, height=101, camera_fit="crop")[0]["skeletons"][0]["points"]
    assert fit == approx_points([(125.12437810945273, 25, 1)])
    assert crop == approx_points([(150, 0.24752475247525, 1)])


def test_frustum_clips_near_far_behind_and_outside_points():
    stage, *_ = scene(((0, 0, -0.5), (0, 0, -101), (0, 0, 10), (20, 0, -10),
                      (0, 20, -10), (0, 0, -1), (0, 0, -100)))
    data = points(stage)
    assert data[:5] == [(0, 0, 0)] * 5
    assert data[5:] == approx_points([(50, 50, 1), (50, 50, 1)])


def test_additional_clipping_planes_are_evaluated_in_camera_space():
    stage, _, _, camera = scene(((0, 0, -10), (2, 0, -10)))
    camera.AddTranslateOp().Set((1, 0, 0))
    camera.CreateClippingPlanesAttr([(1, 0, 0, 0)])
    assert points(stage) == approx_points([(0, 0, 0), (55, 50, 1)])


def test_stage_axis_and_units_do_not_relabel_authored_camera_space():
    stage, *_ = scene(((2, 3, -10),))
    expected = points(stage)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 100.0)
    assert points(stage) == expected


def test_rotated_camera_uses_world_view_transform():
    stage, _, skeleton, camera = scene(((0, 0, 0),))
    skeleton.CreateRestTransformsAttr([matrix(10, 2, 0)])
    camera.AddRotateYOp().Set(-90)
    assert points(stage) == approx_points([(50, 40, 1)])


def test_instance_proxy_skeletons_and_cameras_are_available():
    stage, root, _, _ = scene()
    instance = stage.DefinePrim("/Copy")
    instance.GetReferences().AddInternalReference(root.GetPath())
    instance.SetInstanceable(True)
    assert "/Copy/Rig" in [item["path"] for item in inspect_stage(stage)["skeletons"]]
    data = sample_stage(stage, skeleton_paths="/Copy/Rig", width=101, height=101)
    assert data[0]["skeletons"][0]["path"] == "/Copy/Rig"
    assert data[0]["skeletons"][0]["points"] == approx_points([(50, 50, 1)])


def test_camera_selection_requires_disambiguation():
    stage, *_ = scene()
    UsdGeom.Camera.Define(stage, "/OtherCamera")
    with pytest.raises(ValueError, match="Several USD cameras.*specify camera_path"):
        sample_stage(stage)
    assert sample_stage(stage, camera_path="/Camera")[0]["camera_path"] == "/Camera"
    with pytest.raises(ValueError, match="does not identify a USD Camera"):
        sample_stage(stage, camera_path="/Character")
    stage.RemovePrim("/Camera")
    stage.RemovePrim("/OtherCamera")
    with pytest.raises(ValueError, match="No USD Camera"):
        sample_stage(stage)


def test_explicit_skeleton_selection_order_and_errors():
    stage, root, _, _ = scene()
    copy = stage.DefinePrim("/Copy")
    copy.GetReferences().AddInternalReference(root.GetPath())
    data = sample_stage(stage, skeleton_paths="/Copy/Rig, /Character/Rig, /Copy/Rig")
    assert [item["path"] for item in data[0]["skeletons"]] == ["/Copy/Rig", "/Character/Rig"]
    with pytest.raises(ValueError, match="Skeleton not found"):
        sample_stage(stage, skeleton_paths="/Missing")
    with pytest.raises(ValueError, match="absolute USD prim path"):
        sample_stage(stage, skeleton_paths="Character/Rig")


@pytest.mark.parametrize("kwargs", [
    {"width": 0}, {"height": -1}, {"width": 10.5}, {"width": True}, {"height": 16385},
    {"frames": []}, {"frames": [math.nan]}, {"frames": [math.inf]}, {"frames": [True]},
    {"frames": ["1"]}, {"frames": "1,2"}, {"frames": None}, {"frames": [10 ** 400]},
    {"camera_fit": "stretch"},
])
def test_invalid_inputs_are_rejected(kwargs):
    stage, *_ = scene()
    with pytest.raises(ValueError):
        sample_stage(stage, **kwargs)


@pytest.mark.parametrize("fault", ["empty", "duplicate", "hierarchy", "short_rest", "nan_rest", "no_rest"])
def test_invalid_skeletons_fail_instead_of_silently_emitting_bad_poses(fault):
    stage, _, skeleton, _ = scene()
    if fault == "empty":
        skeleton.GetJointsAttr().Set([])
    elif fault == "duplicate":
        skeleton.GetJointsAttr().Set(["Root", "Root"])
    elif fault == "hierarchy":
        skeleton.GetJointsAttr().Set(["Root/Child", "Root"])
    elif fault == "short_rest":
        skeleton.GetRestTransformsAttr().Set([])
    elif fault == "nan_rest":
        skeleton.GetRestTransformsAttr().Set([matrix(math.nan, 0, -10)])
    elif fault == "no_rest":
        skeleton.GetRestTransformsAttr().Clear()
    with pytest.raises(ValueError, match="Skeleton|skeleton"):
        sample_stage(stage)


def test_malformed_bound_animation_does_not_silently_fall_back_to_rest():
    stage, root, skeleton, _ = scene()
    animation = animate(root, skeleton, {1: [(0, 0, -10)]})
    animation.GetTranslationsAttr().Set([], 1)
    with pytest.raises(ValueError, match="translations.*one value per animation joint"):
        sample_stage(stage)


def test_blendshape_only_binding_keeps_the_skeleton_rest_pose():
    stage, root, _, _ = scene(((2, 0, -10),))
    animation = UsdSkel.Animation.Define(stage, "/Character/BlendAnimation")
    animation.CreateBlendShapesAttr(["Smile"])
    animation.CreateBlendShapeWeightsAttr().Set([0], 1)
    animation.GetBlendShapeWeightsAttr().Set([1], 10)
    UsdSkel.BindingAPI.Apply(root.GetPrim()).CreateAnimationSourceRel().SetTargets([animation.GetPath()])
    data = sample_stage(stage, frames=[1, 5, 10], width=101, height=101)
    for frame in data:
        assert frame["skeletons"][0]["points"] == approx_points([(60, 50, 1)])
    assert inspect_stage(stage)["skeletons"][0]["animated"] is False


def test_joint_animation_resolves_authored_constant_default_components():
    stage, root, skeleton, _ = scene()
    animation = animate(root, skeleton, {1: [(2, 0, -10)], 3: [(4, 0, -10)]})
    # Non-varying components need only one authored default, rather than a
    # time sample at every frame. UsdAttribute.Get(time) resolves it correctly.
    animation.GetRotationsAttr().Clear()
    animation.GetRotationsAttr().Set([Gf.Quatf(1)])
    animation.GetScalesAttr().Clear()
    animation.GetScalesAttr().Set([Gf.Vec3h(1)])
    data = sample_stage(stage, frames=[1, 2, 3], width=101, height=101)
    for frame, expected in zip(data, (60, 65, 70), strict=True):
        assert frame["skeletons"][0]["points"] == approx_points([(expected, 50, 1)])


@pytest.mark.parametrize("mode", ["missing", "blocked"])
def test_missing_or_blocked_joint_components_nullify_animation_and_use_rest(mode):
    stage, root, skeleton, _ = scene()
    animation = animate(root, skeleton, {1: [(2, 0, -10)]})
    if mode == "blocked":
        animation.GetScalesAttr().Block()
    else:
        animation.GetScalesAttr().Clear()
    assert points(stage) == approx_points([(50, 50, 1)])
    skeleton.GetRestTransformsAttr().Clear()
    with pytest.raises(ValueError, match="no rest pose"):
        sample_stage(stage)


def test_complete_joint_animation_can_supply_a_skeleton_without_rest_transforms():
    stage, root, skeleton, _ = scene()
    skeleton.GetRestTransformsAttr().Clear()
    animate(root, skeleton, {1: [(2, 0, -10)]})
    assert points(stage) == approx_points([(60, 50, 1)])


def test_discovery_sorts_paths_independently_of_authored_prim_order():
    stage, _, _, _ = scene()
    UsdGeom.Camera.Define(stage, "/AAA_Camera")
    skeleton = UsdSkel.Skeleton.Define(stage, "/AAA_Rig")
    skeleton.CreateJointsAttr(["joint0"])
    skeleton.CreateRestTransformsAttr([matrix(0, 0, -10)])
    skeleton.CreateBindTransformsAttr([matrix()])
    metadata = inspect_stage(stage)
    assert metadata["cameras"] == ["/AAA_Camera", "/Camera"]
    assert [item["path"] for item in metadata["skeletons"]] == ["/AAA_Rig", "/Character/Rig"]
    assert [item["path"] for item in sample_stage(stage, camera_path="/Camera")[0]["skeletons"]] == ["/AAA_Rig", "/Character/Rig"]


@pytest.mark.parametrize("attribute,value", [
    ("horizontalAperture", 0), ("focalLength", -1), ("clippingRange", (10, 1)),
    ("horizontalApertureOffset", math.nan), ("projection", "fisheye"),
])
def test_invalid_cameras_fail_clearly(attribute, value):
    stage, _, _, camera = scene()
    camera.GetPrim().GetAttribute(attribute).Set(value)
    with pytest.raises(ValueError, match="Camera"):
        sample_stage(stage)


@pytest.mark.parametrize("extension", [".usd", ".usda", ".usdc", ".usdz"])
def test_open_stage_supports_all_usd_file_formats(tmp_path, extension):
    stage, *_ = scene()
    target = tmp_path / ("scene" + extension)
    if extension == ".usdz":
        source = tmp_path / "source.usda"
        stage.Export(str(source))
        assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(source)), str(target))
    else:
        stage.Export(str(target))
    original = target.read_bytes()
    opened = open_stage(target)
    assert points(opened) == approx_points([(50, 50, 1)])
    assert Path(inspect_stage(opened)["path"]) == target
    assert target.read_bytes() == original


def test_reopening_refreshes_referenced_layers_even_with_retained_stage(tmp_path):
    stage, *_ = scene()
    reference = tmp_path / "character.usda"
    stage.Export(str(reference))
    source = tmp_path / "scene.usda"
    source.write_text('#usda 1.0\n( subLayers = [@character.usda@] )\n', encoding="utf-8")
    first = open_stage(source)
    assert points(first) == approx_points([(50, 50, 1)])
    changed, *_ = scene(((2, 0, -10),))
    changed.Export(str(reference))
    second = open_stage(source)
    assert second is not first
    assert points(second) == approx_points([(60, 50, 1)])


def test_open_stage_never_discards_another_stages_dirty_edits(tmp_path):
    stage, *_ = scene()
    source = tmp_path / "scene.usda"
    stage.Export(str(source))
    opened = Usd.Stage.Open(str(source))
    opened.GetPrimAtPath("/Character/Rig").GetAttribute("restTransforms").Set([matrix(3, 0, -10)])
    original = source.read_bytes()
    with pytest.raises(ValueError, match="unsaved edits"):
        open_stage(source)
    assert points(opened) == approx_points([(65, 50, 1)])
    assert source.read_bytes() == original


def test_open_stage_path_and_composition_errors_are_actionable(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        open_stage("relative.usda")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        open_stage(tmp_path / "absent.usda")
    with pytest.raises(ValueError, match="Unsupported USD extension"):
        open_stage(tmp_path / "bad.obj")
    malformed = tmp_path / "bad.usda"
    malformed.write_text("not USD", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not open USD stage"):
        open_stage(malformed)
    incomplete = tmp_path / "incomplete.usda"
    incomplete.write_text('#usda 1.0\n( subLayers = [@missing.usda@] )\n', encoding="utf-8")
    with pytest.raises(ValueError, match="composition is incomplete"):
        open_stage(incomplete)
