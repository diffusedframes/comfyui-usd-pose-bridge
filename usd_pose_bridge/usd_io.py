"""Read UsdSkel poses and project them through authored USD cameras.

OpenUSD is optional at import time. Coordinates are pixels with a top-left
origin; invisible joints use the OpenPose sentinel ``(0, 0, 0)``. The USD
camera defines the view, so stage up-axis and unit metadata are never used
to rotate or rescale joint positions independently of the camera.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import threading
from functools import lru_cache
from numbers import Integral, Real
from typing import Sequence


_OPEN_LOCK = threading.RLock()
_EXTENSIONS = {".usd", ".usda", ".usdc", ".usdz"}


@lru_cache(maxsize=1)
def _usd():
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel
    except (ImportError, OSError) as exc:
        raise ImportError(
            "OpenUSD could not be loaded. Install usd-core in the Python environment "
            "that runs ComfyUI (python -m pip install usd-core). If pxr is already "
            "installed, use a wheel matching that Python version and remove conflicting "
            "standalone USD entries from PYTHONPATH/PATH; incompatible USD DLLs can "
            f"cause this error. Original error: {exc}"
        ) from exc
    return Gf, Sdf, Usd, UsdGeom, UsdSkel


def open_stage(path: str | os.PathLike):
    """Open a fresh stage without authoring or saving any source file.

    USD stages share file-backed Sdf layers process-wide. Non-dirty used layers
    are explicitly refreshed, including referenced/payload/sublayer files. Dirty
    layers are rejected rather than discarding another caller's unsaved edits.
    As with USD's Reload API, refresh can update other stages sharing those
    layers; callers should not concurrently edit/read the same files elsewhere.
    """
    if not isinstance(path, (str, os.PathLike)) or not os.fspath(path).strip():
        raise ValueError("Provide an absolute path to a .usd, .usda, .usdc, or .usdz file.")
    source = Path(path).expanduser()
    if not source.is_absolute():
        raise ValueError(f"USD file path must be absolute: {source}")
    if source.suffix.lower() not in _EXTENSIONS:
        raise ValueError(f"Unsupported USD extension '{source.suffix}'. Use .usd, .usda, .usdc, or .usdz.")
    if not source.is_file():
        raise FileNotFoundError(f"USD file does not exist or is not a file: {source}")
    source = source.resolve()
    _, Sdf, Usd, _, _ = _usd()
    with _OPEN_LOCK:
        # A surrounding ComfyUI extension may install a UsdStageCacheContext.
        # Block it so selecting/re-evaluating a source never reuses that stage.
        try:
            with Usd.StageCacheContext(Usd.BlockStageCaches):
                stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadAll)
            if not stage:
                raise ValueError("OpenUSD returned no stage.")
            refreshed = set()
            while True:
                pending = [
                    layer for layer in stage.GetUsedLayers()
                    if not layer.anonymous and layer.identifier not in refreshed
                ]
                if not pending:
                    break
                # Check the whole batch first: never erase unsaved edits.
                for layer in pending:
                    if layer.dirty:
                        raise ValueError(
                            f"USD layer has unsaved edits in this Python process: {layer.identifier}. "
                            "Save or close the editing stage before loading it here."
                        )
                # Reloading the root can introduce references absent from the
                # first composition; the loop also refreshes newly used layers.
                root_id = stage.GetRootLayer().identifier
                pending.sort(key=lambda layer: layer.identifier != root_id)
                for layer in pending:
                    if not layer.Reload(True):
                        raise ValueError(f"Could not refresh USD layer: {layer.identifier}")
                    refreshed.add(layer.identifier)
            errors = stage.GetCompositionErrors()
            if errors:
                raise ValueError("USD composition is incomplete: " + "; ".join(str(error) for error in errors))
            return stage
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Could not open USD stage '{source}': {exc}") from exc


def _prims(stage):
    _, _, Usd, _, _ = _usd()
    if not isinstance(stage, Usd.Stage):
        raise TypeError("Expected a Usd.Stage returned by open_stage().")
    return list(stage.Traverse(Usd.TraverseInstanceProxies()))


def _cache_and_skeletons(stage):
    _, _, Usd, _, UsdSkel = _usd()
    prims = _prims(stage)
    cache = UsdSkel.Cache()
    for prim in prims:
        if prim.IsA(UsdSkel.Root):
            cache.Populate(UsdSkel.Root(prim), Usd.TraverseInstanceProxies())
    skeletons = [UsdSkel.Skeleton(prim) for prim in prims if prim.IsA(UsdSkel.Skeleton)]
    return cache, sorted(skeletons, key=lambda skeleton: str(skeleton.GetPath()))


def inspect_stage(stage) -> dict:
    """Return JSON-serializable stage, camera, and skeleton metadata."""
    _, _, _, UsdGeom, _ = _usd()
    cache, skeletons = _cache_and_skeletons(stage)
    root = stage.GetRootLayer()
    result = {
        "path": str(root.realPath or root.identifier),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "time_start": float(stage.GetStartTimeCode()),
        "time_end": float(stage.GetEndTimeCode()),
        "time_codes_per_second": float(stage.GetTimeCodesPerSecond()),
        "cameras": sorted(str(prim.GetPath()) for prim in _prims(stage) if prim.IsA(UsdGeom.Camera)),
        "skeletons": [],
    }
    for skeleton in skeletons:
        query = cache.GetSkelQuery(skeleton)
        anim = query.GetAnimQuery() if query else None
        result["skeletons"].append({
            "path": str(skeleton.GetPath()),
            "joints": [str(joint) for joint in (skeleton.GetJointsAttr().Get() or [])],
            "animated": bool((anim and anim.JointTransformsMightBeTimeVarying())
                             or _transform_might_vary(skeleton.GetPrim())),
        })
    return result


def _transform_might_vary(prim):
    _, _, _, UsdGeom, _ = _usd()
    while prim and not prim.IsPseudoRoot():
        xformable = UsdGeom.Xformable(prim)
        if xformable:
            if xformable.TransformMightBeTimeVarying():
                return True
            if xformable.GetResetXformStack():
                break
        prim = prim.GetParent()
    return False


def _absolute_prim_path(value: str, label: str):
    _, Sdf, _, _, _ = _usd()
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be an absolute USD prim path.")
    valid, reason = Sdf.Path.IsValidPathString(value)
    if not valid:
        raise ValueError(f"Invalid {label} '{value}': {reason}")
    path = Sdf.Path(value)
    if not path.IsAbsolutePath() or not path.IsPrimPath():
        raise ValueError(f"{label} must be an absolute USD prim path: {value}")
    return path


def _select_camera(stage, camera_path):
    _, _, _, UsdGeom, _ = _usd()
    if not isinstance(camera_path, str):
        raise ValueError("camera_path must be an absolute USD prim path or an empty string.")
    camera_path = camera_path.strip()
    if camera_path:
        prim = stage.GetPrimAtPath(_absolute_prim_path(camera_path, "Camera path"))
        if not prim or not prim.IsA(UsdGeom.Camera):
            raise ValueError(f"Camera path does not identify a USD Camera: {camera_path}")
        return UsdGeom.Camera(prim)
    cameras = [UsdGeom.Camera(prim) for prim in _prims(stage) if prim.IsA(UsdGeom.Camera)]
    if not cameras:
        raise ValueError("No USD Camera found. Author a camera in the stage and select its prim path.")
    if len(cameras) != 1:
        raise ValueError(
            "Several USD cameras are available; specify camera_path: "
            + ", ".join(str(camera.GetPath()) for camera in cameras)
        )
    return cameras[0]


def _select_skeletons(skeletons, skeleton_paths):
    if not isinstance(skeleton_paths, str):
        raise ValueError("skeleton_paths must be '*' or comma-separated absolute skeleton prim paths.")
    available = {str(skeleton.GetPath()): skeleton for skeleton in skeletons}
    if not available:
        raise ValueError("No UsdSkel Skeleton found in the stage. Export a USD skeleton, not only a mesh.")
    if skeleton_paths.strip() in ("", "*"):
        return skeletons
    chosen = []
    for raw in skeleton_paths.split(","):
        path = str(_absolute_prim_path(raw.strip(), "Skeleton path"))
        if path not in available:
            raise ValueError(f"Skeleton not found: {path}. Available: {', '.join(available)}")
        if available[path] not in chosen:
            chosen.append(available[path])
    return chosen


def _finite_matrix(matrix) -> bool:
    return all(math.isfinite(float(component)) for row in matrix for component in row)


def _validated_query(cache, skeleton):
    _, _, _, _, UsdSkel = _usd()
    path = str(skeleton.GetPath())
    joints = [str(joint) for joint in (skeleton.GetJointsAttr().Get() or [])]
    if not joints or len(set(joints)) != len(joints):
        raise ValueError(f"Skeleton {path} must have a nonempty, unique joints array.")
    valid, reason = UsdSkel.Topology(joints).Validate()
    if not valid:
        raise ValueError(f"Skeleton {path} has invalid joint hierarchy: {reason}")
    for label, attr in (("restTransforms", skeleton.GetRestTransformsAttr()),
                        ("bindTransforms", skeleton.GetBindTransformsAttr())):
        values = attr.Get()
        if values is not None:
            if len(values) != len(joints) or any(not _finite_matrix(matrix) for matrix in values):
                raise ValueError(f"Skeleton {path} {label} must contain one finite matrix per joint ({len(joints)}).")
    query = cache.GetSkelQuery(skeleton)
    if not query:
        raise ValueError(f"OpenUSD could not evaluate skeleton {path}. Check its joints and animation binding.")
    if not query.HasRestPose() and not query.GetAnimQuery():
        raise ValueError(f"Skeleton {path} has neither a valid rest pose nor bound animation.")
    return joints, query


def _validate_animation(query, time, path):
    """Prevent UsdSkel's fallback to rest from hiding malformed animation arrays."""
    _, _, _, _, UsdSkel = _usd()
    anim_query = query.GetAnimQuery()
    if not anim_query:
        return
    animation = UsdSkel.Animation(anim_query.GetPrim())
    order = list(anim_query.GetJointOrder())
    components = [(label, attr.Get(time)) for label, attr in (
        ("translations", animation.GetTranslationsAttr()),
        ("rotations", animation.GetRotationsAttr()),
        ("scales", animation.GetScalesAttr()),
    )]
    if not order:
        # SkelAnimation may provide only blendshape weights. It then has no
        # joint pose to override and UsdSkel uses the skeleton's rest pose.
        if any(values is not None and len(values) for _, values in components):
            raise ValueError(f"Animation {animation.GetPath()} provides joint transforms without a joints array.")
        if not query.HasRestPose():
            raise ValueError(f"Skeleton {path} has no rest pose and its bound animation supplies no joint transforms.")
        return
    if len(set(order)) != len(order):
        raise ValueError(f"Animation bound to {path} needs a unique joints array.")
    if any(values is None for _, values in components):
        # Components have no schema identity defaults. A missing or blocked
        # component nullifies the entire joint animation; all joints fall
        # back to the rest pose, matching native UsdSkel evaluation.
        if not query.HasRestPose():
            raise ValueError(f"Skeleton {path} has no rest pose and its bound animation is missing joint transform components.")
        return
    for label, values in components:
        if values is not None and len(values) != len(order):
            raise ValueError(
                f"Animation {animation.GetPath()} {label} at frame {time.GetValue()} must "
                f"contain one value per animation joint ({len(order)})."
            )
    # Compute on the animation query itself: the skeleton query otherwise
    # silently substitutes the rest pose if animation evaluation fails.
    transforms = anim_query.ComputeJointLocalTransforms(time)
    if len(transforms) != len(order) or any(not _finite_matrix(matrix) for matrix in transforms):
        raise ValueError(f"Animation bound to {path} has invalid transforms at frame {time.GetValue()}.")


def _camera_matrices(camera_schema, time, width, height, camera_fit):
    Gf, _, _, _, _ = _usd()
    if str(camera_schema.GetProjectionAttr().Get(time)) not in ("perspective", "orthographic"):
        raise ValueError(f"Camera {camera_schema.GetPath()} uses an unsupported projection; use perspective or orthographic.")
    camera = camera_schema.GetCamera(time)
    if (not _finite_matrix(camera.transform)
            or camera.transform.GetDeterminant() == 0
            or not all(math.isfinite(float(value)) and value > 0 for value in
                       (camera.horizontalAperture, camera.verticalAperture))
            or not all(math.isfinite(float(value)) for value in
                       (camera.horizontalApertureOffset, camera.verticalApertureOffset))
            or (camera.projection == Gf.Camera.Perspective
                and (not math.isfinite(camera.focalLength) or camera.focalLength <= 0))):
        raise ValueError(f"Camera {camera_schema.GetPath()} has invalid optics or transforms at frame {time.GetValue()}.")
    near, far = camera.clippingRange.min, camera.clippingRange.max
    if not (math.isfinite(near) and math.isfinite(far) and 0 < near < far):
        raise ValueError(f"Camera {camera_schema.GetPath()} needs a finite clipping range with 0 < near < far.")
    frustum = camera.frustum
    window = frustum.GetWindow()
    size = window.GetSize()
    center = (window.GetMin() + window.GetMax()) / 2
    aspect = width / height
    # Equivalent to CameraUtilConformWindow(GfFrustum*, Fit/Crop, aspect).
    # Implemented on the frustum so usd-core need not provide imaging modules.
    match_width = (camera_fit == "fit") == (size[0] > aspect * size[1])
    if match_width:
        half = Gf.Vec2d(size[0] / 2, size[0] / aspect / 2)
    else:
        half = Gf.Vec2d(size[1] * aspect / 2, size[1] / 2)
    frustum.SetWindow(Gf.Range2d(center - half, center + half))
    view = frustum.ComputeViewMatrix()
    projection = frustum.ComputeProjectionMatrix()
    planes = list(camera.clippingPlanes)
    if (not _finite_matrix(view) or not _finite_matrix(projection)
            or any(not all(math.isfinite(float(value)) for value in plane) for plane in planes)):
        raise ValueError(f"Camera {camera_schema.GetPath()} has an invalid projection or clipping planes.")
    return view, projection, planes


def _project(point, view, projection, planes, width, height):
    Gf, _, _, _, _ = _usd()
    hidden = (0.0, 0.0, 0.0)
    eye = Gf.Vec4d(float(point[0]), float(point[1]), float(point[2]), 1.0) * view
    if eye[2] >= 0:
        return hidden
    # USD clippingPlanes are in camera space, with negative half-spaces clipped.
    if any(sum(float(plane[index]) * eye[index] for index in range(4)) < 0 for plane in planes):
        return hidden
    clip = eye * projection
    if not all(math.isfinite(float(value)) for value in clip) or clip[3] <= 0:
        return hidden
    ndc = [float(clip[index] / clip[3]) for index in range(3)]
    if any(value < -1.000000001 or value > 1.000000001 for value in ndc):
        return hidden
    # Pixel centers span [0, width-1] / [0, height-1], including the frustum edges.
    x = min(1.0, max(0.0, (ndc[0] + 1) / 2)) * (width - 1)
    y = min(1.0, max(0.0, (1 - ndc[1]) / 2)) * (height - 1)
    return (x, y, 1.0)


def sample_stage(
    stage,
    skeleton_paths: str = "*",
    camera_path: str = "",
    frames: Sequence[float] = (1,),
    width: int = 512,
    height: int = 512,
    camera_fit: str = "fit",
) -> list[dict]:
    """Evaluate selected skeletons and camera at each requested USD time code.

    Confidence denotes only camera-frustum visibility, not mesh occlusion.
    Results preserve the requested frame and skeleton selection order.
    """
    for label, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= 16384:
            raise ValueError(f"{label} must be an integer between 1 and 16384.")
    if camera_fit not in ("fit", "crop"):
        raise ValueError("camera_fit must be 'fit' or 'crop'.")
    if isinstance(frames, (str, bytes)):
        raise ValueError("frames must be a nonempty sequence of finite numeric USD time codes.")
    try:
        frame_values = list(frames)
    except TypeError as exc:
        raise ValueError("frames must be a nonempty sequence of finite numeric USD time codes.") from exc
    try:
        valid_frames = bool(frame_values) and all(
            not isinstance(frame, bool) and isinstance(frame, Real) and math.isfinite(float(frame))
            for frame in frame_values
        )
    except (ValueError, TypeError, OverflowError):
        valid_frames = False
    if not valid_frames:
        raise ValueError("frames must be a nonempty sequence of finite numeric USD time codes.")
    _, _, Usd, UsdGeom, _ = _usd()
    cache, skeletons = _cache_and_skeletons(stage)
    camera = _select_camera(stage, camera_path)
    selected = _select_skeletons(skeletons, skeleton_paths)
    queries = [(skeleton, *_validated_query(cache, skeleton)) for skeleton in selected]
    snapshots = []
    for frame in frame_values:
        time = Usd.TimeCode(float(frame))
        xforms = UsdGeom.XformCache(time)
        view, projection, planes = _camera_matrices(camera, time, width, height, camera_fit)
        snapshot = {"frame": float(frame), "camera_path": str(camera.GetPath()), "skeletons": []}
        for skeleton, joints, query in queries:
            path = str(skeleton.GetPath())
            _validate_animation(query, time, path)
            try:
                world = query.ComputeJointWorldTransforms(xforms)
            except Exception as exc:
                raise ValueError(f"Could not evaluate skeleton {path} at frame {frame}: {exc}") from exc
            if len(world) != len(joints) or any(not _finite_matrix(matrix) for matrix in world):
                raise ValueError(f"Skeleton {path} did not produce one finite world transform per joint at frame {frame}.")
            snapshot["skeletons"].append({
                "path": path,
                "joints": list(joints),
                "points": [_project(matrix.ExtractTranslation(), view, projection, planes, width, height)
                           for matrix in world],
            })
        snapshots.append(snapshot)
    return snapshots
