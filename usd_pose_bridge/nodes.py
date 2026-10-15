"""ComfyUI nodes. The USD and Torch runtimes are loaded only on execution."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .mapping import build_person, parse_mapping
from .pose_io import load_pose_json, resolve_input_path, save_pose_json
from .render import render_frames
from .usd_io import inspect_stage, open_stage, sample_stage

PACK_ROOT = Path(__file__).resolve().parent.parent
MAX_BATCH_PIXELS = 32_000_000
CATEGORY = "DiffusedFrames/USD Pose"


def _report(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)


def _tensor(frames, stick_width=4, draw_hands=True, draw_face=True):
    _check_batch(frames)
    import torch

    return torch.from_numpy(render_frames(frames, stick_width, draw_hands, draw_face))


def _check_batch(frames):
    if not isinstance(frames, list) or not frames:
        raise ValueError("A nonempty list of OpenPose frames is required.")
    pixels = 0
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("Each pose frame must be an object.")
        for key in ("canvas_width", "canvas_height"):
            value = frame.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 4096:
                raise ValueError(f"{key} must be an integer from 1 to 4096.")
        pixels += frame["canvas_width"] * frame["canvas_height"]
    if pixels > MAX_BATCH_PIXELS:
        raise ValueError("Image batch exceeds 32 million pixels. Reduce resolution or frame_count and process in chunks.")


class USDPBLoadStage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"stage_path": ("STRING", {
            "default": "examples/demo.usda",
            "tooltip": "USD file on the ComfyUI server. Relative paths use ComfyUI/input; examples/ uses this pack.",
        })}}

    RETURN_TYPES = ("USD_STAGE", "STRING")
    RETURN_NAMES = ("stage", "stage_info")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Read a USD scene with skeletons and cameras. Does not modify the source scene."

    @classmethod
    def IS_CHANGED(cls, stage_path):
        # A composed stage can depend on many externally edited references and clips.
        return float("nan")

    def load(self, stage_path):
        path = resolve_input_path(stage_path, PACK_ROOT)
        stage = open_stage(str(path))
        info = _report(inspect_stage(stage))
        return {"ui": {"usd_pose_report": [info]}, "result": (stage, info)}


class USDPBInspectStage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"stage": ("USD_STAGE",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("scene_info",)
    FUNCTION = "inspect"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = "List skeleton paths, joint names, cameras, units, and the scene timeline."

    def inspect(self, stage):
        info = _report(inspect_stage(stage))
        return {"ui": {"usd_pose_report": [info]}, "result": (info,)}


class USDPBJointMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mapping_json": ("STRING", {
            "default": "{}", "multiline": True,
            "tooltip": 'Example: {"body": {"left_shoulder": "Hips/Spine/LeftArm"}}. Empty means automatic mapping.',
        })}}

    RETURN_TYPES = ("USD_JOINT_MAP",)
    RETURN_NAMES = ("joint_map",)
    FUNCTION = "create"
    CATEGORY = CATEGORY
    DESCRIPTION = "Override automatic joint mapping with precise USD joint paths. See examples/joint_map.json."

    def create(self, mapping_json):
        return (parse_mapping(mapping_json),)


class USDPBProjectPose:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage": ("USD_STAGE",),
                "camera_path": ("STRING", {"default": "", "tooltip": "Blank selects the camera if the scene has exactly one."}),
                "skeleton_paths": ("STRING", {"default": "*", "tooltip": "* for all skeletons, or comma-separated absolute prim paths."}),
                "start_frame": ("FLOAT", {"default": 1.0, "min": -1_000_000.0, "max": 1_000_000.0, "step": 1.0}),
                "frame_count": ("INT", {"default": 1, "min": 1, "max": 512}),
                "frame_step": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 10000.0, "step": 0.1}),
                "width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 8}),
                "pose_format": (["COCO_18", "BODY_25"], {"default": "COCO_18"}),
                "camera_fit": (["fit", "crop"], {"default": "fit"}),
                "coordinate_space": (["normalized", "pixel"], {"default": "normalized"}),
                "stick_width": ("INT", {"default": 4, "min": 1, "max": 32}),
                "draw_hands": ("BOOLEAN", {"default": True}),
                "draw_face": ("BOOLEAN", {"default": True}),
            },
            "optional": {"joint_map": ("USD_JOINT_MAP",)},
        }

    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT", "STRING")
    RETURN_NAMES = ("pose_images", "pose_keypoint", "mapping_report")
    FUNCTION = "project"
    CATEGORY = CATEGORY
    DESCRIPTION = "Project animated USD skeletons through a USD camera into an image batch and OpenPose keypoints."

    def project(self, stage, camera_path="", skeleton_paths="*", start_frame=1.0,
                frame_count=1, frame_step=1.0, width=768, height=768, pose_format="COCO_18",
                camera_fit="fit", coordinate_space="normalized", stick_width=4,
                draw_hands=True, draw_face=True, joint_map=None):
        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or not 1 <= frame_count <= 512:
            raise ValueError("frame_count must be an integer from 1 to 512.")
        if not math.isfinite(start_frame) or not math.isfinite(frame_step) or frame_step <= 0:
            raise ValueError("start_frame must be finite and frame_step must be positive and finite.")
        if coordinate_space not in {"pixel", "normalized"}:
            raise ValueError("coordinate_space must be normalized or pixel.")
        if pose_format not in {"COCO_18", "BODY_25"}:
            raise ValueError("pose_format must be COCO_18 or BODY_25.")
        # Bound image memory before sampling an expensive scene.
        _check_batch([{"canvas_width": width, "canvas_height": height}] * frame_count)
        frames = [float(start_frame + i * frame_step) for i in range(frame_count)]
        snapshots = sample_stage(stage, skeleton_paths=skeleton_paths, camera_path=camera_path,
                                 frames=frames, width=width, height=height, camera_fit=camera_fit)
        pose_frames = []
        mapping_reports = {}
        for snapshot in snapshots:
            people = []
            for skeleton in snapshot["skeletons"]:
                points = skeleton["points"]
                if coordinate_space == "normalized":
                    points = [(x / width, y / height, confidence) for x, y, confidence in points]
                person, mapping_report = build_person(skeleton["joints"], points, pose_format, joint_map)
                person["person_id"] = [len(people)]
                person["usd_skeleton_path"] = skeleton["path"]
                people.append(person)
                mapping_reports[skeleton["path"]] = mapping_report
            pose_frames.append({
                "version": 1.3, "people": people, "canvas_width": width, "canvas_height": height,
                "pose_format": pose_format, "coordinate_space": coordinate_space,
                "usd_frame": snapshot["frame"], "usd_camera_path": snapshot["camera_path"],
                "time_codes_per_second": float(stage.GetTimeCodesPerSecond()),
            })
        info = _report({
            "frame_count": len(pose_frames), "frames": frames,
            "camera_path": snapshots[0]["camera_path"], "pose_format": pose_format,
            "coordinate_space": coordinate_space, "skeletons": mapping_reports,
            "visibility": "Camera frustum only; mesh occlusion is not evaluated.",
        })
        return {"ui": {"usd_pose_report": [info]},
                "result": (_tensor(pose_frames, stick_width, draw_hands, draw_face), pose_frames, info)}


class USDPBRenderPose:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pose_keypoint": ("POSE_KEYPOINT",),
            "stick_width": ("INT", {"default": 4, "min": 1, "max": 32}),
            "draw_hands": ("BOOLEAN", {"default": True}),
            "draw_face": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pose_images",)
    FUNCTION = "render"
    CATEGORY = CATEGORY
    DESCRIPTION = "Render an OpenPose sequence, including POSE_KEYPOINT data from compatible pose preprocessors."

    def render(self, pose_keypoint, stick_width=4, draw_hands=True, draw_face=True):
        return (_tensor(pose_keypoint, stick_width, draw_hands, draw_face),)


class USDPBLoadPose:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pose_path": ("STRING", {"default": "pose.json"}),
            "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
            "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
            "coordinate_space": (["auto", "pixel", "normalized"], {"default": "auto"}),
        }}

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoint",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Read a frame or sequence of OpenPose JSON. Width/height supply missing canvas metadata."

    @classmethod
    def IS_CHANGED(cls, pose_path, **kwargs):
        path = resolve_input_path(pose_path, PACK_ROOT)
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size

    def load(self, pose_path, width=512, height=512, coordinate_space="auto"):
        frames = load_pose_json(resolve_input_path(pose_path, PACK_ROOT), width, height, coordinate_space)
        _check_batch(frames)
        return (frames,)


class USDPBSavePose:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pose_keypoint": ("POSE_KEYPOINT",),
            "filename_prefix": ("STRING", {"default": "usd_pose/pose"}),
            "format": (["sequence", "per_frame"], {"default": "sequence"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_paths",)
    FUNCTION = "save"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = "Save a sequence JSON or one OpenPose JSON per frame under ComfyUI/output, without overwriting."

    def save(self, pose_keypoint, filename_prefix="usd_pose/pose", format="sequence"):
        import folder_paths

        paths = save_pose_json(pose_keypoint, Path(folder_paths.get_output_directory()), filename_prefix, format)
        info = _report({"saved_files": paths})
        return {"ui": {"usd_pose_report": [info]}, "result": (info,)}


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    USDPBLoadStage, USDPBInspectStage, USDPBJointMap, USDPBProjectPose,
    USDPBRenderPose, USDPBLoadPose, USDPBSavePose,
)}

NODE_DISPLAY_NAME_MAPPINGS = {
    "USDPBLoadStage": "Load USD Stage",
    "USDPBInspectStage": "Inspect USD Stage",
    "USDPBJointMap": "USD Joint Map",
    "USDPBProjectPose": "USD to OpenPose",
    "USDPBRenderPose": "Render OpenPose",
    "USDPBLoadPose": "Load OpenPose JSON",
    "USDPBSavePose": "Save OpenPose JSON",
}
