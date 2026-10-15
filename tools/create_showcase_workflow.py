"""Author the optional USD -> OpenPose -> SDXL photographic-plate showcase.

No models are downloaded and no inference is started by this script. The API
graphs use built-in ComfyUI nodes plus this pack; the UI graphs have identical
settings. Run with --verify-url to check port schemas against a running ComfyUI.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
CONTROLNET = "SDXL\\xinsir-openpose-sdxl-1.0.safetensors"
POSITIVE = (
    "cinematic live action film still, a single full body astronaut wearing a "
    "weathered graphite gray and burnt orange EVA spacesuit, dark reflective "
    "helmet visor, realistic human proportions, detailed practical costume, "
    "natural running stride, one foot naturally lifted, believable grounded "
    "foot contact, industrial harbor location, overcast natural "
    "daylight, soft contact shadow, realistic photography, restrained colors, "
    "matching the existing photograph's perspective and lighting"
)
NEGATIVE = (
    "two people, multiple characters, extra person, duplicate body, extra limbs, "
    "extra arms, extra legs, missing limbs, malformed hands, dislocated joints, "
    "bad anatomy, giant head, cropped body, cut off feet, floating person, "
    "cartoon, illustration, plastic render, text, watermark, blurry"
)


@dataclass
class Node:
    id: int
    type: str
    title: str
    pos: tuple[int, int]
    size: tuple[int, int]
    # Input sockets and outputs are explicit so authoring also works offline.
    sockets: list[tuple[str, str]]
    outputs: list[tuple[str, str]]
    values: dict[str, Any]
    connections: dict[str, tuple[int, int]]
    optional: tuple[str, ...] = ()


def author_case(case: str, info: dict[str, Any], mapping: dict[str, Any], args: Any):
    """Return a portable UI workflow and the corresponding API request body."""
    frame = float(args.frame if args.frame is not None else info.get("usd_frame", 1))
    camera = args.camera or info.get("camera_path", f"/World/Camera_{case}")
    width = int(info.get("width", 1024))
    height = int(info.get("height", 768))
    # ComfyUI's input folder is independent of the installed custom-node folder.
    plate = f"diffusedframes_showcase/{Path(info.get('plate', f'plate_{case}.png')).name}"
    mask = f"diffusedframes_showcase/{Path(info.get('mask', f'mask_{case}.png')).name}"
    prefix = f"diffusedframes_showcase/case_{case}"
    definitions: list[Node] = []

    def add(id_, type_, title, pos, sockets=(), outputs=(), values=None, links=None,
            size=(310, 150), optional=()):
        definitions.append(Node(id_, type_, title, pos, size, list(sockets),
                                list(outputs), values or {}, links or {}, optional))

    add(1, "USDPBLoadStage", "01 | Load animated USD character", (30, 160),
        outputs=[("stage", "USD_STAGE"), ("stage_info", "STRING")],
        values={"stage_path": args.stage}, size=(345, 110))
    add(2, "USDPBJointMap", "Map the source rig's body joints", (30, 330),
        outputs=[("joint_map", "USD_JOINT_MAP")],
        values={"mapping_json": json.dumps(mapping, indent=2)}, size=(345, 300))
    add(3, "USDPBProjectPose", "02 | USD camera + frame -> OpenPose", (420, 160),
        sockets=[("stage", "USD_STAGE"), ("joint_map", "USD_JOINT_MAP")],
        outputs=[("pose_images", "IMAGE"), ("pose_keypoint", "POSE_KEYPOINT"),
                 ("mapping_report", "STRING")],
        values={"camera_path": camera,
                "skeleton_paths": info.get("skeleton_paths", info.get("skeleton_path", "*")),
                "start_frame": frame, "frame_count": 1, "frame_step": 1.0,
                "width": width, "height": height, "pose_format": "COCO_18",
                "camera_fit": "fit", "coordinate_space": "normalized",
                "stick_width": 8, "draw_hands": False, "draw_face": False},
        links={"stage": (1, 0), "joint_map": (2, 0)}, size=(355, 520),
        optional=("joint_map",))
    add(4, "SaveImage", "Save the actual USD pose control", (420, 740),
        sockets=[("images", "IMAGE")], outputs=[("images", "IMAGE")],
        values={"filename_prefix": f"{prefix}/pose"}, links={"images": (3, 0)},
        size=(355, 280))
    add(5, "LoadImage", "03 | Original photographic background", (30, 1120),
        outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], values={"image": plate},
        size=(345, 330))
    add(6, "LoadImage", "Actor / ground-contact region (white)", (30, 1520),
        outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], values={"image": mask},
        size=(345, 300))
    add(7, "ImageToMask", "Read white pixels as the edit mask", (420, 1520),
        sockets=[("image", "IMAGE")], outputs=[("MASK", "MASK")],
        values={"channel": "red"}, links={"image": (6, 0)}, size=(310, 90))
    add(25, "ThresholdMask", "Remove render noise; keep a binary actor mask", (420, 1680),
        sockets=[("mask", "MASK")], outputs=[("MASK", "MASK")],
        values={"value": 0.5}, links={"mask": (7, 0)}, size=(310, 95))
    add(8, "GrowMask", "Allow suit volume and contact shadows", (790, 1520),
        sockets=[("mask", "MASK")], outputs=[("MASK", "MASK")],
        values={"expand": 32, "tapered_corners": True}, links={"mask": (25, 0)},
        size=(315, 115))
    add(9, "MaskToImage", "Mask as image for feathering", (790, 1690),
        sockets=[("mask", "MASK")], outputs=[("IMAGE", "IMAGE")],
        links={"mask": (8, 0)}, size=(315, 70))
    add(10, "ImageBlur", "Feather the finite mask boundary", (1160, 1520),
        sockets=[("image", "IMAGE")], outputs=[("IMAGE", "IMAGE")],
        values={"blur_radius": 15, "sigma": 5.0}, links={"image": (9, 0)},
        size=(315, 115))
    add(11, "ImageToMask", "Final generation + composite mask", (1520, 1520),
        sockets=[("image", "IMAGE")], outputs=[("MASK", "MASK")],
        values={"channel": "red"}, links={"image": (10, 0)}, size=(315, 90))
    add(12, "CheckpointLoaderSimple", "04 | Optional SDXL image model", (850, 160),
        outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")],
        values={"ckpt_name": args.checkpoint}, size=(365, 140))
    add(13, "CLIPTextEncode", "Character appearance and shot lighting", (1270, 160),
        sockets=[("clip", "CLIP")], outputs=[("CONDITIONING", "CONDITIONING")],
        values={"text": args.positive}, links={"clip": (12, 1)}, size=(395, 250))
    add(14, "CLIPTextEncode", "Avoid duplicate figures and anatomy defects", (1270, 470),
        sockets=[("clip", "CLIP")], outputs=[("CONDITIONING", "CONDITIONING")],
        values={"text": args.negative}, links={"clip": (12, 1)}, size=(395, 220))
    add(15, "ControlNetLoader", "Matching SDXL OpenPose ControlNet", (850, 380),
        outputs=[("CONTROL_NET", "CONTROL_NET")],
        values={"control_net_name": args.controlnet}, size=(365, 110))
    add(16, "ControlNetApplyAdvanced", "05 | Guide SDXL with projected USD pose", (1730, 160),
        sockets=[("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                 ("control_net", "CONTROL_NET"), ("image", "IMAGE"), ("vae", "VAE")],
        outputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING")],
        values={"strength": args.strength, "start_percent": 0.0, "end_percent": 1.0},
        links={"positive": (13, 0), "negative": (14, 0), "control_net": (15, 0),
               "image": (3, 0), "vae": (12, 2)}, size=(345, 235), optional=("vae",))
    add(17, "VAEEncodeForInpaint", "Encode plate; add noise only inside mask", (1730, 760),
        sockets=[("pixels", "IMAGE"), ("vae", "VAE"), ("mask", "MASK")],
        outputs=[("LATENT", "LATENT")], values={"grow_mask_by": 0},
        links={"pixels": (5, 0), "vae": (12, 2), "mask": (11, 0)}, size=(345, 150))
    add(18, "KSampler", "06 | Generate the astronaut", (2150, 160),
        sockets=[("model", "MODEL"), ("positive", "CONDITIONING"),
                 ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
        outputs=[("LATENT", "LATENT")],
        values={"seed": args.seed, "steps": args.steps, "cfg": 6.0,
                "sampler_name": "dpmpp_2m_sde", "scheduler": "karras", "denoise": 1.0},
        links={"model": (12, 0), "positive": (16, 0), "negative": (16, 1),
               "latent_image": (17, 0)}, size=(335, 310))
    add(19, "VAEDecode", "Decode the generated image", (2150, 560),
        sockets=[("samples", "LATENT"), ("vae", "VAE")], outputs=[("IMAGE", "IMAGE")],
        links={"samples": (18, 0), "vae": (12, 2)}, size=(335, 100))
    add(20, "ImageCompositeMasked", "07 | Preserve the original outside the mask", (2560, 160),
        sockets=[("destination", "IMAGE"), ("source", "IMAGE"), ("mask", "MASK")],
        outputs=[("IMAGE", "IMAGE")], values={"x": 0, "y": 0, "resize_source": False},
        links={"destination": (5, 0), "source": (19, 0), "mask": (11, 0)},
        size=(370, 190), optional=("mask",))
    add(21, "SaveImage", f"08 | Final plate composite - case {case}", (2560, 460),
        sockets=[("images", "IMAGE")], outputs=[("images", "IMAGE")],
        values={"filename_prefix": f"{prefix}/final"}, links={"images": (20, 0)},
        size=(460, 430))
    add(22, "SaveImage", "Save final feathered edit mask for comparison", (1520, 1760),
        sockets=[("images", "IMAGE")], outputs=[("images", "IMAGE")],
        values={"filename_prefix": f"{prefix}/edit_mask"}, links={"images": (10, 0)},
        size=(315, 280))
    add(23, "USDPBSavePose", "Export frame + camera + skeleton metadata", (420, 1070),
        sockets=[("pose_keypoint", "POSE_KEYPOINT")], outputs=[("saved_paths", "STRING")],
        values={"filename_prefix": f"usd_pose/showcase_{case}", "format": "sequence"},
        links={"pose_keypoint": (3, 1)}, size=(355, 150))

    api = {str(node.id): {"class_type": node.type,
                        "inputs": {**node.values, **{name: [str(link[0]), link[1]]
                                                      for name, link in node.connections.items()}},
                        "_meta": {"title": node.title}}
           for node in definitions}
    links: list[list[Any]] = []
    ui_nodes = []
    by_id = {node.id: node for node in definitions}
    for node in definitions:
        inputs = []
        for index, (name, type_) in enumerate(node.sockets):
            source_id, source_slot = node.connections[name]
            source_type = by_id[source_id].outputs[source_slot][1]
            if source_type != type_:
                raise ValueError(f"Type mismatch at {node.type}.{name}: {source_type} != {type_}")
            link_id = len(links) + 1
            links.append([link_id, source_id, source_slot, node.id, index, type_])
            inputs.append({"name": name, "type": type_, "link": link_id,
                           **({"shape": 7} if name in node.optional else {})})
        widgets = []
        for name, value in node.values.items():
            widgets.append(value)
            if node.type == "KSampler" and name == "seed":
                widgets.append("fixed")
        ui_nodes.append({"id": node.id, "type": node.type, "title": node.title,
                         "pos": list(node.pos), "size": list(node.size), "flags": {},
                         "order": len(ui_nodes), "mode": 0, "inputs": inputs,
                         "outputs": [{"name": name, "type": type_, "links": None,
                                      "slot_index": index}
                                     for index, (name, type_) in enumerate(node.outputs)],
                         "properties": {"Node name for S&R": node.type},
                         "widgets_values": widgets})
    ui_by_id = {node["id"]: node for node in ui_nodes}
    for link in links:
        output = ui_by_id[link[1]]["outputs"][link[2]]
        output["links"] = (output["links"] or []) + [link[0]]
    note = (
        "DIFFUSEDFRAMES | USD POSE BRIDGE - OPTIONAL SDXL SHOWCASE\n\n"
        f"Case {case}: {camera}, USD time code {frame:g}; {width} x {height}.\n"
        "The colored control is projected from the animated USD rig, not estimated from the photo.\n\n"
        "SETUP\n1. Install the USD Pose Bridge pack.\n"
        "2. Obtain the separately licensed showcase source assets as described in examples/showcase.\n"
        "3. Copy plate_*.png and mask_*.png to ComfyUI/input/diffusedframes_showcase/.\n"
        "4. Select an SDXL checkpoint and a matching SDXL OpenPose ControlNet in the two loaders.\n\n"
        f"Recorded checkpoint: {args.checkpoint}\nRecorded ControlNet: {args.controlnet}\n\n"
        "Run saves the projected pose, OpenPose JSON, feathered edit mask and final composite.\n"
        "Only the feathered edit region is replaced; original background pixels outside it are preserved.\n"
        "Change the USD frame or authored camera and regenerate the matching plate/mask for a new case.\n"
        "This is a still-image example; independent SDXL frames do not guarantee temporal consistency.\n"
        "No model download is required to use the pack's pose-export nodes."
    )
    ui_nodes.append({"id": 24, "type": "Note", "title": "Read first | Setup and what this proves",
                     "pos": [2120, 1060], "size": [890, 650], "flags": {}, "order": len(ui_nodes),
                     "mode": 0, "properties": {}, "widgets_values": [note],
                     "color": "#243447", "bgcolor": "#152536"})
    workflow = {"last_node_id": 25, "last_link_id": len(links), "nodes": ui_nodes,
                "links": links, "groups": [
                    {"title": "USD: skeleton, camera, frame -> OpenPose", "bounding": [0, 80, 805, 950],
                     "color": "#3f789e", "font_size": 24, "flags": {}},
                    {"title": "Photographic plate + editable region", "bounding": [0, 1050, 1870, 1040],
                     "color": "#8a7452", "font_size": 24, "flags": {}},
                    {"title": "Optional SDXL pose-controlled inpainting", "bounding": [825, 80, 1690, 900],
                     "color": "#5e7e5c", "font_size": 24, "flags": {}},
                    {"title": "Original plate composite", "bounding": [2530, 80, 530, 900],
                     "color": "#86627a", "font_size": 24, "flags": {}}],
                "config": {}, "extra": {"ds": {"scale": 0.42, "offset": [60, 60]}},
                "version": 0.4}
    return workflow, {"prompt": api}


def verify_schema(api: dict[str, Any], url: str):
    """Check actual server inputs and connection output types, without inference."""
    with urlopen(f"{url.rstrip('/')}/object_info", timeout=30) as response:
        schema = json.load(response)
    for id_, node in api["prompt"].items():
        spec = schema[node["class_type"]]
        required = spec["input"].get("required", {})
        accepted = {**required, **spec["input"].get("optional", {})}
        missing = set(required) - set(node["inputs"])
        unknown = set(node["inputs"]) - set(accepted)
        if missing or unknown:
            raise ValueError(f"{id_} {node['class_type']}: missing={missing}, unknown={unknown}")
        for name, value in node["inputs"].items():
            if isinstance(value, list):
                source = api["prompt"][value[0]]
                output_type = schema[source["class_type"]]["output"][value[1]]
                if accepted[name][0] != output_type:
                    raise ValueError(f"{id_}.{name}: {output_type} != {accepted[name][0]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-info", type=Path, default=ROOT / "examples/showcase/scene_info.json")
    parser.add_argument("--joint-map", type=Path, default=ROOT / "examples/showcase/joint_map.json")
    parser.add_argument("--stage", default="examples/showcase/shot.usdc")
    parser.add_argument("--output", type=Path, default=ROOT / "examples")
    parser.add_argument("--case", choices=("A", "B", "all"), default="all")
    parser.add_argument("--camera")
    parser.add_argument("--frame", type=float)
    parser.add_argument("--seed", type=int, default=2718281828)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--controlnet", default=CONTROLNET)
    parser.add_argument("--positive", default=POSITIVE)
    parser.add_argument("--negative", default=NEGATIVE)
    parser.add_argument("--verify-url", help="Optional running ComfyUI, e.g. http://127.0.0.1:8000")
    args = parser.parse_args()
    scene = json.loads(args.scene_info.read_text(encoding="utf-8")) if args.scene_info.exists() else {}
    cases = scene.get("cases", scene)
    if isinstance(cases, list):
        cases = {case.get("id", case.get("case")): case for case in cases}
    mapping = json.loads(args.joint_map.read_text(encoding="utf-8")) if args.joint_map.exists() else {}
    args.output.mkdir(parents=True, exist_ok=True)
    for case in ("A", "B") if args.case == "all" else (args.case,):
        workflow, api = author_case(case, {**scene, **cases.get(case, {})}, mapping, args)
        if args.verify_url:
            verify_schema(api, args.verify_url)
        suffix = "" if case == "A" else f"_{case}"
        for stem, content in (("workflow", workflow), ("api", api)):
            destination = args.output / f"{stem}_showcase{suffix}.json"
            destination.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
            print(destination)


if __name__ == "__main__":
    main()
