"""Build a real-asset USD / photograph showcase with Blender 5.2 or later.

Run from a shell (the downloads are cached outside the distributable examples)::

    blender --background --factory-startup --python tools/create_showcase.py

Only the generated USD, PNGs, joint map, and credit file need to be distributed.
The source actor is Cesium Man, copyright 2017 Cesium, CC BY 4.0. All original
materials, including the Cesium logo texture, are replaced. The background is
Philip Modin's CC0 ``abandoned_slipway`` HDRI from Poly Haven.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "dist" / "showcase_source"
OUTPUT = ROOT / "examples" / "showcase"
MODEL_URL = (
    "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/"
    "Models/CesiumMan/glTF-Binary/CesiumMan.glb"
)
HDRI_URL = "https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/4k/abandoned_slipway_4k.exr"


def download(url, path):
    if not path.is_file():
        print(f"Downloading {url}", flush=True)
        urllib.request.urlretrieve(url, path)


def material(name, color, metallic=0.0, roughness=0.5):
    result = bpy.data.materials.new(name)
    result.diffuse_color = (*color, 1)
    result.use_nodes = True
    shader = result.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    return result


def emission(name, color):
    result = bpy.data.materials.new(name)
    result.use_nodes = True
    tree = result.node_tree
    tree.nodes.clear()
    shader = tree.nodes.new("ShaderNodeEmission")
    shader.inputs["Color"].default_value = (*color, 1)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(shader.outputs[0], output.inputs["Surface"])
    return result


def aim(obj, point):
    obj.rotation_euler = (Vector(point) - obj.location).to_track_quat("-Z", "Y").to_euler()


def camera(name, location, target, focal_length):
    data = bpy.data.cameras.new(name)
    data.lens = focal_length
    data.sensor_width = 36
    data.sensor_fit = "HORIZONTAL"
    data.clip_start = 0.05
    data.clip_end = 1000
    result = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(result)
    result.location = location
    aim(result, target)
    return result


def setup_world(path):
    world = bpy.data.worlds.new("AbandonedSlipway_CC0_PhilipModin")
    world.use_nodes = True
    bpy.context.scene.world = world
    tree = world.node_tree
    tree.nodes.clear()
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    mapping = tree.nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value[2] = math.radians(120)
    texture = tree.nodes.new("ShaderNodeTexEnvironment")
    texture.image = bpy.data.images.load(str(path), check_existing=True)
    background = tree.nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = 0.9
    output = tree.nodes.new("ShaderNodeOutputWorld")
    tree.links.new(coordinates.outputs["Generated"], mapping.inputs["Vector"])
    tree.links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
    tree.links.new(texture.outputs["Color"], background.inputs["Color"])
    tree.links.new(background.outputs[0], output.inputs[0])
    return world


def set_plain_world(color):
    world = bpy.data.worlds.new("SilhouetteBackground")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (*color, 1)
    bpy.context.scene.world = world


def render(path):
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def joint_mapping(armature):
    def path(name):
        bone = armature.data.bones[name]
        segments = [bone.name]
        while bone.parent:
            bone = bone.parent
            segments.append(bone.name)
        return "/".join(reversed(segments))

    body = {
        "neck": path("Skeleton_neck_joint_1"),
        "mid_hip": path("Skeleton_torso_joint_1"),
    }
    for side, refs in (
        ("left", ("Skeleton_arm_joint_L__4_", "Skeleton_arm_joint_L__3_", "Skeleton_arm_joint_L__2_")),
        ("right", ("Skeleton_arm_joint_R", "Skeleton_arm_joint_R__2_", "Skeleton_arm_joint_R__3_")),
    ):
        for semantic, name in zip(("shoulder", "elbow", "wrist"), refs, strict=True):
            body[f"{side}_{semantic}"] = path(name)
        for index, semantic in enumerate(("hip", "knee", "ankle"), 1):
            body[f"{side}_{semantic}"] = path(f"leg_joint_{side[0].upper()}_{index}")
    for name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear"):
        body[name] = None
    return {"body": body, "left_hand": None, "right_hand": None, "face": None}


def main():
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--save-blend", action="store_true")
    args = parser.parse_args(arguments)
    CACHE.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model_path = CACHE / "CesiumMan.glb"
    hdri_path = CACHE / "abandoned_slipway_4k.exr"
    download(MODEL_URL, model_path)
    download(HDRI_URL, hdri_path)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(model_path))
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 48
    scene.render.fps = 24
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1
    scene.frame_set(1)

    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    armature.name = "RunnerRig"
    actor = bpy.data.objects["Cesium_Man"]
    actor.name = "Runner"
    # The imported file also contains an invisible helper mesh: keep only the
    # skinned character and its transform hierarchy in the deliverable.
    for obj in list(bpy.data.objects):
        if obj.type == "MESH" and obj != actor:
            bpy.data.objects.remove(obj, do_unlink=True)
    ancestor = armature
    while ancestor.parent:
        ancestor = ancestor.parent
    ancestor.name = "Performer"
    ancestor.scale *= 1.48
    # Keep the original two-second performance and proportions. Position the
    # lowest animated foot above the ground without changing the animation.
    ancestor.location.z += 0.035
    bpy.context.view_layer.update()

    actor.data.materials.clear()
    materials = [
        material("Graphite fabric", (0.055, 0.075, 0.095), 0.1, 0.55),
        material("Safety orange panels", (0.8, 0.115, 0.018), 0.12, 0.42),
        material("Silver protection", (0.40, 0.48, 0.53), 0.30, 0.38),
    ]
    for item in materials:
        actor.data.materials.append(item)
    group_names = {group.index: group.name for group in actor.vertex_groups}
    orange_groups = {"Skeleton_torso_joint_1", "Skeleton_torso_joint_2", "torso_joint_3"}
    silver_groups = {"Skeleton_neck_joint_1", "Skeleton_neck_joint_2", "leg_joint_L_3", "leg_joint_R_3"}
    for polygon in actor.data.polygons:
        weights = {}
        for index in polygon.vertices:
            for group in actor.data.vertices[index].groups:
                name = group_names[group.group]
                weights[name] = weights.get(name, 0) + group.weight
        dominant = max(weights, key=weights.get) if weights else ""
        polygon.material_index = 1 if dominant in orange_groups else 2 if dominant in silver_groups else 0
        polygon.use_smooth = True
    # Delete orphan source textures/materials to ensure no logo is exported.
    for source_material in list(bpy.data.materials):
        if source_material not in materials:
            bpy.data.materials.remove(source_material)
    for image in list(bpy.data.images):
        if image.users == 0:
            bpy.data.images.remove(image)

    photographic_world = setup_world(hdri_path)
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, 0))
    ground = bpy.context.object
    ground.name = "ShadowGround"
    ground.is_shadow_catcher = True
    ground.data.materials.append(material("Neutral ground", (0.18, 0.18, 0.18), roughness=1))

    cam_a = camera("Camera_A", (2.9, -5.8, 1.45), (0, 0, 0.95), 49)
    cam_b = camera("Camera_B", (-3.9, -5.0, 1.35), (0, 0, 0.94), 48)
    scene.camera = cam_a
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = max(1, args.samples)
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.04
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1
    scene.render.pixel_aspect_y = 1
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "AgX"
    scene.render.film_transparent = False

    # A local rim light increases leg/hand visibility, independent of the source
    # panorama. It is not included in the clean plate or in the USD deliverable.
    light_data = bpy.data.lights.new("Soft rim", "AREA")
    light_data.energy = 250
    light_data.shape = "DISK"
    light_data.size = 3
    light = bpy.data.objects.new("Soft rim", light_data)
    scene.collection.objects.link(light)
    light.location = (2, 1.8, 3.2)
    aim(light, (0, 0, 1))

    scene.frame_set(7)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in (actor, armature, ancestor, cam_a, cam_b):
        obj.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.wm.usd_export(
        filepath=str(OUTPUT / "shot.usdc"),
        selected_objects_only=True,
        export_animation=True,
        export_armatures=True,
        only_deform_bones=False,
        export_materials=True,
        generate_preview_surface=True,
        generate_materialx_network=False,
        export_textures_mode="KEEP",
        convert_world_material=False,
        export_cameras=True,
        export_lights=False,
        root_prim_path="/World",
    )

    info = {
        "asset": "DiffusedFrames industrial runner showcase",
        "usd": "shot.usdc",
        "joint_map": "joint_map.json",
        "frames_per_second": 24,
        "start_frame": 1,
        "end_frame": 48,
        "camera_fit": "fit",
        "pose_format": "COCO_18",
        "coordinate_space": "normalized",
        "skeleton_path": "/World/Performer/RunnerRig/Armature",
        "cases": [],
        "credits": [
            "Cesium Man © 2017 Cesium, CC BY 4.0; original animation retained; converted to USD, scaled, and all materials replaced by DiffusedFrames.",
            "abandoned_slipway HDRI by Philip Modin / Poly Haven, CC0; used for lighting and photographic background.",
        ],
    }
    white = emission("Silhouette white", (1, 1, 1))
    projection_reference = {}
    for label, cam, frame in (("A", cam_a, 7), ("B", cam_b, 19)):
        scene.camera = cam
        scene.frame_set(frame)
        # Kept only in the development cache for independent USD / Blender
        # camera-alignment verification; not required by the workflow.
        projection_reference[label] = {}
        for bone in armature.pose.bones:
            point = world_to_camera_view(scene, cam, armature.matrix_world @ bone.head)
            projection_reference[label][bone.name] = [point.x * 1024, (1 - point.y) * 768]
        scene.world = photographic_world
        scene.view_settings.view_transform = "AgX"
        actor.hide_render = False
        ground.hide_render = False
        scene.view_layers[0].material_override = None
        render(OUTPUT / f"source_{label}.png")

        actor.hide_render = True
        ground.hide_render = True
        render(OUTPUT / f"plate_{label}.png")

        actor.hide_render = False
        set_plain_world((0, 0, 0))
        scene.view_settings.view_transform = "Standard"
        scene.view_layers[0].material_override = white
        render(OUTPUT / f"mask_{label}.png")
        scene.view_layers[0].material_override = None
        info["cases"].append(
            {
                "id": label,
                "camera_path": f"/World/Camera_{label}/Camera_{label}",
                "usd_frame": frame,
                "width": 1024,
                "height": 768,
                "source": f"source_{label}.png",
                "plate": f"plate_{label}.png",
                "mask": f"mask_{label}.png",
            }
        )

    actor.hide_render = False
    ground.hide_render = False
    scene.world = photographic_world
    scene.camera = cam_a
    scene.frame_set(7)
    scene.view_settings.view_transform = "AgX"
    (OUTPUT / "scene_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "joint_map.json").write_text(json.dumps(joint_mapping(armature), indent=2) + "\n", encoding="utf-8")
    (CACHE / "blender_projection.json").write_text(json.dumps(projection_reference, indent=2) + "\n", encoding="utf-8")
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(CACHE / "showcase.blend"))
    print("SHOWCASE_COMPLETE", json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
