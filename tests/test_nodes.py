import importlib.util
import json
import sys
from types import SimpleNamespace

import pytest

from usd_pose_bridge.nodes import (
    NODE_CLASS_MAPPINGS, PACK_ROOT, USDPBInspectStage, USDPBJointMap,
    USDPBLoadPose, USDPBLoadStage, USDPBProjectPose, USDPBRenderPose, USDPBSavePose,
)


def test_comfy_entrypoint_loads_by_arbitrary_package_name():
    spec = importlib.util.spec_from_file_location("usd_pack_entry", PACK_ROOT / "__init__.py",
                                                submodule_search_locations=[str(PACK_ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        assert len(module.NODE_CLASS_MAPPINGS) == 7
        assert module.WEB_DIRECTORY == "./web"
        for cls in module.NODE_CLASS_MAPPINGS.values():
            assert callable(getattr(cls, cls.FUNCTION))
            assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES)
            assert cls.INPUT_TYPES()["required"]
    finally:
        for key in list(sys.modules):
            if key == spec.name or key.startswith(spec.name + "."):
                del sys.modules[key]


@pytest.fixture
def stage():
    return USDPBLoadStage().load("examples/demo.usda")["result"][0]


def test_full_demo_animation_pipeline(stage, tmp_path, monkeypatch):
    import torch

    inspector = USDPBInspectStage().inspect(stage)
    assert "/World/Camera" in inspector["result"][0]
    result = USDPBProjectPose().project(stage, frame_count=3, frame_step=8, width=256, height=256)
    image, frames, report = result["result"]
    assert image.shape == (3, 256, 256, 3)
    assert image.dtype == torch.float32
    assert 0 <= image.min() <= image.max() <= 1
    assert not torch.equal(image[0], image[1])
    assert len(frames) == 3 and len(frames[0]["people"]) == 2
    assert len(frames[0]["people"][0]["pose_keypoints_2d"]) == 54
    assert frames[0]["coordinate_space"] == "normalized"
    assert [frame["usd_frame"] for frame in frames] == [1, 9, 17]
    assert json.loads(report)["frame_count"] == 3
    assert torch.equal(USDPBRenderPose().render(frames)[0], image)
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(get_output_directory=lambda: str(tmp_path)))
    saved = USDPBSavePose().save(frames)["result"][0]
    path = json.loads(saved)["saved_files"][0]
    loaded = USDPBLoadPose().load(path)[0]
    assert loaded == frames
    assert torch.equal(USDPBRenderPose().render(loaded)[0], image)


def test_body25_and_pixel_normalized_render_agree(stage):
    import torch

    node = USDPBProjectPose()
    pixel = node.project(stage, width=128, height=128, coordinate_space="pixel", pose_format="BODY_25")["result"]
    normalized = node.project(stage, width=128, height=128, pose_format="BODY_25")["result"]
    assert len(pixel[1][0]["people"][0]["pose_keypoints_2d"]) == 75
    assert torch.equal(pixel[0], normalized[0])


@pytest.mark.parametrize("kwargs", [{"frame_count": 0}, {"frame_count": 1.5}, {"frame_step": 0},
                                    {"start_frame": float("nan")}, {"coordinate_space": "bad"},
                                    {"width": 4096, "height": 4096, "frame_count": 3}])
def test_projection_rejects_invalid_requests_before_sampling(kwargs):
    with pytest.raises(ValueError):
        USDPBProjectPose().project(None, **kwargs)


def test_empty_joint_map_is_automatic():
    assert USDPBJointMap().create("{}")[0] == {}


def test_example_workflow_inputs_match_registered_nodes():
    paths = list((PACK_ROOT / "examples").glob("api_*.json"))
    assert paths
    for path in paths:
        prompt = json.loads(path.read_text())
        prompt = prompt.get("prompt", prompt)
        for data in prompt.values():
            cls = NODE_CLASS_MAPPINGS.get(data["class_type"])
            if cls is None:
                continue
            schema = cls.INPUT_TYPES()
            required = set(schema["required"])
            available = required | set(schema.get("optional", {}))
            assert required <= data["inputs"].keys(), (path, required - data["inputs"].keys())
            assert data["inputs"].keys() <= available, (path, data["inputs"].keys() - available)
