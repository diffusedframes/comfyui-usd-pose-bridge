import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from usd_pose_bridge.pose_io import load_pose_json, resolve_input_path, save_pose_json


def frame():
    return {"canvas_width": 64, "canvas_height": 64, "people": [
        {"pose_keypoints_2d": [0.0, 0.0, 0.0] * 18}
    ], "coordinate_space": "pixel"}


def test_sequence_and_per_frame_roundtrip_without_overwrite(tmp_path):
    frames = [frame(), frame()]
    first = save_pose_json(frames, tmp_path, "poses/demo", "sequence")
    second = save_pose_json(frames, tmp_path, "poses/demo", "sequence")
    assert first != second
    assert json.loads(Path(first[0]).read_text()) == frames
    loaded = load_pose_json(Path(first[0]))
    assert loaded[0]["pose_format"] == "COCO_18"
    assert loaded[0]["coordinate_space"] == "pixel"
    files = save_pose_json(frames, tmp_path, "poses/demo", "per_frame")
    assert len(files) == 2
    assert all(json.loads(Path(path).read_text()) == frame() for path in files)


@pytest.mark.parametrize("prefix", ["../escape", "x/../../escape", "/absolute", "C:\\escape", "x/./pose", ""])
def test_output_prefix_cannot_escape(tmp_path, prefix):
    with pytest.raises(ValueError):
        save_pose_json([frame()], tmp_path, prefix, "sequence")


def test_invalid_json_creates_no_output(tmp_path):
    data = frame()
    data["people"][0]["pose_keypoints_2d"][0] = float("nan")
    with pytest.raises(ValueError):
        save_pose_json([data], tmp_path / "out", "pose", "sequence")
    assert not (tmp_path / "out").exists()


def test_standard_openpose_without_canvas_metadata(tmp_path):
    path = tmp_path / "pose.json"
    path.write_text(json.dumps({"people": frame()["people"]}), encoding="utf-8")
    loaded = load_pose_json(path, 768, 512, "pixel")
    assert loaded[0]["canvas_width"] == 768
    assert loaded[0]["canvas_height"] == 512
    assert loaded[0]["coordinate_space"] == "pixel"


def test_input_paths_use_comfy_input_and_bundled_examples(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    pack = tmp_path / "pack"
    input_dir.mkdir()
    (pack / "examples").mkdir(parents=True)
    (input_dir / "scene.usda").touch()
    (pack / "examples" / "demo.usda").touch()
    monkeypatch.setitem(__import__("sys").modules, "folder_paths", SimpleNamespace(get_input_directory=lambda: str(input_dir)))
    assert resolve_input_path("scene.usda", pack) == input_dir / "scene.usda"
    assert resolve_input_path("examples/demo.usda", pack) == pack / "examples" / "demo.usda"


def test_reject_mixed_body_formats(tmp_path):
    data = frame()
    data["people"].append({"pose_keypoints_2d": [0, 0, 0] * 25})
    path = tmp_path / "pose.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="same body format"):
        load_pose_json(path)


@pytest.mark.parametrize("field,values", [
    ("hand_left_keypoints_2d", [0, 0, 1]),
    ("face_keypoints_2d", [0, 0, 1] * 69),
    ("pose_keypoints_2d", [float("nan"), 0, 1] * 18),
    ("pose_keypoints_2d", [0, 0, 2] * 18),
])
def test_import_rejects_malformed_landmarks(tmp_path, field, values):
    data = frame()
    data["people"][0][field] = values
    path = tmp_path / "pose.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_pose_json(path)
