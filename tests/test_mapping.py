import json

import pytest

from usd_pose_bridge.mapping import BODY_25, COCO_18, build_person, parse_mapping


def body_point(person, name, names=COCO_18):
    i = names.index(name) * 3
    return person["pose_keypoints_2d"][i:i + 3]


def test_mixamo_mapping_excludes_clavicle_and_does_not_invent_face():
    names = ["Root/mixamorig:LeftShoulder", "Root/mixamorig:LeftArm", "Root/mixamorig:RightArm",
             "Root/mixamorig:LeftForeArm", "Root/mixamorig:LeftHand", "Root/mixamorig:LeftUpLeg", "Root/Head"]
    points = [(0.01, 0.01, 1), (.2, .3, .9), (.8, .5, .7), (.1, .4, 1), (0, .5, 1), (.3, .8, 1), (.5, .1, 1)]
    person, report = build_person(names, points)
    assert body_point(person, "left_shoulder") == [.2, .3, .9]
    assert body_point(person, "left_elbow") == [.1, .4, 1]
    assert body_point(person, "left_wrist") == [0, .5, 1]
    assert body_point(person, "neck") == [.5, .4, .7]
    assert body_point(person, "nose") == [0, 0, 0]
    assert person["face_keypoints_2d"] is None
    assert person["hand_left_keypoints_2d"] is None
    assert report["derived"]["neck"]["sources"] == ["left_shoulder", "right_shoulder"]
    assert "Root/mixamorig:LeftShoulder" in report["unmapped_joints"]


@pytest.mark.parametrize("names", [
    ["upper_arm.L", "forearm.L", "hand.L", "thigh.L", "shin.L", "foot.L"],
    ["upperarm_l", "lowerarm_l", "hand_l", "thigh_l", "calf_l", "foot_l"],
    ["left_shoulder", "left_elbow", "left_wrist", "left_hip", "left_knee", "left_ankle"],
])
def test_rig_aliases(names):
    semantics = ["left_shoulder", "left_elbow", "left_wrist", "left_hip", "left_knee", "left_ankle"]
    points = [(float(i), float(i + 1), 1) for i in range(len(names))]
    person, report = build_person(names, points)
    for name, semantic, point in zip(names, semantics, points, strict=True):
        assert report["mapped"][semantic] == name
        assert body_point(person, semantic) == list(point)


def test_body25_midhip_and_toes_and_standard_order():
    names = ["left_hip", "right_hip", "left_big_toe", "right_heel", "Head"]
    person, report = build_person(names, [(.2, .8, .9), (.8, .8, .7), (.2, 1, 1), (.8, 1, 1), (.5, .1, 1)], "BODY_25")
    assert len(person["pose_keypoints_2d"]) == 75
    assert body_point(person, "mid_hip", BODY_25) == [.5, .8, .7]
    assert body_point(person, "right_hip", BODY_25) == [.8, .8, .7]
    assert body_point(person, "left_big_toe", BODY_25) == [.2, 1, 1]
    assert "mid_hip" in report["derived"]


def test_null_body_override_disables_autodetection_and_derivation():
    names = ["Neck", "LeftArm", "RightArm"]
    person, report = build_person(names, [(1, 1, 1)] * 3, overrides={"body": {"neck": None, "left_shoulder": None}})
    assert body_point(person, "neck") == [0, 0, 0]
    assert body_point(person, "left_shoulder") == [0, 0, 0]
    assert "neck" not in report["derived"]


def test_actual_neck_preserved_even_if_hidden():
    person, report = build_person(["Neck", "LeftArm", "RightArm"], [(4, 4, 0), (2, 3, 1), (8, 5, 1)])
    assert body_point(person, "neck") == [4, 4, 0]
    assert "neck" not in report["derived"]


def test_ambiguous_auto_names_are_reported_and_explicit_full_path_resolves():
    names = ["Alice/LeftArm", "Bob/LeftArm", "RightArm"]
    person, report = build_person(names, [(1, 1, 1), (2, 2, 1), (3, 3, 1)])
    assert "left_shoulder" in report["ambiguous"]
    assert body_point(person, "left_shoulder") == [0, 0, 0]
    with pytest.raises(ValueError, match="Ambiguous.*full path"):
        build_person(names, [(1, 1, 1)] * 3, overrides={"body": {"left_shoulder": "LeftArm"}})
    person, _ = build_person(names, [(1, 1, 1), (2, 2, 1), (3, 3, 1)],
                             overrides={"body": {"left_shoulder": "Bob/LeftArm"}})
    assert body_point(person, "left_shoulder") == [2, 2, 1]


def test_ambiguous_neck_does_not_trigger_fallback():
    person, report = build_person(["a/Neck", "b/Neck", "LeftArm", "RightArm"], [(1, 1, 1)] * 4)
    assert "neck" in report["ambiguous"]
    assert body_point(person, "neck") == [0, 0, 0]
    assert "neck" not in report["derived"]


def test_unknown_override_is_error_even_if_landmark_not_in_selected_format():
    with pytest.raises(ValueError, match="Unknown joint reference 'Typo'"):
        build_person(["Neck"], [(1, 1, 1)], overrides={"body": {"left_heel": "Typo"}})


def test_partial_fingers_do_not_invent_fingertip():
    names = ["LeftHand", "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "left_index4"]
    person, report = build_person(names, [(i, i, 1) for i in range(5)])
    hand = person["hand_left_keypoints_2d"]
    assert len(hand) == 63
    assert hand[:3] == [0, 0, 1]
    assert hand[9:12] == [3, 3, 1]
    assert hand[12:15] == [0, 0, 0]
    assert hand[24:27] == [4, 4, 1]
    assert "left_hand[4]" in report["missing"]
    assert person["hand_right_keypoints_2d"] is None


@pytest.mark.parametrize("name", ["thumb.04.L", "thumb.03.End.L", "LeftHandThumb3_End", "thumb_04_l"])
def test_finger_tip_explicit_end_aliases(name):
    person, report = build_person([name], [(0.5, 0.7, 1)])
    assert person["hand_left_keypoints_2d"][12:15] == [.5, .7, 1]


def test_explicit_hands_and_faces():
    names = ["a/point", "Face_00"]
    refs = [None] * 21
    refs[0] = "point"
    face = ["Face_00"] + [None] * 69
    person, report = build_person(names, [(7, 8, .5), (9, 10, .8)], overrides={"left_hand": refs, "face": face})
    assert len(person["hand_left_keypoints_2d"]) == 63
    assert person["hand_left_keypoints_2d"][:3] == [7, 8, .5]
    assert len(person["face_keypoints_2d"]) == 210
    assert person["face_keypoints_2d"][:3] == [9, 10, .8]
    assert "face[69]" in report["missing"]
    person, _ = build_person(["LeftHandThumb1"], [(1, 1, 1)], overrides={"left_hand": None})
    assert person["hand_left_keypoints_2d"] is None


@pytest.mark.parametrize("bad, message", [
    ('{"bodi": {}}', "Unknown mapping key"), ('{"body":{"left_sholder":"a"}}', "Unknown body landmark"),
    ('{"body":[]}', "must be an object"), ('{"left_hand":[]}', "list of 21"),
    ('{"face":[null]}', "68 or 70"), ('{"body":{"neck":4}}', "nonempty joint"),
    ('[]', "JSON object"), ('{"body":', "line 1, column"),
])
def test_strict_mapping_validation(bad, message):
    with pytest.raises(ValueError, match=message):
        parse_mapping(bad)


def test_mapping_round_trip_and_empty():
    data = {"body": {"neck": "a/Neck", "nose": None}, "right_hand": None, "face": [None] * 68}
    assert parse_mapping(json.dumps(data)) == data
    assert parse_mapping(" \n ") == {}


@pytest.mark.parametrize("point", [(1, 2), (float("nan"), 1, 1), (0, 0, -1), (0, 0, 1.1), (0, float("inf"), 0)])
def test_invalid_projected_points(point):
    with pytest.raises(ValueError, match="Projected joint"):
        build_person(["Neck"], [point])


def test_length_mismatch_and_unknown_pose_format():
    with pytest.raises(ValueError, match="same length"):
        build_person(["Neck"], [])
    with pytest.raises(ValueError, match="COCO_18 or BODY_25"):
        build_person([], [], "COCO17")
