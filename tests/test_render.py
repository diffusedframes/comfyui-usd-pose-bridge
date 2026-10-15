import numpy as np
import pytest

from usd_pose_bridge.mapping import BODY_25, COCO_18
from usd_pose_bridge.render import render_frames


def person_with(points, names=COCO_18):
    body = [0.0] * (len(names) * 3)
    for name, point in points.items():
        index = names.index(name) * 3
        body[index:index + 3] = point
    return {"pose_keypoints_2d": body}


def frame(person=None, **kwargs):
    result = {"people": [] if person is None else [person], "canvas_width": 101, "canvas_height": 81,
              "coordinate_space": "normalized", "pose_format": "COCO_18"}
    result.update(kwargs)
    return result


def test_empty_people_and_arbitrary_batch():
    result = render_frames([frame(), frame(), frame()], 2)
    assert result.shape == (3, 81, 101, 3)
    assert result.dtype == np.float32
    assert result.min() == result.max() == 0


def test_missing_landmarks_do_not_create_origin_artifacts():
    result = render_frames([frame(person_with({"left_wrist": (.5, .5, 1)}))], 2)[0]
    assert result[0:10, 0:10].max() == 0
    assert result[40, 50].sum() > 0


def test_visible_x_zero_is_rendered_and_confidence_zero_is_not():
    result = render_frames([frame(person_with({"left_wrist": (0, .5, 1), "right_wrist": (.5, .5, 0)}))], 2)[0]
    assert result[40, 0].sum() > 0
    assert result[40, 50].sum() == 0


def test_normalized_pixel_and_inferred_coordinates_agree():
    normalized = frame(person_with({"left_shoulder": (20 / 101, 24 / 81, 1), "left_elbow": (40 / 101, 40 / 81, 1)}))
    pixels = frame(person_with({"left_shoulder": (20, 24, 1), "left_elbow": (40, 40, 1)}), coordinate_space="pixel")
    inferred = dict(pixels)
    del inferred["coordinate_space"]
    result = render_frames([normalized, pixels, inferred], 2)
    np.testing.assert_array_equal(result[0], result[1])
    np.testing.assert_array_equal(result[1], result[2])


def test_coordinate_inference_ignores_missing_sentinels():
    person = person_with({"left_wrist": (.5, .5, 1), "right_wrist": (999, 999, 0)})
    inferred = frame(person)
    del inferred["coordinate_space"]
    np.testing.assert_array_equal(render_frames([inferred]), render_frames([frame(person)]))


def test_body25_colors_use_semantics_for_legs_and_face():
    points = {"left_knee": (.3, .5, 1), "left_ankle": (.3, .8, 1), "right_eye": (.7, .2, 1)}
    coco = render_frames([frame(person_with(points))], 2)
    body25 = render_frames([frame(person_with(points, BODY_25), pose_format="BODY_25")], 2)
    np.testing.assert_array_equal(coco, body25)


def test_body25_without_metadata_is_inferred():
    explicit = frame(person_with({"mid_hip": (.5, .5, 1)}, BODY_25), pose_format="BODY_25")
    inferred = dict(explicit)
    del inferred["pose_format"]
    np.testing.assert_array_equal(render_frames([inferred]), render_frames([explicit]))


def test_hands_face_and_toggles():
    person = person_with({})
    hand = [0.] * 63
    hand[0:3] = [.4, .4, 1]
    hand[3:6] = [.6, .4, 1]
    face = [0.] * 204
    face[0:3] = [.5, .7, 1]
    person.update(hand_left_keypoints_2d=hand, face_keypoints_2d=face)
    all_parts = render_frames([frame(person)], 2)[0]
    assert all_parts[32, 50].sum() > 0
    assert all_parts[56, 50].sum() == 3
    disabled = render_frames([frame(person)], 2, False, False)
    assert disabled.max() == 0
    assert all_parts.min() >= 0 and all_parts.max() <= 1


def test_multiple_people_and_no_mutation():
    left = person_with({"left_wrist": (.2, .5, 1)})
    right = person_with({"right_wrist": (.8, .5, 1)})
    original = list(left["pose_keypoints_2d"])
    result = render_frames([frame(people=[left, right])], 2)[0]
    assert result[40, 20].sum() > 0
    assert result[40, 80].sum() > 0
    assert left["pose_keypoints_2d"] == original


def test_normalized_last_pixel_center():
    result = render_frames([frame(person_with({"left_wrist": (100 / 101, 80 / 81, 1)}))], 1)[0]
    assert result[-1, -1].sum() > 0


@pytest.mark.parametrize("key,value", [("canvas_width", 0), ("canvas_height", -2), ("canvas_width", float("nan")),
                                        ("canvas_width", float("inf")), ("canvas_height", 5.5), ("canvas_height", True)])
def test_invalid_dimensions(key, value):
    with pytest.raises(ValueError, match="finite positive integer"):
        render_frames([frame(**{key: value})])


@pytest.mark.parametrize("points,message", [([0, 0], "multiple of 3"), ([[0, 0, 1]], "flat"),
                                          ([0, 0, 1], "18 landmarks"), ([0, 0, float("nan")] * 18, "non-finite"),
                                          ([0, 0, 1.1] * 18, "confidence")])
def test_invalid_keypoints(points, message):
    with pytest.raises(ValueError, match=message):
        render_frames([frame({"pose_keypoints_2d": points})])


def test_dimension_mismatch_and_empty_batch():
    with pytest.raises(ValueError, match="same canvas"):
        render_frames([frame(), frame(canvas_width=99)])
    with pytest.raises(ValueError, match="at least one"):
        render_frames([])


def test_unknown_format_space_and_invalid_stick_width():
    with pytest.raises(ValueError, match="pose_format"):
        render_frames([frame(pose_format="COCO17")])
    with pytest.raises(ValueError, match="coordinate_space"):
        render_frames([frame(coordinate_space="guess")])
    with pytest.raises(ValueError, match="stick_width"):
        render_frames([frame()], 0)
