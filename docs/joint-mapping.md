# DiffusedFrames USD Pose Bridge: joint mapping

Automatic mapping matches semantic joint leaves while removing namespaces and common punctuation. Left/right refer to the character's anatomy, not screen position. A camera looking at a character from the front naturally places their left side on the right of the image.

Important aliases include Mixamo `LeftArm` for shoulder, `LeftForeArm` for elbow, and `LeftHand` for wrist; Unreal `upperarm_l`, `lowerarm_l`, `hand_l`, `thigh_l`, `calf_l`, `foot_l`; and Blender names such as `upper_arm.L`. Mixamo `LeftShoulder` is a clavicle and is not automatically used as the shoulder landmark.

The mapper leaves ambiguous automatic matches unresolved. Reports distinguish mapped, derived, missing, and ambiguous joints. Explicit unknown or ambiguous references raise an error. A full USD joint path is safest; unique leaf names also work. Joint paths are skeleton-relative tokens listed in the scene report, rather than absolute USD prim paths.

Connect **USD Joint Map** to the optional `joint_map` input of **USD to OpenPose**. Empty `{}` keeps automatic mapping. Override only the fields you need:

```json
{
  "body": {
    "left_shoulder": "Hips/Spine/Chest/LeftArm",
    "left_elbow": "Hips/Spine/Chest/LeftArm/LeftForeArm",
    "left_wrist": "Hips/Spine/Chest/LeftArm/LeftForeArm/LeftHand",
    "nose": null
  }
}
```

`null` explicitly disables a landmark. The body keys, in BODY 25 order, are:

```text
nose, neck, right_shoulder, right_elbow, right_wrist,
left_shoulder, left_elbow, left_wrist, mid_hip,
right_hip, right_knee, right_ankle, left_hip, left_knee, left_ankle,
right_eye, left_eye, right_ear, left_ear,
left_big_toe, left_small_toe, left_heel,
right_big_toe, right_small_toe, right_heel
```

COCO 18 omits `mid_hip` and the six foot landmarks, retaining the standard COCO/OpenPose ordering. When a neck or pelvis joint is absent, the mapper can derive the midpoint of the two mapped shoulders or hips. It does not substitute head center for nose or fabricate facial landmarks. Midpoints use available projected coordinates, so they are a 2D fallback; map the actual joint for exact perspective placement.

`left_hand` and `right_hand` optionally contain 21 references: wrist, then four landmarks for thumb, index, middle, ring, and pinky. Missing fingertips stay missing. If omitted, common finger names are mapped automatically; `null` disables the whole hand. Standard three-bone finger chains may lack a fourth fingertip marker, so export end joints or supply explicit tip landmarks for full hands.

`face` optionally contains 68 or 70 references in the downstream OpenPose facial-landmark order. Face arrays are explicit: rig facial bones rarely correspond directly to those surface landmarks. The demo includes a dedicated face mapping to demonstrate this path.

One override map is applied to every selected character. For different rig hierarchies, project each rig separately using its own map. An override that does not exist on a selected rig is an error rather than a silent mismatch.
