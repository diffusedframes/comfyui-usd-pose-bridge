# DiffusedFrames USD Pose Bridge: technical notes

The pack uses the supported ComfyUI V1 custom-node interface. Nodes return standard image tensors with shape `[batch, height, width, 3]`, RGB float32 in `[0, 1]`. Projection and rasterization run on the CPU. OpenUSD and Torch are imported on execution, so an incompatible USD installation does not prevent node discovery.

## USD evaluation

The reader evaluates bound `UsdSkel.Animation` with `UsdSkel.Cache` and concatenates the skeleton's composed world transform. Animation source inheritance and joint-order remapping are handled by USD. Rest transforms provide the USD-defined fallback. Character and camera transforms are evaluated at every requested time.

The authored USD camera supplies the world/view/projection relationship. Stage units and Y/Z up-axis metadata describe the scene; independently rotating the joints would break camera matching. Frustum-window conforming implements fit/crop for a square-pixel output canvas. Camera aperture offsets and orthographic projection are preserved.

Projected positions use top-left image coordinates and pixel centers from 0 to width−1 / height−1. Normalized output divides those positions by width and height. A clipped, offscreen, invalid, or missing landmark is `[0, 0, 0]`. A visible projected joint has confidence 1; confidence is a visibility flag, not a detector score. No mesh occlusion test is performed.

Skeleton discovery is stable by path, so person ordering remains stable between animation frames. Additional metadata such as `usd_skeleton_path`, `usd_frame`, and `usd_camera_path` identifies the source. Each OpenPose body/hand/face array is flat `[x, y, confidence, ...]`. The `POSE_KEYPOINT` value is a list of frame objects; it is not ComfyUI's outer execution-list mechanism.

## Files and caching

The loader always re-executes when its workflow runs because USD references, sublayers, payloads, and clips can change independently of the root file. It opens outside any installed stage cache and refreshes non-dirty file-backed used layers. USD shares these layer objects process-wide: a refresh can update another stage using the same layers in that Python process. The reader refuses to discard dirty layers. Concurrent editing of the same layers in another extension is outside this release's tested scope. No source file is written.

Exports are confined to ComfyUI's output directory, use exclusive creation, and increment numbered names instead of overwriting. A sequence export is one JSON list of frames, matching the Auxiliary save convention. Per-frame export is one OpenPose frame object per file, ordered by batch index; its `usd_frame` field preserves the original USD sample time. Frame data and coordinate metadata are preserved.

## Verification scope

Tests use real OpenUSD stages and Torch tensors, including animation, camera transforms, clipping, mapping, rendering, and JSON round trips. The included demo is rendered through the same nodes. An interactive ComfyUI session is a separate installation check; backend tests do not establish compatibility with every frontend version or third-party pose editor. COCO 18 is the intended default for Auxiliary tools that assume 18 body landmarks; check each downstream tool before passing BODY 25.
