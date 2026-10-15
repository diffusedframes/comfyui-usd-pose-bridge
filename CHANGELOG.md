# Changelog

## 0.1.0 — development preview

Initial DiffusedFrames USD Pose Bridge implementation. The still and 24-frame
animation demos passed through a running ComfyUI Desktop 0.38.2 execution API on
Windows. Broader rig and interactive interface testing is ongoing.

- Seven ComfyUI nodes for USD loading, scene inspection, joint mapping, pose
  projection, rendering, and OpenPose JSON import/export.
- Animated USD skeleton evaluation with authored camera projection, multiple
  characters, COCO 18/BODY 25 layouts, hands, and explicit face mapping.
- A two-character animated demo and ready-to-use still/animation workflows.
- A cinematic, real-asset showcase with two USD cameras, photographic plates,
  explicit rig mapping, and reproducible SDXL OpenPose inpainting workflows.
- Automated Windows/Linux test configuration, installable ZIP packaging, and
  workflows for reviewing GitHub releases and manually publishing to Registry.
