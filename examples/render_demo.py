"""Render this pack's demo through the actual ComfyUI node implementations.

Run with the Python environment used by ComfyUI, or the development environment:
    python examples/render_demo.py
No ComfyUI server is needed. Pillow, Torch, NumPy, and usd-core must be installed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

PACK_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACK_ROOT))


def render_demo():
    from usd_pose_bridge.nodes import USDPBJointMap, USDPBLoadStage, USDPBProjectPose

    request = json.loads((PACK_ROOT / "examples/api_animation.json").read_text(encoding="utf-8"))["prompt"]
    stage = USDPBLoadStage().load(**request["1"]["inputs"])["result"][0]
    mapping = USDPBJointMap().create(**request["6"]["inputs"])[0]
    options = {key: value for key, value in request["2"]["inputs"].items() if key not in {"stage", "joint_map"}}
    images, frames, report = USDPBProjectPose().project(stage=stage, joint_map=mapping, **options)["result"]
    pil_frames = [Image.fromarray((frame.numpy() * 255).round().astype("uint8")) for frame in images]
    still = PACK_ROOT / "examples/demo_preview.png"
    animation = PACK_ROOT / "examples/demo_animation.gif"
    pil_frames[0].save(still)
    pil_frames[0].save(animation, save_all=True, append_images=pil_frames[1:],
                       duration=1000 / stage.GetTimeCodesPerSecond(), loop=0, disposal=2)
    print(f"Rendered {len(frames)} frames: {still} and {animation}")
    return report


if __name__ == "__main__":
    render_demo()
