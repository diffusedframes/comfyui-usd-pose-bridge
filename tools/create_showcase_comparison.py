"""Make a comparison figure from actual ComfyUI showcase outputs.

The manifest is the local execution record, never a source of invented results.
Use --comfy-output to select the same output directory as the running server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


def font(size: int):
    for candidate in ("C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "dist/showcase_results.json")
    parser.add_argument("--comfy-output", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.manifest.read_text(encoding="utf-8"))["cases"]
    scene = json.loads((ROOT / "examples/showcase/scene_info.json").read_text(encoding="utf-8"))
    scene_cases = {item["id"]: item for item in scene["cases"]}
    cell_width, cell_height, gap, margin = 640, 480, 24, 36
    board = Image.new("RGB", (margin * 2 + cell_width * 3 + gap * 2, 1340), "#101722")
    draw = ImageDraw.Draw(board)
    draw.text((margin, 25), "DIFFUSEDFRAMES / USD POSE BRIDGE", font=font(23), fill="#aebbc9")
    draw.text((margin, 65), "USD animation → OpenPose → AI composite", font=font(42), fill="#f3f5f7")
    metrics = []
    for row, case in enumerate(cases):
        identifier = case["case"]
        settings = scene_cases[identifier]
        y = 155 + row * 555
        draw.text((margin, y - 34), f"Camera {identifier} · USD frame {settings['usd_frame']}",
                  font=font(22), fill="#aebbc9")
        paths = {}
        for key in ("final", "pose", "mask"):
            result = case[key]
            assert result["type"] == "output"
            path = (args.comfy_output / result["subfolder"] / result["filename"]).resolve()
            assert path.is_relative_to(args.comfy_output.resolve())
            paths[key] = path
        plate = np.asarray(Image.open(ROOT / "examples/showcase" / settings["plate"]).convert("RGB"))
        final = np.asarray(Image.open(paths["final"]).convert("RGB"))
        mask = np.asarray(Image.open(paths["mask"]).convert("RGB")).max(axis=2)
        outside = mask == 0
        error = np.abs(final.astype(int) - plate.astype(int)).max(axis=2)
        # SaveImage quantization may differ by one code value at a float boundary.
        assert error[outside].max() <= 1, "The photographed background changed outside the edit region."
        metrics.append({"case": identifier, "unchanged_region_pixels": int(outside.sum()),
                        "outside_mask_max_channel_error": int(error[outside].max())})
        for key in ("final", "pose"):
            with Image.open(paths[key]) as result_image:
                result_image.convert("RGB").save(ROOT / f"examples/showcase/{key}_{identifier}.png", optimize=True)
        for column, (label, path) in enumerate((
            ("3D SOURCE + PHOTOGRAPH", ROOT / "examples/showcase" / settings["source"]),
            ("POSE PROJECTED FROM USD", paths["pose"]),
            ("SDXL + OPENPOSE CONTROLNET", paths["final"]),
        )):
            x = margin + column * (cell_width + gap)
            with Image.open(path) as panel:
                board.paste(panel.convert("RGB").resize((cell_width, cell_height), Image.Resampling.LANCZOS), (x, y))
            draw.text((x, y + cell_height + 10), label, font=font(19), fill="#dbe1e8")
    draw.text((margin, 1280), "Source: Cesium (CC BY 4.0) · Photograph: Philip Modin / Poly Haven (CC0)",
              font=font(20), fill="#aebbc9")
    draw.text((margin, 1309), "Actual workflow outputs · Background preserved outside the feathered edit region · Still-image studies",
              font=font(19), fill="#aebbc9")
    target = ROOT / "examples/showcase_comparison.jpg"
    board.save(target, quality=92, subsampling=0, optimize=True)
    (ROOT / "dist/showcase_background_check.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(target)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
