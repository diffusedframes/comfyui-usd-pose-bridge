"""Build a portable custom_nodes ZIP without development environments or caches."""

from __future__ import annotations

import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

PACK_ROOT = Path(__file__).resolve().parent.parent
INSTALL_FOLDER = "comfyui-usd-pose-bridge"
PACKAGE_NAME = "diffusedframes-usd-pose-bridge"


def build_release():
    metadata = (PACK_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version_match = re.search(r'^version\s*=\s*"([^"\r\n]+)"\s*$', metadata, re.MULTILINE)
    if version_match is None:
        raise ValueError("Cannot read the release version from pyproject.toml")
    version = version_match.group(1)
    files = [PACK_ROOT / name for name in (
        "__init__.py", "README.md", "LICENSE", "pyproject.toml", "requirements.txt", ".comfyignore",
    )]
    for directory in ("usd_pose_bridge", "web", "docs", "examples"):
        files.extend(path for path in (PACK_ROOT / directory).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
    destination = PACK_ROOT / "dist" / f"{PACKAGE_NAME}-{version}.zip"
    destination.parent.mkdir(exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, Path(INSTALL_FOLDER) / path.relative_to(PACK_ROOT))
    print(destination)
    return destination


if __name__ == "__main__":
    build_release()
