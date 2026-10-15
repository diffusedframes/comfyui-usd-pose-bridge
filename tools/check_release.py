"""Check package metadata before building a release or publishing to Registry."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 development dependency
    import tomli as tomllib

PACK_ROOT = Path(__file__).resolve().parent.parent
_PRERELEASE_ID = r"(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
SEMVER = re.compile(
    r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    rf"(?:-{_PRERELEASE_ID}(?:\.{_PRERELEASE_ID})*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)


def check_release(root: Path = PACK_ROOT, *, tag: str | None = None, registry: bool = False) -> str:
    """Return the version, raising ValueError when release inputs disagree."""
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    version = project["version"]
    if not isinstance(version, str) or not SEMVER.fullmatch(version):
        raise ValueError("project.version must be a semantic version, for example 0.1.0")
    if tag is not None and tag != f"v{version}":
        raise ValueError(f"Tag {tag!r} does not match project.version {version!r}; expected v{version}")

    source = ast.parse((root / "usd_pose_bridge" / "__init__.py").read_text(encoding="utf-8"))
    package_version = next(
        (ast.literal_eval(node.value) for node in source.body
         if isinstance(node, ast.Assign)
         and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)),
        None,
    )
    if package_version != version:
        raise ValueError("usd_pose_bridge.__version__ must match project.version")

    requirements = {
        line.split("#", 1)[0].strip()
        for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    }
    if requirements != set(project.get("dependencies", [])):
        raise ValueError("requirements.txt and project.dependencies must list the same runtime dependencies")

    if registry:
        comfy = metadata.get("tool", {}).get("comfy", {})
        publisher = comfy.get("PublisherId", "")
        if not isinstance(publisher, str) or not publisher.strip():
            raise ValueError("Set tool.comfy.PublisherId to your actual Comfy Registry publisher ID first")
        if not comfy.get("DisplayName"):
            raise ValueError("Set tool.comfy.DisplayName before publishing")
        if not project.get("urls", {}).get("Repository"):
            raise ValueError("Set project.urls.Repository before publishing")
    return version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Require this release tag to match the package version")
    parser.add_argument("--registry", action="store_true", help="Also require Comfy Registry metadata")
    args = parser.parse_args()
    try:
        version = check_release(tag=args.tag, registry=args.registry)
    except (ValueError, KeyError) as error:
        parser.exit(1, f"Release check failed: {error}\n")
    print(f"Release metadata verified for {version}.")


if __name__ == "__main__":
    main()
