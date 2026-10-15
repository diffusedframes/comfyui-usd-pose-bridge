# Releasing DiffusedFrames USD Pose Bridge

This pack is an early development preview. Automated tests exercise real USD,
camera projection, pose files and image output. The still and 24-frame animation
demos also passed through a running ComfyUI Desktop 0.38.2 execution API on Windows.
This verifies backend integration; broader rig and interactive interface testing
is ongoing.

## Routine development

The **Tests** workflow runs on pull requests and branch pushes. It checks source
style, package metadata and the full test suite on Windows and Ubuntu with Python
3.10 and 3.12, using CPU Torch. GitHub runs in its own environment; these checks do
not install dependencies into a user's ComfyUI Desktop environment.

Keep `requirements.txt` and `[project].dependencies` in `pyproject.toml` together.
Torch belongs to ComfyUI and is only a development/test extra here.

## Prepare a GitHub release

1. Set the same semantic version in `pyproject.toml` and
   `usd_pose_bridge/__init__.py`. Update `CHANGELOG.md` and the examples if needed.
2. Run `python -m ruff check .`, `python -m pytest -q` and
   `python tools/check_release.py` in the development environment. On Python 3.10,
   install the `test` extra first so the metadata check has `tomli`.
3. Merge the reviewed changes and wait for all four GitHub test jobs to pass.
4. Create and push a version tag matching the metadata exactly, such as `v0.1.0`.
   The **Prepare GitHub release** workflow repeats the test matrix, validates the
   tag, builds the installable ZIP and creates a **draft** GitHub release.
5. Review the release notes, download the ZIP and test installation in ComfyUI.
   Publish the draft when the build is ready for users. Versions before 1.0 and
   versions with a prerelease suffix are marked as prereleases automatically.

The ZIP contains one `comfyui-usd-pose-bridge` folder, ready to extract under
`custom_nodes`. Its file name includes the package version from metadata.
Building a ZIP locally with `python tools/build_release.py` does not publish it.

## Enable ComfyUI Manager installation

GitHub publication and Registry publication are separate steps. The `diffusedframes`
publisher is registered on [Comfy Registry](https://registry.comfy.org/), and
`[tool.comfy].PublisherId` is configured in this repository. Follow the official
[publishing guide](https://docs.comfy.org/registry/publishing) to create a publishing
API key for that publisher.

1. Confirm the intended package ID in `[project].name` before the first publication;
   that identifier is also permanent after creation.
2. Add the API key directly under the GitHub repository's **Settings → Secrets and
   variables → Actions → New repository secret**, named `REGISTRY_ACCESS_TOKEN`.
   Do not put the key in source files or chat.
3. Commit the metadata, make a new matching version tag, and complete the release
   checks above. The tagged source must contain the publisher ID.
4. Open **Actions → Publish to Comfy Registry → Run workflow** and enter that
   existing tag. The workflow validates the metadata and key configuration,
   tests the tagged commit on all four environments, and publishes that same commit.

Registry publication is manual. Pushing a branch or changing `pyproject.toml`
does not publish a package. Until the repository secret is configured, the
publishing workflow stops with a clear setup message.

The publishing action is pinned to a reviewed commit from
[Comfy-Org/publish-node-action](https://github.com/Comfy-Org/publish-node-action).
Dependabot opens monthly update proposals for actions and Python dependencies;
those proposals still go through the normal tests and review.
