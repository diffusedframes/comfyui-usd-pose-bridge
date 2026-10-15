from pathlib import Path

import pytest

from tools.check_release import check_release


@pytest.fixture
def release_source(tmp_path):
    (tmp_path / "usd_pose_bridge").mkdir()
    (tmp_path / "usd_pose_bridge" / "__init__.py").write_text('__version__ = "0.2.0"\n')
    (tmp_path / "requirements.txt").write_text("# Runtime only\nusd-core>=25.5,<27\n")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "example"\nversion = "0.2.0"\ndependencies = ["usd-core>=25.5,<27"]\n'
        '[project.urls]\nRepository = "https://github.com/diffusedframes/comfyui-usd-pose-bridge"\n'
        '[tool.comfy]\nDisplayName = "Example"\n'
    )
    return tmp_path


def test_current_source_has_consistent_release_metadata():
    assert check_release(Path(__file__).resolve().parents[1])


def test_version_can_be_released_without_registry_identity(release_source):
    assert check_release(release_source, tag="v0.2.0") == "0.2.0"


def test_wrong_tag_cannot_release_another_version(release_source):
    with pytest.raises(ValueError, match="does not match"):
        check_release(release_source, tag="v0.1.0")


def test_package_version_drift_blocks_release(release_source):
    (release_source / "usd_pose_bridge" / "__init__.py").write_text('__version__ = "0.1.0"\n')
    with pytest.raises(ValueError, match="__version__"):
        check_release(release_source)


def test_dependency_drift_blocks_release(release_source):
    (release_source / "requirements.txt").write_text("usd-core>=26\n")
    with pytest.raises(ValueError, match="runtime dependencies"):
        check_release(release_source)


def test_registry_requires_real_publisher_configuration(release_source):
    with pytest.raises(ValueError, match="PublisherId"):
        check_release(release_source, registry=True)
    with (release_source / "pyproject.toml").open("a") as stream:
        stream.write('PublisherId = "diffusedframes"\n')
    assert check_release(release_source, tag="v0.2.0", registry=True) == "0.2.0"
