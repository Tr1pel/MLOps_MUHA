import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("image_metadata", Path("scripts/image_metadata.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
image_metadata = module.image_metadata
SHA = "a" * 40


def test_main_image_uses_lowercase_name_and_full_sha():
    metadata = image_metadata("Tr1pel/MLOps_MUHA", SHA, "branch", "main", "0.1.0")
    assert metadata == {
        "image": "ghcr.io/tr1pel/mlops_muha",
        "tags": f"ghcr.io/tr1pel/mlops_muha:sha-{SHA}",
        "version": "0.1.0",
    }


def test_release_has_package_version_and_sha_tags():
    metadata = image_metadata("Tr1pel/MLOps_MUHA", SHA, "tag", "v0.1.0", "0.1.0")
    assert metadata["tags"].split(",") == [
        f"ghcr.io/tr1pel/mlops_muha:sha-{SHA}",
        "ghcr.io/tr1pel/mlops_muha:0.1.0",
    ]


@pytest.mark.parametrize(
    ("repository", "sha", "ref_type", "ref_name", "error"),
    [
        ("invalid", SHA, "branch", "main", "repository"),
        ("owner/repo", "abc123", "branch", "main", "full commit SHA"),
        ("owner/repo", SHA, "tag", "v0.2.0", "does not match"),
        ("owner/repo", SHA, "tag", "v0.1.0-rc1", "vMAJOR.MINOR.PATCH"),
        ("owner/repo", SHA, "branch", "feature", "Only main"),
        ("owner/repo", SHA, "pull", "42", "Only main"),
    ],
)
def test_invalid_release_metadata_is_rejected(repository, sha, ref_type, ref_name, error):
    with pytest.raises(ValueError, match=error):
        image_metadata(repository, sha, ref_type, ref_name, "0.1.0")
