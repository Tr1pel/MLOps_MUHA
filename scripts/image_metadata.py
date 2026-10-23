"""Validate release versions and derive traceable image tags using only the stdlib."""

import os
import re
import tomllib
from pathlib import Path


def image_metadata(
    repository: str, sha: str, ref_type: str, ref_name: str, package_version: str
) -> dict[str, str]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Invalid GitHub repository")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("A full commit SHA is required")
    image = f"ghcr.io/{repository.lower()}"
    tags = [f"{image}:sha-{sha}"]
    if ref_type == "tag":
        if not re.fullmatch(r"v\d+\.\d+\.\d+", ref_name):
            raise ValueError("Release tags must use vMAJOR.MINOR.PATCH")
        if ref_name != f"v{package_version}":
            raise ValueError("Release tag does not match [project].version in pyproject.toml")
        tags.append(f"{image}:{package_version}")
    elif ref_type != "branch" or ref_name != "main":
        raise ValueError("Only main and release tags may publish images")
    return {"image": image, "tags": ",".join(tags), "version": package_version}


def main() -> None:
    with Path("pyproject.toml").open("rb") as file:
        package_version = tomllib.load(file)["project"]["version"]
    metadata = image_metadata(
        os.environ["GITHUB_REPOSITORY"],
        os.environ["GITHUB_SHA"],
        os.environ["GITHUB_REF_TYPE"],
        os.environ["GITHUB_REF_NAME"],
        package_version,
    )
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        for key, value in metadata.items():
            output.write(f"{key}={value}\n")


if __name__ == "__main__":
    main()
