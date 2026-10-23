from importlib.metadata import version


def application_version() -> str:
    """Read the installed distribution; pyproject.toml is the version source."""
    return version("agent-platform")
