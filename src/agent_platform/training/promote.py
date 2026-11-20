"""Registry CLI: show versions/aliases, or point an alias at a version.

python -m agent_platform.training.promote                 # print the registry state
python -m agent_platform.training.promote --version 3     # champion -> v3
python -m agent_platform.training.promote --version 2 --alias candidate
"""

import argparse
import logging
import os
from collections.abc import Sequence

from agent_platform.training.config import DEFAULT_ALIAS, DEFAULT_MODEL_NAME
from agent_platform.training.registry import format_registry, registry_rows, set_alias


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-name", default=os.environ.get("FLY_MODEL_NAME", DEFAULT_MODEL_NAME)
    )
    parser.add_argument("--alias", default=os.environ.get("FLY_MODEL_ALIAS", DEFAULT_ALIAS))
    parser.add_argument("--version", type=int, help="version the alias should point to")
    args = parser.parse_args(argv)
    if args.version is not None:
        set_alias(args.model_name, args.version, args.alias)
    print(format_registry(args.model_name, registry_rows(args.model_name)))  # noqa: T201


if __name__ == "__main__":
    main()
