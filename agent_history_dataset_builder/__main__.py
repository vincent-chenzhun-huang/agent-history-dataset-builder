"""Run the entry points directly from a checkout."""

from __future__ import annotations

import argparse

from .cli import scan_main, verify_main
from .hook import main as review_main


COMMANDS = {
    "scan-history": scan_main,
    "verify-attempt": verify_main,
    "dataset-review-hook": review_main,
}


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m agent_history_dataset_builder")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    return COMMANDS[options.command](options.args)


if __name__ == "__main__":
    raise SystemExit(main())
