"""Command line entry points for scanning and verifying the dataset."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from . import dataset
from .analysis import utc
from .model import StructuredModel
from .prs import GhPullRequests, parse_pr_url
from .scanner import scan


def scan_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scan-history")
    parser.add_argument("--codex-home", action="append", type=Path, required=True)
    parser.add_argument("--pr")
    parser.add_argument("--since", metavar="UTC_TIMESTAMP")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--codex-bin", default=os.environ.get("AGENT_HISTORY_CODEX_BIN", "codex"))
    parser.add_argument("--model")
    parser.add_argument("--model-command", help="Trusted JSON-in/JSON-out executable and arguments")
    args = parser.parse_args(argv)
    if args.output is None and not args.dry_run:
        parser.error("--output is required when publishing")
    cutoff = utc(args.since)
    if args.since and not cutoff:
        parser.error("--since must be an ISO 8601 timestamp with a UTC offset")
    try:
        command = shlex.split(args.model_command) if args.model_command is not None else None
        if command == []:
            parser.error("--model-command must name an executable")
        host = parse_pr_url(args.pr)[0].host if args.pr else None
        report = scan(
            args.codex_home, GhPullRequests(host), StructuredModel(args.codex_bin, args.model, command),
            pr_url=args.pr, since=cutoff or None, output=args.output, dry_run=args.dry_run,
        )
    except (OSError, ValueError) as error:
        print(f"scan-history: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def verify_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="verify-attempt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("attempt_id")
    args = parser.parse_args(argv)
    try:
        if dataset.verify(args.output, args.attempt_id):
            return 0
        print("verify-attempt: attempt ID not found", file=sys.stderr)
    except (OSError, ValueError) as error:
        print(f"verify-attempt: {error}", file=sys.stderr)
    return 1
