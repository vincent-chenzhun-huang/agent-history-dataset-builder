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
from .config import load_settings
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
    parser.add_argument("--config", type=Path)
    parser.add_argument("--codex-bin")
    parser.add_argument("--model")
    parser.add_argument("--model-command", help="Trusted JSON-in/JSON-out executable and arguments")
    parser.add_argument("--model-timeout", type=int, metavar="SECONDS")
    args = parser.parse_args(argv)
    cutoff = utc(args.since)
    if args.since and not cutoff:
        parser.error("--since must be an ISO 8601 timestamp with a UTC offset")
    try:
        settings = load_settings(args.config)
        output = args.output if args.output is not None else settings.output
        if output is None and not args.dry_run:
            parser.error("--output is required when publishing without a configured output")
        model = settings.model
        command = list(model.command) or None
        if args.model_command is not None:
            command = shlex.split(args.model_command)
            if not command:
                parser.error("--model-command must name an executable")
        timeout = model.timeout_seconds if args.model_timeout is None else args.model_timeout
        if timeout < 1:
            parser.error("--model-timeout must be a positive integer")
        host = parse_pr_url(args.pr)[0].host if args.pr else None
        report = scan(
            args.codex_home, GhPullRequests(host, hosts=settings.github_hosts), StructuredModel(
                args.codex_bin or os.environ.get("AGENT_HISTORY_CODEX_BIN") or model.codex_bin,
                args.model or model.name, command, timeout=timeout, max_input_chars=model.max_input_chars,
            ),
            pr_url=args.pr, since=cutoff or None, output=output, dry_run=args.dry_run, settings=settings,
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
