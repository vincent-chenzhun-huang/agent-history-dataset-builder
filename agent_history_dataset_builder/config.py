"""Load optional settings shared by the scanner and review hooks."""

from __future__ import annotations

import json
import re
from collections.abc import Set
from dataclasses import dataclass, field
from pathlib import Path


PUBLICATION_ADAPTERS = ("git", "gh")
EVIDENCE_KINDS = frozenset({
    "user_message", "assistant_message", "command_execution", "file_change", "tool_result", "mcp_tool_call",
})


@dataclass(frozen=True)
class Limits:
    event_chars: int = 6000
    context_chars: int = 10_000_000
    max_events: int = 100_000
    lookback_events: int = 48
    lookback_by_kind: dict[str, int] = field(default_factory=dict)
    batch_size: int = 4
    task_limit: int = 16


@dataclass(frozen=True)
class ModelSettings:
    adapter: str = "codex"
    codex_bin: str = "codex"
    name: str | None = None
    command: tuple[str, ...] = ()
    timeout_seconds: int = 240
    max_input_chars: int = 1_048_576


@dataclass(frozen=True)
class HookSettings:
    every: int = 5
    context_chars: int = 5000
    span_limit: int = 3


@dataclass(frozen=True)
class PublicationFormats:
    argv_variants: tuple[tuple[str, ...], ...]
    output_formats: tuple[str, ...]
    line_prefix: str | None = None
    branch_change_variants: tuple[tuple[str, ...], ...] = ()


_OUTPUT_FORMATS = {"legacy-status", "branch-update", "prefixed-url"}

_HOSTNAME = re.compile(
    r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+", re.ASCII | re.IGNORECASE,
)
_REPOSITORY = re.compile(r"[a-z0-9_-][a-z0-9_.-]*/[a-z0-9_.-]+", re.ASCII)
_BRANCH_PREFIX = re.compile(r"(?:[A-Za-z0-9_-][A-Za-z0-9_.-]*/)+")


@dataclass(frozen=True)
class PublicationBranchMapping:
    host: str
    repository: str
    adapter: str
    remote_prefix: str


@dataclass(frozen=True)
class Settings:
    publication_adapters: tuple[str, ...] = PUBLICATION_ADAPTERS
    analysis: Limits = field(default_factory=Limits)
    model: ModelSettings = field(default_factory=ModelSettings)
    hook: HookSettings = field(default_factory=HookSettings)
    output: Path | None = None
    publication_formats: dict[str, PublicationFormats] = field(default_factory=dict)
    github_hosts: tuple[str, ...] = ()
    publication_branch_mappings: tuple[PublicationBranchMapping, ...] = ()


def _section(value: object, name: str, allowed: Set[str]) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown {name} setting: {', '.join(sorted(unknown))}")
    return value


def _number(value: object, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _argv_variants(value: object, key: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list) or any(
        not isinstance(argv, list) or not argv
        or any(not isinstance(arg, str) or not arg.strip() or any(ord(c) < 32 for c in arg) for arg in argv)
        for argv in value
    ):
        raise ValueError(f"{key} must list argv arrays")
    variants = tuple(tuple(argv) for argv in value)
    if len(set(variants)) != len(variants):
        raise ValueError(f"{key} must list unique argv arrays")
    return variants


def _publication_formats(value: object) -> dict[str, PublicationFormats]:
    if not isinstance(value, dict):
        raise ValueError("publication_formats must be a JSON object")
    result = {}
    published_commands: set[tuple[str, ...]] = set()
    branch_commands: set[tuple[str, ...]] = set()
    for name, options in value.items():
        if name in PUBLICATION_ADAPTERS or not re.fullmatch(r"[a-z][a-z0-9-]*", name):
            raise ValueError(f"Invalid custom publication adapter name: {name}")
        key = f"publication_formats.{name}"
        fields = _section(options, key, {"argv_variants", "output_formats", "line_prefix", "branch_change_variants"})
        variants = _argv_variants(fields.get("argv_variants"), f"{key}.argv_variants")
        branch_variants = _argv_variants(fields.get("branch_change_variants", []), f"{key}.branch_change_variants")
        if not variants or published_commands.intersection(variants):
            raise ValueError(f"{key}.argv_variants must provide unambiguous commands")
        if any(argv[:2] in {("git", "push"), ("git", "commit")} or argv[:3] == ("gh", "pr", "create")
               or {"--help", "-h", "--dry-run"}.intersection(argv) for argv in variants):
            raise ValueError(f"{key}.argv_variants cannot redefine built-in or non-publishing commands")
        published_commands.update(variants)
        branch_commands.update(branch_variants)
        formats = fields.get("output_formats")
        if (
            not isinstance(formats, list) or not formats
            or any(not isinstance(fmt, str) or fmt not in _OUTPUT_FORMATS for fmt in formats)
            or len(set(formats)) != len(formats)
        ):
            raise ValueError(f"{key}.output_formats must list unique supported format names")
        prefix = fields.get("line_prefix")
        if "prefixed-url" in formats:
            prefix = _text(prefix, f"{key}.line_prefix")
            if not prefix.strip().endswith(":") or any(ord(c) < 32 for c in prefix):
                raise ValueError(f"{key}.line_prefix must be a literal single-line label ending in ':'")
            prefix = prefix.strip()
        elif "line_prefix" in fields:
            raise ValueError(f"{key}.line_prefix requires the prefixed-url output format")
        result[name] = PublicationFormats(variants, tuple(formats), prefix, branch_variants)
    if published_commands & branch_commands:
        raise ValueError("Publication commands cannot also change branches")
    return result


def _publication_branch_mappings(
    value: object, adapters: list[str],
) -> tuple[PublicationBranchMapping, ...]:
    if not isinstance(value, list):
        raise ValueError("publication_branch_mappings must be an array")
    mappings = []
    scopes = set()
    for index, raw in enumerate(value):
        label = f"publication_branch_mappings[{index}]"
        rule = _section(raw, label, {"host", "repository", "adapter", "remote_prefix"})
        host = _text(rule.get("host"), f"{label}.host").lower()
        repo = _text(rule.get("repository"), f"{label}.repository").lower()
        adapter = _text(rule.get("adapter"), f"{label}.adapter")
        prefix = _text(rule.get("remote_prefix"), f"{label}.remote_prefix")
        if not _HOSTNAME.fullmatch(host) or not _REPOSITORY.fullmatch(repo) or repo.endswith(("/.", "/..")):
            raise ValueError(f"{label} requires a hostname and an owner/repository")
        if adapter in PUBLICATION_ADAPTERS or adapter not in adapters:
            raise ValueError(f"{label}.adapter must be an enabled custom adapter")
        if (
            not _BRANCH_PREFIX.fullmatch(prefix) or prefix.startswith("-") or ".." in prefix
            or any(part.endswith((".", ".lock")) for part in prefix[:-1].split("/"))
        ):
            raise ValueError(f"{label}.remote_prefix must be a literal branch prefix ending in '/'")
        scope = (host, repo, adapter)
        if scope in scopes:
            raise ValueError(f"{label} duplicates a host, repository, and adapter scope")
        scopes.add(scope)
        mappings.append(PublicationBranchMapping(host, repo, adapter, prefix))
    return tuple(mappings)


def load_settings(path: Path | None) -> Settings:
    if path is None:
        return Settings()
    path = path.expanduser().resolve()
    data = _section(json.loads(path.read_text(encoding="utf-8")), "config", {
        "version", "publication_adapters", "publication_formats", "github_hosts", "publication_branch_mappings",
        "analysis", "model", "hook", "output",
    })
    if _number(data.get("version", 1), "version") != 1:
        raise ValueError("Unsupported config version")
    adapters = data.get("publication_adapters", list(PUBLICATION_ADAPTERS))
    publication_formats = _publication_formats(data.get("publication_formats", {}))
    if (
        not isinstance(adapters, list)
        or any(not isinstance(name, str) or name not in (*PUBLICATION_ADAPTERS, *publication_formats)
               for name in adapters)
        or len(set(adapters)) != len(adapters)
    ):
        raise ValueError("publication_adapters must list unique supported adapter names")
    if publication_formats.keys() - set(adapters):
        raise ValueError("publication_formats can only configure enabled adapters")
    hosts = data.get("github_hosts", [])
    if (
        not isinstance(hosts, list)
        or any(not isinstance(host, str) or not _HOSTNAME.fullmatch(host) for host in hosts)
        or len({host.lower() for host in hosts}) != len(hosts)
    ):
        raise ValueError("github_hosts must list unique hostnames")

    analysis = _section(data.get("analysis", {}), "analysis", set(Limits.__dataclass_fields__))
    lookback = _section(analysis.get("lookback_by_kind", {}), "analysis.lookback_by_kind", EVIDENCE_KINDS)
    numbers = {
        key: _number(value, f"analysis.{key}", 0 if key == "lookback_events" else 1)
        for key, value in analysis.items() if key != "lookback_by_kind"
    }
    limits = Limits(**numbers, lookback_by_kind={
        kind: _number(count, f"analysis.lookback_by_kind.{kind}", 0)
        for kind, count in lookback.items()
    })

    model = _section(data.get("model", {}), "model", set(ModelSettings.__dataclass_fields__))
    adapter = model.get("adapter", "command" if "command" in model else "codex")
    if adapter not in ("codex", "command"):
        raise ValueError("model.adapter must be codex or command")
    command = model.get("command", [])
    if not isinstance(command, list):
        raise ValueError("model.command must be an array of arguments")
    argv = tuple(_text(value, "model.command argument") for value in command)
    if (adapter == "command") != bool(argv):
        raise ValueError("model.command is required for, and only applies to, the command adapter")
    model_settings = ModelSettings(
        adapter=adapter,
        codex_bin=_text(model.get("codex_bin", "codex"), "model.codex_bin"),
        name=_text(model["name"], "model.name") if "name" in model else None,
        command=argv,
        timeout_seconds=_number(model.get("timeout_seconds", 240), "model.timeout_seconds"),
        max_input_chars=_number(model.get("max_input_chars", ModelSettings.max_input_chars), "model.max_input_chars"),
    )
    hook = _section(data.get("hook", {}), "hook", set(HookSettings.__dataclass_fields__))
    output = None
    if "output" in data:
        output = (path.parent / Path(_text(data["output"], "output")).expanduser()).resolve()
    hook_settings = HookSettings(
        every=_number(hook.get("every", 5), "hook.every"),
        context_chars=_number(hook.get("context_chars", 5000), "hook.context_chars"),
        span_limit=_number(hook.get("span_limit", 3), "hook.span_limit"),
    )
    return Settings(
        publication_adapters=tuple(adapters), analysis=limits, model=model_settings,
        hook=hook_settings, output=output, publication_formats=publication_formats,
        github_hosts=tuple(host.lower() for host in hosts),
        publication_branch_mappings=_publication_branch_mappings(
            data.get("publication_branch_mappings", []), adapters,
        ),
    )
