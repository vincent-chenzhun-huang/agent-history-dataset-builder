"""Match transcript work spans to GitHub PRs using observed command results."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import NamedTuple, Protocol
from urllib.parse import unquote, urlsplit

from .config import PUBLICATION_ADAPTERS, PublicationBranchMapping, PublicationFormats
from .transcripts import Event


_PR_URL = re.compile(r"https://([\w.-]+)/([\w.-]+)/([\w.-]+)/pull/([1-9]\d*)")
_COMMIT_RESULT = re.compile(r"^\[[^\r\n\]]+ ([0-9a-f]{7,64})\](?:\s|$)", re.I | re.M)
_PUBLICATION_REASONS = {"git": "work_commit", "gh": "pr_created"}


class GitHubError(ValueError):
    """GitHub evidence was unavailable."""


@dataclass(frozen=True)
class Repo:
    host: str
    owner: str
    name: str

    @property
    def path(self) -> str:
        return f"repos/{self.owner}/{self.name}"


@dataclass(frozen=True)
class PullRequest:
    url: str
    repo: Repo
    number: int
    commits: frozenset[str]
    repository_id: str = ""
    head_branch: str | None = None

    @property
    def identity(self) -> str:
        return f"{self.repo.host}/{self.repository_id or self.repo.path}/{self.number}"


class PublicationEvidence(NamedTuple):
    reason: str
    event: Event
    branch: str | None = None
    remote_branch: str | None = None
    branch_mapping: PublicationBranchMapping | None = None
    adapter: str | None = None


@dataclass(frozen=True)
class WorkSpan:
    events: tuple[Event, ...]
    pr: PullRequest | None = None
    basis: tuple[PublicationEvidence, ...] = ()
    issue: str | None = None
    worktree: str | None = None
    worktree_repo: Repo | None = None
    repository: Repo | None = None
    has_publication: bool = False
    earlier_requests: tuple[Event, ...] = ()

    @property
    def session_id(self) -> str:
        return self.events[0].anchor.session_id


class PullRequests(Protocol):
    def get(self, url: str) -> PullRequest: ...

    def for_commit(self, repo: Repo, sha: str) -> tuple[PullRequest, ...]: ...


def parse_pr_url(url: str) -> tuple[Repo, int]:
    match = _PR_URL.fullmatch(url.rstrip("/"))
    if match is None:
        raise GitHubError("Expected a GitHub pull request URL such as https://github.com/owner/repo/pull/1")
    host, owner, repo, number = match.groups()
    return Repo(host.lower(), owner.lower(), repo.lower()), int(number)


def parse_repo(remote: object) -> Repo | None:
    if not isinstance(remote, str):
        return None
    ssh = re.fullmatch(r"git@([\w.-]+):([\w.-]+)/([\w.-]+)", remote)
    if ssh:
        host, owner, name = ssh.groups()
    else:
        try:
            url = urlsplit(remote)
        except ValueError:
            return None
        parts = url.path.strip("/").split("/")
        if url.scheme not in {"https", "ssh"} or not url.hostname or len(parts) != 2:
            return None
        host, (owner, name) = url.hostname, parts
    name = name.removesuffix(".git")
    if not all(re.fullmatch(r"[\w.-]+", value) for value in (host, owner, name)):
        return None
    return Repo(host.lower(), owner.lower(), name.lower())


class GhPullRequests:
    """Read GitHub using the caller's gh authentication."""

    def __init__(self, additional_host: str | None = None, *, hosts: tuple[str, ...] = ()) -> None:
        # These caches are per scan and do not write state.
        self.get = lru_cache(maxsize=512)(self.get)
        self.for_commit = lru_cache(maxsize=512)(self.for_commit)
        self.hosts = {"github.com", additional_host, *hosts}

    def _api(self, repo: Repo, suffix: str, *, paginate: bool = False) -> object:
        if repo.host not in self.hosts:
            raise GitHubError(f"No authorized GitHub host for {repo.path}")
        command = ["gh", "api", "--hostname", repo.host, "--method", "GET"]
        if paginate:
            command += ["--paginate", "--slurp"]
        command.append(f"{repo.path}/{suffix}")
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            if result.returncode:
                raise GitHubError(f"GitHub lookup failed for {repo.path}/{suffix}: {result.stderr.strip()}")
            return json.loads(result.stdout)
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
            raise GitHubError(f"Could not read GitHub evidence for {repo.path}/{suffix}") from error

    def get(self, url: str) -> PullRequest:
        repo, number = parse_pr_url(url)
        data = self._api(repo, f"pulls/{number}")
        if not isinstance(data, dict) or not isinstance(data.get("html_url"), str):
            raise GitHubError("GitHub did not return a pull request")
        canonical_repo, canonical_number = parse_pr_url(data["html_url"])
        if canonical_repo.host != repo.host or canonical_number != number:
            raise GitHubError("GitHub returned an unexpected pull request")
        pages = self._api(canonical_repo, f"pulls/{number}/commits?per_page=100", paginate=True)
        commits = {entry["sha"].lower() for page in pages for entry in page if entry.get("sha")}
        head_data = data.get("head", {})
        head = head_data.get("sha")
        head_branch = head_data.get("ref")
        if isinstance(head, str):
            commits.add(head.lower())
        return PullRequest(
            data["html_url"], canonical_repo, number, frozenset(commits),
            str(data.get("base", {}).get("repo", {}).get("id", "")),
            head_branch if _branch_name(head_branch) else None,
        )

    def for_commit(self, repo: Repo, sha: str) -> tuple[PullRequest, ...]:
        pages = self._api(repo, f"commits/{sha}/pulls?per_page=100", paginate=True)
        return tuple(
            self.get(entry["html_url"])
            for page in pages
            for entry in page
            if isinstance(entry.get("html_url"), str)
        )


@dataclass(frozen=True)
class Command:
    event: Event
    text: str
    output: str
    successful: bool
    cwd: str | None


def _native_command(event: Event) -> Command | None:
    item = event.payload.get("item", {})
    command = item.get("command")
    if isinstance(command, list) and all(isinstance(x, str) for x in command):
        # Codex records the shell wrapper as argv.
        command = command[-1] if len(command) == 3 and command[1] in {"-c", "-lc"} else shlex.join(command)
    if not isinstance(command, str):
        return None
    output = item.get("aggregated_output") or item.get("stdout") or ""
    return Command(event, command, str(output), item.get("exit_code") == 0, item.get("cwd") or item.get("workdir"))


def _paired_command(event: Event, call: Event) -> Command | None:
    if call.payload.get("name") not in {"exec_command", "shell_command", "shell"}:
        return None
    args = call.payload.get("arguments", call.payload.get("input"))
    try:
        args = json.loads(args) if isinstance(args, str) else args
    except json.JSONDecodeError:
        return None
    if not isinstance(args, dict):
        return None
    command = args.get("cmd", args.get("command"))
    if isinstance(command, list) and all(isinstance(x, str) for x in command):
        command = shlex.join(command)
    output = event.payload.get("output")
    if not isinstance(command, str) or not isinstance(output, str):
        return None
    # The adapter's process status provides evidence. Arbitrary output alone does not.
    boundary = re.search(r"(?m)^(?:Final output|Output):\n", output)
    if boundary is None:
        return None
    status = output[:boundary.start()]
    return Command(
        event, command, output[boundary.end():],
        bool(re.search(r"(?m)^Process exited with code 0\s*$", status)),
        args.get("workdir"),
    )


def _argv(command: str) -> list[str]:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="();<>|&\n")
        lexer.whitespace = " \t\r"
        lexer.whitespace_split = True
        lexer.commenters = ""
        words = list(lexer)
    except ValueError:
        return []
    # A composite command has only one outer exit status.
    if any(word and all(char in "();<>|&\n" for char in word) for word in words):
        return []
    if words and words[0].startswith("/") and Path(words[0]).name in {"gh", "git"}:
        words[0] = Path(words[0]).name
    return words


def _action(argv: list[str], *prefix: str) -> bool:
    return argv[:len(prefix)] == list(prefix)


def _sha_matches(commit: str, pr: PullRequest) -> bool:
    return sum(sha.lower().startswith(commit.lower()) for sha in pr.commits) == 1


def _git_target(words: list[str]) -> tuple[list[str], str | None]:
    if _action(words, "git", "-C") and len(words) >= 4:
        return ["git", *words[3:]], words[2]
    return words, None


def _worktree_path(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if value.startswith("file:"):
        try:
            uri = urlsplit(value)
            if uri.netloc not in {"", "localhost"} or uri.query or uri.fragment:
                return None
            value = unquote(uri.path, errors="strict")
        except (ValueError, UnicodeError):
            return None
        if not value.startswith("/"):
            return None
    return os.path.normpath(value) if "\x00" not in value else None


def _recorded_argv(text: str) -> list[str]:
    words = _argv(text.rstrip("\n"))
    if words:
        return words
    # The only accepted command batch is staging followed by one final commit.
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2 or any(char in text for char in ("`", "$", "\\")):
        return []
    commands = [_argv(line) for line in lines]
    if all(_action(argv, "git", "add") and len(argv) > 2 for argv in commands[:-1]):
        if _action(commands[-1], "git", "commit"):
            return commands[-1]
    return []


def _branch_name(value: object) -> str | None:
    if not isinstance(value, str) or not re.fullmatch(r"[\w.][\w./-]*", value):
        return None
    if value in {"HEAD", "@"} or ".." in value or "//" in value or value.endswith(("/", ".", ".lock")):
        return None
    return value.removeprefix("refs/heads/") or None


def _branch_change(words: list[str]) -> bool:
    if _action(words, "git", "switch"):
        return True
    if _action(words, "gh", "pr", "checkout"):
        return True
    if _action(words, "git", "checkout"):
        return len(words) > 2 and "--" not in words[2:]
    return False


def _repo_option(words: list[str]) -> Repo | None:
    for index, word in enumerate(words[3:], start=3):
        value = None
        if word in {"-R", "--repo"} and index + 1 < len(words):
            value = words[index + 1]
        elif word.startswith("--repo="):
            value = word.partition("=")[2]
        if value:
            return parse_repo("https://" + (value if value.count("/") == 2 else f"github.com/{value}"))
    return None


def _git_publication(words: list[str], output: str) -> str | None:
    match = _COMMIT_RESULT.search(output) if _action(words, "git", "commit") else None
    return match.group(1).lower() if match else None


def _gh_publication(words: list[str], output: str) -> str | None:
    if not _action(words, "gh", "pr", "create"):
        return None
    target = _repo_option(words)
    urls = [
        url for line in output.splitlines()
        if (url := line.strip()) and _PR_URL.fullmatch(url)
        and (target is None or parse_pr_url(url)[0] == target)
    ]
    return urls[0] if len(urls) == 1 else None


_STATUS_PR = re.compile(
    r"^[ \t]*(?:created|updated|published|pushed)[ \t]+(?:pull request|PR)\b"
    r"(?:[ \t]+#?([1-9]\d*))?[: \t]+"
    r"(https://[\w.-]+/[\w.-]+/[\w.-]+/pull/[1-9]\d*)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_BRANCH_PR = re.compile(r"^Updating PR for branch (\S+) \[(\S+)\][ \t]*$", re.MULTILINE)
_NONPUBLICATION = re.compile(
    r"^[ \t]*(?:usage:|options:|\[?dry[- ]run\b|would\s+(?:push|publish|create|update)\b)"
    r"|\(dry[- ]run\)",
    re.IGNORECASE | re.MULTILINE,
)
_PUBLICATION_MARKERS = re.compile(
    r"^[ \t]*(?:creat(?:ed|ing)|updat(?:ed|ing)|publish(?:ed|ing)|push(?:ed|ing))"
    r"[ \t]+(?:PR|pull request)\b", re.IGNORECASE,
)


def _configured_publication(output: str, formats: PublicationFormats) -> tuple[str, str | None] | None:
    output = output.replace("\r\n", "\n")
    lines = [line.strip() for line in output.splitlines()]
    prefixed = [line for line in lines if formats.line_prefix and line.startswith(formats.line_prefix)]
    marker_lines = set(prefixed)
    if _NONPUBLICATION.search(output) or sum(
        line in marker_lines or bool(_PUBLICATION_MARKERS.match(line)) for line in lines
    ) != 1:
        return None
    published = []
    if "legacy-status" in formats.output_formats:
        published.extend((url, None) for number, url in _STATUS_PR.findall(output)
                         if not number or number == url.rsplit("/", 1)[-1])
    if "branch-update" in formats.output_formats:
        published.extend((url, branch) for branch_name, url in _BRANCH_PR.findall(output)
                         if (branch := _branch_name(branch_name)) is not None)
    if "prefixed-url" in formats.output_formats and prefixed and formats.line_prefix:
        parts = prefixed[0].removeprefix(formats.line_prefix).split(maxsplit=1)
        if parts and not (len(parts) == 2 and re.search(r"\bdry[- ]run\b|\bwould\b", parts[1], re.IGNORECASE)):
            published.append((parts[0], None))
    if len(published) != 1:
        return None
    url, branch = published[0]
    if not _PR_URL.fullmatch(url) or any(match.group() != url for match in _PR_URL.finditer(output)):
        return None
    return url, branch


def worktree_snapshot(span: WorkSpan) -> dict[str, str]:
    if not span.worktree or span.worktree_repo is None:
        return {}
    directory = Path(span.worktree)
    if not directory.is_absolute() or not directory.is_dir():
        return {}

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "-C", str(directory), *args],
            check=True, capture_output=True,
            text=True, timeout=5, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        ).stdout.strip()

    try:
        if parse_repo(git("remote", "get-url", "origin")) != span.worktree_repo:
            return {}
        head = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain"))
    except (OSError, subprocess.SubprocessError):
        return {}
    return {"recorded_head_sha": head, "recorded_worktree_dirty": str(dirty).lower()}


def _confirmed_branch(
    evidence: PublicationEvidence, url: str, pr: PullRequest,
    mappings: tuple[PublicationBranchMapping, ...],
) -> PublicationEvidence:
    adapter = evidence.adapter
    if adapter is None:
        return evidence
    remote = pr.head_branch
    if not remote:
        raise GitHubError("Publication has no attributable PR branch")
    local = evidence.branch
    if not local or local == remote:
        return evidence._replace(branch=local or remote, remote_branch=remote)
    # The branch rule applies only to the exact authenticated publication repository.
    if parse_pr_url(url) == (pr.repo, pr.number):
        repository = f"{pr.repo.owner}/{pr.repo.name}"
        for mapping in mappings:
            if (mapping.host, mapping.repository, mapping.adapter) == (pr.repo.host, repository, adapter):
                if mapping.remote_prefix + local == remote:
                    return evidence._replace(remote_branch=remote, branch_mapping=mapping)
    raise GitHubError("Published branch does not match authenticated PR head")


def match_spans(
    events: list[Event], github: PullRequests, selected: PullRequest | None = None, *,
    adapters: tuple[str, ...] = PUBLICATION_ADAPTERS,
    formats: dict[str, PublicationFormats] | None = None,
    branch_mappings: tuple[PublicationBranchMapping, ...] = (),
) -> list[WorkSpan]:
    if not events:
        return []
    spans: list[WorkSpan] = []
    pending: list[Event] = []
    observed: list[tuple[str, Repo | None, PublicationEvidence]] = []
    configured = {name: options for name, options in (formats or {}).items() if name in adapters}
    publication_commands = {argv: name for name, options in configured.items() for argv in options.argv_variants}
    branch_commands = {argv for options in configured.values() for argv in options.branch_change_variants}
    calls = {}
    repo: Repo | None = None
    cwd: str | None = None
    branch: str | None = None
    recorded_worktree: str | None = None
    worktree_repos: dict[str, Repo] = {}
    worktree_branches: dict[str, str] = {}

    def boundary(event: Event) -> None:
        request = None
        if event.turn_id is not None:
            request = next((entry for entry in reversed(pending)
                            if entry.kind == "user_message" and entry.turn_id == event.turn_id), None)
        if request is not None and not any(item.event.position > request.position for _, _, item in observed):
            pending.remove(request)
        else:
            request = None
        finish()
        if request is not None:
            pending.append(request)

    def finish() -> None:
        nonlocal recorded_worktree
        if not pending:
            return
        candidates: dict[str, tuple[PullRequest, list[PublicationEvidence]]] = {}
        issue = None
        urls = {parse_pr_url(value) for value, _, _ in observed if value.startswith("https://")}
        for value, evidence_repo, evidence in observed:
            try:
                if value.startswith("https://"):
                    if selected is None:
                        prs = (github.get(value),)
                    else:
                        prs = (selected,) if parse_pr_url(value) == parse_pr_url(selected.url) else ()
                else:
                    if evidence_repo is None:
                        issue = "A work commit has no recorded GitHub repository"
                        continue
                    if selected is not None:
                        # A fork's commit can be looked up against the chosen PR.
                        prs = (selected,) if selected.repo.host == evidence_repo.host else ()
                    else:
                        prs = github.for_commit(evidence_repo, value)
                    prs = tuple(pr for pr in prs if _sha_matches(value, pr))
            except GitHubError as error:
                issue = str(error)
                continue
            for pr in prs:
                try:
                    confirmed = _confirmed_branch(evidence, value, pr, branch_mappings)
                except GitHubError as error:
                    issue = str(error)
                    continue
                _, refs = candidates.setdefault(pr.identity, (pr, []))
                refs.append(confirmed)
        if len(candidates) == 1 and len(urls) <= 1:
            pr, refs = next(iter(candidates.values()))
            path = recorded_worktree or cwd
            spans.append(WorkSpan(
                tuple(pending), pr, tuple(refs), worktree=path, worktree_repo=worktree_repos.get(path),
                repository=repo or pr.repo, has_publication=bool(urls),
            ))
        else:
            if len(candidates) > 1 or len(urls) > 1:
                issue = "Competing PR matches"
            spans.append(WorkSpan(
                tuple(pending), issue=issue or "No confirmed PR evidence", repository=repo,
                has_publication=bool(urls),
            ))
        pending.clear()
        observed.clear()
        recorded_worktree = None

    for event in events:
        if event.kind == "session_meta":
            raw_git = event.payload.get("git")
            header_repo = header_branch = None
            if isinstance(raw_git, dict):
                header_repo = parse_repo(raw_git.get("repository_url"))
                header_branch = _branch_name(raw_git.get("branch"))
                if (repo and header_repo and repo != header_repo) or (branch and header_branch and branch != header_branch):
                    boundary(event)
                repo = header_repo or repo
                branch = header_branch or branch
            header_cwd = _worktree_path(event.payload.get("cwd"))
            if header_cwd:
                cwd = header_cwd
                if header_repo:
                    worktree_repos[cwd] = header_repo
                if header_branch:
                    worktree_branches[cwd] = header_branch
        if event.kind == "turn_context":
            turn_cwd = _worktree_path(event.payload.get("cwd"))
            turn_branch = event.payload.get("git", {})
            turn_branch = _branch_name(turn_branch.get("branch")) if isinstance(turn_branch, dict) else None
            if turn_cwd and turn_cwd != cwd:
                turn_repo = worktree_repos.get(turn_cwd)
                if turn_repo and repo and turn_repo != repo:
                    boundary(event)
                cwd = turn_cwd
                repo = turn_repo or repo
                recorded_worktree = None
            turn_branch = turn_branch or worktree_branches.get(cwd)
            if turn_branch:
                if branch and turn_branch != branch:
                    boundary(event)
                branch = turn_branch
                if cwd:
                    worktree_branches[cwd] = branch
        command = None
        if event.kind == "tool_call":
            calls[event.anchor] = event
        elif event.kind == "tool_result" and event.call_anchor in calls:
            command = _paired_command(event, calls[event.call_anchor])
        elif event.kind == "command_execution":
            command = _native_command(event)

        if command and command.successful:
            words, explicit_cwd = _git_target(_recorded_argv(command.text))
            base_cwd = _worktree_path(command.cwd) if command.cwd else cwd
            command_cwd = _worktree_path(explicit_cwd) if explicit_cwd else base_cwd
            if command_cwd and explicit_cwd and base_cwd and not PurePosixPath(command_cwd).is_absolute():
                command_cwd = os.path.normpath(str(PurePosixPath(base_cwd) / command_cwd))
            command_repo = worktree_repos.get(command_cwd)
            if _action(words, "git", "remote", "get-url") and not event.output_truncated:
                discovered = parse_repo(command.output.strip())
                if discovered is not None:
                    if repo is not None and repo != discovered and command_cwd == cwd:
                        boundary(event)
                    command_repo = discovered
                    if command_cwd == cwd:
                        repo = discovered
                    if isinstance(command_cwd, str):
                        worktree_repos[command_cwd] = discovered
            if _branch_change(words) or tuple(words) in branch_commands:
                boundary(event)
                branch = None
                worktree_branches.pop(command_cwd, None)
            elif not event.output_truncated and words:
                adapter = publication_commands.get(tuple(words), words[0])
                value = published_branch = None
                if adapter in adapters:
                    if adapter == "git":
                        value = _git_publication(words, command.output)
                    elif adapter == "gh":
                        value = _gh_publication(words, command.output)
                    elif tuple(words) in publication_commands:
                        result = _configured_publication(command.output, configured[adapter])
                        if result:
                            value, published_branch = result
                            command_branch = worktree_branches.get(command_cwd)
                            if command_branch and published_branch and command_branch != published_branch:
                                boundary(event)
                            published_branch = published_branch or command_branch
                if value:
                    reason = _PUBLICATION_REASONS.get(adapter, "configured_publication")
                    if command_repo and repo and command_repo != repo:
                        boundary(event)
                        repo = command_repo
                    recorded_worktree = command_cwd or recorded_worktree
                    observed.append((value, command_repo, PublicationEvidence(
                        reason, event, published_branch,
                        adapter=adapter if reason == "configured_publication" else None,
                    )))
        pending.append(event)
    finish()
    return spans
