"""Match transcript work spans to GitHub PRs using observed command results."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from .transcripts import Event


_PR_URL = re.compile(r"https://([\w.-]+)/([\w.-]+)/([\w.-]+)/pull/([1-9]\d*)")
_SHA = re.compile(r"[0-9a-f]{7,64}", re.I)
_COMMIT_RESULT = re.compile(r"^\[[^\r\n\]]+ ([0-9a-f]{7,64})\](?:\s|$)", re.I | re.M)


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

    @property
    def identity(self) -> str:
        return f"{self.repo.host}/{self.repository_id or self.repo.path}/{self.number}"


@dataclass(frozen=True)
class WorkSpan:
    events: tuple[Event, ...]
    pr: PullRequest | None = None
    basis: tuple[tuple[str, Event], ...] = ()
    issue: str | None = None
    worktree: str | None = None
    worktree_repo: Repo | None = None

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

    def __init__(self, additional_host: str | None = None) -> None:
        # These caches are per scan and do not write state.
        self.get = lru_cache(maxsize=512)(self.get)
        self.for_commit = lru_cache(maxsize=512)(self.for_commit)
        self.hosts = {"github.com", additional_host}

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
        head = data.get("head", {}).get("sha")
        if isinstance(head, str):
            commits.add(head.lower())
        return PullRequest(
            data["html_url"], canonical_repo, number, frozenset(commits),
            str(data.get("base", {}).get("repo", {}).get("id", "")),
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
    return Command(event, command, str(output), item.get("exit_code") == 0, item.get("cwd"))


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
        lexer = shlex.shlex(command, posix=True, punctuation_chars="();<>|&")
        lexer.whitespace_split = True
        lexer.commenters = ""
        words = list(lexer)
    except ValueError:
        return []
    # A composite command has only one outer exit status.
    if any(word and all(char in "();<>|&" for char in word) for word in words):
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


def _branch_change(words: list[str]) -> bool:
    if _action(words, "git", "switch") or _action(words, "git", "worktree", "add"):
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


def match_spans(
    events: list[Event], github: PullRequests, selected: PullRequest | None = None
) -> list[WorkSpan]:
    if not events:
        return []
    spans: list[WorkSpan] = []
    pending: list[Event] = []
    observed: list[tuple[str, Event, Repo | None]] = []
    calls = {}
    repo: Repo | None = None
    cwd: str | None = None
    branch: str | None = None
    recorded_worktree: str | None = None
    worktree_repos: dict[str, Repo] = {}

    def boundary(event: Event) -> None:
        request = next((entry for entry in reversed(pending)
                        if entry.kind == "user_message" and entry.turn_id == event.turn_id), None)
        if request is not None and not any(entry.position > request.position for _, entry, _ in observed):
            pending.remove(request)
        else:
            request = None
        finish()
        if request is not None:
            pending.append(request)

    def finish() -> None:
        if not pending:
            return
        candidates: dict[str, tuple[PullRequest, list[tuple[str, Event]]]] = {}
        issue = None
        for value, evidence, evidence_repo in observed:
            try:
                if value.startswith("https://"):
                    if selected is None:
                        prs = (github.get(value),)
                    else:
                        prs = (selected,) if parse_pr_url(value) == parse_pr_url(selected.url) else ()
                    reason = "pr_created"
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
                    reason = "work_commit"
            except GitHubError as error:
                issue = str(error)
                continue
            for pr in prs:
                _, refs = candidates.setdefault(pr.identity, (pr, []))
                refs.append((reason, evidence))
        if len(candidates) == 1:
            pr, refs = next(iter(candidates.values()))
            path = recorded_worktree or cwd
            spans.append(WorkSpan(
                tuple(pending), pr, tuple(refs), worktree=path, worktree_repo=worktree_repos.get(path),
            ))
        else:
            if candidates:
                issue = "Competing PR matches"
            spans.append(WorkSpan(tuple(pending), issue=issue or "No confirmed PR evidence"))
        pending.clear()
        observed.clear()

    for event in events:
        if event.kind == "session_meta":
            raw_git = event.payload.get("git")
            if isinstance(raw_git, dict):
                repo = parse_repo(raw_git.get("repository_url"))
                branch = raw_git.get("branch")
            cwd = event.payload.get("cwd")
            if repo and isinstance(cwd, str):
                worktree_repos[cwd] = repo
        if event.kind == "turn_context":
            turn_cwd = event.payload.get("cwd")
            turn_branch = event.payload.get("git", {})
            turn_branch = turn_branch.get("branch") if isinstance(turn_branch, dict) else None
            if isinstance(turn_cwd, str) and turn_cwd != cwd:
                boundary(event)
                cwd = turn_cwd
                repo = worktree_repos.get(cwd)
                branch = None
                recorded_worktree = None
            if isinstance(turn_branch, str) and branch is not None and turn_branch != branch:
                boundary(event)
            if isinstance(turn_branch, str):
                branch = turn_branch
        command = None
        if event.kind == "tool_call":
            calls[event.anchor] = event
        elif event.kind == "tool_result" and event.call_anchor in calls:
            command = _paired_command(event, calls[event.call_anchor])
        elif event.kind == "command_execution":
            command = _native_command(event)

        if command and command.successful:
            words, explicit_cwd = _git_target(_argv(command.text))
            command_cwd = explicit_cwd or command.cwd or cwd
            # Distinct command worktrees partition the event stream too.
            if isinstance(command_cwd, str) and (
                (recorded_worktree is not None and recorded_worktree != command_cwd)
                or (recorded_worktree is None and cwd is not None and cwd != command_cwd)
            ):
                boundary(event)
            if isinstance(command_cwd, str):
                recorded_worktree = command_cwd
            command_repo = worktree_repos.get(command_cwd, repo if command_cwd == cwd else None)
            if _action(words, "git", "remote", "get-url") and not event.output_truncated:
                discovered = parse_repo(command.output.strip())
                if discovered is not None:
                    if command_repo is not None and command_repo != discovered:
                        finish()
                    command_repo = discovered
                    if isinstance(command_cwd, str):
                        worktree_repos[command_cwd] = discovered
            if _branch_change(words):
                boundary(event)
            elif _action(words, "git", "commit"):
                match = _COMMIT_RESULT.search(command.output)
                if match:
                    observed.append((match.group(1).lower(), event, command_repo))
            elif _action(words, "gh", "pr", "create"):
                target = _repo_option(words)
                urls = [
                    url for line in command.output.splitlines()
                    if (url := line.strip()) and _PR_URL.fullmatch(url)
                    and (target is None or parse_pr_url(url)[0] == target)
                ]
                if len(urls) == 1:
                    observed.append((urls[0], event, command_repo))
        pending.append(event)
    finish()
    return spans
