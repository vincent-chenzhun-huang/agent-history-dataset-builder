"""Exercise the CUJs through the CLI and review boundary with scripted adapters."""

import contextlib
import csv
import io
import json
import shutil
import shlex
import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agent_history_dataset_builder import dataset
from agent_history_dataset_builder.analysis import (
    ModelError, ModelInputTooLarge, ModelTimeout, Task, analyze_span, apply_task_decisions, ref,
)
from agent_history_dataset_builder.cli import scan_main, verify_main
from agent_history_dataset_builder.config import Limits, PublicationBranchMapping, PublicationFormats, Settings, load_settings
from agent_history_dataset_builder.hook import (
    REVIEW_PROTOCOL, main as hook_main, prompt_response, publish_review, receive_review, review_context, review_due,
)
from agent_history_dataset_builder.model import StructuredModel
from agent_history_dataset_builder.prs import GhPullRequests, GitHubError, PullRequest, Repo, match_spans
from agent_history_dataset_builder.scanner import scan
from agent_history_dataset_builder.transcripts import REVIEW_PREFIX, page_anchors, read_rollout


REPO = Repo("github.com", "example", "project")
FIRST = PullRequest("https://github.com/example/project/pull/1", REPO, 1, frozenset({"a" * 40}), "1", "first")
SECOND = PullRequest("https://github.com/example/project/pull/2", REPO, 2, frozenset({"b" * 40}), "1")
ORIGINAL = "  Keep the old output stable.\nHandle missing input.\n"
TURN = "turn-1"
TEST_FORMATS = {
    "review-publish": PublicationFormats(
        (("review", "publish"), ("review", "publish", "--only")),
        ("legacy-status", "branch-update"),
        branch_change_variants=(("review", "next"), ("review", "checkout", "first")),
    ),
    "review-submit": PublicationFormats(
        (("review", "submit"), ("review", "submit", "origin", "HEAD"),
         ("review", "submit", "origin", "HEAD:refs/heads/first")),
        ("prefixed-url",),
        "Review result:",
    ),
}


class FixtureGitHub:
    def __init__(self, *pulls):
        self.pulls = pulls

    def get(self, url):
        return next(pr for pr in self.pulls if pr.url == url)

    def for_commit(self, repo, sha):
        return tuple(pr for pr in self.pulls if pr.repo == repo and any(s.startswith(sha) for s in pr.commits))


def _record(record_type, second, **payload):
    return json.dumps({
        "timestamp": f"2026-01-01T00:00:{second:02}Z",
        "type": record_type,
        "payload": payload,
    }) + "\n"


def meta(session="session-a", **extra):
    return _record("session_meta", 0, id=session, cwd="/old machine/worktree", git={
        "branch": "first", "repository_url": "git@github.com:example/project.git",
    }, **extra)


def turn(turn_id, second, subtype="task_started", **extra):
    return _record("event_msg", second, type=subtype, turn_id=turn_id, **extra)


def user(message, turn_id, second):
    return _record("event_msg", second, type="item_completed", turn_id=turn_id, item={
        "type": "UserMessage", "id": f"u-{turn_id}-{second}",
        "content": [{"type": "text", "text": message}],
    })


def assistant(message, turn_id, second):
    return _record("response_item", second, type="message", role="assistant", turn_id=turn_id,
                   content=[{"type": "output_text", "text": message}])


def command(name, output, turn_id, second, exit_code=0, *, cwd="/old machine/worktree"):
    return _record("event_msg", second, type="item_completed", turn_id=turn_id, item={
        "type": "CommandExecution", "id": f"cmd-{turn_id}-{second}", "command": ["sh", "-c", name],
        "cwd": cwd, "aggregated_output": output, "exit_code": exit_code,
    })


def failure_turn(turn_id=TURN, second=1, pr=FIRST):
    return (
        turn(turn_id, second)
        + user(ORIGINAL, turn_id, second + 1)
        + assistant("I replaced the output format.", turn_id, second + 2)
        + user("The old output must remain stable. Fix the replaced format.", turn_id, second + 3)
        + user("It still changed the output. Keep the existing format.", turn_id, second + 4)
        + command("git commit -m fix", f"[work {next(iter(pr.commits))[:8]}] fix", turn_id, second + 5)
        + turn(turn_id, second + 6, "task_complete")
    )


class FixtureModel:
    def decide(self, phase, data):
        events = data["events"]
        current = set(data["current"])
        if phase == "tasks":
            decisions = []
            for event in events:
                if event["id"] in current and event["text"] == ORIGINAL:
                    decisions.append({"operation": "open", "source": event["id"],
                                      "quote": "Keep the old output stable.", "description": "Keep output stable"})
            return {"decisions": decisions}
        request = next((e for e in events if e["text"] == ORIGINAL), None)
        action = next((e for e in events if "I replaced the output format." in e["text"]), None)
        corrections = [e["id"] for e in events if e["id"] in current and e["kind"] == "user_message"
                       and ("Fix the replaced format." in e["text"] or "It still changed" in e["text"])]
        if not request or not action or not corrections or not data["existing_tasks"]:
            return {"candidates": []}
        return {"candidates": [{
            "task_id": data["existing_tasks"][0]["id"], "attempts": [action["id"]],
            "corrections": corrections, "requirement": request["id"], "requirement_quote": "Keep the old output stable.",
            "evidence": [], "trigger": "user_correction", "label": "incorrect_design_decision",
            "summary": "Replaced the output format", "correction_summary": "Restore the existing format",
            "reason": "The new format broke the original output requirement",
        }]}


class Acceptance(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="dataset acceptance ")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "history root"
        self.captures = self.root / "sessions"
        self.captures.mkdir(parents=True)
        self.path = self.captures / "rollout.jsonl"
        self.output = self.root.parent / "dataset output.csv"
        self.github = FixtureGitHub(FIRST, SECOND)

    def scan_cli(self, *args, github=None, root=None):
        with (
            patch("agent_history_dataset_builder.cli.GhPullRequests", return_value=github or self.github),
            patch("agent_history_dataset_builder.cli.StructuredModel", return_value=FixtureModel()),
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):
            status = scan_main(["--codex-home", str(root or self.root), *args])
        self.assertEqual(status, 0)
        return json.loads(stdout.getvalue())

    def write(self, content):
        self.path.write_text(content, encoding="utf-8")
        return list(read_rollout(self.path))

    def hook_result(self, events):
        span, = [span for span in match_spans(events, self.github) if span.pr]
        request = next(e for e in span.events if e.prompt == ORIGINAL)
        action = next(e for e in span.events if e.kind == "assistant_message")
        correction = next(e for e in span.events if "Fix the replaced" in (e.prompt or ""))
        return {
            "decisions": [{"operation": "open", "source": ref(request), "quote": "Keep the old output stable.",
                           "description": "Keep output stable"}],
            "candidates": [{
                "task_source": ref(request), "task_quote": "Keep the old output stable.",
                "attempts": [ref(action)], "corrections": [ref(correction)],
                "requirement": ref(request), "requirement_quote": "Keep the old output stable.", "evidence": [],
                "trigger": "user_correction", "label": "incorrect_design_decision",
                "summary": "Changed output", "correction_summary": "Restore output", "reason": "Broke existing output",
            }],
        }

    def test_cuj_1_preview_has_source_span_and_leaves_output_absent(self):
        self.write(meta() + failure_turn())
        report = self.scan_cli("--pr", FIRST.url, "--dry-run", "--output", str(self.output))
        self.assertFalse(self.output.exists())
        self.assertEqual(report["spans"][0]["session_id"], "session-a")
        self.assertEqual(report["spans"][0]["last_turn"], TURN)
        self.assertEqual(report["spans"][0]["match_reasons"], ["work_commit"])
        row, = report["rows"]
        self.assertEqual(row["original_prompt"], ORIGINAL)
        self.assertEqual(row["failure_label"], "incorrect_design_decision")
        self.assertEqual(sum(ref["role"] == "correction" for ref in json.loads(row["evidence_refs"])), 2)

    def test_pr_matches_creation_and_commit_in_separate_sessions(self):
        self.write(meta() + failure_turn())
        separate = self.captures / "another.jsonl"
        separate.write_text(
            meta("session-pr-command") + turn("create", 1)
            + user("Open the PR", "create", 2)
            + command("gh pr create", FIRST.url, "create", 3), encoding="utf-8",
        )
        report = self.scan_cli("--pr", FIRST.url, "--dry-run")
        self.assertEqual(
            {span["session_id"]: span["match_reasons"] for span in report["spans"]},
            {"session-a": ["work_commit"], "session-pr-command": ["pr_created"]},
        )

    def test_disabled_and_truncated_publication_adapters_do_not_match(self):
        events = self.write(meta() + failure_turn())
        self.assertIsNone(match_spans(events, self.github, adapters=("gh",))[0].pr)
        self.assertIs(match_spans(events, self.github, adapters=("git",))[0].pr, FIRST)
        truncated = self.write(meta() + command(
            "gh pr create", FIRST.url + "\n... output truncated for persistence ...", TURN, 1,
        ))
        self.assertTrue(truncated[-1].output_truncated)
        self.assertIsNone(match_spans(truncated, self.github)[0].pr)
        self.assertTrue(review_context(truncated, self.github)["evidence_gaps"])

    def test_configured_publication_requires_a_successful_single_pr_result_and_confirmation(self):
        published = f"Updated pull request #1: {FIRST.url}"
        positive = self.write(meta() + command("review publish", published, TURN, 1))
        self.assertIsNone(match_spans(positive, self.github)[0].pr)
        adapter = ("review-publish",)
        span, = match_spans(positive, self.github, adapters=adapter, formats=TEST_FORMATS)
        self.assertIs(span.pr, FIRST)
        self.assertEqual(span.basis[0][0], "configured_publication")
        self.assertIsNone(match_spans(positive, self.github, SECOND, adapters=adapter, formats=TEST_FORMATS)[0].pr)
        for args, out, status in [
            ("review publish", FIRST.url, 0),
            ("review publish", f"Updated PR mention: {FIRST.url}", 0),
            ("review publish", f"Updated PR #2: {FIRST.url}", 0),
            ("review publish", f"{published}\nUpdated PR: {SECOND.url}", 0),
            ("review publish", published, 1),
            ("review publish --all", published, 0),
        ]:
            with self.subTest(args=args, out=out, status=status):
                events = self.write(meta() + command(args, out, TURN, 1, status))
                self.assertIsNone(match_spans(events, self.github, adapters=adapter, formats=TEST_FORMATS)[0].pr)

    def test_enterprise_command_variants_and_output_require_positive_evidence(self):
        pr = replace(FIRST, head_branch="first")
        github = FixtureGitHub(pr)
        options = {"adapters": tuple(TEST_FORMATS), "formats": TEST_FORMATS}
        branch_status = f"Updating PR for branch first [{pr.url}]"
        labeled_status = f"Review result: {pr.url}"
        for argv, output, reason in [
            ("review publish --only", branch_status, "configured_publication"),
            ("review publish", branch_status, "configured_publication"),
            ("review submit", labeled_status, "configured_publication"),
            ("review submit origin HEAD:refs/heads/first", labeled_status + " (updated)", "configured_publication"),
        ]:
            with self.subTest(argv=argv):
                events = self.write(meta() + command(argv, output, TURN, 1))
                self.assertIsNone(match_spans(events, github)[0].pr)
                with patch.object(github, "get", wraps=github.get) as get:
                    span, = match_spans(events, github, **options)
                    get.assert_called_once_with(pr.url)
                self.assertEqual(span.pr, pr)
                self.assertEqual(span.basis[0].reason, reason)
                self.assertEqual(span.basis[0].branch, "first")

        for argv, output, status in [
            ("review publish --only", f"Updating PR for branch unrelated [{pr.url}]", 0),
            ("review publish --only", f"{branch_status}\nCreating PR for branch other", 0),
            ("review publish --only", f"{branch_status}\nUpdating PR for branch other [{SECOND.url}]", 0),
            ("review publish --only", f"Usage: push\n{branch_status}", 0),
            ("review publish --only", f"{branch_status}\n... output truncated for persistence", 0),
            ("review publish --only", branch_status, 1),
            ("review publish --only --dry-run", branch_status, 0),
            ("review submit origin HEAD", labeled_status + " (dry-run)", 0),
            ("review submit origin HEAD", f"{labeled_status}\nReview result: {SECOND.url}", 0),
            ("review submit origin HEAD", pr.url, 0),
            ("review submit --help", labeled_status, 0),
            ("echo 'review submit origin HEAD'", labeled_status, 0),
            ("git push origin HEAD", labeled_status, 0),
            ("review submit origin HEAD\necho done", labeled_status, 0),
        ]:
            with self.subTest(argv=argv, output=output, status=status):
                events = self.write(meta() + command(argv, output, TURN, 1, status))
                self.assertTrue(all(span.pr is None for span in match_spans(events, github, **options)))
        events = self.write(meta() + command("review submit origin HEAD", labeled_status, TURN, 1))
        with patch.object(github, "get", side_effect=GitHubError("No authorization")):
            self.assertIsNone(match_spans(events, github, **options)[0].pr)

    def test_publication_branch_mapping_requires_an_exact_scope_and_full_branch(self):
        repo = Repo("code.example.com", "example", "project")
        pr = replace(FIRST, repo=repo, url=f"https://{repo.host}/example/project/pull/1",
                     head_branch="publisher/topic/first")
        published = command("review publish --only", f"Updating PR for branch topic/first [{pr.url}]",
                            TURN, 1, cwd="/other worktree")
        events = self.write(meta() + published)
        github = FixtureGitHub(pr)
        rule = PublicationBranchMapping(repo.host, "example/project", "review-publish", "publisher/")

        def matches(service, *rules):
            return match_spans(events, service, adapters=tuple(TEST_FORMATS), formats=TEST_FORMATS, branch_mappings=rules)[0]

        self.assertIsNone(matches(github).pr)
        span = matches(github, rule)
        self.assertEqual(span.pr, pr)
        self.assertEqual(span.basis[0].branch, "topic/first")
        self.assertEqual(span.basis[0].remote_branch, "publisher/topic/first")
        self.assertEqual(span.basis[0].branch_mapping, rule)
        for other_rule in [
            replace(rule, host="another.example.com"),
            replace(rule, repository="example/other"),
            replace(rule, adapter="review-submit"),
            replace(rule, remote_prefix="other/"),
        ]:
            self.assertIsNone(matches(github, other_rule).pr)
        for refname in (None, "publisher/first", "publisher/topic/first/extra", "extra/publisher/topic/first"):
            self.assertIsNone(matches(FixtureGitHub(replace(pr, head_branch=refname)), rule).pr)
        exact = matches(FixtureGitHub(replace(pr, head_branch="topic/first")), rule)
        self.assertIsNotNone(exact.pr)
        self.assertIsNone(exact.basis[0].branch_mapping)
        with patch.object(github, "get", return_value=replace(pr, repo=replace(repo, name="renamed"))):
            alias_rule = replace(rule, repository="example/renamed")
            self.assertIsNone(matches(github, alias_rule).pr)
        with patch("agent_history_dataset_builder.prs.subprocess.run") as run:
            self.assertIsNone(matches(GhPullRequests(), rule).pr)
            run.assert_not_called()

        for output, exit_code in [
            (f"Updating PR for branch topic/first [{pr.url}]\nCreating PR for branch other", 0),
            (f"Updating PR for branch topic/first [{pr.url}]\n... output truncated for persistence", 0),
            (f"Updating PR for branch topic/first [{pr.url}]", 1),
        ]:
            events = self.write(meta() + command("review publish --only", output, TURN, 1, exit_code,
                                                 cwd="/other worktree"))
            with patch.object(github, "get", wraps=github.get) as get:
                self.assertIsNone(matches(github, rule).pr)
                get.assert_not_called()

    def test_branch_mapping_config_rejects_invalid_and_conflicting_rules(self):
        path = self.root / "publication.json"
        rule = {
            "host": "Code.example.com", "repository": "Example/Project",
            "adapter": "review-submit", "remote_prefix": "publisher/",
        }
        for rules in [
            rule,
            [{**rule, "host": "https://code.example.com"}],
            [{**rule, "repository": "project"}],
            [{**rule, "adapter": "gh"}],
            [{**rule, "remote_prefix": ""}],
            [{**rule, "remote_prefix": "publisher"}],
            [{**rule, "remote_prefix": "publisher//"}],
            [{**rule, "remote_prefix": "../"}],
            [{**rule, "remote_prefix": "publisher.*/"}],
            [{**rule, "suffix_match": True}],
            [rule, {**rule, "host": "code.EXAMPLE.com", "remote_prefix": "another/"}],
        ]:
            with self.subTest(rules=rules):
                path.write_text(json.dumps({
                    "publication_adapters": ["gh", "review-submit"], "publication_branch_mappings": rules,
                    "publication_formats": {"review-submit": {
                        "argv_variants": [["review", "submit"]],
                        "output_formats": ["prefixed-url"], "line_prefix": "Review result:",
                    }},
                }), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_settings(path)
        path.write_text(json.dumps({"publication_branch_mappings": [rule]}), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_settings(path)

    def test_branch_mapping_is_shared_between_scan_and_review(self):
        path = self.root / "publication.json"
        rule = {
            "host": "github.com", "repository": "example/project",
            "adapter": "review-submit", "remote_prefix": "publisher/",
        }
        path.write_text(json.dumps({
            "publication_adapters": ["review-submit"], "publication_branch_mappings": [rule],
            "publication_formats": {"review-submit": {
                "argv_variants": [["review", "submit", "origin", "HEAD:refs/heads/first"]],
                "output_formats": ["prefixed-url"], "line_prefix": "Review result:",
            }},
        }), encoding="utf-8")
        pr = replace(FIRST, head_branch="publisher/first")
        github = FixtureGitHub(pr)
        base = meta() + user(ORIGINAL, TURN, 1) + assistant("I replaced the output format.", TURN, 2)
        base += user("The old output must remain stable. Fix the replaced format.", TURN, 3)
        child_result = self.hook_result(self.write(base + command("git commit -m fix", "[first aaaaaaaa] fix", TURN, 4)))
        events = self.write(base + command("review submit origin HEAD:refs/heads/first", f"Review result: {pr.url}", TURN, 4))
        report = self.scan_cli("--config", str(path), "--pr", pr.url, "--output", str(self.output), github=github)
        self.assertEqual(sum(span["pr_url"] == pr.url for span in report["spans"]), 1)
        self.assertEqual(report["publication"]["inserted"], 1)
        offline, = dataset.load(self.output)
        settings = load_settings(path)
        self.assertEqual(review_context(events, github, settings)["spans"][0]["pr"], pr.url)
        hook_output = self.output.with_name("review.csv")
        published = publish_review(events, child_result, github, hook_output, settings=settings)
        self.assertEqual(published["inserted"], 1)
        online, = dataset.load(hook_output)
        self.assertEqual(online["pr_link_evidence"], offline["pr_link_evidence"])
        link, = json.loads(offline["pr_link_evidence"])
        self.assertEqual(link["branch"], "first")
        self.assertEqual(link["remote_branch"], "publisher/first")
        self.assertEqual(link["branch_mapping"], rule)
        self.assertEqual(offline["failure_label"], "incorrect_design_decision")
        self.assertEqual(offline["human_verified"], "false")

    def test_publication_options_select_only_configured_variants_and_formats(self):
        path = self.root / "publication.json"
        config = {
            "publication_adapters": ["review-publish"],
            "publication_formats": {"review-publish": {
                "argv_variants": [["review", "publish", "--only"]],
                "output_formats": ["branch-update"],
            }},
            "github_hosts": ["Enterprise.example.com"],
        }
        path.write_text(json.dumps(config), encoding="utf-8")
        settings = load_settings(path)
        self.assertEqual(settings.github_hosts, ("enterprise.example.com",))
        for argv, output, matched in [
            ("review publish --only", f"Updating PR for branch first [{FIRST.url}]", True),
            ("review publish", f"Updating PR for branch first [{FIRST.url}]", False),
            ("review publish --only", f"Updated PR: {FIRST.url}", False),
        ]:
            with self.subTest(argv=argv, output=output):
                self.write(meta() + command(argv, output, TURN, 1))
                report = self.scan_cli("--dry-run", "--config", str(path))
                self.assertEqual(bool(report["spans"][0]["pr_url"]), matched)
        for invalid in [
            {"argv_variants": ["review publish --only"]},
            {"argv_variants": [["git", "push", "--all"]]},
            {"argv_variants": [["review", "publish", None]]},
            {"output_formats": ["https://.*"]},
            {"output_formats": ["unsupported-format"]},
            {"output_formats": ["prefixed-url"]},
            {"branch_change_variants": [["review", "publish", "--only"]]},
        ]:
            path.write_text(json.dumps({
                **config, "publication_formats": {"review-publish": {
                    **config["publication_formats"]["review-publish"], **invalid,
                }},
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_settings(path)

    def test_staged_newline_commit_batch_with_file_uri_uses_public_git_evidence(self):
        output = "[first aaaaaaaa] Fix output"
        workdir = "file:///old%20machine/worktree"
        for argv, matched in [
            ('git add -- "report output.py"\ngit commit -m "Fix output"', True),
            ('git add output.py\ngit add report.py\ngit commit -m "Fix output"', True),
            ('git add output.py\ngit commit -m "Fix output"\necho done', False),
            ('git add output.py && git commit -m "Fix output"', False),
            ('git add output.py\necho "git commit -m fix"', False),
            ('git add output.py\ngit commit -m "${message}"', False),
        ]:
            with self.subTest(argv=argv):
                events = self.write(meta() + command(argv, output, TURN, 1, cwd=workdir))
                span, = match_spans(events, self.github)
                self.assertEqual(bool(span.pr), matched)
                if matched:
                    self.assertEqual(span.worktree, "/old machine/worktree")

    def test_unmatched_earlier_span_supplies_request_and_requirement_without_its_edits(self):
        header = meta(history_mode="paginated")
        requirement = "Use a single output format for downstream clients."
        self.write(
            header + user(ORIGINAL, TURN, 1) + user(requirement, TURN, 2)
            + assistant("An edit from a different branch.", TURN, 3)
            + command("review next", "Switched branch", TURN, 4),
        )
        (self.captures / "resumed.jsonl").write_text(
            header + _record("turn_context", 5, turn_id="resumed", cwd="file:///other%20worktree",
                             git={"branch": "first"})
            + command("review checkout first", "Switched branch", "resumed", 6, cwd="/other worktree")
            + _record("event_msg", 7, type="item_completed", turn_id="resumed", item={
                "type": "FileChange", "id": "output-edit",
                "changes": {"output.py": "I replaced the output format."},
            })
            + user("The old output must remain stable. Fix the replaced format.", "resumed", 8)
            + command("review publish --only", f"Updating PR for branch first [{FIRST.url}]",
                      "resumed", 9, cwd="file:///other%20worktree")
            + command("review publish --only", f"Updating PR for branch first [{FIRST.url}]",
                      "resumed", 10, cwd="file:///other%20worktree"),
            encoding="utf-8",
        )
        case = self

        class EarlierRequestModel(FixtureModel):
            def decide(self, phase, data):
                case.assertFalse(any("different branch" in event["text"] for event in data["events"]))
                if phase == "tasks":
                    case.assertTrue(data["earlier_requests"])
                    return super().decide(phase, {**data, "current": data["earlier_requests"]})
                result = super().decide(phase, data)
                for candidate in result["candidates"]:
                    requirement_event = next(event for event in data["events"] if event["text"] == requirement)
                    case.assertTrue(requirement_event["context_only"])
                    attempt = next(event for event in data["events"] if event["id"] == candidate["attempts"][0])
                    case.assertNotIn("context_only", attempt)
                    candidate.update(requirement=requirement_event["id"], requirement_quote=requirement)
                if result["candidates"]:
                    correct = result["candidates"][0]
                    original = next(event for event in data["events"] if event["text"] == ORIGINAL)
                    result["candidates"].insert(0, {**correct, "requirement": original["id"]})
                return result

        rule = PublicationBranchMapping("github.com", "example/project", "review-publish", "publisher/")
        pr = replace(FIRST, head_branch="publisher/first")
        github = FixtureGitHub(pr)
        settings = Settings(publication_adapters=("review-publish",), publication_formats=TEST_FORMATS,
                            publication_branch_mappings=(rule,))
        report = scan([self.root], github, EarlierRequestModel(), pr_url=pr.url,
                      output=self.output, dry_run=False,
                      settings=settings)
        self.assertIsNone(report["spans"][0]["pr_url"])
        self.assertEqual(report["spans"][-1]["pr_url"], FIRST.url)
        self.assertEqual(report["publication"]["inserted"], 1)
        self.assertTrue(any("failed evidence checks" in gap["issue"] for gap in report["evidence_gaps"]))
        row, = dataset.load(self.output)
        self.assertEqual(row["original_prompt"], ORIGINAL)
        self.assertEqual(row["human_verified"], "false")
        refs = {entry["role"]: entry for entry in json.loads(row["evidence_refs"])}
        self.assertEqual(refs["requirement"]["quote"], requirement)
        self.assertNotEqual(refs["request"]["anchor"]["rollout_id"], refs["implementation"]["anchor"]["rollout_id"])
        self.assertEqual(refs["request"]["anchor"]["position"], 2)
        self.assertEqual(refs["requirement"]["anchor"]["position"], 3)
        self.assertEqual(refs["implementation"]["anchor"]["item_id"], "output-edit")
        self.assertEqual(refs["implementation"]["anchor"]["position"], 4)
        self.assertEqual(refs["correction"]["anchor"]["position"], 5)
        links = json.loads(row["pr_link_evidence"])
        self.assertEqual([link["anchor"]["position"] for link in links], [6, 7])
        self.assertEqual({link["branch"] for link in links}, {"first"})
        self.assertEqual({link["remote_branch"] for link in links}, {"publisher/first"})

    def test_resumed_page_preserves_requirements_and_edits_across_worktrees(self):
        page_header = meta(history_mode="paginated")
        requirement = "Retain the old format for downstream readers."
        self.write(page_header + user(ORIGINAL, TURN, 1) + user(requirement, TURN, 2))
        resumed = self.captures / "earlier-filename.jsonl"
        resumed.write_text(
            page_header + _record("turn_context", 3, turn_id="resumed", cwd="/other worktree")
            + _record("event_msg", 4, type="item_completed", turn_id="resumed", item={
                "type": "FileChange", "id": "edit", "changes": {"output.py": "I replaced the output format."},
            })
            + "".join(command("gh run view", "Late CI chatter", "resumed", second,
                              cwd="/ci checkout") for second in range(5, 12))
            + user("The old output must remain stable. Fix the replaced format.", "resumed", 12)
            + command("review publish", f"Updated PR: {FIRST.url}", "resumed", 13, cwd="/other worktree"),
            encoding="utf-8",
        )

        class RequirementsModel(FixtureModel):
            def decide(self, phase, data):
                answer = super().decide(phase, data)
                for candidate in answer.get("candidates", []):
                    source = next(event for event in data["events"] if event["text"] == requirement)
                    candidate.update(requirement=source["id"], requirement_quote=requirement)
                return answer

        settings = Settings(publication_adapters=("review-publish",), publication_formats=TEST_FORMATS, analysis=Limits(
            batch_size=1, max_events=5, lookback_events=4,
        ))
        report = scan([self.root], self.github, RequirementsModel(), pr_url=FIRST.url,
                      settings=settings, output=self.output, dry_run=False)
        row, = report["rows"]
        self.assertEqual(row["original_prompt"], ORIGINAL)
        self.assertTrue(self.output.is_file())
        refs = {entry["role"]: entry for entry in json.loads(row["evidence_refs"])}
        self.assertEqual(refs["requirement"]["quote"], requirement)
        self.assertNotEqual(refs["request"]["anchor"]["rollout_id"], refs["implementation"]["anchor"]["rollout_id"])
        self.assertEqual(refs["implementation"]["anchor"]["position"], 3)
        self.assertEqual(refs["implementation"]["anchor"]["item_id"], "edit")
        self.assertTrue(any("omitted" in gap["issue"] for gap in report["evidence_gaps"]))
        self.path.rename(self.captures / "new-name.jsonl")
        replay = scan([self.root], self.github, RequirementsModel(), pr_url=FIRST.url, settings=settings)
        self.assertEqual(replay["rows"][0]["attempt_id"], row["attempt_id"])

    def test_commit_in_another_worktree_uses_only_its_observed_remote(self):
        start = meta() + user(ORIGINAL, TURN, 1)
        commit = command("git -C ../other commit -m fix", "[work aaaaaaaa] fix", TURN, 3)
        unknown = self.write(start + commit)
        self.assertIsNone(match_spans(unknown, self.github)[0].pr)
        remote = command("git -C ../other remote get-url origin",
                         "ssh://git@github.com:22/example/project.git", TURN, 2)
        span, = match_spans(self.write(start + remote + commit), self.github)
        self.assertIs(span.pr, FIRST)
        self.assertEqual(span.worktree, "/old machine/other")
        self.assertTrue(any(event.prompt == ORIGINAL for event in span.events))

    def test_overlapping_pages_report_uncertain_order(self):
        page_header = meta(history_mode="paginated")
        self.write(page_header + user(ORIGINAL, TURN, 1) + assistant("First page", TURN, 3))
        (self.captures / "overlap.jsonl").write_text(
            page_header + user("Overlapping page", TURN, 2) + assistant("Second page", TURN, 4),
            encoding="utf-8",
        )
        report = self.scan_cli("--dry-run")
        self.assertEqual(len(report["spans"]), 2)
        self.assertIn("Could not establish rollout page order", [gap["issue"] for gap in report["evidence_gaps"]])

    def test_rollout_copies_keep_only_the_complete_prefix_or_report_a_conflict(self):
        page_header = meta(history_mode="paginated")
        prefix = page_header + user(ORIGINAL, TURN, 1)
        self.write(prefix)
        copy = self.captures / "copy.jsonl"
        copy.write_text(prefix + assistant("I replaced the output format.", TURN, 2), encoding="utf-8")
        report = self.scan_cli("--dry-run")
        self.assertEqual(len(report["spans"]), 1)
        self.assertEqual(report["spans"][0]["last_record"], 3)
        self.write(prefix + assistant("Different source text in a copy", TURN, 2))
        conflict = self.scan_cli("--dry-run")
        self.assertFalse(conflict["spans"])
        self.assertIn("Conflicting copies of the same rollout page",
                      [gap["issue"] for gap in conflict["evidence_gaps"]])

    def test_shell_batches_and_mcp_results_do_not_confirm_prs(self):
        events = self.write(
            meta() + command("gh pr create && echo done", FIRST.url, TURN, 1)
            + _record("event_msg", 2, type="item_completed", turn_id=TURN, item={
                "type": "McpToolCall", "id": "mcp1", "name": "publish", "result": {"url": FIRST.url},
            })
        )
        self.assertTrue(all(span.pr is None for span in match_spans(events, self.github)))

    def test_shared_json_settings_and_cli_model_overrides(self):
        path = self.root.parent / "local config.json"
        path.write_text(json.dumps({
            "output": "dataset output.csv",
            "publication_adapters": ["git"],
            "analysis": {"batch_size": 1, "lookback_by_kind": {"tool_result": 2}},
            "model": {
                "adapter": "command", "command": ["local-model"], "timeout_seconds": 7, "max_input_chars": 2048,
            },
            "hook": {"every": 3},
        }))
        settings = load_settings(path)
        self.assertEqual(settings.analysis.lookback_by_kind, {"tool_result": 2})
        with (
            patch("agent_history_dataset_builder.cli.GhPullRequests", return_value=self.github),
            patch("agent_history_dataset_builder.cli.StructuredModel", return_value=FixtureModel()) as model,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(scan_main([
                "--codex-home", str(self.root), "--config", str(path),
                "--model-command", "override --flag", "--model", "explicit-name",
                "--codex-bin", "explicit-codex", "--model-timeout", "9",
            ]), 0)
        model.assert_called_once_with(
            "explicit-codex", "explicit-name", ["override", "--flag"], timeout=9, max_input_chars=2048,
        )
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(hook_main(["config", "--config", str(path)]), 0)
        hooks = json.loads(stdout.getvalue())["hooks"]
        for name in ("UserPromptSubmit", "SubagentStop"):
            args = shlex.split(hooks[name][0]["hooks"][0]["command"])
            self.assertEqual(args[args.index("--output") + 1], str(self.output.resolve()))
            self.assertEqual(args[args.index("--config") + 1], str(path.resolve()))
        prompt = hooks["UserPromptSubmit"][0]["hooks"][0]["command"]
        self.assertIn("--every 3", prompt)

    def test_settings_reject_unsupported_adapters_and_validation_overrides(self):
        path = self.root.parent / "bad.json"
        for settings in (
            {"version": 2},
            {"publication_adapters": ["mcp"]},
            {"analysis": {"exact_quote_validation": False}},
            {"analysis": {"max_events": True}},
            {"analysis": {"lookback_by_kind": {"tool_result": -1}}},
            {"model": {"adapter": "command"}},
            {"model": {"max_input_chars": 0}},
            {"model": {"max_input_chars": True}},
            {"hook": {"every": 0}},
        ):
            with self.subTest(settings=settings):
                path.write_text(json.dumps(settings))
                with self.assertRaises(ValueError):
                    load_settings(path)

    def test_budget_omissions_and_model_timeout_are_reported(self):
        events = self.write(meta() + failure_turn())
        span, = match_spans(events, self.github)
        analysis = analyze_span(span, FixtureModel(), [], Limits(event_chars=5, context_chars=10, max_events=1))
        self.assertFalse(analysis.rows)
        self.assertTrue(any("clipped" in gap for gap in analysis.issues))
        self.assertTrue(any("omitted" in gap for gap in analysis.issues))
        self.assertTrue(any("model call skipped" in gap for gap in analysis.issues))
        lookback = analyze_span(span, FixtureModel(), [], Limits(batch_size=1, lookback_by_kind={"user_message": 0}))
        self.assertTrue(any("Lookback budget omitted" in gap for gap in lookback.issues))
        hook = review_context(events, self.github, Settings(analysis=Limits(max_events=1)))
        self.assertEqual(len(hook["spans"][0]["events"]), 1)
        self.assertTrue(any("omitted" in gap for gap in hook["evidence_gaps"]))
        with patch("agent_history_dataset_builder.model.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(["fixture-model"], 7)) as run:
            model = StructuredModel("unused", command=["fixture-model"], timeout=7)
            with self.assertRaises(ModelTimeout):
                model.decide("tasks", {})
            self.assertEqual(run.call_args.kwargs["timeout"], 7)
            analysis = analyze_span(span, model, [])
        self.assertTrue(any("timed out after 7s" in gap for gap in analysis.issues))

    def test_task_limit_omits_extra_tasks_from_model_context(self):
        events = self.write(meta() + failure_turn())
        span, = match_spans(events, self.github)
        request = next(event for event in events if event.prompt == ORIGINAL)
        prior = [Task(str(index), request, "Keep the old output stable.", 2, "Preserve output")
                 for index in range(2)]
        with patch.object(FixtureModel, "decide", side_effect=[{"decisions": []}, {"candidates": []}]) as decide:
            result = analyze_span(span, FixtureModel(), prior, Limits(task_limit=1))
        self.assertTrue(any("Task budget omitted 1 task(s)" in issue for issue in result.issues))
        self.assertTrue(all(len(call.args[1]["existing_tasks"]) == 1 for call in decide.call_args_list))

    def test_filled_context_budget_keeps_task_available_for_failure_analysis(self):
        logs = "".join(command("git status", f"status line {index}", TURN, 3) for index in range(650))
        events = self.write(
            meta() + user(ORIGINAL, TURN, 1) + assistant("I replaced the output format.", TURN, 2)
            + logs + user("The old output must remain stable. Fix the replaced format.", TURN, 4)
            + command("git commit -m fix", "[first aaaaaaaa] fix", TURN, 5)
        )
        span, = match_spans(events, self.github)

        class DetailedTaskModel(FixtureModel):
            def decide(self, phase, data):
                answer = super().decide(phase, data)
                for decision in answer.get("decisions", []):
                    decision["description"] = (
                        "Keep the existing output format stable for downstream clients. "
                        "Handle missing input by following the established behavior so callers "
                        "can continue reading the output without changes on their side. "
                        "Verify compatibility both with complete input and with missing input."
                    )
                return answer

        result = analyze_span(span, DetailedTaskModel(), [], Limits(context_chars=100_000))
        self.assertEqual(len(result.rows), 1, result.issues)
        self.assertEqual(result.rows[0]["original_prompt"], ORIGINAL)
        self.assertTrue(any("Context budget omitted" in issue for issue in result.issues))
        self.assertFalse(any("no visible tasks" in issue for issue in result.issues))

    def test_native_model_caps_the_complete_prompt_in_characters(self):
        settings = load_settings(None)
        self.assertEqual(settings.analysis.context_chars, 10_000_000)
        self.assertEqual(settings.analysis.max_events, 100_000)
        self.assertEqual(settings.model.max_input_chars, 1_048_576)
        data = {"events": [{"text": "🧰 café"}], "current": [], "existing_tasks": []}

        def respond(argv, prompt, environment, cwd):
            Path(argv[argv.index("--output-last-message") + 1]).write_text('{"decisions": []}')

        with patch.object(StructuredModel, "_run", side_effect=respond) as run:
            self.assertEqual(StructuredModel("unused").decide("tasks", data), {"decisions": []})
            prompt = run.call_args.args[1]
            self.assertIn("🧰 café", prompt)
            self.assertIn('"instructions"', prompt)
            self.assertIn('"schema"', prompt)
            self.assertGreater(len(prompt.encode("utf-8")), len(prompt))
            StructuredModel("unused", max_input_chars=len(prompt)).decide("tasks", data)
            run.reset_mock()
            with self.assertRaises(ModelInputTooLarge):
                StructuredModel("unused", max_input_chars=len(prompt) - 1).decide("tasks", data)
            run.assert_not_called()

        with patch.object(StructuredModel, "_run", return_value='{"decisions": []}') as run:
            model = StructuredModel("unused", command=["model"], max_input_chars=1)
            self.assertEqual(model.decide("tasks", data), {"decisions": []})
            self.assertEqual(run.call_args.args[0], ["model"])

    def test_oversized_batches_fall_back_and_split_in_source_order(self):
        needle = "Indivisible log"
        correction = "The old output must remain stable. Fix the replaced format."
        events = self.write(
            meta() + user(ORIGINAL, TURN, 1) + assistant("I replaced the output format.", TURN, 2)
            + user("Inspect the test result.", TURN, 3)
            + command("git status", needle, TURN, 4)
            + user(correction, TURN, 5) + command("git commit -m fix", "[first aaaaaaaa] fix", TURN, 6)
        )
        span, = match_spans(events, self.github)
        accepted = {"tasks": [], "failures": []}
        expected = [ref(event) for event in events if event.kind in {
            "user_message", "assistant_message", "command_execution",
        } and needle not in event.payload.get("item", {}).get("aggregated_output", "")]

        class SizeLimitedModel(FixtureModel):
            def decide(self, phase, data):
                if any(needle in record["text"] for record in data["events"]):
                    raise ModelInputTooLarge("Prompt exceeded test limit")
                accepted[phase].extend(data["current"])
                return super().decide(phase, data)

        limits = Limits(batch_size=1, lookback_events=4, lookback_by_kind={"command_execution": 0})
        result = analyze_span(span, SizeLimitedModel(), [], limits)
        self.assertEqual(accepted, {"tasks": expected, "failures": expected})
        for phase in accepted:
            issues = [issue for issue in result.issues if issue.startswith(phase)]
            self.assertTrue(any("bounded lookback" in issue for issue in issues))
            self.assertTrue(any("Split 2 current events" in issue for issue in issues))
            self.assertTrue(any("One current event" in issue for issue in issues))
        self.assertEqual(len(result.rows), 1, result.issues)

    def test_failed_model_batches_do_not_prevent_later_valid_publication(self):
        self.write(meta() + user("Inspect a separate issue.", TURN, 1) + failure_turn(TURN, 2))

        class IntermittentModel(FixtureModel):
            def decide(self, phase, data):
                window = {event["text"] for event in data["events"] if event["id"] in data["current"]}
                if "Inspect a separate issue." in window:
                    if phase == "tasks":
                        raise ModelError("Adapter unavailable")
                    return {"candidates": "invalid"}
                return super().decide(phase, data)

        report = scan([self.root], self.github, IntermittentModel(), output=self.output, dry_run=False,
                      settings=Settings(analysis=Limits(batch_size=1)))
        issues = [gap["issue"] for gap in report["evidence_gaps"]]
        self.assertTrue(any("tasks batch 1: Adapter unavailable" in issue for issue in issues))
        self.assertTrue(any("failures batch 1:" in issue and "candidates array" in issue for issue in issues))
        row, = dataset.load(self.output)
        self.assertEqual(row["original_prompt"], ORIGINAL)

    def test_task_quotes_are_unique_and_utf8_offsets_survive(self):
        _, request = self.write(meta() + user("🧰 Add tests. Then add docs.", TURN, 1))
        tasks = {}
        evidence = {ref(request): request}
        decisions = [
            {"operation": "open", "source": ref(request), "quote": quote, "description": quote}
            for quote in ("Add tests", "add docs", "Invented work")
        ]
        issues = apply_task_decisions(decisions, evidence, FIRST.identity, tasks)
        self.assertEqual(len(issues), 1)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(sorted(task.offset for task in tasks.values()), [5, 21])

    def test_cuj_2_backfill_partitions_prs_and_defers_mentions(self):
        switch = turn("turn-2", 10) + command("git switch second", "Switched", "turn-2", 11)
        unrelated = (
            turn("turn-3", 30)
            + command("git switch other", "Switched", "turn-3", 31)
            + user(f"Read {FIRST.url} for context", "turn-3", 32)
        )
        self.write(meta() + failure_turn() + switch + failure_turn("turn-2", 12, SECOND) + unrelated)
        report = self.scan_cli("--output", str(self.output))
        self.assertEqual(len(dataset.load(self.output)), 2)
        self.assertEqual({row["pr_url"] for row in report["rows"]}, {FIRST.url, SECOND.url})
        self.assertTrue(any(span["issue"] == "No confirmed PR evidence" for span in report["spans"]))

    def test_cuj_3_verify_and_replay_preserves_row_after_relocation(self):
        events = self.write(meta() + failure_turn())
        self.scan_cli("--pr", FIRST.url, "--output", str(self.output))
        row, = dataset.load(self.output)
        anchors = {tuple(event.anchor.__dict__.items()) for event in events}
        evidence = json.loads(row["evidence_refs"])
        self.assertTrue(all(tuple(entry["anchor"].items()) in anchors for entry in evidence))
        self.assertEqual(verify_main(["--output", str(self.output), row["attempt_id"]]), 0)
        saved = self.output.read_bytes()
        mtime = self.output.stat().st_mtime_ns
        moved = self.root.parent / "relocated history"
        shutil.copytree(self.root, moved)
        moved_capture = moved / "sessions" / "relocated.jsonl"
        (moved / "sessions" / self.path.name).rename(moved_capture)
        self.scan_cli("--pr", FIRST.url, "--output", str(self.output), root=moved)
        self.assertEqual(self.output.read_bytes(), saved)
        self.assertEqual(self.output.stat().st_mtime_ns, mtime)
        self.assertEqual(dataset.load(self.output)[0]["human_verified"], "true")

    def test_saved_row_links_to_attempt_thread_and_old_csv_can_be_updated(self):
        thread_id = "01a1de30-15b3-7912-b3ee-ae0b3b4b2749"
        self.write(meta(thread_id) + failure_turn())
        self.scan_cli("--pr", FIRST.url, "--output", str(self.output))
        row, = dataset.load(self.output)
        self.assertEqual(row["codex_thread_id"], thread_id)
        self.assertEqual(row["codex_thread_url"], f"codex://threads/{thread_id}")

        with self.output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=dataset.LEGACY_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(row)
        old_file = self.output.read_bytes()
        self.assertEqual(dataset.load(self.output), [row])
        self.assertEqual(self.output.read_bytes(), old_file)
        self.assertEqual(verify_main(["--output", str(self.output), row["attempt_id"]]), 0)
        self.assertEqual(dataset.load(self.output), [{**row, "human_verified": "true"}])

    def test_new_page_anchors_never_overwrite_a_verified_row(self):
        self.write(meta() + failure_turn())
        report = self.scan_cli("--pr", FIRST.url, "--output", str(self.output))
        candidate, = report["rows"]
        self.assertTrue(dataset.verify(self.output, candidate["attempt_id"]))
        verified, = dataset.load(self.output)

        refs = json.loads(candidate["evidence_refs"])
        for entry in refs:
            entry["anchor"]["rollout_id"] = "resumed-page"
        rescanned = {
            **candidate,
            "attempt_id": "page-aware-attempt",
            "task_id": "page-aware-task",
            "evidence_refs": json.dumps(refs),
        }
        self.assertEqual(dataset.publish(self.output, [rescanned])["inserted"], 1)
        previous, new = dataset.load(self.output)
        self.assertEqual(previous, verified)
        self.assertEqual(new["human_verified"], "false")

    def test_cuj_4_third_turn_requests_child_and_result_publishes(self):
        earlier = "".join(
            turn(f"warm-{n}", n * 3)
            + user("Warm up", f"warm-{n}", n * 3 + 1)
            + turn(f"warm-{n}", n * 3 + 2, "task_complete")
            for n in range(1, 3)
        )
        source = meta() + earlier + failure_turn(TURN, 15)
        events = self.write(source)
        payload = {
            "hook_event_name": "UserPromptSubmit", "session_id": "session-a", "turn_id": TURN,
            "transcript_path": str(self.path), "prompt": "Finish the work.",
        }
        first_record = next(e.position for e in events if e.turn_id == TURN and e.kind == "turn_started")
        prefix = [event for event in events if event.position <= first_record]
        self.assertFalse(review_due(payload, prefix, 5))
        response = prompt_response(payload, prefix, 3, self.github)
        self.assertEqual(response["hookSpecificOutput"]["hookEventName"], "UserPromptSubmit")
        self.assertIn("spawn one read-only subagent", response["hookSpecificOutput"]["additionalContext"])
        self.assertFalse(review_due({**payload, "agent_id": "child"}, prefix, 3))
        self.assertFalse(review_due(payload, events, 3))
        self.assertEqual(len(review_context(events, self.github)["spans"]), 1)
        result = {
            **self.hook_result(events), "protocol": REVIEW_PROTOCOL,
            "session": "session-a", "turn": TURN,
        }
        reply = receive_review({
            "hook_event_name": "SubagentStop", "session_id": "session-a",
            "transcript_path": str(self.path), "last_assistant_message": json.dumps(result),
        }, self.github, self.output)
        self.assertEqual(reply, {})
        self.assertEqual(len(dataset.load(self.output)), 1)

    def test_due_turn_feedback_resolves_to_the_recorded_user_anchor(self):
        earlier_turn = "earlier"
        correction = "The old output must remain stable. Fix the replaced format."
        source = (meta(history_mode="paginated") + turn(earlier_turn, 1) + user(ORIGINAL, earlier_turn, 2)
                  + assistant("I replaced the output format.", earlier_turn, 3)
                  + command("git commit -m fix", "[work aaaaaaaa] fix", earlier_turn, 4)
                  + turn(earlier_turn, 5, "task_complete"))
        payload = {"hook_event_name": "UserPromptSubmit", "session_id": "session-a",
                   "turn_id": TURN, "prompt": correction, "transcript_path": str(self.path)}
        self.write(source)
        with (
            patch("agent_history_dataset_builder.hook.GhPullRequests", return_value=self.github),
            patch("sys.stdin", io.StringIO(json.dumps(payload))),
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):
            self.assertEqual(hook_main(["prompt", "--every", "2", "--output", str(self.output)]), 0)
        review = json.loads(stdout.getvalue())["hookSpecificOutput"]["additionalContext"]
        self.assertIn(f"incoming:{TURN}", review)
        events = page_anchors(self.write(source + turn(TURN, 6) + user(correction, TURN, 7)))
        result = self.hook_result(events)
        self.assertIn(result["decisions"][0]["source"], review)
        result["candidates"][0]["corrections"] = [f"incoming:{TURN}"]
        receive_review({
            "hook_event_name": "SubagentStop", "session_id": "session-a", "transcript_path": str(self.path),
            "last_assistant_message": json.dumps({
                **result, "protocol": REVIEW_PROTOCOL, "session": "session-a", "turn": TURN,
            }),
        }, self.github, self.output)
        row, = dataset.load(self.output)
        refs = json.loads(row["evidence_refs"])
        correction_ref, = [entry for entry in refs if entry["role"] == "correction"]
        self.assertEqual(correction_ref["anchor"]["item_id"], f"u-{TURN}-7")

    def test_hook_ignores_review_only_and_subagent_turns(self):
        events = self.write(
            meta() + turn("synthetic", 1)
            + user(REVIEW_PREFIX + "\nSynthetic instruction", "synthetic", 2)
            + turn("synthetic", 3, "task_complete")
            + turn("sub", 4, root_turn_id="parent") + user("Subagent prompt", "sub", 5)
            + turn("sub", 6, "task_complete", root_turn_id="parent")
            + turn(TURN, 7)
        )
        payload = {"hook_event_name": "UserPromptSubmit", "session_id": "session-a", "turn_id": TURN,
                   "prompt": "Real input"}
        self.assertTrue(review_due(payload, events, 1))
        self.assertFalse(review_due(payload, events, 2))

    def test_cuj_6_no_findings_never_opens_writer(self):
        events = self.write(meta() + failure_turn())
        result = {"status": "no_findings"}
        with patch("agent_history_dataset_builder.hook.dataset.publish") as writer:
            self.assertEqual(publish_review(events, result, self.github, self.output), result)
            self.assertFalse(self.output.exists())
            self.output.write_bytes(b"existing file")
            before = self.output.stat().st_mtime_ns
            self.assertEqual(publish_review(events, result, self.github, self.output), result)
            self.assertEqual(self.output.read_bytes(), b"existing file")
            self.assertEqual(self.output.stat().st_mtime_ns, before)
            writer.assert_not_called()

    def test_cuj_7_late_pr_recovery_and_replay(self):
        events = self.write(meta() + failure_turn())
        result = self.hook_result(events)
        missing = publish_review(events, result, FixtureGitHub(), self.output)
        self.assertEqual(missing["inserted"], 0)
        self.assertFalse(self.output.exists())
        self.scan_cli("--output", str(self.output))
        row, = dataset.load(self.output)
        replay = publish_review(events, result, self.github, self.output)
        self.assertEqual(replay["unchanged"], 1)
        self.assertEqual(dataset.load(self.output)[0]["attempt_id"], row["attempt_id"])
        self.assertEqual(len(dataset.load(self.output)), 1)

    def test_review_rejects_requirement_only_added_after_attempt(self):
        events = self.write(meta() + failure_turn())
        result = self.hook_result(events)
        correction = next(event for event in events if "Fix the replaced" in (event.prompt or ""))
        result["candidates"][0].update(requirement=ref(correction), requirement_quote="Fix the replaced format.")
        report = publish_review(events, result, self.github, self.output)
        self.assertEqual(report["inserted"], 0)
        self.assertTrue(report["issues"])
        self.assertFalse(self.output.exists())

    def test_csv_concurrent_updates_and_atomic_failure(self):
        self.write(meta() + failure_turn())
        row, = self.scan_cli("--pr", FIRST.url, "--dry-run")["rows"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: dataset.publish(self.output, [row]), range(2)))
        self.assertEqual(sum(result["inserted"] for result in results), 1)
        before = self.output.read_bytes()
        with patch("agent_history_dataset_builder.dataset.os.replace", side_effect=OSError("failed replace")):
            with self.assertRaises(OSError):
                dataset.verify(self.output, row["attempt_id"])
        self.assertEqual(before, self.output.read_bytes())
        self.assertEqual(list(self.output.parent.glob(f".{self.output.name}.*")), [])


if __name__ == "__main__":
    unittest.main()
