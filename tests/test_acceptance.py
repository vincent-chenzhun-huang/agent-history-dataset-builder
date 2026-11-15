"""Exercise the CUJs through the CLI and review boundary with scripted adapters."""

import contextlib
import csv
import io
import json
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from agent_history_dataset_builder import dataset
from agent_history_dataset_builder.analysis import apply_task_decisions, ref
from agent_history_dataset_builder.cli import scan_main, verify_main
from agent_history_dataset_builder.hook import (
    REVIEW_PROTOCOL, prompt_response, publish_review, receive_review, review_context, review_due,
)
from agent_history_dataset_builder.prs import PullRequest, Repo, match_spans
from agent_history_dataset_builder.transcripts import REVIEW_PREFIX, read_rollout


REPO = Repo("github.com", "example", "project")
FIRST = PullRequest("https://github.com/example/project/pull/1", REPO, 1, frozenset({"a" * 40}), "1")
SECOND = PullRequest("https://github.com/example/project/pull/2", REPO, 2, frozenset({"b" * 40}), "1")
ORIGINAL = "  Keep the old output stable.\nHandle missing input.\n"
TURN = "turn-1"


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


def meta(session="session-a"):
    return _record("session_meta", 0, id=session, cwd="/old machine/worktree", git={
        "branch": "first", "repository_url": "git@github.com:example/project.git",
    })


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


def command(name, output, turn_id, second, exit_code=0):
    return _record("event_msg", second, type="item_completed", turn_id=turn_id, item={
        "type": "CommandExecution", "id": f"cmd-{turn_id}-{second}", "command": ["sh", "-c", name],
        "cwd": "/old machine/worktree", "aggregated_output": output, "exit_code": exit_code,
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
        source = (meta() + turn(earlier_turn, 1) + user(ORIGINAL, earlier_turn, 2)
                  + assistant("I replaced the output format.", earlier_turn, 3)
                  + command("git commit -m fix", "[work aaaaaaaa] fix", earlier_turn, 4)
                  + turn(earlier_turn, 5, "task_complete"))
        payload = {"hook_event_name": "UserPromptSubmit", "session_id": "session-a",
                   "turn_id": TURN, "prompt": correction}
        review = prompt_response(payload, self.write(source), 2, self.github)
        self.assertIn(f"incoming:{TURN}", review["hookSpecificOutput"]["additionalContext"])
        events = self.write(source + turn(TURN, 6) + user(correction, TURN, 7))
        result = self.hook_result(events)
        result["candidates"][0]["corrections"] = [f"incoming:{TURN}"]
        self.assertEqual(publish_review(events, result, self.github, self.output, incoming_turn=TURN)["inserted"], 1)
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
