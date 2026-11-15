import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from agent_history_dataset_builder.transcripts import (
    RolloutError,
    discover_rollouts,
    read_rollout,
)


def line(record_type, **payload):
    return (json.dumps({"type": record_type, "payload": payload}) + "\n").encode()


def event(subtype, **fields):
    return line("event_msg", type=subtype, **fields)


def response(subtype, **fields):
    return line("response_item", type=subtype, **fields)


def item(subtype, *, turn_id="turn-a", **fields):
    return event("item_completed", turn_id=turn_id, item={"type": subtype, **fields})


HEADER = line("session_meta", id="session-a", rollout_id="rollout-a")


class TranscriptReaderTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="codex history ")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "sessions" / "new" / "capture.jsonl"
        self.path.parent.mkdir(parents=True)

    def test_discovery_uses_supplied_roots_and_archives(self):
        archive = self.root / "archived_sessions" / "old.jsonl.zst"
        archive.parent.mkdir()
        for path in (self.path, archive, archive.with_suffix(".bak")):
            path.touch()

        self.assertEqual(
            list(discover_rollouts([self.root, self.root])),
            [self.path.resolve(), archive.resolve()],
        )

    def test_preserves_user_text_and_steering_in_source_order(self):
        prompt = "  Original request 👋\nwith exact whitespace\n"
        attachment = {"type": "local_image", "path": "/machine-a/attachment.png"}
        self.path.write_bytes(
            HEADER
            + event("task_started", turn_id="turn-a")
            + response(
                "message", role="user", content=[{"type": "input_text", "text": "Injected context"}]
            )
            + item("UserMessage", id="request-1", content=[{"type": "text", "text": prompt}, attachment])
            + item("UserMessage", id="request-2", content=[{"type": "text", "text": "Use the helper."}])
            + event("task_complete", turn_id="turn-a")
        )

        events = list(read_rollout(self.path))
        self.assertEqual(
            [record.kind for record in events],
            ["session_meta", "turn_started", "context_message", "user_message", "user_message", "turn_completed"],
        )
        user_events = [record for record in events if record.kind == "user_message"]
        self.assertEqual([record.prompt for record in user_events], [prompt, "Use the helper."])
        self.assertEqual([record.turn_id for record in user_events], ["turn-a", "turn-a"])
        self.assertEqual(user_events[0].payload["item"]["content"][1], attachment)
        self.assertEqual(user_events[0].anchor.item_id, "request-1")

    def test_active_reader_replays_context_for_the_completed_tail(self):
        call = response("function_call", id="tool-a", call_id="call-a", name="command")
        result = response(
            "function_call_output", call_id="call-a", output="… 101 tokens truncated …"
        )
        self.path.write_bytes(HEADER + event("task_started", turn_id="turn-a") + call + result[:10])

        initial = list(read_rollout(self.path, active=True))
        self.assertEqual(initial[-1].kind, "tool_call")
        with self.path.open("ab") as stream:
            stream.write(result[10:])

        (tail,) = list(read_rollout(self.path, active=True, after_record=initial[-1].position))
        self.assertEqual(tail.call_anchor, initial[-1].anchor)
        self.assertEqual(tail.turn_id, "turn-a")
        self.assertTrue(tail.output_truncated)
        self.assertEqual(list(read_rollout(self.path, active=True, after_record=tail.position)), [])

    def test_native_command_and_legacy_prompt_are_available_as_evidence(self):
        self.path.write_bytes(
            HEADER
            + event("user_message", message="Legacy prompt\n")
            + item(
                "CommandExecution",
                id="cmd-a",
                command="git status",
                exit_code=0,
                aggregated_output="... command output truncated for persistence ...",
            )
            + item("FileChange", id="edit-a", status="completed", changes={})
        )

        _, prompt, command, edit = read_rollout(self.path)
        self.assertEqual(prompt.prompt, "Legacy prompt\n")
        self.assertEqual(command.kind, "command_execution")
        self.assertEqual(command.payload["item"]["exit_code"], 0)
        self.assertTrue(command.output_truncated)
        self.assertEqual(edit.kind, "file_change")

    def test_preserves_gap_and_final_complete_record_in_saved_rollout(self):
        self.path.write_bytes(HEADER + b'{broken\n' + event("task_started", turn_id="turn-a").rstrip(b"\n"))

        _, gap, start = read_rollout(self.path)
        self.assertEqual(gap.kind, "gap")
        self.assertEqual(gap.anchor.position, 2)
        self.assertEqual(start.position, 3)

    def test_reused_item_ids_have_distinct_anchors(self):
        self.path.write_bytes(
            HEADER
            + response("message", role="assistant", id="message-a")
            + item("AgentMessage", id="message-a")
            + item("AgentMessage", id="message-a")
        )

        _, first, second, third = read_rollout(self.path)
        self.assertEqual(len({first.anchor, second.anchor, third.anchor}), 3)
        self.assertIsNone(second.anchor.position)
        self.assertEqual(third.anchor.position, third.position)
        self.assertEqual(list(read_rollout(self.path, after_record=second.position)), [third])

    def test_rejects_rollouts_without_a_session_header(self):
        self.path.write_bytes(event("task_started", turn_id="turn-a"))

        with self.assertRaisesRegex(RolloutError, "Missing session header"):
            list(read_rollout(self.path))

    @unittest.skipUnless(shutil.which("zstd"), "zstd is required for compressed rollouts")
    def test_moved_compressed_rollout_has_identical_anchors(self):
        self.path.write_bytes(HEADER + event("user_message", message="First\n"))
        expected = list(read_rollout(self.path))
        moved = self.root / "copied from another machine" / "renamed.jsonl.zst"
        moved.parent.mkdir()
        subprocess.run(
            ["zstd", "-q", "-o", str(moved), "--", str(self.path)],
            check=True,
        )
        self.assertEqual(list(read_rollout(moved)), expected)


if __name__ == "__main__":
    unittest.main()
