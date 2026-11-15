"""Read Codex rollout records without depending on the checkout or rollout path."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO
from uuid import UUID


REVIEW_PREFIX = "[agent-history-dataset review continuation]"


def codex_thread_fields(session_id: str) -> dict[str, str]:
    """Identify the stored thread and link to it when its ID is a Codex UUID."""
    url = ""
    try:
        thread_id = str(UUID(session_id))
    except ValueError:
        pass
    else:
        if thread_id == session_id.lower():
            url = f"codex://threads/{thread_id}"
    return {"codex_thread_id": session_id, "codex_thread_url": url}


class RolloutError(ValueError):
    """A rollout cannot be read or identified reliably."""


@dataclass(frozen=True)
class Anchor:
    session_id: str
    rollout_id: str
    record_type: str
    item_id: str | None
    position: int | None


@dataclass(frozen=True)
class Event:
    anchor: Anchor
    position: int
    kind: str
    turn_id: str | None
    timestamp: str | None
    payload: dict[str, Any]
    call_id: str | None = None
    call_anchor: Anchor | None = None
    output_truncated: bool = False
    gap: str | None = None

    @property
    def prompt(self) -> str | None:
        """Return the recorded user text. Attachments remain in the payload."""
        if self.kind != "user_message":
            return None
        item = self.payload.get("item")
        if not isinstance(item, dict):
            message = self.payload.get("message")
            return message if isinstance(message, str) else None
        parts = item.get("content")
        if not isinstance(parts, list):
            return None
        text = [
            part["text"]
            for part in parts
            if isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        ]
        return "".join(text) if text else None


_TURN_EVENTS = {
    "task_started": "turn_started",
    "turn_started": "turn_started",
    "task_complete": "turn_completed",
    "turn_completed": "turn_completed",
    "turn_aborted": "turn_aborted",
}
_TOOL_CALLS = {
    "function_call",
    "custom_tool_call",
    "FunctionCall",
    "CustomToolCall",
}
_TOOL_RESULTS = {
    "function_call_output",
    "custom_tool_call_output",
    "FunctionCallOutput",
    "CustomToolCallOutput",
}
_COMPLETED_ITEMS = {
    "AgentMessage": "assistant_message",
    "CommandExecution": "command_execution",
    "ContextCompaction": "compaction",
    "FileChange": "file_change",
    "McpToolCall": "mcp_tool_call",
}
_OUTPUT_FIELDS = {
    "tool_result": ("output",),
    "command_execution": ("aggregated_output", "formatted_output", "stdout", "stderr"),
    "mcp_tool_call": ("result",),
}
_TRUNCATION = re.compile(
    r"output truncated for persistence|"
    r"(?:…|\.{3})\s*(?:[\d,]+\s+tokens\s+truncated|output truncated)",
    re.IGNORECASE,
)


def discover_rollouts(history_roots: Iterable[Path]) -> Iterator[Path]:
    """Find saved rollouts below caller-supplied Codex history roots."""
    seen: set[Path] = set()
    for root in history_roots:
        root = root.expanduser()
        if not root.is_dir():
            raise RolloutError(f"History root is not a directory: {root}")
        for name in ("sessions", "archived_sessions"):
            for path in sorted((root / name).rglob("*.jsonl*")):
                if path.name.endswith((".jsonl", ".jsonl.zst")) and path.is_file():
                    resolved = path.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        yield resolved


def read_rollout(
    path: Path, *, active: bool = False, after_record: int = 0
) -> Iterator[Event]:
    """Stream events after a physical record position.

    Rebuild turn and tool-call context from the prefix. For an active JSONL
    file, wait for a newline before exposing its last record to the caller.
    """
    if after_record < 0:
        raise ValueError("after_record must be nonnegative")

    session_id: str | None = None
    rollout_id: str | None = None
    current_turn: str | None = None
    calls: dict[str, Anchor] = {}
    seen_items: set[tuple[str, str]] = set()

    with _open_rollout(path) as stream:
        for position, line in enumerate(stream, start=1):
            if active and not line.endswith(b"\n"):
                break
            if not line.strip():
                continue

            record = _record(line)
            if record is None:
                if session_id is None or rollout_id is None:
                    raise RolloutError(f"Missing session header in {path}")
                if position > after_record:
                    yield Event(
                        Anchor(session_id, rollout_id, "gap", None, position),
                        position,
                        "gap",
                        current_turn,
                        None,
                        {},
                        gap="Malformed rollout record",
                    )
                continue

            record_type, payload = record["type"], record["payload"]
            if session_id is None:
                if record_type != "session_meta":
                    raise RolloutError(f"Missing session header in {path}")
                session_id = _string(payload.get("id")) or _string(payload.get("session_id"))
                if session_id is None:
                    raise RolloutError(f"Missing session ID in {path}")
                rollout_id = _string(payload.get("rollout_id")) or session_id

            kind, item = _event_data(record_type, payload)
            turn_id = _turn_id(payload) or current_turn
            if kind in {"turn_started", "turn_context"}:
                current_turn = turn_id
            if kind in {"turn_completed", "turn_aborted"} and turn_id == current_turn:
                current_turn = None

            item_id = _string(item.get("id")) if record_type != "session_meta" else None
            anchor_type = f"{record_type}/{kind}"
            anchor_position: int | None = position
            if item_id is not None:
                key = (anchor_type, item_id)
                if key not in seen_items:
                    anchor_position = None
                    seen_items.add(key)
            anchor = Anchor(session_id, rollout_id, anchor_type, item_id, anchor_position)
            call_id = _string(item.get("call_id")) if kind in {"tool_call", "tool_result"} else None
            if kind == "tool_call" and call_id is not None:
                calls[call_id] = anchor
            if position > after_record:
                event = Event(
                    anchor=anchor,
                    position=position,
                    kind=kind,
                    turn_id=turn_id,
                    timestamp=_string(record.get("timestamp")),
                    payload=payload,
                    call_id=call_id,
                    call_anchor=calls.get(call_id) if kind == "tool_result" and call_id else None,
                    output_truncated=any(
                        _is_truncated(item.get(field)) for field in _OUTPUT_FIELDS.get(kind, ())
                    ),
                )
                if event.prompt and event.prompt.startswith(REVIEW_PREFIX + "\n"):
                    event = Event(anchor, position, "review_control", turn_id, event.timestamp, payload)
                yield event


@contextmanager
def _open_rollout(path: Path) -> Iterator[BinaryIO]:
    if not path.name.endswith(".zst"):
        with path.open("rb") as stream:
            yield stream
        return

    try:
        process = subprocess.Popen(
            ["zstd", "-q", "-d", "-c", "--", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError as error:
        raise RolloutError("Install zstd to read compressed rollouts") from error
    with process:
        assert process.stdout is not None
        yield process.stdout
        if process.wait() != 0:
            raise RolloutError(f"Could not decompress rollout: {path}")


def _record(line: bytes) -> dict[str, Any] | None:
    try:
        record = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if (
        isinstance(record, dict)
        and isinstance(record.get("type"), str)
        and isinstance(record.get("payload"), dict)
    ):
        return record
    return None


def _event_data(
    record_type: str, payload: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    subtype = _string(payload.get("type"))
    item = payload
    if record_type == "event_msg":
        if subtype == "user_message":
            return "user_message", item
        if subtype == "agent_message":
            return "assistant_message", item
        if subtype in _TURN_EVENTS:
            return _TURN_EVENTS[subtype], item
        nested = payload.get("item")
        if subtype == "item_completed" and isinstance(nested, dict):
            item = nested
            subtype = _string(item.get("type"))
            if subtype in {"UserMessage", "user_message"}:
                return "user_message", item
            if subtype in _COMPLETED_ITEMS:
                return _COMPLETED_ITEMS[subtype], item
    elif record_type == "response_item" and subtype == "message":
        # Model-facing user messages can also contain injected context.
        role = payload.get("role")
        return ("assistant_message" if role == "assistant" else "context_message"), item

    if subtype in _TOOL_CALLS:
        return "tool_call", item
    if subtype in _TOOL_RESULTS:
        return "tool_result", item
    return record_type, item


def _turn_id(payload: dict[str, Any]) -> str | None:
    metadata = payload.get("internal_chat_message_metadata_passthrough")
    return _string(payload.get("turn_id")) or (
        _string(metadata.get("turn_id")) if isinstance(metadata, dict) else None
    )


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _is_truncated(output: Any) -> bool:
    if isinstance(output, str):
        return bool(_TRUNCATION.search(output))
    if isinstance(output, dict):
        return output.get("truncated") is True or any(
            _is_truncated(value) for value in output.values()
        )
    if isinstance(output, list):
        return any(_is_truncated(value) for value in output)
    return False
