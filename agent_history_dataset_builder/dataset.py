"""Publish and verify the portable CSV dataset."""

from __future__ import annotations

import csv
import fcntl
import json
import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .transcripts import codex_thread_fields


LEGACY_COLUMNS = (
    "attempt_id", "task_id", "pr_url", "pr_link_evidence", "task_description",
    "original_prompt", "task_started_at", "task_ended_at", "attempt_summary",
    "attempt_started_at", "attempt_ended_at", "detection_trigger", "correction_summary",
    "correction_started_at", "correction_ended_at", "failure_label", "failure_reason",
    "evidence_refs", "attempt_head_sha", "recorded_head_sha", "recorded_worktree_dirty",
    "created_at", "human_verified",
)
COLUMNS = (*LEGACY_COLUMNS, "codex_thread_id", "codex_thread_url")


def load(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = tuple(reader.fieldnames or ())
            if columns not in (COLUMNS, LEGACY_COLUMNS):
                raise ValueError("Unexpected CSV dataset columns")
            rows = list(reader)
            if any(None in row or None in row.values() for row in rows):
                raise ValueError("Malformed CSV dataset row")
            ids = [row["attempt_id"] for row in rows]
            if any(not value for value in ids) or len(ids) != len(set(ids)):
                raise ValueError("Dataset attempt IDs must be present and unique")
            if columns == LEGACY_COLUMNS:
                for row in rows:
                    try:
                        evidence = _refs(row)
                    except ValueError:
                        evidence = []
                    implementation = next(
                        (entry.get("anchor") for entry in evidence if entry.get("role") == "implementation"),
                        None,
                    )
                    session_id = implementation.get("session_id") if isinstance(implementation, dict) else None
                    row.update(codex_thread_fields(session_id if isinstance(session_id, str) else ""))
            return rows
    except FileNotFoundError:
        return []


def _refs(row: dict, field: str = "evidence_refs") -> list[dict]:
    values = json.loads(row[field])
    if not isinstance(values, list) or any(not isinstance(value, dict) for value in values):
        raise ValueError(f"Invalid {field} in dataset")
    return values


def _anchor_key(ref: dict) -> str:
    return json.dumps(ref.get("anchor"), sort_keys=True, separators=(",", ":"))


def _ref_key(ref: dict) -> tuple[str, str]:
    return ref.get("role", ""), _anchor_key(ref)


def _pr_identity(row: dict) -> str:
    return next((item["pr_id"] for item in _refs(row, "pr_link_evidence") if item.get("pr_id")), row["pr_url"])


def _action_anchors(row: dict) -> set[str]:
    return {_anchor_key(item) for item in _refs(row) if item.get("role") == "implementation"}


def _request_anchors(row: dict) -> set[str]:
    return {_anchor_key(item) for item in _refs(row) if item.get("role") == "request"}


def _task_anchors(row: dict) -> set[tuple[str, object]]:
    return {
        (_anchor_key(item), item.get("outcome_offset"))
        for item in _refs(row)
        if item.get("role") == "request"
    }


def _overlaps(first: dict, other: dict) -> bool:
    times = [row[column] for row in (first, other) for column in ("attempt_started_at", "attempt_ended_at")]
    return all(times) and max(times[0], times[2]) <= min(times[1], times[3])


def _rows_changed(existing: list[dict], candidates: list[dict]) -> tuple[list[dict], dict]:
    rows = [dict(row) for row in existing]
    report = {"inserted": 0, "updated": 0, "unchanged": 0, "conflicts": []}
    for candidate in candidates:
        candidate = dict(candidate)
        same_pr = [row for row in rows if _pr_identity(row) == _pr_identity(candidate)]
        same_task = [row for row in same_pr if _task_anchors(row) & _task_anchors(candidate)]
        task_ids = {row["task_id"] for row in same_task}
        if len(task_ids) == 1:
            candidate["task_id"] = next(iter(task_ids))
        actions = _action_anchors(candidate)
        matches = [
            row for row in same_pr
            if row["attempt_id"] == candidate["attempt_id"]
            or actions & _action_anchors(row)
        ]
        if not matches:
            could_match = any(
                _request_anchors(row) & _request_anchors(candidate) and _overlaps(row, candidate)
                for row in same_pr
            )
            if could_match or len(task_ids) > 1:
                report["conflicts"].append(candidate["attempt_id"])
                continue
            new_row = {column: candidate.get(column, "") for column in COLUMNS}
            new_row["created_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            new_row["human_verified"] = "false"
            rows.append(new_row)
            report["inserted"] += 1
            continue
        if len(matches) != 1 or matches[0]["task_id"] != candidate["task_id"]:
            report["conflicts"].append(candidate["attempt_id"])
            continue
        current = matches[0]
        old = _refs(current)
        known = {_ref_key(item) for item in old}
        added = [
            item for item in _refs(candidate)
            if item.get("role") != "request" and _ref_key(item) not in known
        ]
        if current["human_verified"] == "true" or not added:
            report["unchanged"] += 1
            continue
        if current["failure_label"] != candidate["failure_label"]:
            report["conflicts"].append(current["attempt_id"])
            continue
        current["evidence_refs"] = json.dumps([*old, *added], ensure_ascii=False)
        for column in ("attempt_started_at", "correction_started_at"):
            times = [value for value in (current[column], candidate[column]) if value]
            if times:
                current[column] = min(times)
        for column in ("attempt_ended_at", "correction_ended_at", "task_ended_at"):
            current[column] = max(current[column], candidate[column])
        report["updated"] += 1
    return rows, report


@contextmanager
def _locked_directory(path: Path) -> Iterator[int]:
    # The directory inode survives atomic replacement and requires no lock file.
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield descriptor
    finally:
        os.close(descriptor)


def _replace(path: Path, rows: list[dict], directory_fd: int) -> None:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", encoding="utf-8",
            newline="", mode="w", delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            if path.exists():
                os.fchmod(stream.fileno(), stat.S_IMODE(path.stat().st_mode))
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        os.fsync(directory_fd)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def publish(path: Path, candidates: list[dict]) -> dict:
    if not candidates:
        return {"inserted": 0, "updated": 0, "unchanged": 0, "conflicts": []}
    path = path.expanduser().resolve()
    with _locked_directory(path) as descriptor:
        rows, report = _rows_changed(load(path), candidates)
        if report["inserted"] or report["updated"]:
            _replace(path, rows, descriptor)
        return report


def verify(path: Path, attempt_id: str) -> bool:
    path = path.expanduser().resolve()
    with _locked_directory(path) as descriptor:
        rows = load(path)
        for row in rows:
            if row["attempt_id"] != attempt_id:
                continue
            if row["human_verified"] != "true":
                row["human_verified"] = "true"
                _replace(path, rows, descriptor)
            return True
    return False


def preview(path: Path | None, candidates: list[dict]) -> dict:
    return _rows_changed(load(path) if path is not None else [], candidates)[1]
