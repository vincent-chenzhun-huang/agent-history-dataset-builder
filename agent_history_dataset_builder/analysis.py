"""Select evidence for the model and validate its decisions against transcripts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Protocol

from .prs import WorkSpan
from .transcripts import Event, codex_thread_fields


LABELS = (
    "known_rules_violation",
    "code_smell_precedent",
    "code_smell_invented",
    "incorrect_design_decision",
    "missing_local_repo_context",
    "missing_cross_system_dependency_knowledge",
)
_IMPLEMENTATION = {"assistant_message", "command_execution", "file_change", "tool_result", "mcp_tool_call"}
_EVIDENCE = _IMPLEMENTATION | {"user_message"}
_CONTEXT_LABELS = {
    "code_smell_precedent",
    "missing_local_repo_context", "missing_cross_system_dependency_knowledge",
}


class ModelError(ValueError):
    """An analyzer did not return usable results."""


class Model(Protocol):
    def decide(self, phase: str, data: dict) -> dict: ...


def _digest(*values: object) -> str:
    encoded = json.dumps(values, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:32]


def ref(event: Event) -> str:
    return _digest(asdict(event.anchor))


def evidence_ref(role: str, event: Event, **extra: object) -> dict:
    return {"role": role, "anchor": asdict(event.anchor), **extra}


def utc(value: str | None) -> str:
    if not value:
        return ""
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if stamp.tzinfo is None:
        return ""
    return stamp.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _content(event: Event) -> str:
    if event.prompt is not None:
        return event.prompt
    item = event.payload.get("item", event.payload)
    # Send only evidence fields. Transport metadata and reasoning records stay local.
    fields = (
        "content", "message", "command", "exit_code", "status", "changes",
        "aggregated_output", "output", "result", "stderr", "stdout",
    )
    return json.dumps({field: item[field] for field in fields if field in item}, ensure_ascii=False)


def _quote_in(event: Event, quote: object) -> bool:
    if not isinstance(quote, str) or not quote.strip():
        return False
    # Interpret quoted tool text without JSON escaping.
    if event.prompt is not None:
        return quote in event.prompt

    def contains(value: object) -> bool:
        if isinstance(value, str):
            return quote in value
        if isinstance(value, list):
            return any(contains(part) for part in value)
        if isinstance(value, dict):
            return any(contains(part) for part in value.values())
        return False
    return contains(json.loads(_content(event)))


def _block_data(events: list[Event], limit: int = 100_000) -> list[dict]:
    data = []
    for event in events:
        content = _content(event)
        # Preserve both ends of large results and tell the model what was clipped.
        maximum = min(6000, limit // 2)
        clipped = len(content) > maximum
        if clipped:
            content = content[:maximum // 2] + "\n[preview clipped]\n" + content[-maximum // 2:]
        data.append({
            "id": ref(event), "kind": event.kind, "text": content,
            "truncated": event.output_truncated or clipped,
        })
    remaining = limit
    selected = set()
    for index in sorted(range(len(data)), key=lambda i: (data[i]["kind"] == "user_message", i), reverse=True):
        size = len(data[index]["text"]) + 100
        if size <= remaining:
            selected.add(index)
            remaining -= size
    return [item for index, item in enumerate(data) if index in selected]


def blocks(events: tuple[Event, ...]) -> list[list[Event]]:
    result: list[list[Event]] = []
    for event in events:
        if event.kind == "user_message":
            result.append([event])
        elif result and event.kind in _EVIDENCE:
            result[-1].append(event)
    return result


def _batches(source: list[list[Event]]):
    for start in range(0, len(source), 4):
        current = [event for block in source[start:start + 4] for event in block]
        context = source[start - 1] if start else []
        yield current, context


def _task_rank(task: Task, current: list[Event]) -> tuple[int, int, str]:
    prompt = "\n".join(event.prompt or "" for event in current if event.kind == "user_message")
    direct = int(ref(task.request) in prompt or task.id in prompt)
    path_pattern = r"(?<![\w/])[\w.-]+(?:/[\w.-]+)+"
    requested_paths = set(re.findall(path_pattern, prompt))
    shared_path = int(bool(requested_paths & set(re.findall(path_pattern, task.request.prompt or ""))))
    return direct, shared_path, utc(task.request.timestamp)


@dataclass
class Task:
    id: str
    request: Event
    quote: str
    offset: int
    description: str
    end: Event | None = None

    def model_data(self) -> dict:
        return {
            "id": self.id, "request": ref(self.request), "quote": self.quote,
            "description": self.description, "closed": self.end is not None,
        }


@dataclass
class Analysis:
    tasks: list[Task]
    rows: list[dict]
    issues: list[str]


def _list_of_objects(data: object, field: str) -> list[dict]:
    if isinstance(data, dict) and isinstance(data.get(field), list):
        if all(isinstance(entry, dict) for entry in data[field]):
            return data[field]
    raise ModelError(f"Expected a JSON object with a {field} array")


def apply_task_decisions(
    decisions: list[dict], events: dict[str, Event], pr_id: str, tasks: dict[str, Task]
) -> list[str]:
    issues = []
    for decision in decisions:
        operation = decision.get("operation")
        if operation in ("unrelated", "uncertain"):
            if operation == "uncertain":
                issues.append("Uncertain task boundary")
            continue
        source = events.get(str(decision.get("source")))
        if source is None:
            issues.append("Unknown task source")
            continue
        if operation == "open":
            quote, description = decision.get("quote"), decision.get("description")
            prompt = source.prompt
            if (
                prompt is None or not isinstance(quote, str) or not quote.strip()
                or prompt.find(quote) == -1 or prompt.find(quote, prompt.index(quote) + 1) != -1
                or not isinstance(description, str) or not description.strip()
            ):
                issues.append("Task outcome must quote a unique part of a recorded user request")
                continue
            offset = len(prompt[:prompt.index(quote)].encode())
            task_id = _digest(pr_id, ref(source), offset)
            tasks.setdefault(task_id, Task(task_id, source, quote, offset, description))
        else:
            selected_id = decision.get("task_id")
            task = tasks.get(str(selected_id))
            # A decision for a task opened in this response may have no ID yet.
            if selected_id is None and len(tasks) == 1:
                task = next(iter(tasks.values()))
            if task is None or operation not in ("close", "continue", "revise"):
                issues.append("Task decision refers to an unknown task or operation")
            elif operation == "close":
                if source.kind not in _EVIDENCE or not _at_or_before(task.request, source):
                    issues.append("Invalid task closure evidence")
                else:
                    task.end = source
            elif source.kind != "user_message" or not _at_or_before(task.request, source):
                issues.append("Invalid task continuation evidence")
            else:
                task.end = None
    return issues


def _at_or_before(first: Event, second: Event) -> bool:
    if first.anchor.rollout_id == second.anchor.rollout_id and first.anchor.session_id == second.anchor.session_id:
        return first.position <= second.position
    return bool(utc(first.timestamp) and utc(second.timestamp) and utc(first.timestamp) < utc(second.timestamp))


def _refs(value: object, events: dict[str, Event]) -> list[Event] | None:
    if not isinstance(value, list) or any(not isinstance(key, str) or key not in events for key in value):
        return None
    return [events[key] for key in dict.fromkeys(value)]


def validate_failure(
    candidate: dict, tasks: dict[str, Task], span: WorkSpan, events: dict[str, Event]
) -> dict | None:
    if span.pr is None:
        return None
    task = tasks.get(str(candidate.get("task_id")))
    actions = _refs(candidate.get("attempts"), events)
    feedback = _refs(candidate.get("corrections"), events)
    extra = _refs(candidate.get("evidence"), events)
    requirement = events.get(str(candidate.get("requirement")))
    trigger, label = candidate.get("trigger"), candidate.get("label")
    if not task or not actions or not feedback or extra is None or not requirement or label not in LABELS:
        return None
    if not _quote_in(requirement, candidate.get("requirement_quote")):
        return None

    local = {ref(event) for event in span.events}
    if any(ref(event) not in local or event.kind not in _IMPLEMENTATION for event in actions):
        return None
    if not all(ref(event) in local for event in feedback):
        return None
    actions.sort(key=lambda event: event.position)
    feedback.sort(key=lambda event: event.position)
    if not all((
        _at_or_before(task.request, actions[0]),
        _at_or_before(requirement, actions[0]),
        _at_or_before(actions[-1], feedback[0]),
        actions[-1].position < feedback[0].position,
    )):
        return None
    if trigger == "user_correction":
        if any(event.kind != "user_message" for event in feedback):
            return None
    elif trigger == "agent_self_correction":
        if any(event.kind != "assistant_message" for event in feedback):
            return None
        independent = [
            event for event in extra
            if event.kind in _IMPLEMENTATION - {"assistant_message"}
            and ref(event) in local
            and _at_or_before(actions[0], event)
            and all(event.anchor != action.anchor for action in actions)
        ]
        if not independent:
            return None
    else:
        return None

    if label in _CONTEXT_LABELS and requirement == task.request and not extra:
        return None
    texts = [candidate.get(key) for key in ("summary", "correction_summary", "reason")]
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        return None

    refs = [evidence_ref("request", task.request, outcome_offset=task.offset),
            evidence_ref("requirement", requirement, quote=candidate["requirement_quote"])]
    request_item = task.request.payload.get("item", {})
    if isinstance(request_item, dict) and isinstance(request_item.get("content"), list):
        attachments = [
            index for index, part in enumerate(request_item["content"])
            if isinstance(part, dict) and part.get("type") in {"image", "local_image", "file"}
        ]
        if attachments:
            refs.append(evidence_ref("attachments", task.request, content_indices=attachments))
    refs.extend(evidence_ref("implementation", event) for event in actions)
    refs.extend(evidence_ref("correction", event) for event in feedback)
    refs.extend(evidence_ref("support", event) for event in extra)
    return {
        "attempt_id": _digest(span.pr.identity, ref(actions[0])),
        "task_id": task.id,
        "pr_url": span.pr.url,
        "pr_link_evidence": json.dumps([
            evidence_ref(reason, event, pr_id=span.pr.identity) for reason, event in span.basis
        ], ensure_ascii=False),
        "task_description": task.description,
        "original_prompt": task.request.prompt,
        "task_started_at": utc(task.request.timestamp),
        "task_ended_at": utc(task.end.timestamp) if task.end else "",
        "attempt_summary": texts[0],
        "attempt_started_at": utc(actions[0].timestamp),
        "attempt_ended_at": utc(actions[-1].timestamp),
        "detection_trigger": trigger,
        "correction_summary": texts[1],
        "correction_started_at": utc(feedback[0].timestamp),
        "correction_ended_at": utc(feedback[-1].timestamp),
        "failure_label": label,
        "failure_reason": texts[2],
        "evidence_refs": json.dumps(refs, ensure_ascii=False),
        **codex_thread_fields(actions[0].anchor.session_id),
    }


def analyze_span(span: WorkSpan, model: Model, prior: list[Task]) -> Analysis:
    if span.pr is None:
        return Analysis([], [], [])
    tasks = {task.id: task for task in prior}
    rows, issues = [], []
    batches = list(_batches(blocks(span.events)))
    for current, context in batches:
        nearby = sorted(tasks.values(), key=lambda task: _task_rank(task, current), reverse=True)[:16]
        sources = [*context, *current]
        for task in nearby:
            if task.request not in sources:
                sources.insert(0, task.request)
        event_map = {ref(event): event for event in sources}
        data = {
            "current": [ref(event) for event in current],
            "events": _block_data(sources),
            "existing_tasks": [task.model_data() for task in nearby],
        }
        decisions = _list_of_objects(model.decide("tasks", data), "decisions")
        issues.extend(apply_task_decisions(decisions, event_map, span.pr.identity, tasks))

    for current, context in batches:
        nearby = sorted(tasks.values(), key=lambda task: _task_rank(task, current), reverse=True)[:16]
        if not nearby:
            continue
        sources = [*context, *current]
        # Later feedback can refer to an attempt several blocks earlier.
        first_position = sources[0].position
        earlier = [event for event in span.events if event.position < first_position and event.kind in _EVIDENCE]
        sources = [*earlier[-48:], *sources]
        for task in nearby:
            if task.request not in sources:
                sources.insert(0, task.request)
        event_map = {ref(event): event for event in sources}
        data = {
            "current": [ref(event) for event in current],
            "events": _block_data(sources),
            "existing_tasks": [task.model_data() for task in nearby],
            "labels": LABELS,
        }
        for candidate in _list_of_objects(model.decide("failures", data), "candidates"):
            feedback = candidate.get("corrections")
            current_refs = set(data["current"])
            if not isinstance(feedback, list) or not any(isinstance(entry, str) and entry in current_refs for entry in feedback):
                issues.append("Failure requires feedback in the current window")
                continue
            row = validate_failure(candidate, tasks, span, event_map)
            if row is None:
                issues.append("Failure candidate failed evidence checks")
            else:
                rows.append(row)
    return Analysis(list(tasks.values()), rows, issues)
