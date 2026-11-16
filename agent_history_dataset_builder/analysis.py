"""Select evidence for the model and validate its decisions against transcripts."""

from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Protocol

from .config import EVIDENCE_KINDS, Limits
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
_EVIDENCE = EVIDENCE_KINDS
_EVIDENCE_PRIORITY = {"user_message": 3, "file_change": 2, "assistant_message": 1}
_CONTEXT_LABELS = {
    "code_smell_precedent",
    "missing_local_repo_context", "missing_cross_system_dependency_knowledge",
}


class ModelError(ValueError):
    """An analyzer did not return usable results."""


class ModelTimeout(ModelError):
    """The model exhausted its configured time budget."""


class ModelInputTooLarge(ModelError):
    """The complete native prompt exceeds the transport limit."""


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


def _block_data(
    events: list[Event], limit: int = Limits.context_chars, *, event_chars: int = Limits.event_chars,
    max_events: int = Limits.max_events, gaps: list[str] | None = None, preferred: frozenset[str] = frozenset(),
    local_refs: frozenset[str] | None = None,
) -> list[dict]:
    records = []
    maximum = min(event_chars, max(1, limit // 2))
    clipped_count = 0
    for event in events:
        content = _content(event)
        clipped = len(content) > maximum
        if clipped:
            clipped_count += 1
            tail = (maximum - 1) // 2
            content = content[:maximum - 1 - tail] + "…" + (content[-tail:] if tail else "")
        record = {
            "id": ref(event), "kind": event.kind, "text": content,
            "truncated": event.output_truncated or clipped,
        }
        if local_refs is not None and record["id"] not in local_refs:
            record["context_only"] = True
        records.append(record)
    remaining = max(0, limit - 2)
    selected = set()
    ranked = sorted(
        range(len(records)),
        key=lambda i: (records[i]["id"] in preferred, _EVIDENCE_PRIORITY.get(events[i].kind, 0), i),
        reverse=True,
    )
    for index in ranked:
        size = len(json.dumps(records[index], ensure_ascii=False)) + 2
        if len(selected) < max_events and size <= remaining:
            selected.add(index)
            remaining -= size
    if gaps is not None:
        if clipped_count:
            gaps.append(f"Event text budget clipped {clipped_count} evidence event(s)")
        if len(selected) < len(records):
            gaps.append(f"Context budget omitted {len(records) - len(selected)} evidence event(s)")
    return [record for index, record in enumerate(records) if index in selected]


def _previous_evidence(
    span: WorkSpan, before: int, limits: Limits, gaps: list[str], current: list[Event],
) -> list[Event]:
    earlier = [*span.earlier_requests,
               *(event for event in span.events if event.position < before and event.kind in _EVIDENCE)]
    opening = span.earlier_requests[0] if span.earlier_requests else None
    terms = set(re.findall(r"[\w-]{5,}", " ".join(event.prompt or "" for event in current).lower()))
    kept = []
    counts: dict[str, int] = {}
    ranked = sorted(
        reversed(earlier),
        key=lambda event: (
            event is opening, _EVIDENCE_PRIORITY.get(event.kind, 0),
            len(terms.intersection(re.findall(r"[\w-]{5,}", (event.prompt or "").lower()))),
        ),
        reverse=True,
    )
    for event in ranked:
        if len(kept) >= limits.lookback_events:
            break
        count = counts.get(event.kind, 0)
        if count < limits.lookback_by_kind.get(event.kind, limits.lookback_events):
            counts[event.kind] = count + 1
            kept.append(event)
    if len(kept) < len(earlier):
        gaps.append(f"Lookback budget omitted {len(earlier) - len(kept)} earlier evidence event(s)")
    return sorted(kept, key=lambda event: event.position)


def blocks(events: tuple[Event, ...]) -> list[list[Event]]:
    result: list[list[Event]] = []
    for event in events:
        if event.kind == "user_message":
            result.append([event])
        elif result and event.kind in _EVIDENCE:
            result[-1].append(event)
    return result


def _batches(source: list[list[Event]], size: int):
    for start in range(0, len(source), size):
        current = [event for block in source[start:start + size] for event in block]
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
            evidence_ref(
                item.reason, item.event, pr_id=span.pr.identity,
                **({"branch": item.branch} if item.branch else {}),
                **({"adapter": item.adapter} if item.adapter else {}),
                **({"remote_branch": item.remote_branch} if item.remote_branch else {}),
                **({"branch_mapping": asdict(item.branch_mapping)} if item.branch_mapping else {}),
            )
            for item in span.basis
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


def analyze_span(span: WorkSpan, model: Model, prior: list[Task], limits: Limits = Limits()) -> Analysis:
    if span.pr is None:
        return Analysis([], [], [])
    tasks = {task.id: task for task in prior}
    rows, issues = [], []
    batches = list(_batches(blocks(span.events), limits.batch_size))
    local_refs = frozenset(ref(event) for event in span.events if event.kind in _EVIDENCE)

    for phase, field in (("tasks", "decisions"), ("failures", "candidates")):
        pending: deque[tuple[list[Event], list[Event] | None]] = deque(batches)
        number = 0
        while pending:
            current, context = pending.popleft()
            number += 1
            prefix = f"{phase} batch {number}: "
            gaps: list[str] = []
            nearby = sorted(tasks.values(), key=lambda task: _task_rank(task, current), reverse=True)
            if len(nearby) > limits.task_limit:
                gaps.append(f"Task budget omitted {len(nearby) - limits.task_limit} task(s)")
            nearby = nearby[:limits.task_limit]
            if phase == "failures" and not nearby:
                continue
            sources = [*(context or []), *current]
            if phase == "failures" or span.earlier_requests or context is None:
                sources = [*_previous_evidence(span, sources[0].position, limits, gaps, current), *sources]
            for task in nearby:
                if task.request not in sources:
                    sources.insert(0, task.request)
            task_options = [task.model_data() for task in nearby]
            evidence_budget = limits.context_chars
            if phase == "failures":
                # Leave room for the top-ranked task even when history fills the window.
                task_size = len(json.dumps(task_options[0], ensure_ascii=False)) + 4
                evidence_budget -= min(task_size, evidence_budget // 4)
            records = _block_data(
                sources, evidence_budget, event_chars=limits.event_chars,
                max_events=limits.max_events, gaps=gaps,
                preferred=frozenset(
                    [ref(task.request) for task in nearby]
                    + [ref(event) for event in span.earlier_requests[:1]]
                ),
                local_refs=local_refs,
            )
            visible = {record["id"] for record in records}
            current_refs = [ref(event) for event in current if ref(event) in visible]
            task_records = []
            remaining = limits.context_chars - len(json.dumps(records, ensure_ascii=False)) - 2
            for record in task_options:
                if record["request"] not in visible:
                    continue
                size = len(json.dumps(record, ensure_ascii=False)) + 2
                if size <= remaining:
                    task_records.append(record)
                    remaining -= size
            if len(task_records) < len(nearby):
                gaps.append(f"Context budget omitted {len(nearby) - len(task_records)} task record(s)")
            if not current_refs:
                gaps.append("Budget left no current evidence; model call skipped")
            elif phase == "failures" and not task_records:
                gaps.append("Budget left no visible tasks; model call skipped")
            issues.extend(prefix + gap for gap in gaps)
            if not current_refs or (phase == "failures" and not task_records):
                continue
            event_map = {ref(event): event for event in sources if ref(event) in visible}
            data = {
                "current": current_refs,
                "events": records,
                "existing_tasks": task_records,
            }
            if phase == "tasks" and span.earlier_requests:
                data["earlier_requests"] = [
                    ref(event) for event in span.earlier_requests if ref(event) in visible
                ]
            if phase == "failures":
                data["labels"] = LABELS
            try:
                answers = _list_of_objects(model.decide(phase, data), field)
            except ModelInputTooLarge as error:
                if context:
                    issues.append(prefix + f"{error}. Replaced previous-block context with bounded lookback")
                    pending.appendleft((current, None))
                elif len(current) > 1:
                    middle = len(current) // 2
                    issues.append(prefix + f"{error}. Split {len(current)} current events into two batches")
                    pending.appendleft((current[middle:], None))
                    pending.appendleft((current[:middle], None))
                else:
                    issues.append(prefix + f"{error}. One current event with its context could not fit")
                continue
            except ModelError as error:
                issues.append(prefix + str(error))
                continue
            if phase == "tasks":
                issues.extend(apply_task_decisions(answers, event_map, span.pr.identity, tasks))
                continue

            for candidate in answers:
                feedback = candidate.get("corrections")
                if not isinstance(feedback, list) or not any(
                    isinstance(entry, str) and entry in current_refs for entry in feedback
                ):
                    issues.append("Failure requires feedback in the current window")
                    continue
                row = validate_failure(candidate, tasks, span, event_map)
                if row is None:
                    issues.append("Failure candidate failed evidence checks")
                else:
                    rows.append(row)
    return Analysis(list(tasks.values()), rows, issues)
