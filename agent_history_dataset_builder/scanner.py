"""Run the shared offline scan pipeline."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from . import dataset
from .analysis import Model, Task, analyze_span, utc
from .config import Settings
from .prs import PullRequest, PullRequests, WorkSpan, match_spans, worktree_snapshot
from .transcripts import Event, RolloutError, discover_rollouts, order_pages, page_anchors, read_rollout


def span_report(span: WorkSpan) -> dict:
    turns = list(dict.fromkeys(event.turn_id for event in span.events if event.turn_id))
    return {
        "session_id": span.session_id,
        "first_turn": turns[0] if turns else None,
        "last_turn": turns[-1] if turns else None,
        "first_record": span.events[0].position,
        "last_record": span.events[-1].position,
        "pr_url": span.pr.url if span.pr else None,
        "match_reasons": sorted({item.reason for item in span.basis}),
        "issue": span.issue,
    }


def _event_date(event: Event) -> str:
    return utc(event.timestamp)


def _with_earlier_requests(spans: list[WorkSpan]) -> list[WorkSpan]:
    result = []
    earlier = []
    repo = pr_id = None
    for span in spans:
        if span.repository is None or span.repository != repo:
            earlier.clear()
            pr_id = None
            repo = span.repository
        if span.pr:
            if pr_id is not None and span.pr.identity != pr_id:
                earlier.clear()
            pr_id = span.pr.identity
            span = replace(span, earlier_requests=tuple(earlier))
        elif span.has_publication:
            earlier.clear()
            pr_id = None
        elif repo is not None:
            earlier.extend(event for event in span.events if event.prompt)
        result.append(span)
    return result


def _load_spans(
    roots: list[Path], github: PullRequests, selected: PullRequest | None, gaps: list[dict],
    settings: Settings,
) -> list[WorkSpan]:
    result = []
    sessions: dict[str, dict[str, list[Event]]] = defaultdict(dict)
    conflicting: set[tuple[str, str]] = set()
    for path in discover_rollouts(roots):
        try:
            events = list(read_rollout(path, active=path.suffix == ".jsonl" and "archived_sessions" not in path.parts))
        except (OSError, RolloutError) as error:
            gaps.append({"file": str(path), "issue": str(error)})
            continue
        if not events:
            continue
        events = page_anchors(events)
        anchor = events[0].anchor
        key = (anchor.session_id, anchor.rollout_id)
        pages = sessions[anchor.session_id]
        if key in conflicting:
            continue
        previous = pages.get(anchor.rollout_id)
        if previous is None or previous == events[:len(previous)]:
            pages[anchor.rollout_id] = events
        elif events != previous[:len(events)]:
            del pages[anchor.rollout_id]
            conflicting.add(key)
            gaps.append({
                "session_id": anchor.session_id, "rollout_id": anchor.rollout_id,
                "issue": "Conflicting copies of the same rollout page",
            })
    for session_id, rollouts in sessions.items():
        pages = list(rollouts.values())
        if not pages:
            continue
        for page in pages:
            for event in page:
                if event.gap or event.output_truncated:
                    gaps.append({
                        "session_id": session_id, "rollout_id": event.anchor.rollout_id,
                        "record": event.position, "issue": event.gap or "Tool output was truncated",
                    })
        ordered = order_pages(pages)
        if ordered is None:
            gaps.append({"session_id": session_id, "issue": "Could not establish rollout page order"})
        for events in [ordered] if ordered is not None else pages:
            spans = match_spans(
                events, github, selected, adapters=settings.publication_adapters,
                formats=settings.publication_formats, branch_mappings=settings.publication_branch_mappings,
            )
            result.extend(_with_earlier_requests(spans))
    return sorted(result, key=lambda span: (
        _event_date(span.events[0]), span.session_id, span.events[0].position,
    ))


def scan(
    roots: list[Path], github: PullRequests, model: Model, *,
    pr_url: str | None = None, since: str | None = None, output: Path | None = None,
    dry_run: bool = True, settings: Settings = Settings(),
) -> dict:
    if output is None and not dry_run:
        raise ValueError("Publication requires an explicit output path")
    selected = github.get(pr_url) if pr_url else None
    gaps: list[dict] = []
    spans = _load_spans(roots, github, selected, gaps, settings)
    prior: dict[str, list[Task]] = defaultdict(list)
    report = {
        "dry_run": dry_run, "spans": [], "tasks": [], "rows": [], "evidence_gaps": gaps,
    }
    tasks = {}
    for span in spans:
        report["spans"].append(span_report(span))
        if span.pr is None:
            continue
        # Keep old tasks from this PR as context for a later correction.
        result = analyze_span(span, model, prior[span.pr.identity], settings.analysis)
        prior[span.pr.identity] = result.tasks
        tasks.update((task.id, task) for task in result.tasks)
        rows = [
            row for row in result.rows
            if since is None or row["correction_ended_at"] >= since
        ]
        if rows and not dry_run:
            snapshot = worktree_snapshot(span)
            for row in rows:
                row.update(snapshot)
        report["rows"].extend(rows)
        gaps.extend({"session_id": span.session_id, "issue": issue} for issue in result.issues)
    report["tasks"] = [
        {
            "task_id": task.id, "description": task.description,
            "original_prompt": task.request.prompt, "started_at": utc(task.request.timestamp),
            "ended_at": utc(task.end.timestamp) if task.end else "",
        }
        for task in tasks.values()
    ]
    report["publication"] = (
        dataset.preview(output, report["rows"]) if dry_run else dataset.publish(output, report["rows"])
    )
    return report
