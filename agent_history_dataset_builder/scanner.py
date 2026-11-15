"""Run the shared offline scan pipeline."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from . import dataset
from .analysis import Model, Task, analyze_span, utc
from .prs import PullRequest, PullRequests, WorkSpan, match_spans, worktree_snapshot
from .transcripts import Event, RolloutError, discover_rollouts, read_rollout


def span_report(span: WorkSpan) -> dict:
    turns = list(dict.fromkeys(event.turn_id for event in span.events if event.turn_id))
    return {
        "session_id": span.session_id,
        "first_turn": turns[0] if turns else None,
        "last_turn": turns[-1] if turns else None,
        "first_record": span.events[0].position,
        "last_record": span.events[-1].position,
        "pr_url": span.pr.url if span.pr else None,
        "match_reasons": sorted({reason for reason, _ in span.basis}),
        "issue": span.issue,
    }


def _event_date(event: Event) -> str:
    return utc(event.timestamp)


def _load_spans(
    roots: list[Path], github: PullRequests, selected: PullRequest | None, gaps: list[dict],
) -> list[WorkSpan]:
    result = []
    copies: set[tuple] = set()
    for path in discover_rollouts(roots):
        try:
            events = list(read_rollout(path, active=path.suffix == ".jsonl" and "archived_sessions" not in path.parts))
        except (OSError, RolloutError) as error:
            gaps.append({"file": str(path), "issue": str(error)})
            continue
        if not events:
            continue
        signature = tuple(event.anchor for event in events)
        if signature in copies:
            continue
        copies.add(signature)
        for event in events:
            if event.gap or event.output_truncated:
                gaps.append({
                    "session_id": event.anchor.session_id, "record": event.position,
                    "issue": event.gap or "Tool output was truncated",
                })
        result.extend(match_spans(events, github, selected))
    return sorted(result, key=lambda span: (
        _event_date(span.events[0]), span.session_id, span.events[0].position,
    ))


def scan(
    roots: list[Path], github: PullRequests, model: Model, *,
    pr_url: str | None = None, since: str | None = None, output: Path | None = None,
    dry_run: bool = True,
) -> dict:
    if output is None and not dry_run:
        raise ValueError("Publication requires an explicit output path")
    selected = github.get(pr_url) if pr_url else None
    gaps: list[dict] = []
    spans = _load_spans(roots, github, selected, gaps)
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
        result = analyze_span(span, model, prior[span.pr.identity])
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
