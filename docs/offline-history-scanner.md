# Offline history scanner

The scanner discovers historical Codex conversations, resolves the PRs they worked on, and proposes one CSV row per failed implementation attempt. A PR can have several tasks and span several conversations. Each task retains a description and its original user prompt verbatim.

## Entry point

```text
scan-history --codex-home PATH --output PATH [--pr PR_URL] [--since UTC] [--dry-run]
```

With no PR filter, scan all discoverable local history. Accept additional history roots from other hosts when supplied. A dry run reports matched, unresolved, and ambiguous associations, candidate rows, and unavailable snapshots without changing the CSV.

## Processing

1. **Read rollouts.** Stream plain and Zstandard-compressed JSONL from `sessions/` and `archived_sessions/`. Identify threads and rollouts explicitly. Treat compressed and plain representations of the same rollout as one source, and retain distinct rollout revisions. Normalize ordered messages, edits, tool calls/results, and turn boundaries, preferring structured item IDs and falling back to rollout ID plus record ordinal. Pair results to calls by `call_id`, track stored-output truncation, and retain access to raw payloads. Use SQLite for supplemental Git metadata and parent–subagent links. Retry incomplete tails in active files.
2. **Resolve PRs deterministically.** Extract successful PR creation results, commits observed in the work, and exact head-repository and branch identities. Query GitHub for associated PRs. Treat session-start Git metadata as a candidate when there is evidence it represents the work. A default-branch base SHA or a bare PR URL in arbitrary tool output is insufficient. Keep the association method and source reference. Multiple supported links per thread are valid. Unresolved or conflicting links stay pending.
3. **Extract tasks with the model.** For each confirmed PR, provide ordered candidate conversation windows. Ask for task time bounds, descriptions, and source anchors for the initiating requests. Copy `original_prompt` verbatim from the initiating user item in code, preserving whitespace and line breaks. Retain attachment references in the evidence.
4. **Find failed attempts with the model.** Identify an implementation or design decision and evidence that it missed the original requirement. A user correction is the primary signal. Merge corrections to the same attempt and split subsequent implementations into new attempts. A scope change or clarification alone adds no failure row. The shared labeler also accepts the evidence-backed self-correction case defined in the [hook design](online-codex-hook.md).
5. **Validate and write.** Validate that the model's refs resolve to the supplied history, the events appear in the expected order, and the label has supporting evidence. Save uncertain findings for review without assigning a failure label. Use the shared writer below.

## Shared dataset contract

| Columns | Contents |
|---|---|
| `attempt_id`, `task_id`, `pr_url`, `pr_link_evidence` | Canonical PR identity, join provenance, and stable IDs. Derive the task ID from the PR and request anchor. Derive the attempt ID from the PR and first implementation anchor. |
| `task_description`, `original_prompt`, `task_started_at`, `task_ended_at` | Task description, exact initiating prompt, and UTC bounds. Leave the end blank while the task is open. |
| `attempt_summary`, `attempt_started_at`, `attempt_ended_at` | Implementation or decision and its UTC bounds. |
| `detection_trigger`, `correction_summary`, `correction_started_at`, `correction_ended_at` | `user_correction` or `agent_self_correction`, with the summary and UTC bounds of the corresponding feedback or acknowledgment. |
| `failure_label`, `failure_reason`, `evidence_refs` | One primary label, short rationale, and a JSON array of role-tagged thread, turn, and item refs in one quoted CSV cell. Include request, implementation, and correction or observed failure. Include rule or context evidence where relevant. |
| `attempt_head_sha`, `recorded_head_sha`, `recorded_worktree_dirty` | Observed attempt-time SHA and the original worktree's HEAD and dirty flag at first write. Leave unavailable values blank. A dirty flag alone cannot reconstruct uncommitted changes. |
| `created_at`, `human_verified` | First-write UTC timestamp and verification boolean. The scanner initializes it to `false`. A reviewer can set it to `true` after checking the evidence. |

Use UTF-8 CSV with standard quoting for multiline prompts. Under a shared lock, reload the current CSV, upsert by `attempt_id`, and atomically replace the file. Preserve the first write's timestamp and repo snapshot. Keep verified rows intact. Send unverified candidates with competing implementation anchors to review before inserting a possible duplicate. A review command uses the same lock so verification can run alongside the hook.

### Allowed labels

| Label | Required evidence |
|---|---|
| `known_rules_violation` | An explicit rule applied and the attempt contradicted it. |
| `code_smell_precedent` | The attempt followed or extended a problematic pattern already present. |
| `code_smell_invented` | The attempt introduced a new code smell. |
| `incorrect_design_decision` | A design choice failed under the known requirements and facts. |
| `missing_local_repo_context` | A fact available in the repo would have changed the implementation. |
| `missing_cross_system_dependency_knowledge` | A fact about an external system or dependency would have changed the implementation. |

Pick the most specific label supported by the evidence. If duplicated logic resulted from missing an existing helper, use `missing_local_repo_context` and explain the duplication in the reason.

## Acceptance checks

- Plain and compressed representations produce identical event anchors. Calls and outputs pair correctly, and known truncation is visible.
- PR mentions alone create no associations. Multiple confirmed PRs in a thread remain distinct.
- Two corrections to one attempt yield one row. Two failed implementations yield two rows. Scope changes alone yield none.
- Original multiline prompts round-trip exactly through CSV.
- Offline rescans and online replays converge on the same row without changing verified data.
