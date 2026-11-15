# Online Codex hook

Detect failed implementation attempts during active Codex sessions and write reviewable rows through the same labeler and dataset writer as the [offline scanner](offline-history-scanner.md).

## Capture and cadence

Register a Codex `Stop` command hook. It reads the session ID, turn ID, transcript path, and working directory available in the hook input. It durably enqueues a small event keyed by session and turn, then returns without adding instructions to the active conversation. A local worker does model and GitHub calls outside the hook.

Count distinct completed root turns per session and schedule analysis every configurable `X` turns. Confirm completion against the transcript and retry when the transcript is still being written. Subagent turns contribute evidence but do not advance the root cadence. Process the remainder at session end when available. The offline scanner can recover events missed during an interruption.

## Detect failure

Provide the model with new user messages and enough preceding history to recover the original request and implementation. The model returns structured candidate failures and source refs. A candidate can have either trigger:

- **User correction.** The user identifies a miss and asks the agent to correct or redirect the implementation.
- **Agent self-correction.** The agent explicitly recognizes that a completed implementation or design decision fell short, and a diff, failing verification, or other recorded evidence supports the acknowledgment.

A failed command followed by an ordinary retry, an early design exploration, or a new requirement alone is insufficient. The worker validates each source reference and copies the original task prompt directly from its source item. Use the same six labels and row schema as the offline scanner. If Codex itself runs the labeler, isolate its invocation from the hook and exclude its sessions from both scanners.

## Resolve and persist

Use the shared deterministic PR resolver. Pending findings are keyed by the thread and first implementation anchor until a PR is confirmed. Retry them on later turns or when a PR association is refreshed. Once resolved, compute the canonical attempt ID and write the CSV row with `human_verified=false`. A later correction to the same attempt can update an unverified row. A new implementation receives a new attempt ID.

Look up the original worktree's HEAD and dirty flag at the first successful dataset write. Store an attempt-time SHA only when the transcript or captured evidence establishes it. Keep unavailable fields blank and preserve the original write-time snapshot on retries.

Use a local SQLite queue with retry state and acknowledge work only after its results have been written or durably marked pending. A crash after the CSV write and before queue acknowledgment is safe because both ingestion routes upsert by the same attempt ID. Keep the queue, raw transcripts, and CSV outside tracked source files.

## Acceptance checks

- Duplicate hook deliveries and worker crash replay produce one row.
- The worker runs after `X` distinct completed root turns and can tie a correction to an attempt from a previous batch.
- User corrections and supported self-corrections yield rows. Clarifications and routine retries alone yield none.
- A late PR link releases the pending finding. Offline backfill resolves the same attempt ID.
- Verified rows survive rescans, and labeler invocations cannot trigger recursive analysis.
