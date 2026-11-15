# Agent History Dataset Builder

Build a human-verified CSV of failed Codex attempts linked to GitHub pull requests. Each row preserves the original task prompt and the evidence of the failure.

The [implementation plan](docs/implementation-plan.md) covers the offline scan, online hook, dataset schema, and testing strategy.

## Install and run

Use Python 3.10 or newer on macOS or Linux. Install and authenticate `gh` and Codex. Install `zstd` to read compressed transcripts.

```bash
python3 -m pip install -e .
scan-history --codex-home "$CODEX_HISTORY_ROOT" --pr "$PR_URL" --dry-run
scan-history --codex-home "$CODEX_HISTORY_ROOT" --pr "$PR_URL" --output "$OUTPUT_CSV"
verify-attempt --output "$OUTPUT_CSV" "$ATTEMPT_ID"
```

Set the paths and PR URL for your machine. Omit `--pr` to discover PRs from local evidence. If you use another Codex executable, pass its path with `--codex-bin`. Use `--model-command` to supply an executable that accepts one JSON request on stdin and returns one JSON response on stdout.

Use `codex_thread_url` to open the stored attempt in Codex Desktop, or run `codex resume "$THREAD_ID"` with the row's `codex_thread_id`. The thread must be available to that client. The evidence references identify the records within the thread. Existing CSVs gain the new columns when the next update writes the file.

## Enable the conversation hook

```bash
dataset-review-hook config --output "$OUTPUT_CSV" --every 5
```

Merge the generated `UserPromptSubmit` and `SubagentStop` entries into your repository's `.codex/hooks.json`. Review and trust both in Codex. Enable native subagents in the client. The generated config uses the installed executable and resolves the CSV path at configuration time.

The hook can request in-session delegation after each configured interval. Pre-compaction review is excluded on stock Codex. Offline backfill can recover supported attempts after compaction.

## Run tests

```bash
python3 -m unittest discover -s tests -v
```
