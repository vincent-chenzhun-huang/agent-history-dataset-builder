# Agent History Dataset Builder

Build a human-verified CSV of failed Codex attempts linked to GitHub pull requests. Each row preserves the original task prompt and the evidence of the failure.

The [implementation plan](docs/implementation-plan.md) covers the offline scan, online hook, dataset schema, and testing strategy.

## Install and run

Use Python 3.10 or newer on macOS or Linux. Set `DATASET_PYTHON` to a compatible interpreter. For an existing editable install, use the interpreter from its environment. A Python 3.12 install works even if the system `python3` is too old. Each machine needs its own `gh` and Codex authentication, local history, and `zstd` for compressed transcripts.

```bash
"$DATASET_PYTHON" -m pip install -e .
scan-history --codex-home "$CODEX_HISTORY_ROOT" --pr "$PR_URL" --dry-run
scan-history --codex-home "$CODEX_HISTORY_ROOT" --pr "$PR_URL" --output "$OUTPUT_CSV"
verify-attempt --output "$OUTPUT_CSV" "$ATTEMPT_ID"
```

Set the paths and PR URL for your machine. Omit `--pr` to discover PRs from local evidence. If you use another Codex executable, pass its path with `--codex-bin`. Use `--model-command` to supply an executable that accepts one JSON request on stdin and returns one JSON response on stdout.

`--since` filters the returned candidates by correction time. The scanner still reads the supplied history. See [publication evidence and history limits](#publication-evidence-and-history-limits) before rescanning older paginated exports.

Use `codex_thread_url` to open the stored attempt in Codex Desktop, or run `codex resume "$THREAD_ID"` with the row's `codex_thread_id`. The thread must be available to that client. The evidence references identify the records within the thread. Existing CSVs gain the new columns when the next update writes the file.

## Optional settings

Pass the same JSON file to `scan-history --config "$DATASET_CONFIG"` and `dataset-review-hook config --config "$DATASET_CONFIG"`. This example sets an output path and a per-kind lookback. The other values shown are defaults. Relative output paths resolve from the config file's directory. Both entry points accept `--output` as an override.

```json
{
  "version": 1,
  "output": "attempts.csv",
  "publication_adapters": ["git", "gh"],
  "analysis": {
    "event_chars": 6000,
    "context_chars": 10000000,
    "max_events": 100000,
    "lookback_events": 48,
    "lookback_by_kind": {"tool_result": 8},
    "batch_size": 4,
    "task_limit": 16
  },
  "model": {"adapter": "codex", "codex_bin": "codex", "timeout_seconds": 240, "max_input_chars": 1048576},
  "hook": {"every": 5, "context_chars": 5000, "span_limit": 3}
}
```

The `lookback_by_kind` example limits older tool results to eight. By default, all kinds share the overall lookback limit. `batch_size` counts user-message blocks. `context_chars` bounds serialized evidence and task records per model request, and `max_events` limits the selected evidence records. Each hook span uses the smaller of the analysis and hook context limits.

The native Codex adapter checks the complete prompt, including the wrapper, instructions, and schema, against `model.max_input_chars` before execution. A prompt at the limit is allowed. Custom JSON command adapters manage their own input limits.

For oversized native prompts, the scanner first replaces the previous block with bounded lookback when available. If the prompt still exceeds the limit, the scanner splits the current events in source order, including within a user-message block. The existing evidence and task budgets still apply. Reports record clipping, omitted evidence, fallbacks, splits, and model errors in `evidence_gaps`. If a single event still cannot fit or a model call fails, the scanner records the gap and continues with later batches. Successful batches can still publish validated rows.

Set `model.name` to select a Codex model. To use a JSON-in/JSON-out executable, set `model.adapter` to `command` and provide `model.command` as an array of arguments. `--codex-bin`, `--model`, `--model-command`, and `--model-timeout` override the corresponding settings. `AGENT_HISTORY_CODEX_BIN` applies when `--codex-bin` is omitted. The hook uses a native Codex child reviewer.

## Publication evidence and history limits

| Adapter | Accepted evidence in this checkout |
|---|---|
| `git` | A successful `git commit` result with a commit SHA and a known repository. A newline-separated batch may stage files with `git add` and end with one `git commit`. Authenticated GitHub data must confirm that the PR contains the commit. |
| `gh` | A successful `gh pr create` result containing one unambiguous URL in the form `https://<host>/<owner>/<repo>/pull/<number>`. Authenticated GitHub data must confirm the PR. |

Custom publishers are opt-in. Define an adapter and list it in `publication_adapters`. The scanner matches exact recorded argument arrays and checks the output. It never runs the publisher.

```json
{
  "publication_adapters": ["git", "gh", "local-review"],
  "github_hosts": ["github-enterprise.example.com"],
  "publication_formats": {
    "local-review": {
      "argv_variants": [["review", "publish"], ["review", "publish", "--only"]],
      "branch_change_variants": [["review", "switch", "main"]],
      "output_formats": ["branch-update"]
    }
  }
}
```

Choose `legacy-status` for output such as `Updated pull request #1: URL`, `branch-update` for `Updating PR for branch BRANCH [URL]`, or `prefixed-url` for a URL preceded by a literal `line_prefix`. List branch-changing commands as exact argument arrays in `branch_change_variants`. A match needs complete, successful output for one PR and authenticated GitHub confirmation. The CSV records the adapter and branch. PR mentions, ordinary `git push`, help or dry-run output, arbitrary shell batches, and ambiguous publication output provide insufficient evidence.

For a single-PR scan, `--pr` authorizes that URL's host for lookup. Backfills and hooks use the configured hosts. Update local configs for custom publishers to supply their own adapter names, arguments, and formats.

Branch names match exactly by default. If your publisher adds a fixed prefix, add a rule to `publication_branch_mappings` in the same config. Each rule has `host`, `repository` in `owner/repo` form, an enabled custom `adapter`, and a literal `remote_prefix` ending in `/`. A prefix of `publisher/` maps `topic/fix` to `publisher/topic/fix`. Configure one rule per host, repository, and adapter.

The authenticated PR must belong to the publication repository. The rule grants no host access and creates no repository aliases. The CSV records the local branch, authenticated remote branch, and applied rule in `pr_link_evidence`.

Supply all available rollout pages for a conversation. The offline scanner joins pages marked paginated when they share a session ID and their recorded time ranges establish a strict order. It reports ambiguous page order and reads those pages independently. When copies of a page disagree, it reports the conflict and skips that page. The hook reads the active transcript supplied by Codex. Missing pages can hide requests, constraints, changes, and publication evidence. The scanner cannot reconstruct missing records or recover page ancestry from a resumed page. An available page can still yield a row if it has enough evidence.

Working-directory changes alone do not end a PR work span. Recorded branch or repository changes still bound the span. The scanner resolves local file-URI working directories from the transcript and requires repository evidence for the commit's worktree. An earlier unmatched span in the same ordered session and recorded repository can supply user requests and constraints to the model. The implementation and correction must still come from the matched span. Within the configured preview budgets, user messages and recorded file changes take priority over command output. The model receives the selected evidence in source order.

Failure analysis reserves room in the context budget for the highest-ranked task. Records outside the matched span are marked as context so the model can cite an earlier requirement while selecting the attempt and correction from the matched span.

When rescanning history with a reader that distinguishes page anchors, rows produced by an older reader may get new task or attempt IDs. IDs depend on the source anchors, so a replay can add a second row if the old and new anchors cannot be matched. Existing human-verified rows retain their values. There is no ID migration. Use a dry run or a separate output file to compare the two runs.

## Enable the conversation hook

```bash
dataset-review-hook config --output "$OUTPUT_CSV" --every 5
# Or use a shared config with an output path.
dataset-review-hook config --config "$DATASET_CONFIG"
```

Merge the generated `UserPromptSubmit` and `SubagentStop` entries into your repository's `.codex/hooks.json`. Review and trust both in Codex. Enable native subagents in the client. The generated config uses the installed executable and resolves the CSV path at configuration time.

The hook can request in-session delegation after each configured interval. Pre-compaction review is excluded on stock Codex. Offline backfill can recover supported attempts after compaction.

## Run tests

```bash
"$DATASET_PYTHON" -m unittest discover -s tests -v
git diff --check
```
