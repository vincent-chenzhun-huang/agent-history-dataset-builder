# Codex Attempt Dataset

Build a human-verified dataset of failed coding-agent attempts, grouped by GitHub pull request. The dataset preserves the original task prompt, the implementation, the evidence that it fell short, and a failure label.

| Component | Design |
|---|---|
| Offline scanner | [Scan historical sessions and backfill the dataset](docs/offline-history-scanner.md) |
| Online hook | [Detect and record failed attempts during active sessions](docs/online-codex-hook.md) |

Both components use the same transcript reader, PR resolver, labeler, and CSV writer. The first implementation milestone is a dry run against one confirmed PR. Review its candidate rows before enabling the hook.

The repository starts with design documents. Runtime data belongs in a separate local data directory.
