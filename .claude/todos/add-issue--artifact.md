# TODO: open GitHub issue — `artifacts/` directory

Submit with:

```sh
gh issue create --repo marawangamal/actmat \
  --title "Route generated outputs through artifacts/ dir" \
  --body-file .claude/todos/add-issue--artifact.md
```

(Or paste the body below into the web UI.)

---

## Motivation

Generated outputs are currently scattered:
- Covariance / Fisher `.pt` files land next to checkpoints
  (e.g. `checkpoints-analysis-drift/ViT-B-16/CarsVal/covariance_checkpoint_200.pt`).
- SLURM logs go to `logs/` at repo root.
- Results live in `results/` and `results-analysis/`.

This mixes durable inputs (checkpoints), reproducible outputs (cov/fisher/metrics), and ephemeral logs in different places — hard to clean up, hard to sync, hard to gitignore.

## Proposal

Introduce a single `artifacts/` root, organized by run:

```
artifacts/
  <run-id>/
    covariance/<task>/<ckpt>.pt
    fisher/<task>/<ckpt>.pt
    logs/<job-id>.out
    metrics.json
```

## Scope

- Update `resolve_run_dir` (or add a sibling `resolve_artifact_dir`) to return the artifacts root for a given run.
- Update `scripts/vision/{covariance,fisher}.py`, `scripts/language/{covariance,fisher}.py`, and the SLURM drivers (`*-drifts.sh`, `eval_*.sh`) to write under `artifacts/`.
- Update `_TaskVector` auto-discovery to look in the artifact path, not next to the checkpoint.
- Add `artifacts/` to `.gitignore`.
