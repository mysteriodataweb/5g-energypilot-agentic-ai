# results/

Outputs of `scripts/run_evaluation.py`. Everything here comes from executed simulations on
synthetic data.

* `reference/` — the run cited in the README (committed for traceability): `report.md`,
  per-episode CSV files (`*_runs.csv`), aggregated statistics (`*_summary.csv`, `*_paired.csv`,
  `*_handling.csv`), benchmark results and `metadata.json` (commit, seeds, solver, LLM used).
* any other sub-directory — local runs (git-ignored).

Reproduce the reference run:

```bash
python scripts/run_evaluation.py --seeds 10 --out results/reference
```
