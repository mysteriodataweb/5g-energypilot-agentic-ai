# data/

* `instruction_benchmark.json` — labelled operational instructions (synthetic, written by the
  project author) used to evaluate instruction interpretation. See `docs/instruction_interface.md`.
* `generated/` — CSV exports of the synthetic scenario profiles, created by
  `python scripts/generate_data.py` (git-ignored; fully reproducible from the seed).

No real measurement or third-party dataset is included. See `docs/data_provenance.md` for the
generation methodology and the distinction between measured, synthetic, simulated, modelled,
estimated and assumed quantities.
