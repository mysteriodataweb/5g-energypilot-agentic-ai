"""Smoke tests: the command-line entry points run end to end."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "LLM_PROVIDER": "none"}


def _run(*args: str, cwd: Path = ROOT) -> str:
    out = subprocess.run([sys.executable, *args], cwd=cwd, env=ENV, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_generate_data(tmp_path):
    _run("scripts/generate_data.py", "--seeds", "0", "--scenarios", "normal", "--out", str(tmp_path))
    meta = json.loads((tmp_path / "normal_seed0.json").read_text())
    assert meta["provenance"] and "SYNTHETIC" in meta["note"]
    assert (tmp_path / "normal_seed0.csv").read_text().startswith("time_h,")


def test_run_simulation(tmp_path):
    stdout = _run("scripts/run_simulation.py", "--scenario", "infeasible_instruction", "--approach", "agent",
                  "--runs-dir", str(tmp_path))  # fmt: skip
    assert '"status": "infeasible"' in stdout and "hash chain valid: True" in stdout
    assert len(list(tmp_path.glob("*.jsonl"))) == 1


def test_run_simulation_custom_instruction(tmp_path):
    stdout = _run("scripts/run_simulation.py", "--scenario", "normal", "--approach", "optimizer", "--instruction",
                  "Keep at least 60% battery charge between 18:00 and 20:00.", "--runs-dir", str(tmp_path))  # fmt: skip
    assert '"status": "accepted"' in stdout and "correct_handling                   nan" in stdout


def test_instruction_benchmark_script():
    stdout = _run("scripts/run_instruction_benchmark.py")
    assert "deterministic_rules" in stdout and "unsafe_acceptance_rate" in stdout


def test_evaluation_script_small(tmp_path):
    _run("scripts/run_evaluation.py", "--seeds", "1", "--experiments", "main", "benchmark",
         "--input-modes", "natural_language", "--out", str(tmp_path))  # fmt: skip
    report = (tmp_path / "report.md").read_text()
    assert "synthetic" in report and "LLM: none configured" in report
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["llm"] is None and meta["seeds"] == [0]
