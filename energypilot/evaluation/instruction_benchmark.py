"""Instruction-interpretation benchmark (deterministic parser vs LLM interpreter).

For each labelled instruction the interpreter's status and structured output
are compared with the label. Reported rates:

* ``status_accuracy``: ok / needs_clarification / rejected matches the label;
* ``exact_accuracy``: status matches and, for ``ok`` items, the operational
  content equals the label exactly;
* ``unsafe_acceptance_rate``: share of items accepted although the label is
  not ``ok``, or accepted with wrong content (the dangerous error);
* ``over_caution_rate``: share of ``ok`` items that were not accepted
  (safe, but the operator must rephrase).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from energypilot.config.settings import AppConfig
from energypilot.instructions.schema import ParseResult
from energypilot.instructions.validation import parse_structured

DEFAULT_BENCHMARK = Path(__file__).resolve().parents[2] / "data" / "instruction_benchmark.json"

Interpreter = Callable[[str, int, AppConfig], ParseResult]


def load_benchmark(path: str | Path = DEFAULT_BENCHMARK) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["items"]


def _minute(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def expected_requirements(item: dict, config: AppConfig) -> dict | None:
    if item["expected_status"] != "ok":
        return None
    result = parse_structured(item["expected"], _minute(item["issued_at"]), config)
    if result.instruction is None:
        raise ValueError(f"invalid label for {item['id']}: {result.issues}")
    return result.instruction.requirement_dict()


def evaluate_interpreter(name: str, interpreter: Interpreter, items: list[dict], config: AppConfig) -> pd.DataFrame:
    rows = []
    for item in items:
        issued = _minute(item["issued_at"])
        started = time.perf_counter()
        try:
            result = interpreter(item["text"], issued, config)
            status, error = result.status.value, None
            got = result.instruction.requirement_dict() if result.instruction is not None else None
            method = result.method
        except Exception as exc:  # noqa: BLE001 - recorded as a failure of the interpreter
            status, got, error, method = "error", None, f"{type(exc).__name__}: {exc}", name
        latency = time.perf_counter() - started
        expected = expected_requirements(item, config)
        status_ok = status == item["expected_status"]
        content_ok = status_ok and (expected is None or got == expected)
        accepted = status == "ok"
        rows.append(
            {
                "interpreter": name,
                "method": method,
                "id": item["id"],
                "split": item["split"],
                "lang": item["lang"],
                "expected_status": item["expected_status"],
                "status": status,
                "status_correct": status_ok,
                "exact_correct": content_ok,
                "unsafe_acceptance": accepted and not content_ok,
                "over_caution": item["expected_status"] == "ok" and not accepted,
                "latency_s": latency,
                "error": error,
                "text": item["text"],
            }
        )
    return pd.DataFrame(rows)


def summarize_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (interp, split), g in df.groupby(["interpreter", "split"]):
        ok_items = g[g.expected_status == "ok"]
        rows.append(
            {
                "interpreter": interp,
                "split": split,
                "n": len(g),
                "status_accuracy": g.status_correct.mean(),
                "exact_accuracy": g.exact_correct.mean(),
                "unsafe_acceptance_rate": g.unsafe_acceptance.mean(),
                "over_caution_rate": ok_items.over_caution.mean() if len(ok_items) else float("nan"),
                "mean_latency_s": g.latency_s.mean(),
            }
        )
    return pd.DataFrame(rows)
