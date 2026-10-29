"""Structured, tamper-evident audit log (JSON Lines).

Every instruction, parse result, tool invocation, solver result, validation
outcome, LLM call and final decision is recorded as one JSON object with a
sequence number, a UTC timestamp and a SHA-256 hash chained to the previous
record, so that any later modification or deletion is detectable
(:meth:`AuditLogger.verify_chain`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

EVENT_TYPES = frozenset(
    {
        "run_started",
        "instruction_received",
        "instruction_parsed",
        "llm_call",
        "llm_failure",
        "tool_call",
        "solver_result",
        "infeasibility_diagnosis",
        "validation_result",
        "decision",
        "instruction_activated",
        "shield_intervention",
        "mpc_replan",
        "mpc_replan_failed",
        "mpc_plan_validation",
        "mpc_fallback_rule_based",
        "run_completed",
    }
)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None if np.isnan(value) else ("inf" if value > 0 else "-inf")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(dataclasses.asdict(value))
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump(mode="json"))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class AuditLogger:
    """Append-only audit trail kept in memory and optionally mirrored to a JSONL file."""

    def __init__(self, run_id: str | None = None, path: str | Path | None = None) -> None:
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        self._last_hash = "0" * 64

    def log(self, event_type: str, **payload: Any) -> dict[str, Any]:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown audit event type '{event_type}'")
        record = {
            "seq": len(self.records),
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event": event_type,
            "payload": _to_jsonable(payload),
            "prev_hash": self._last_hash,
        }
        record["hash"] = self._hash(record)
        self._last_hash = record["hash"]
        self.records.append(record)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.debug("audit %s #%d", event_type, record["seq"])
        return record

    @staticmethod
    def _hash(record: dict[str, Any]) -> str:
        body = {k: v for k, v in record.items() if k != "hash"}
        return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    @classmethod
    def verify_records(cls, records: list[dict[str, Any]]) -> bool:
        previous = "0" * 64
        for i, rec in enumerate(records):
            if rec.get("seq") != i or rec.get("prev_hash") != previous or cls._hash(rec) != rec.get("hash"):
                return False
            previous = rec["hash"]
        return True

    def verify_chain(self) -> bool:
        return self.verify_records(self.records)

    @staticmethod
    def load(path: str | Path) -> list[dict[str, Any]]:
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        return [r for r in self.records if event_type is None or r["event"] == event_type]
