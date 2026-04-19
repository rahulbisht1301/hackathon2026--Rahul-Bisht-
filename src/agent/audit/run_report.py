from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_run_report(
    results: list[dict[str, Any]],
    run_started: datetime,
    run_completed: datetime,
) -> dict[str, Any]:
    total = len(results)
    statuses = Counter(str(item.get("status", "failed")) for item in results)
    actions = Counter(
        str(item.get("resolution_action", ""))
        for item in results
        if str(item.get("resolution_action", "")).strip()
    )
    escalation_reasons = Counter(
        str(item.get("escalation_reason_code", ""))
        for item in results
        if str(item.get("status", "")) == "escalated"
        and str(item.get("escalation_reason_code", "")).strip()
    )

    average_confidence = (
        round(sum(float(item.get("confidence_score", 0.0)) for item in results) / total, 4)
        if total
        else 0.0
    )

    return {
        "started_at": run_started.astimezone(timezone.utc).isoformat(),
        "completed_at": run_completed.astimezone(timezone.utc).isoformat(),
        "duration_seconds": round((run_completed - run_started).total_seconds(), 1),
        "total_tickets": total,
        "resolved": statuses.get("resolved", 0),
        "escalated": statuses.get("escalated", 0),
        "failed": statuses.get("failed", 0),
        "resolution_actions": dict(sorted(actions.items())),
        "escalation_reason_codes": dict(sorted(escalation_reasons.items())),
        "average_confidence_score": average_confidence,
    }


def write_run_report(
    path: str,
    results: list[dict[str, Any]],
    run_started: datetime,
    run_completed: datetime,
) -> dict[str, Any]:
    report = build_run_report(results, run_started, run_completed)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

