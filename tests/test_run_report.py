from __future__ import annotations

from datetime import datetime, timezone

from agent.audit.run_report import build_run_report, write_run_report


def test_build_run_report_counts_and_actions():
    started = datetime(2024, 3, 15, 10, 0, tzinfo=timezone.utc)
    completed = datetime(2024, 3, 15, 10, 1, 30, tzinfo=timezone.utc)
    results = [
        {"status": "resolved", "resolution_action": "refund_issued", "confidence_score": 0.9},
        {
            "status": "escalated",
            "resolution_action": "escalated",
            "escalation_reason_code": "tool_failures_exhausted",
            "confidence_score": 0.4,
        },
        {"status": "failed", "resolution_action": "", "confidence_score": 0.0},
    ]

    report = build_run_report(results, started, completed)

    assert report["total_tickets"] == 3
    assert report["resolved"] == 1
    assert report["escalated"] == 1
    assert report["failed"] == 1
    assert report["resolution_actions"]["refund_issued"] == 1
    assert report["resolution_actions"]["escalated"] == 1
    assert report["escalation_reason_codes"]["tool_failures_exhausted"] == 1
    assert report["duration_seconds"] == 90.0


def test_write_run_report_creates_file(tmp_path):
    started = datetime(2024, 3, 15, 10, 0, tzinfo=timezone.utc)
    completed = datetime(2024, 3, 15, 10, 0, 10, tzinfo=timezone.utc)
    results = [{"status": "resolved", "resolution_action": "info_provided", "confidence_score": 0.8}]
    output_path = tmp_path / "run_report.json"

    report = write_run_report(str(output_path), results, started, completed)

    assert output_path.exists()
    assert report["total_tickets"] == 1
    assert report["resolved"] == 1
