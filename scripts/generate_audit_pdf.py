import json
from pathlib import Path


def generate_minimal_pdf(text: str, out_path: Path) -> None:
    content = f"BT /F1 10 Tf 50 760 Td ({text[:180].replace('(', '[').replace(')', ']')}) Tj ET"
    pdf = f"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length {len(content)} >> stream
{content}
endstream endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f
0000000010 00000 n
0000000060 00000 n
0000000117 00000 n
0000000248 00000 n
0000000360 00000 n
trailer << /Size 6 /Root 1 0 R >>
startxref
430
%%EOF
"""
    out_path.write_text(pdf, encoding="utf-8")


def main() -> None:
    audit_path = Path("audit_log.json")
    if not audit_path.exists():
        raise FileNotFoundError("audit_log.json not found")
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    run = payload.get("run_metadata", {})
    summary = (
        f"Run {run.get('run_id', '')}: total={run.get('total_tickets', 0)}, "
        f"resolved={run.get('resolved_autonomously', 0)}, escalated={run.get('escalated', 0)}"
    )
    generate_minimal_pdf(summary, Path("audit_log_report.pdf"))
    print("Generated audit_log_report.pdf")


if __name__ == "__main__":
    main()

