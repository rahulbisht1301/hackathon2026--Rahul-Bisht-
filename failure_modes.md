# Failure Modes and Handling

## 1. Tool Timeout

- **Scenario:** `get_order()` hits injected timeout.
- **Detection:** `TimeoutError` in tool runtime.
- **Response:** retry with exponential backoff (up to 3 attempts).
- **Fallback:** escalate with attempted-call summary if retries exhausted.
- **Logging:** `WARNING` on retry, `ERROR` on exhaustion.

## 2. Malformed Tool Response

- **Scenario:** `check_refund_eligibility()` returns malformed payload (e.g., missing `eligible`).
- **Detection:** schema/key validation failure in node logic.
- **Response:** log warning, infer conservatively from known fields when safe, else escalate.
- **Guarantee:** never crash the graph from malformed tool output.

## 3. Non-Existent Order ID (TKT-017)

- **Scenario:** customer references `ORD-9999`.
- **Detection:** `get_order()` returns `order_not_found`.
- **Response:** no refund, fraud/abuse signal raised on legal-threat language, request corrected order details.

## 4. Social Engineering Attempt (TKT-018)

- **Scenario:** customer claims premium tier and cites non-existent instant refund policy.
- **Detection:** customer tier mismatch vs `get_customer()` result + policy mismatch from KB.
- **Response:** `fraud_flag=True`, urgent escalation, polite decline without exposing internal fraud flag.

## 5. Already Refunded Order (TKT-009)

- **Scenario:** customer asks about refund status for already-refunded order.
- **Detection:** `refund_status == "refunded"` or eligibility reason `already_refunded`.
- **Response:** provide confirmation and 5-7 business day settlement timeline; prevent duplicate refund.

