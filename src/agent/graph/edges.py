from agent.config import settings


def route_resolution(state: dict) -> str:
    if state.get("fraud_flag"):
        return "escalate"
    if state.get("status") == "failed":
        return "escalate"
    if state.get("resolution_action") in {"clarification_requested", "info_requested"}:
        return "resolve"
    confidence = float(state.get("confidence_score", 0.0))
    if confidence >= settings.agent_confidence_threshold:
        return "resolve"
    return "escalate"

