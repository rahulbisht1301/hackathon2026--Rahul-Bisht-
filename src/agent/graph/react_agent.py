from __future__ import annotations

from typing import Any

from langgraph.prebuilt import create_react_agent

from agent.config import settings
from agent.prompts.system_prompt import SYSTEM_PROMPT
from agent.tools.lc_tools import (
    check_refund_eligibility,
    escalate,
    get_customer,
    get_order,
    get_product,
    issue_refund,
    search_knowledge_base,
    send_reply,
)

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:  # pragma: no cover - exercised in runtime environments without package
    ChatGoogleGenerativeAI = None  # type: ignore

ALL_TOOLS = [
    get_order,
    get_customer,
    get_product,
    search_knowledge_base,
    check_refund_eligibility,
    issue_refund,
    send_reply,
    escalate,
]

_react_agent: Any | None = None


def build_react_agent() -> Any:
    if ChatGoogleGenerativeAI is None:
        raise RuntimeError("langchain-google-genai is not available.")
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.gemini_api_key,
        temperature=settings.llm_temperature,
        max_output_tokens=settings.llm_max_tokens,
    )
    try:
        return create_react_agent(
            model=llm,
            tools=ALL_TOOLS,
            state_modifier=SYSTEM_PROMPT,
        )
    except TypeError:
        return create_react_agent(
            model=llm,
            tools=ALL_TOOLS,
            prompt=SYSTEM_PROMPT,
        )


def get_react_agent() -> Any:
    global _react_agent
    if _react_agent is None:
        _react_agent = build_react_agent()
    return _react_agent

