from langgraph.graph import END, StateGraph

from agent.graph.edges import route_resolution
from agent.graph.nodes import (
    classify_ticket,
    escalate_ticket,
    reason_and_act,
    resolve_ticket,
    write_audit_entry,
)
from agent.graph.state import TicketState


async def build_graph(checkpointer):
    graph = StateGraph(TicketState)
    graph.add_node("classify", classify_ticket)
    graph.add_node("reason_and_act", reason_and_act)
    graph.add_node("resolve", resolve_ticket)
    graph.add_node("escalate", escalate_ticket)
    graph.add_node("audit_and_end", write_audit_entry)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "reason_and_act")
    graph.add_conditional_edges(
        "reason_and_act",
        route_resolution,
        {"resolve": "resolve", "escalate": "escalate"},
    )
    graph.add_edge("resolve", "audit_and_end")
    graph.add_edge("escalate", "audit_and_end")
    graph.add_edge("audit_and_end", END)
    return graph.compile(checkpointer=checkpointer)

