from __future__ import annotations

from typing import Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


IntentLabel = Literal["greeting", "product_query", "high_intent"]
"""The closed set of intents the classifier may return.

* ``greeting``       — casual hello, small talk, no product question.
* ``product_query``  — user asks about features, pricing, or policies.
* ``high_intent``    — user signals purchase / demo / trial interest,
                       triggering the lead-collection flow.
"""

class LeadData(TypedDict):
    """Structured representation of the three fields collected from a
    high-intent prospect before triggering the lead-capture tool.

    Every field starts as ``None`` and is populated one-per-turn as the
    conversation progresses. The collection order (name -> email -> platform)
    is enforced by ``get_missing_lead_field``, not by this type.
    """

    name: str | None
    email: str | None
    platform: str | None


# Human-friendly display names used by the lead_collect_node to phrase prompts.
LEAD_FIELD_DISPLAY_NAMES: dict[str, str] = {
    "name": "name",
    "email": "email address",
    "platform": "primary content platform (e.g. YouTube, TikTok, Instagram)",
}

# Deterministic ordering for field collection; checked left-to-right.
_LEAD_FIELD_COLLECTION_ORDER: tuple[str, ...] = ("name", "email", "platform")


class AgentState(TypedDict):
    """The single state object shared by every node in the LangGraph"""

    messages: list[BaseMessage]
    intent: str
    lead_data: LeadData
    lead_captured: bool
    rag_context: str
    awaiting_lead_field: str | None


def create_initial_state() -> AgentState:
    """Return a cleanly initialised ``AgentState`` with safe defaults."""
    return AgentState(
        messages=[],
        intent="",
        lead_data=LeadData(name=None, email=None, platform=None),
        lead_captured=False,
        rag_context="",
        awaiting_lead_field=None,
    )


def get_missing_lead_field(state: AgentState) -> str | None:
    """Return the name of the first lead field that is still ``None``, or
    ``None`` if all fields have been collected."""
    lead: LeadData = state["lead_data"]
    for field in _LEAD_FIELD_COLLECTION_ORDER:
        if lead.get(field) is None:
            return field
    return None


def is_lead_complete(state: AgentState) -> bool:
    """Return ``True`` when all three lead fields have been collected"""
    return get_missing_lead_field(state) is None


def format_conversation_for_display(
    state: AgentState,
    *,
    max_messages: int = 12,
) -> str:
    """Render the most recent messages as a human-readable string"""
    recent: list[BaseMessage] = state["messages"][-max_messages:]

    if not recent:
        return "(no messages yet)"

    lines: list[str] = []

    lines.append("┌─── Conversation Transcript ───")
    lines.append(
        f"│  intent={state['intent']!r}  "
        f"lead_captured={state['lead_captured']}  "
        f"awaiting={state['awaiting_lead_field']!r}"
    )
    lead = state["lead_data"]
    lines.append(
        f"│  lead_data: name={lead.get('name')!r}  "
        f"email={lead.get('email')!r}  "
        f"platform={lead.get('platform')!r}"
    )
    lines.append("├───────────────────────────────")

    for msg in recent:
        if isinstance(msg, HumanMessage):
            prefix = "│ User:  "
        elif isinstance(msg, AIMessage):
            prefix = "│ Agent: "
        elif isinstance(msg, SystemMessage):
            prefix = "│ Sys:   "
        else:
            prefix = f"│ {msg.__class__.__name__}: "

        content = str(msg.content)
        first_line, *rest = content.split("\n")
        lines.append(f"{prefix} {first_line}")
        for continuation in rest:
            lines.append(f"│          {continuation}")

    lines.append("└───────────────────────────────")
    return "\n".join(lines)