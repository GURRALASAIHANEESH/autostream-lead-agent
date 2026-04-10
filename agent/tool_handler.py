import logging

from langchain_core.messages import AIMessage

from agent.state_manager import AgentState, get_missing_lead_field, LeadData
from tools.lead_capture import LeadRecord, attempt_lead_capture

logger: logging.Logger = logging.getLogger(__name__)


def tool_node(state: AgentState) -> dict:
    """Terminal action node. Attempts lead capture and returns a confirmation or
    a graceful retry message if the gatekeeper blocks the capture."""

    record: LeadRecord | None = attempt_lead_capture(state)

    if record is None:
        lead = dict(state["lead_data"])

        if lead.get("name") == "":
            lead["name"] = None

        updated_lead = LeadData(
            name=lead.get("name"),
            email=lead.get("email"),
            platform=lead.get("platform"),
        )

        missing = next(
            (f for f in ("name", "email", "platform") if not lead.get(f)),
            "name"
        )
        question = f"Let me just confirm — could you share your {missing}?"
        logger.warning("Lead capture failed, re-requesting field: %s", missing)
        return {
            "messages": [AIMessage(content=question)],
            "lead_data": updated_lead,
            "awaiting_lead_field": missing,
            "lead_captured": False,
            "intent": "high_intent",
        }

    logger.info(
        "Lead captured successfully: name=%s, email=%s, platform=%s",
        record.name,
        record.email,
        record.platform,
    )

    confirmation: str = (
        f"You're all set, {record.name}! We've received your details and our "
        f"team will reach out to your {record.platform} shortly. Check your "
        f"inbox at {record.email} for next steps."
    )

    return {
        "messages": [AIMessage(content=confirmation)],
        "lead_captured": True,
        "intent": state["intent"],
    }