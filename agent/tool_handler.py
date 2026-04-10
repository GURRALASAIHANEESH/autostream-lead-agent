import logging

from langchain_core.messages import AIMessage

from agent.state_manager import AgentState
from tools.lead_capture import LeadRecord, attempt_lead_capture

logger: logging.Logger = logging.getLogger(__name__)


def tool_node(state: AgentState) -> dict:
    """Terminal action node. Attempts lead capture and returns a confirmation or
    a graceful retry message if the gatekeeper blocks the capture."""

    record: LeadRecord | None = attempt_lead_capture(state)

    if record is None:
        logger.warning(
            "Lead capture gatekeeper blocked the attempt; requesting re-confirmation."
        )
        return {
            "messages": [AIMessage(content=(
                "I seem to have lost some of your details. "
                "Could you share your name, email, and platform again?"
            ))],
            "lead_captured": False,
            "lead_data": {"name": None, "email": None, "platform": None},
            "awaiting_lead_field": None,
            "intent": state["intent"],
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