from __future__ import annotations

import logging
from datetime import datetime, timezone

from pydantic import BaseModel, EmailStr, Field, ValidationError, field_validator

from agent.state_manager import AgentState, is_lead_complete

logger: logging.Logger = logging.getLogger(__name__)


class LeadRecord(BaseModel):
    """Immutable, validated snapshot of a captured lead."""

    name: str
    email: EmailStr
    platform: str
    captured_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    @field_validator("name")
    @classmethod
    def _strip_and_validate_name(cls, v: str) -> str:
        stripped: str = v.strip()
        if not stripped:
            raise ValueError(
                "name must not be empty after stripping whitespace"
            )
        return stripped

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, v: str) -> str:

        return v.lower()

    @field_validator("platform")
    @classmethod
    def _strip_and_validate_platform(cls, v: str) -> str:
        stripped: str = v.strip()
        if not stripped:
            raise ValueError(
                "platform must not be empty after stripping whitespace"
            )
        return stripped

def mock_lead_capture(name: str, email: str, platform: str) -> LeadRecord:
    """Validate inputs, persist a mock lead record, and return it."""
    record: LeadRecord = LeadRecord(
        name=name,
        email=email,
        platform=platform,
    )

    # Required by the assignment specification in exactly this format.
    print(f"Lead captured: {record.name} | {record.email} | {record.platform}")

    logger.info(
        "Lead captured: %s",
        record.model_dump_json(),
    )

    return record

def attempt_lead_capture(state: AgentState) -> LeadRecord | None:
    """Pre-flight check → validate → fire the lead capture tool."""
    if not is_lead_complete(state):
        logger.warning(
            "Lead capture attempted with incomplete data: %r",
            state["lead_data"],
        )
        return None

    if state["lead_captured"]:
        logger.warning("Lead already captured, skipping duplicate trigger")
        return None


    name: str | None = state["lead_data"]["name"]
    email: str | None = state["lead_data"]["email"]
    platform: str | None = state["lead_data"]["platform"]

    if name is None or email is None or platform is None:
        logger.error(
            "Lead data contains None values despite passing completeness "
            "check — possible state corruption: name=%r email=%r platform=%r",
            name,
            email,
            platform,
        )
        return None

 
    try:
        record: LeadRecord = mock_lead_capture(
            name=name,
            email=email,
            platform=platform,
        )
    except ValidationError as exc:
        logger.error(
            "Lead validation failed for data (name=%r, email=%r, "
            "platform=%r): %s",
            name,
            email,
            platform,
            exc,
        )
        return None

    return record