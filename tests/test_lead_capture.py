# === FILE: tests/test_lead_capture.py ===

import logging
from datetime import datetime

import pytest
from pydantic import ValidationError

from agent.state_manager import AgentState, create_initial_state
from tools.lead_capture import LeadRecord, attempt_lead_capture, mock_lead_capture

logger: logging.Logger = logging.getLogger(__name__)


@pytest.fixture
def complete_state() -> AgentState:
    """AgentState with all lead fields populated and lead_captured=False."""
    state: AgentState = create_initial_state()
    state["lead_data"]["name"] = "Haneesh"
    state["lead_data"]["email"] = "haneesh@example.com"
    state["lead_data"]["platform"] = "YouTube"
    state["lead_captured"] = False
    return state


@pytest.fixture
def partial_state() -> AgentState:
    """AgentState with only name populated."""
    state: AgentState = create_initial_state()
    state["lead_data"]["name"] = "Haneesh"
    return state


@pytest.fixture
def captured_state(complete_state: AgentState) -> AgentState:
    """AgentState with all fields set and lead already captured."""
    complete_state["lead_captured"] = True
    return complete_state


@pytest.fixture
def invalid_email_state() -> AgentState:
    """AgentState with all fields set but an invalid email value."""
    state: AgentState = create_initial_state()
    state["lead_data"]["name"] = "Haneesh"
    state["lead_data"]["email"] = "bad-email"
    state["lead_data"]["platform"] = "YouTube"
    state["lead_captured"] = False
    return state


def test_mock_lead_capture_returns_lead_record() -> None:
    result: LeadRecord = mock_lead_capture(
        "Haneesh", "haneesh@example.com", "YouTube"
    )

    assert isinstance(result, LeadRecord)
    assert result.name == "Haneesh"
    assert result.email == "haneesh@example.com"
    assert result.platform == "YouTube"


def test_mock_lead_capture_lowercases_email() -> None:
    result: LeadRecord = mock_lead_capture(
        "Haneesh", "HANEESH@EXAMPLE.COM", "YouTube"
    )

    assert result.email == "haneesh@example.com"


def test_mock_lead_capture_strips_whitespace() -> None:
    result: LeadRecord = mock_lead_capture(
        "  Haneesh  ", "haneesh@example.com", "  YouTube  "
    )

    assert result.name == "Haneesh"
    assert result.platform == "YouTube"


def test_mock_lead_capture_raises_on_empty_name() -> None:
    with pytest.raises((ValueError, ValidationError)):
        mock_lead_capture("   ", "haneesh@example.com", "YouTube")


def test_mock_lead_capture_raises_on_invalid_email() -> None:
    with pytest.raises(ValidationError):
        mock_lead_capture("Haneesh", "not-an-email", "YouTube")


def test_attempt_lead_capture_returns_none_when_fields_missing(
    partial_state: AgentState,
) -> None:
    result: LeadRecord | None = attempt_lead_capture(partial_state)

    assert result is None


def test_attempt_lead_capture_returns_none_when_already_captured(
    captured_state: AgentState,
) -> None:
    result: LeadRecord | None = attempt_lead_capture(captured_state)

    assert result is None


def test_attempt_lead_capture_succeeds_when_complete(
    complete_state: AgentState,
) -> None:
    result: LeadRecord | None = attempt_lead_capture(complete_state)

    assert result is not None
    assert isinstance(result, LeadRecord)


def test_attempt_lead_capture_never_raises(
    invalid_email_state: AgentState,
) -> None:
    # Must not raise; the gatekeeper catches validation errors internally.
    result: LeadRecord | None = attempt_lead_capture(invalid_email_state)

    assert result is None



def test_lead_record_captured_at_is_set_automatically() -> None:
    result: LeadRecord = mock_lead_capture(
        "Haneesh", "haneesh@example.com", "YouTube"
    )

    assert result.captured_at is not None
    assert isinstance(result.captured_at, datetime)