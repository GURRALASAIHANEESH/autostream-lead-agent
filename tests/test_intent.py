from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage
import pytest

import agent.intent_classifier as classifier_module
from agent.intent_classifier import IntentClassification, classify_intent
from agent.state_manager import (
    AgentState,
    create_initial_state,
    get_missing_lead_field,
    is_lead_complete,
)


def _make_chain_mock(intent_label: str) -> MagicMock:
    classification = IntentClassification(
        intent=intent_label,
        confidence=0.95,
        reasoning="test",
    )
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = classification
    return chain_mock


def _patch_classifier(intent_label: str):
    """
    Patches get_llm AND resets the module-level cache so _get_classifier_chain()
    rebuilds with the mock on every test.
    """
    chain_mock = _make_chain_mock(intent_label)

    llm_mock = MagicMock()
    llm_mock.with_structured_output.return_value = chain_mock

    return patch("agent.intent_classifier.get_llm", return_value=llm_mock), chain_mock


def test_greeting_classified_correctly() -> None:
    p, _ = _patch_classifier("greeting")
    with p:
        classifier_module._classifier_chain = None  # reset cache
        result = classify_intent("Hey there!", history=[])
    assert result == "greeting"


def test_product_query_classified_correctly() -> None:
    p, _ = _patch_classifier("product_query")
    with p:
        classifier_module._classifier_chain = None
        result = classify_intent("What does the Pro plan include?", history=[])
    assert result == "product_query"


def test_high_intent_classified_correctly() -> None:
    p, _ = _patch_classifier("high_intent")
    with p:
        classifier_module._classifier_chain = None
        result = classify_intent("I want to sign up for Pro", history=[])
    assert result == "high_intent"


def test_context_window_in_history() -> None:
    history = []
    for i in range(5):
        history.append(HumanMessage(content=f"user message {i}"))
        history.append(AIMessage(content=f"assistant response {i}"))
    assert len(history) == 10

    p, chain_mock = _patch_classifier("greeting")
    with p:
        classifier_module._classifier_chain = None
        classify_intent("hello", history=history)

    chain_mock.invoke.assert_called_once()

    call_args = chain_mock.invoke.call_args[0][0]  # positional arg: list[BaseMessage]
    human_message_content = call_args[1].content   # index 1 = HumanMessage
    # 4 history lines means at most 4 "User:"/"Agent:" prefixes
    history_lines = [
        line for line in human_message_content.split("\n")
        if line.startswith("User:") or line.startswith("Agent:")
    ]
    assert len(history_lines) <= 4

@pytest.fixture
def empty_state() -> AgentState:
    """Fresh agent state with all defaults."""
    return create_initial_state()

@pytest.fixture
def partial_state() -> AgentState:
    """State with only name populated."""
    state: AgentState = create_initial_state()
    state["lead_data"]["name"] = "Haneesh"
    return state

@pytest.fixture
def complete_state() -> AgentState:
    """State with all lead fields populated."""
    state: AgentState = create_initial_state()
    state["lead_data"]["name"] = "Haneesh"
    state["lead_data"]["email"] = "haneesh@example.com"
    state["lead_data"]["platform"] = "YouTube"
    return state

def test_get_missing_lead_field_returns_name_first(empty_state: AgentState) -> None:
    assert get_missing_lead_field(empty_state) == "name"

def test_get_missing_lead_field_returns_email_after_name(partial_state: AgentState) -> None:
    assert get_missing_lead_field(partial_state) == "email"

def test_get_missing_lead_field_returns_none_when_complete(complete_state: AgentState) -> None:
    assert get_missing_lead_field(complete_state) is None

def test_is_lead_complete_false_when_partial(partial_state: AgentState) -> None:
    assert is_lead_complete(partial_state) is False

def test_is_lead_complete_true_when_all_set(complete_state: AgentState) -> None:
    assert is_lead_complete(complete_state) is True

def test_create_initial_state_has_correct_defaults(empty_state: AgentState) -> None:
    assert empty_state["messages"] == []
    assert empty_state["intent"] == ""
    assert empty_state["lead_data"] == {"name": None, "email": None, "platform": None}
    assert empty_state["lead_captured"] is False
    assert empty_state["rag_context"] == ""
    assert empty_state["awaiting_lead_field"] is None