from __future__ import annotations

import logging

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from agent.state_manager import IntentLabel
from config import get_llm

logger: logging.Logger = logging.getLogger(__name__)


_SYSTEM_PROMPT: str = (
    "You are the intent classifier for AutoStream, a SaaS video editing platform.\n"
    "\n"
    "Your job: read the user's LATEST message (and recent conversation history "
    "when provided) and assign exactly one intent label.\n"
    "\n"
    "LABELS\n"
    "------\n"
    '"greeting"\n'
    "  The user is saying hello, making small talk, or asking a generic opening "
    "question with no product-related intent.\n"
    "\n"
    '"product_query"\n'
    "  The user is asking about features, pricing, plans, policies, refunds, "
    "support, comparisons, or technical capabilities of AutoStream.\n"
    "\n"
    '"high_intent"\n'
    "  The user expresses a desire to sign up, start a trial, purchase a plan, "
    "get a demo, or explicitly states they want to proceed.  ALSO classify as "
    '"high_intent" if the conversation indicates the agent is in the middle of '
    "collecting lead information (name, email, or content platform) and the "
    "user's message appears to be answering a question the agent previously "
    "asked — even if the answer is just a name, an email address, or a "
    'platform name like "YouTube".\n'
    "\n"
    "DISAMBIGUATION\n"
    "--------------\n"
    '- When in doubt between "product_query" and "high_intent", choose "product_query".\n'
    '- When in doubt between "greeting" and "product_query", choose "product_query".\n'
    "- Never return any label other than the three defined above.\n"
    "\n"
    "Respond with the intent label, a confidence score from 0.0 to 1.0, and a "
    "single sentence of reasoning explaining your choice."
)


_FALLBACK_INTENT: IntentLabel = "product_query"

_MAX_HISTORY_MESSAGES: int = 4


class IntentClassification(BaseModel):
    """Schema enforced on the LLM response via ``with_structured_output``."""

    intent: IntentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

_classifier_chain: Runnable | None = None


def _get_classifier_chain() -> Runnable:
    """Return (and cache) the structured-output runnable."""
    global _classifier_chain
    if _classifier_chain is None:
        llm = get_llm(temperature=0.0)
        _classifier_chain = llm.with_structured_output(IntentClassification)
        logger.info("Intent classifier chain initialised")
    return _classifier_chain


def _format_recent_history(history: list[BaseMessage]) -> str:
    """Only HumanMessage and AIMessage instances are included — system messages are internal scaffolding and would confuse the classifier prompt."""
    recent: list[BaseMessage] = history[-_MAX_HISTORY_MESSAGES:]
    lines: list[str] = []
    for msg in recent:
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Agent: {msg.content}")
    return "\n".join(lines)


def classify_intent(message: str, history: list[BaseMessage]) -> IntentLabel:
    """Classify a user message into ``greeting``, ``product_query``, or ``high_intent``"""
    chain: Runnable = _get_classifier_chain()

    content_parts: list[str] = []

    if history:
        formatted_history: str = _format_recent_history(history)
        if formatted_history:
            content_parts.append(f"Recent conversation:\n{formatted_history}")

    content_parts.append(f"New message to classify:\n{message}")

    human_content: str = "\n\n".join(content_parts)

    prompt_messages: list[BaseMessage] = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    try:
        result: IntentClassification = chain.invoke(prompt_messages)
    except Exception:
        logger.warning(
            "Intent classification LLM call failed for message: %r "
            "— returning fallback '%s'",
            message[:120],
            _FALLBACK_INTENT,
            exc_info=True,
        )
        return _FALLBACK_INTENT

    logger.debug(
        "Intent classified: label=%s  confidence=%.2f  reasoning=%r",
        result.intent,
        result.confidence,
        result.reasoning,
    )

    return result.intent