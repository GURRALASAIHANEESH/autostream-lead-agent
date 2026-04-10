import logging
import re

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from agent.intent_classifier import classify_intent
from agent.state_manager import (
    AgentState,
    LEAD_FIELD_DISPLAY_NAMES,
    get_missing_lead_field,
    is_lead_complete,
)
from agent.tool_handler import tool_node
from config import get_llm, get_settings
from rag.retriever import retriever

logger: logging.Logger = logging.getLogger(__name__)


def _get_last_human_message(messages: list[BaseMessage]) -> str:
    """Walk the message list in reverse and return the content of the most
    recent HumanMessage.  Returns an empty string when none is found."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""

def classify_intent_node(state: AgentState) -> dict:
    """Classifies the latest user message and trims the conversation window."""

    messages: list[BaseMessage] = state["messages"]
    last_human_text: str = _get_last_human_message(messages)
    history: list[BaseMessage] = messages[:-1]

    label: str = classify_intent(last_human_text, history=history)

    max_entries: int = get_settings().MAX_CONVERSATION_TURNS * 2
    trimmed_messages: list[BaseMessage] = messages[-max_entries:]

    logger.info(
        "Intent classified as '%s' for message: %.80s", label, last_human_text
    )

    return {
        "intent": label,
        "messages": trimmed_messages,
    }


def greet_node(state: AgentState) -> dict:
    """Generates a warm, brief greeting using a higher-temperature LLM."""

    llm: BaseChatModel = get_llm(temperature=0.7)

    system_prompt: str = (
        "You are a friendly assistant for AutoStream, a SaaS video editing tool "
        "for content creators. Respond naturally to greetings. Keep it brief — "
        "1 to 2 sentences. Mention you can help with pricing, features, or "
        "getting started."
    )

    last_human_text: str = _get_last_human_message(state["messages"])

    response: BaseMessage = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_human_text),
    ])

    return {
        "messages": [AIMessage(content=str(response.content))],
        "intent": "greeting",
    }


def rag_node(state: AgentState) -> dict:
    """Retrieves KB context and generates a grounded, factual answer."""

    last_human_text: str = _get_last_human_message(state["messages"])
    context: str = retriever.retrieve(last_human_text)

    system_prompt: str = (
        "Answer ONLY using the information in the context below. If the answer "
        "is not in the context, say you don't have that information and offer to "
        "connect them with the team. Do not invent features, prices, or policies."
        "\n\nContext:\n"
        f"{context}"
    )

    llm: BaseChatModel = get_llm(temperature=0.3)

    response: BaseMessage = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_human_text),
    ])

    return {
        "messages": [AIMessage(content=str(response.content))],
        "rag_context": context,
        "intent": "product_query",
    }


def lead_collect_node(state: AgentState) -> dict:
    """Collects one lead field per turn.  When all fields are present the node
    returns without appending a message so the conditional edge can route
    directly to tool_node."""

    lead_data: dict[str, str | None] = dict(state["lead_data"])
    awaiting: str | None = state["awaiting_lead_field"]
    last_human_text: str = _get_last_human_message(state["messages"])

    # Step A -- store the answer to a previously asked field
    if awaiting is not None:
        raw_answer = last_human_text.strip()

        if awaiting == "name":
            extraction_prompt_text = (
                "Extract only the person's name from this message. "
                "Return just the name, nothing else."
            )
        elif awaiting == "email":
            extraction_prompt_text = (
                "Extract only the email address from this message. "
                "Return just the email address, nothing else."
            )
        else:
            extraction_prompt_text = None

        if extraction_prompt_text:
            extraction_llm = get_llm(temperature=0.0)
            extracted = extraction_llm.invoke([
                SystemMessage(content=extraction_prompt_text),
                HumanMessage(content=raw_answer),
            ])
            field_value = str(extracted.content).strip()
        else:
            field_value = raw_answer

        if awaiting == "email" and not _EMAIL_RE.match(field_value):
            question = f"That doesn't look like a valid email address. Could you share a valid email for {state['lead_data'].get('name', 'you')}?"
            return {
                "messages": [AIMessage(content=question)],
                "lead_data": state["lead_data"],
                "awaiting_lead_field": "email",
                "intent": "high_intent",
            }

        lead_data[awaiting] = field_value
        logger.info("Stored lead field '%s'.", awaiting)
        awaiting = None

    # Snapshot used for the completeness check so that get_missing_lead_field
    # sees the value we just stored.
    updated_snapshot: AgentState = {
        **state,
        "lead_data": lead_data,
        "awaiting_lead_field": awaiting,
    }

    # Step B -- if all fields are present, hand off to tool_node
    next_field: str | None = get_missing_lead_field(updated_snapshot)
    if next_field is None:
        logger.info("All lead fields collected; handing off to tool_node.")
        return {
            "lead_data": lead_data,
            "awaiting_lead_field": None,
            "intent": "high_intent",
        }

    # Step C -- ask for the next missing field via LLM
    display_name: str = LEAD_FIELD_DISPLAY_NAMES[next_field]
    llm: BaseChatModel = get_llm(temperature=0.5)

    system_prompt: str = (
        "You are a friendly assistant for AutoStream, a SaaS video editing tool "
        "for content creators. The user has expressed interest in getting started. "
        f"Ask for their {display_name} in a natural, conversational way. "
        "Keep it to one sentence. Do not repeat information the user already provided."
    )

    response: BaseMessage = llm.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"][-4:],
    ])

    logger.info("Requesting lead field '%s' from user.", next_field)

    return {
        "messages": [AIMessage(content=str(response.content))],
        "lead_data": lead_data,
        "awaiting_lead_field": next_field,
        "intent": "high_intent",
    }



_INTENT_TO_NODE: dict[str, str] = {
    "greeting": "greet_node",
    "product_query": "rag_node",
    "high_intent": "lead_collect_node",
}


def route_after_classification(state: AgentState) -> str:
    """Routes to the appropriate handler based on classified intent."""
    return _INTENT_TO_NODE.get(state["intent"], "greet_node")


def route_after_lead_collect(state: AgentState) -> str:
    """Routes to tool_node when lead data is complete and not yet captured,
    otherwise terminates the turn so the user can respond."""
    if is_lead_complete(state) and not state["lead_captured"]:
        return "tool_node"
    return END

graph: StateGraph = StateGraph(AgentState)

graph.add_node("classify_intent_node", classify_intent_node)
graph.add_node("greet_node", greet_node)
graph.add_node("rag_node", rag_node)
graph.add_node("lead_collect_node", lead_collect_node)
graph.add_node("tool_node", tool_node)

graph.set_entry_point("classify_intent_node")

graph.add_conditional_edges(
    "classify_intent_node",
    route_after_classification,
    {
        "greet_node": "greet_node",
        "rag_node": "rag_node",
        "lead_collect_node": "lead_collect_node",
    },
)

graph.add_edge("greet_node", END)
graph.add_edge("rag_node", END)

graph.add_conditional_edges(
    "lead_collect_node",
    route_after_lead_collect,
    {
        "tool_node": "tool_node",
        END: END,
    },
)

graph.add_edge("tool_node", END)

agent = graph.compile()


def run_turn(user_message: str, state: AgentState) -> tuple[str, AgentState]:
    """Executes a single conversational turn.

    Appends the user message, invokes the compiled graph, and extracts the
    assistant's reply from the resulting state.
    """
    input_state: AgentState = {
        **state,
        "messages": [*state["messages"], HumanMessage(content=user_message)],
    }

    new_state: AgentState = agent.invoke(input_state)

    response_text: str = ""
    for message in reversed(new_state["messages"]):
        if isinstance(message, AIMessage):
            response_text = str(message.content)
            break

    return response_text, new_state