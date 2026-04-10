# AutoStream Agent - Social-to-Lead Agentic Workflow

AutoStream Agent is a conversational AI system that qualifies inbound leads from social media content creators. Built on LangGraph and LangChain, it classifies user intent, answers product questions using retrieval-augmented generation against a knowledge base, and collects structured lead information through a multi-turn guided conversation. The system exposes both a CLI interface and a FastAPI service with a Twilio-compatible WhatsApp webhook. It was built as a Machine Learning internship assignment to demonstrate agentic workflow design, strict tool-calling logic, and clean state management in a production-style Python codebase.

## Architecture

### Component Overview

| Component | File | Responsibility |
|---|---|---|
| Orchestrator | `agent/orchestrator.py` | Defines the LangGraph StateGraph, wires all nodes and conditional edges, exposes the compiled graph and `run_turn` helper |
| Intent Classifier | `agent/intent_classifier.py` | Uses structured LLM output to classify each user message as `greeting`, `product_query`, or `high_intent` |
| State Manager | `agent/state_manager.py` | Defines `AgentState` TypedDict, `LeadData`, field-ordering logic, and state factory functions |
| RAG Loader | `rag/loader.py` | Loads product knowledge base documents, chunks them, and builds the FAISS vector store |
| RAG Retriever | `rag/retriever.py` | Singleton `KnowledgeRetriever` that queries the FAISS index and returns joined context chunks |
| Tool Handler | `agent/tool_handler.py` | Terminal action node that calls `attempt_lead_capture` and composes the confirmation or retry message |
| Lead Capture Tool | `tools/lead_capture.py` | Pydantic `LeadRecord` model, `mock_lead_capture` constructor, and `attempt_lead_capture` gatekeeper function |
| FastAPI Layer | `api/app.py` | HTTP interface with `/chat`, `/session`, `/health`, and `/webhook/whatsapp` endpoints; in-memory session store |

### Conversation Flow

A complete conversation follows this six-turn sequence through the graph:

1. The user sends an initial greeting (e.g., "Hi there"). The `classify_intent_node` labels this as `greeting`. The conditional edge routes to `greet_node`, which generates a warm one-to-two sentence response mentioning that the agent can help with pricing, features, or getting started. The turn ends.

2. The user asks a product question (e.g., "What does the Pro plan include?"). The `classify_intent_node` labels this as `product_query`. The conditional edge routes to `rag_node`, which retrieves relevant chunks from the FAISS knowledge base, injects them into a grounded system prompt, and generates an answer strictly from the retrieved context. The turn ends.

3. The user expresses purchase or signup intent (e.g., "I'd like to get started with Pro"). The `classify_intent_node` labels this as `high_intent`. The conditional edge routes to `lead_collect_node`. The node detects that `name` is the first missing field, generates a natural question asking for it via LLM, sets `awaiting_lead_field` to `"name"`, and ends the turn.

4. The user provides their name (e.g., "Haneesh"). The graph routes back through `classify_intent_node`, which again classifies this as `high_intent` given the conversation context. The `lead_collect_node` reads `awaiting_lead_field == "name"`, stores the raw user input into `lead_data["name"]`, determines `email` is the next missing field, asks for it, and sets `awaiting_lead_field` to `"email"`. The turn ends.

5. The user provides their email and, on the subsequent turn, their platform. Each turn follows the same store-then-ask cycle in `lead_collect_node`. After the platform is stored, `get_missing_lead_field` returns `None`, indicating all fields are collected. The node returns without appending a message.

6. The `route_after_lead_collect` conditional edge detects that `is_lead_complete` is `True` and `lead_captured` is `False`, routing to `tool_node`. The tool node calls `attempt_lead_capture`, which validates the data through the `LeadRecord` Pydantic model, returns the record, and the node composes a confirmation message with the user's name, email, and platform. The `lead_captured` flag is set to `True`. The turn ends and the session is complete.

### Why LangGraph

LangGraph was chosen over a linear LangChain chain because the workflow requires explicit branching based on classified intent, multi-turn state accumulation for lead collection, and a clear separation between routing logic and LLM calls. The `AgentState` TypedDict provides a typed contract that every node reads from and writes to, making the data flow between nodes auditable and predictable. Conditional edges are pure Python functions that inspect state fields and return node names, with no hidden prompt engineering driving the routing. Each node is an isolated function with a defined input and output signature, which makes unit testing straightforward without invoking the full graph. The `lead_captured` boolean serves as an idempotency guard inside `attempt_lead_capture`, ensuring the tool node cannot fire twice for the same session even if the graph is re-invoked.

### State Management

`AgentState` is a `TypedDict` with six fields: `messages` (the full conversation as `BaseMessage` objects), `intent` (the latest classification label), `lead_data` (a nested `LeadData` TypedDict with `name`, `email`, and `platform`), `lead_captured` (boolean flag), `rag_context` (the most recent retrieval result), and `awaiting_lead_field` (the field name currently being requested from the user, or `None`).

Each node function returns a partial dictionary containing only the fields it modifies. LangGraph merges this partial return into the existing state, leaving untouched fields intact. This merge behavior means nodes do not need to copy or forward fields they do not own.

The `lead_captured` flag acts as an idempotency guard: `attempt_lead_capture` checks this flag and returns `None` immediately if it is already `True`, preventing duplicate captures regardless of how many times the graph is invoked. The `awaiting_lead_field` mechanism tracks which field the agent last asked about. When the user responds, `lead_collect_node` reads this field to know which slot in `lead_data` should receive the raw user input, then clears it and either sets it to the next missing field or leaves it as `None` when the lead is complete. Fields are collected in a deterministic order enforced by `get_missing_lead_field`: name, then email, then platform.

## Setup

### Prerequisites

- Python 3.11 or later
- An API key for at least one supported LLM provider: OpenAI, Google (Gemini), or Anthropic

### Installation

```bash
git clone https://github.com/your-username/autostream-agent.git
cd autostream-agent
```

```bash
python -m venv .venv
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Open `.env` and set your API key and provider:

```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

### Running the Agent

```bash
python main.py
```

The agent prints a greeting on startup and enters an interactive loop. Type `quit` or `exit` to stop.

### Running the API

```bash
uvicorn api.app:app --reload
```

Send a message to the agent:

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-001", "message": "Hi, what features does AutoStream have?"}' | python -m json.tool
```

Check session state:

```bash
curl -s http://localhost:8000/session/test-001 | python -m json.tool
```

### Running Tests

```bash
pytest tests/ -v
```

All tests mock external dependencies. No LLM API key or FAISS index is required to run the test suite.

## WhatsApp Deployment

The `/webhook/whatsapp` endpoint accepts Twilio-style POST requests with `Body` (the user message text) and `From` (the sender phone number) as form fields. The `From` value is used as the `session_id`, so each phone number gets an isolated conversation state. The endpoint calls `run_turn` internally and returns a TwiML XML response that Twilio forwards back to the user on WhatsApp. To deploy: create a Twilio account and configure a WhatsApp-enabled number, run the API locally behind ngrok for development, and set the webhook URL in the Twilio console to `https://{your-ngrok-subdomain}.ngrok.io/webhook/whatsapp` with the HTTP POST method.

Test the webhook locally:

```bash
curl -s -X POST http://localhost:8000/webhook/whatsapp \
  -d "Body=Hi%2C+I+want+to+sign+up&From=%2B15551234567"
```

## Project Structure

```
autostream-agent/
    .env.example                  Environment variable template
    requirements.txt              Python dependencies
    config.py                     Settings singleton and LLM factory
    main.py                       CLI entrypoint with interactive loop
    agent/
        __init__.py
        state_manager.py          AgentState TypedDict and state helpers
        intent_classifier.py      LLM-based intent classification
        orchestrator.py           LangGraph StateGraph definition and run_turn
        tool_handler.py           tool_node for lead capture confirmation
    rag/
        __init__.py
        loader.py                 Document loading, chunking, FAISS index build
        retriever.py              KnowledgeRetriever singleton
        knowledge_base/           Product knowledge base source documents
    tools/
        __init__.py
        lead_capture.py           LeadRecord model and capture gatekeeper
    api/
        __init__.py
        app.py                    FastAPI routes and WhatsApp webhook
    tests/
        __init__.py
        test_intent.py            Intent classifier and state helper tests
        test_rag.py               RAG retriever tests
        test_lead_capture.py      Lead capture tool and gatekeeper tests
```
## Authors
Gurrala Sai Haneesh
