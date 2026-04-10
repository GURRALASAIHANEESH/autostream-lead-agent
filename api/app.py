import logging
import traceback
from typing import Any

from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, field_validator

from agent.orchestrator import run_turn
from agent.state_manager import AgentState, create_initial_state
from config import get_settings

logger: logging.Logger = logging.getLogger(__name__)

_sessions: dict[str, AgentState] = {}



class ChatRequest(BaseModel):
    session_id: str
    message: str

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message cannot be empty")
        return v.strip()


class ChatResponse(BaseModel):
    session_id: str
    response: str
    intent: str
    lead_captured: bool
    awaiting_field: str | None


class SessionSummary(BaseModel):
    session_id: str
    intent: str
    lead_captured: bool
    awaiting_field: str | None
    lead_data: dict[str, str | None]
    turn_count: int


class DeleteResponse(BaseModel):
    deleted: bool


class HealthResponse(BaseModel):
    status: str
    model: str
    provider: str


app: FastAPI = FastAPI(
    title="AutoStream Agent API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_or_create_session(session_id: str) -> AgentState:
    """Retrieve an existing session or initialise a new one."""
    if session_id not in _sessions:
        logger.info("Creating new session: %s", session_id)
        _sessions[session_id] = create_initial_state()
    return _sessions[session_id]


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a single conversational turn for the given session."""

    logger.debug(
        "Incoming /chat — session_id=%s, message=%.120s",
        request.session_id,
        request.message,
    )

    try:
        state: AgentState = _get_or_create_session(request.session_id)
        response_text, new_state = run_turn(request.message, state)
        _sessions[request.session_id] = new_state

        return ChatResponse(
            session_id=request.session_id,
            response=response_text,
            intent=new_state["intent"],
            lead_captured=new_state["lead_captured"],
            awaiting_field=new_state["awaiting_lead_field"],
        )

    except Exception as exc:
        logger.error(
            "Error in /chat for session %s:\n%s",
            request.session_id,
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again.",
        ) from exc


@app.get("/session/{session_id}", response_model=SessionSummary)
async def get_session(session_id: str) -> SessionSummary:
    """Return a lightweight summary of the current session state."""

    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    state: AgentState = _sessions[session_id]

    return SessionSummary(
        session_id=session_id,
        intent=state["intent"],
        lead_captured=state["lead_captured"],
        awaiting_field=state["awaiting_lead_field"],
        lead_data=dict(state["lead_data"]),
        turn_count=len(state["messages"]) // 2,
    )


@app.delete("/session/{session_id}", response_model=DeleteResponse)
async def delete_session(session_id: str) -> DeleteResponse:
    """Remove a session from the in-memory store."""

    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    del _sessions[session_id]
    logger.info("Deleted session: %s", session_id)

    return DeleteResponse(deleted=True)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness / readiness probe."""

    settings = get_settings()

    return HealthResponse(
        status="ok",
        model=settings.LLM_MODEL,
        provider=settings.LLM_PROVIDER,
    )


_TWIML_TEMPLATE: str = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    "<Response><Message>{body}</Message></Response>"
)


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(
    Body: str = Form(...),
    From: str = Form(...),
) -> Response:
    """Twilio-compatible WhatsApp webhook.

    Uses the sender phone number as the session identifier and returns a
    TwiML XML response that Twilio can forward back to the user.
    """

    session_id: str = From
    message: str = Body.strip()

    logger.debug(
        "Incoming /webhook/whatsapp — session_id=%s, message=%.120s",
        session_id,
        message,
    )

    if not message:
        twiml: str = _TWIML_TEMPLATE.format(
            body="Sorry, I didn't catch that. Could you try again?"
        )
        return Response(content=twiml, media_type="application/xml")

    try:
        state: AgentState = _get_or_create_session(session_id)
        response_text, new_state = run_turn(message, state)
        _sessions[session_id] = new_state
    except Exception:
        logger.error(
            "Error in /webhook/whatsapp for session %s:\n%s",
            session_id,
            traceback.format_exc(),
        )
        response_text = (
            "Something went wrong on our end. Please send your message again."
        )

    # Escape XML-special characters in the agent response to avoid malformed TwiML.
    safe_text: str = (
        response_text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )

    twiml = _TWIML_TEMPLATE.format(body=safe_text)
    return Response(content=twiml, media_type="application/xml")