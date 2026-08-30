"""HTTP API.

Deliberately exposes the scoring engine on its own endpoints, separate from
`/chat`. `/rank` answers with no language model involved anywhere in the call
path - which is what makes "the ranking is deterministic, not LLM output" a
demonstrable claim rather than an assertion.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import service as svc
from ..agent.session import ConversationSession, api_key_present
from ..ingest import load
from ..providers.routes import get_provider

log = logging.getLogger(__name__)

app = FastAPI(
    title="Airport Investment Intelligence",
    description=(
        "Deterministic airport investment scoring, plus a chat agent that "
        "reads from it. The /rank, /airport and /compare endpoints involve no "
        "language model."
    ),
    version="0.1.0",
)

# Sessions are held in memory, keyed by an opaque id the client passes back.
# Fine for a single-process demo; a real deployment would use a shared store.
_SESSIONS: dict[str, ConversationSession] = {}


@app.get("/health")
def health() -> dict[str, Any]:
    airports = load()
    return {
        "status": "ok",
        "airports_scored": len(airports),
        "gemini_key_configured": api_key_present(),
        "long_haul_provider": type(get_provider(airports)).__name__,
    }


# --------------------------------------------------------------------------
# Deterministic endpoints - no LLM in the call path.
# --------------------------------------------------------------------------


@app.get("/rank")
def rank(
    region: str | None = None,
    state: str | None = None,
    hub_class: str | None = None,
    weight_profile: str | None = None,
    top_n: int = Query(default=5, ge=1, le=1000),
    min_enplanements: int = Query(default=svc.DEFAULT_MIN_ENPLANEMENTS, ge=0),
    within_hub_class: bool = False,
) -> dict[str, Any]:
    """Rank airports by investment potential. Pure scoring engine."""
    result = svc.rank(
        region=region,
        state=state,
        hub_class=hub_class,
        profile=weight_profile,
        top_n=top_n,
        min_enplanements=min_enplanements,
        within_hub_class=within_hub_class,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result)
    return result


@app.get("/airport/{iata}")
def airport(iata: str, weight_profile: str | None = None) -> dict[str, Any]:
    """Full scored breakdown for one airport."""
    result = svc.profile_airport(iata, profile=weight_profile)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/compare")
def compare(
    codes: str = Query(description="Comma-separated airports, e.g. LAX,SNA"),
    weight_profile: str | None = None,
) -> dict[str, Any]:
    """Compare airports side by side."""
    parsed = [c.strip() for c in codes.split(",") if c.strip()]
    result = svc.compare(parsed, profile=weight_profile)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result)
    return result


@app.get("/resolve")
def resolve(query: str) -> dict[str, Any]:
    """Resolve free text to an airport."""
    return svc.resolve(query)


@app.get("/long-haul/{iata}")
def long_haul(iata: str) -> dict[str, Any]:
    """Long-haul share of departures, or an explicit statement that it is unavailable."""
    return get_provider(load()).long_haul_share(iata).to_dict()


@app.get("/options")
def options() -> dict[str, Any]:
    """Available weight profiles, regions and hub classes."""
    return {
        "weight_profiles": svc.available_profiles(),
        "regions": svc.available_regions(),
        "hub_classes": ["large_hub", "medium_hub", "small_hub", "nonhub"],
    }


# --------------------------------------------------------------------------
# Agent endpoint.
# --------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(description="The analyst's question.")
    session_id: str | None = Field(
        default=None, description="Omit to start a new conversation."
    )


@app.post("/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    """Ask the agent. Streams newline-delimited JSON events.

    Event types: `session`, `tool_call`, `text`, `error`, `done`.
    """
    if not api_key_present():
        raise HTTPException(
            status_code=503,
            detail=(
                "GEMINI_API_KEY is not set. The deterministic endpoints "
                "(/rank, /airport, /compare) work without it."
            ),
        )

    session_id = request.session_id or uuid.uuid4().hex
    session = _SESSIONS.setdefault(session_id, ConversationSession())

    def events():
        yield json.dumps({"type": "session", "session_id": session_id}) + "\n"
        try:
            for event in session.stream(request.message):
                yield json.dumps(event, default=str) + "\n"
        except Exception as exc:  # surface failures to the client, not just the log
            log.exception("chat stream failed")
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.delete("/chat/{session_id}")
def reset_session(session_id: str) -> dict[str, str]:
    _SESSIONS.pop(session_id, None)
    return {"status": "reset", "session_id": session_id}


# --------------------------------------------------------------------------
# Web terminal UI - static, same-origin, no CORS needed. Mounted last so it
# never shadows an API route above.
# --------------------------------------------------------------------------

_WEB_DIR = Path(__file__).resolve().parents[3] / "ui" / "web"
if _WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")
