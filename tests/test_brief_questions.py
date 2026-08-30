"""End-to-end checks against the four questions in the brief.

These exercise the agent loop and therefore need GEMINI_API_KEY; they skip
cleanly without one so the rest of the suite still runs. The deterministic
behaviour each question depends on is covered without a key in
`test_service.py` - these verify that the model routes to the right tools and
carries the required caveats into its answer.
"""

from __future__ import annotations

import os

import pytest

from airport_iq.agent.session import ConversationSession

pytestmark = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set - agent tests skipped",
)


@pytest.fixture
def session() -> ConversationSession:
    return ConversationSession()


def tool_names(turn) -> set[str]:
    return {call.name for call in turn.tool_calls}


def test_q1_new_england_terminal_expansion(session):
    """Must filter by region and choose the terminal-weighted profile."""
    turn = session.ask(
        "Which airports in New England are strong candidates for terminal expansion?"
    )
    assert "rank_airports" in tool_names(turn)

    ranking = next(c for c in turn.tool_calls if c.name == "rank_airports")
    assert ranking.arguments.get("region") == "new_england"
    assert ranking.arguments.get("weight_profile") == "terminal"

    assert "BOS" in turn.text or "Logan" in turn.text


def test_q2_compare_la_and_santa_ana(session):
    """Must resolve both colloquial names and flag the hub-class mismatch."""
    turn = session.ask("Compare LA and Santa Ana airport congestion levels.")
    used = tool_names(turn)
    assert used & {"compare_airports", "resolve_airport"}

    assert "LAX" in turn.text
    assert "SNA" in turn.text
    lowered = turn.text.lower()
    assert "hub" in lowered, "should note the airports are different hub classes"


def test_q3_anchorage_long_haul_is_reported_as_unavailable(session):
    """The honest answer: say the percentage cannot be computed, do not invent one."""
    turn = session.ask(
        "What is the percentage of long haul flights out of Anchorage airport?"
    )
    assert "get_long_haul_share" in tool_names(turn)

    lowered = turn.text.lower()
    assert any(
        phrase in lowered
        for phrase in ("not available", "cannot", "can't", "unavailable", "do not have")
    ), f"expected an explicit unavailability statement, got: {turn.text[:400]}"


def test_q4_sfo_unmet_demand_explains_why(session):
    """Must call the scoring engine and explain using its components."""
    turn = session.ask("What is the unmet flight demand in SFO airport and why?")
    assert tool_names(turn) & {"get_airport_profile", "explain_score"}

    lowered = turn.text.lower()
    assert "sfo" in lowered
    assert any(term in lowered for term in ("unmet", "demand", "pressure"))


def test_follow_up_keeps_context(session):
    """Conversational follow-ups are a stated requirement."""
    session.ask("Which airports in New England are strong candidates for expansion?")
    history_after_first = len(session._chat.get_history())

    follow_up = session.ask("Why is the first one ranked above the second?")

    assert follow_up.text
    # The model should not need to be told the airports again - Gemini's Chat
    # keeps history internally, so the transcript should simply keep growing
    # across turns rather than the follow-up starting from a blank slate.
    assert len(session._chat.get_history()) > history_after_first


def test_agent_does_not_invent_numbers_for_unknown_airports(session):
    turn = session.ask("What is the investment score for Narnia International Airport?")
    lowered = turn.text.lower()
    assert any(
        phrase in lowered
        for phrase in ("not find", "no airport", "could not", "couldn't", "not in")
    )
