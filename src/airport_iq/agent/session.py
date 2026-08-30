"""Conversation session: the chat wrapper and its history.

Built on Gemini's `chats.create` / `chat.send_message`, with automatic
function calling: passing plain typed functions in `tools` lets the SDK
generate their schemas, call them when the model asks, and feed results back
- all within one `send_message` call. `Chat` keeps its own history, so a
session here only needs to hold the `Chat` object, not a transcript.

Provider and model notes: this project was built against Claude, but the
assignment carries no LLM cost budget, so the chat runs on Gemini's free tier
(no card, no expiry) instead. It was first wired against Gemini's newer
Interactions API per web-fetched docs, which turned out to describe a surface
this SDK version doesn't actually route model calls through - requests
returned successfully but only after being reissued through the SDK's
documented `chats.create` path, which itself surfaced a runtime warning
pointing here. Separately, the newest model (`gemini-3.7-flash`) exhibited
70-160s per-turn latency on the free tier in testing, for a one-token reply
that used only 68 thinking tokens - clearly queuing, not reasoning time.
`gemini-3.1-flash-lite` answered the same prompts, including ones requiring a
tool call, in 2-6s, so it is the pinned default. This task's reasoning is
mostly tool selection and prose explanation over numbers the scoring engine
already computed, which a lite model handles well; a slower, more capable
model would be the better tradeoff for a task that leaned on the model's own
judgment more heavily.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Iterator

from google import genai
from google.genai import types

from .prompt import SYSTEM_PROMPT
from .tools import TOOLS

log = logging.getLogger(__name__)

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

# Fails fast on a stuck connection rather than hanging past the UI's own
# timeout with no information - the exact failure mode that motivated this.
REQUEST_TIMEOUT_MS = 60_000

# The SDK's default ceiling on automatic function calling within one
# send_message is 10. In testing, a single question made 11 tool calls -
# several exact duplicates - hit that ceiling, and the turn came back with no
# final text at all (the last internal step was another function call, which
# has no text part). Tightened to 6 on the theory that a rate-limited free
# tier should fail a misbehaving turn fast and cheap - but a single ranking
# question legitimately needs a resolve + a rank + a couple of per-airport
# follow-ups, and 6 cut off good-faith turns before they reached prose, not
# just runaway ones. Raised to 10 (the SDK's own default): the prompt now
# tells the model rank_airports/compare_airports already carry full component
# detail and resolve names internally, so a well-behaved turn should rarely
# need more than 1-3 calls - but a hard ceiling that eats the final answer,
# leaving the analyst with no response at all, is worse than a few spare
# calls on a misbehaving one.
MAX_REMOTE_CALLS = 10


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class ConversationSession:
    """One analyst conversation, wrapping a single Gemini `Chat`.

    `Chat` keeps the transcript internally (including tool calls and
    results), so follow-ups resolve against what was already said with no
    extra bookkeeping here.
    """

    def __init__(self, client: "genai.Client | None" = None) -> None:
        self._client = client or genai.Client(
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS)
        )
        self._chat = self._new_chat()

    def _new_chat(self):
        return self._client.chats.create(
            model=MODEL,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=TOOLS,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    maximum_remote_calls=MAX_REMOTE_CALLS
                ),
            ),
        )

    def reset(self) -> None:
        self._chat = self._new_chat()

    def ask(self, question: str) -> Turn:
        """Run one question to completion and return the final answer."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for event in self.stream(question):
            if event["type"] == "text":
                text_parts.append(event["text"])
            elif event["type"] == "tool_call":
                tool_calls.append(ToolCall(event["name"], event["arguments"]))

        return Turn(text="".join(text_parts), tool_calls=tool_calls)

    def stream(self, question: str) -> Iterator[dict[str, Any]]:
        """Run one question, yielding events as they occur.

        Not token-level streaming: automatic function calling resolves every
        tool call inside a single `send_message` call, so the tool_call and
        text events below are extracted from the finished turn's history
        rather than observed as they happen. That's enough for the UI to show
        which tools ran before the prose - it just arrives as one settled
        batch instead of a live trickle.
        """
        history_before = len(self._chat.get_history())

        try:
            response = self._chat.send_message(question)
        except Exception as exc:
            # This SDK's error hierarchy for the automatic-function-calling
            # path lives under a module that has moved across releases (see
            # the note above about the abandoned Interactions API), so this
            # duck-types on `status_code` - present on every HTTP error the
            # SDK raises - rather than importing an internal class.
            log.exception("Gemini request failed")
            status = getattr(exc, "status_code", None)
            prefix = f"API error ({status})" if status else "Could not reach the Gemini API"
            yield {"type": "error", "message": f"{prefix}: {exc}"}
            return

        for entry in self._chat.get_history()[history_before:]:
            for part in entry.parts or []:
                call = getattr(part, "function_call", None)
                if call:
                    yield {
                        "type": "tool_call",
                        "name": call.name,
                        "arguments": dict(call.args or {}),
                    }

        if response.text:
            yield {"type": "text", "text": response.text}
        else:
            # Hitting MAX_REMOTE_CALLS ends the turn on an internal function
            # call with no text part, so `response.text` is empty even though
            # tool_call events were already yielded above. Surfacing nothing
            # would look like a hang or a silent failure; say plainly what
            # happened instead of leaving a blank answer.
            log.warning("turn ended with no text after %d tool call(s)", len(
                [p for e in self._chat.get_history()[history_before:]
                 for p in (e.parts or []) if getattr(p, "function_call", None)]
            ))
            yield {
                "type": "text",
                "text": (
                    "I gathered data with several tool calls but didn't reach a "
                    "final answer in this turn. Please try rephrasing the "
                    "question more specifically (e.g. naming a region and a "
                    "project type) so it needs fewer lookups."
                ),
            }


def api_key_present() -> bool:
    """Whether a key is configured, so the UI can fail helpfully instead of 500."""
    return bool(os.getenv("GEMINI_API_KEY"))
