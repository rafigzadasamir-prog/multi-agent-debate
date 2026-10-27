"""LLM backend abstraction for the multi-agent debate framework.

Every agent role (Advocate, Critic, Judge) talks to an ``LLMBackend`` through the
same ``generate(request) -> str`` contract, so the orchestration logic in
:mod:`multi_agent_debate.debate` never needs to know which backend is in play.

Two backends implement that contract:

- :class:`AnthropicLLM` calls the real Anthropic Messages API. Used when an
  ``ANTHROPIC_API_KEY`` is available and real model output is wanted.
- :class:`multi_agent_debate.scripted_llm.ScriptedLLM` is a deterministic,
  offline, rule-based backend. It makes zero network calls, so the entire
  multi-round debate (sequencing, transcript structure, Judge synthesis) is
  fully unit-testable without any API key.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DebateTurn:
    """One already-spoken message, passed back in as context for the next turn."""

    role: str
    round_number: int
    content: str


@dataclass(frozen=True)
class GenerationRequest:
    """Everything a backend needs to produce the next message in the debate."""

    role: str
    round_number: int
    total_rounds: int
    question: str
    history: list[DebateTurn] = field(default_factory=list)
    system_prompt: str = ""


class LLMBackend(ABC):
    """Abstract interface implemented by every debate LLM backend."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> str:
        """Produce the next debate message's text for the given request."""


class AnthropicLLM(LLMBackend):
    """Backend that calls the real Anthropic Messages API.

    The ``client`` parameter exists specifically so tests can inject a fake
    client and exercise the request/response handling without any network
    access or API key.
    """

    DEFAULT_MODEL = "claude-opus-5"

    def __init__(
        self,
        model: str | None = None,
        client: object | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model or self.DEFAULT_MODEL
        self.max_tokens = max_tokens
        if client is not None:
            self._client = client
        else:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
            )

    def generate(self, request: GenerationRequest) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=request.system_prompt,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": self._render_user_content(request)}],
        )
        if getattr(response, "stop_reason", None) == "refusal":
            raise RuntimeError(
                f"Anthropic API declined to respond (stop_details={response.stop_details!r})"
            )
        return self._extract_text(response)

    @staticmethod
    def _render_user_content(request: GenerationRequest) -> str:
        lines = [f"Question under debate: {request.question}", ""]
        if request.history:
            lines.append("Debate transcript so far:")
            for turn in request.history:
                lines.append(f"[round {turn.round_number}] {turn.role.upper()}: {turn.content}")
            lines.append("")
        lines.append(
            f"You are now speaking as the {request.role.upper()} in round "
            f"{request.round_number} of {request.total_rounds}. Respond in that role now."
        )
        return "\n".join(lines)

    @staticmethod
    def _extract_text(response: object) -> str:
        parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        text = "".join(parts).strip()
        if not text:
            raise ValueError("Anthropic response contained no text content block.")
        return text
