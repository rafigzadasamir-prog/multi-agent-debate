"""Tests for the AnthropicLLM backend's request/response handling.

Every test here injects a fake client -- no network call is ever made, and
``anthropic`` itself is never imported unless a real client is constructed
(which these tests avoid).
"""

from __future__ import annotations

import pytest

from multi_agent_debate.llm import AnthropicLLM, DebateTurn, GenerationRequest


class _FakeBlock:
    def __init__(self, type_: str, text: str | None = None) -> None:
        self.type = type_
        self.text = text


class _FakeResponse:
    def __init__(self, content, stop_reason: str = "end_turn", stop_details=None) -> None:
        self.content = content
        self.stop_reason = stop_reason
        self.stop_details = stop_details


class _FakeMessages:
    def __init__(self, response, capture: list | None = None) -> None:
        self._response = response
        self._capture = capture

    def create(self, **kwargs):
        if self._capture is not None:
            self._capture.append(kwargs)
        return self._response


class _FakeClient:
    def __init__(self, response, capture: list | None = None) -> None:
        self.messages = _FakeMessages(response, capture)


def test_anthropic_llm_sends_expected_request_shape():
    captured: list[dict] = []
    response = _FakeResponse(content=[_FakeBlock("text", "The ball costs $0.05.")])
    client = _FakeClient(response, capture=captured)
    backend = AnthropicLLM(model="claude-test-model", client=client)

    request = GenerationRequest(
        role="judge",
        round_number=2,
        total_rounds=2,
        question="A bat and ball together cost $1.10...",
        history=[DebateTurn(role="advocate", round_number=1, content="My answer: $0.10")],
        system_prompt="You are the Judge.",
    )

    result = backend.generate(request)

    assert result == "The ball costs $0.05."
    assert len(captured) == 1
    call_kwargs = captured[0]
    assert call_kwargs["model"] == "claude-test-model"
    assert call_kwargs["system"] == "You are the Judge."
    assert call_kwargs["thinking"] == {"type": "adaptive"}
    assert call_kwargs["max_tokens"] == backend.max_tokens

    messages = call_kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert "A bat and ball together cost $1.10" in content
    assert "[round 1] ADVOCATE: My answer: $0.10" in content
    assert "JUDGE" in content
    assert "round 2 of 2" in content


def test_anthropic_llm_uses_default_model_when_none_given():
    response = _FakeResponse(content=[_FakeBlock("text", "hi")])
    backend = AnthropicLLM(client=_FakeClient(response))
    assert backend.model == AnthropicLLM.DEFAULT_MODEL == "claude-opus-5"


def test_anthropic_llm_respects_custom_max_tokens():
    response = _FakeResponse(content=[_FakeBlock("text", "hi")])
    captured: list[dict] = []
    backend = AnthropicLLM(client=_FakeClient(response, capture=captured), max_tokens=256)
    request = GenerationRequest(role="advocate", round_number=1, total_rounds=1, question="Q?")
    backend.generate(request)
    assert captured[0]["max_tokens"] == 256


def test_anthropic_llm_skips_non_text_content_blocks():
    response = _FakeResponse(
        content=[_FakeBlock("thinking", None), _FakeBlock("text", "Answer.")]
    )
    backend = AnthropicLLM(client=_FakeClient(response))
    request = GenerationRequest(role="advocate", round_number=1, total_rounds=1, question="Q?")
    assert backend.generate(request) == "Answer."


def test_anthropic_llm_concatenates_multiple_text_blocks():
    response = _FakeResponse(
        content=[_FakeBlock("text", "Part one. "), _FakeBlock("text", "Part two.")]
    )
    backend = AnthropicLLM(client=_FakeClient(response))
    request = GenerationRequest(role="advocate", round_number=1, total_rounds=1, question="Q?")
    assert backend.generate(request) == "Part one. Part two."


def test_anthropic_llm_raises_on_refusal_stop_reason():
    response = _FakeResponse(content=[], stop_reason="refusal", stop_details={"category": "test"})
    backend = AnthropicLLM(client=_FakeClient(response))
    request = GenerationRequest(role="advocate", round_number=1, total_rounds=1, question="Q?")
    with pytest.raises(RuntimeError, match="refusal|declined"):
        backend.generate(request)


def test_anthropic_llm_raises_on_response_with_no_usable_text():
    response = _FakeResponse(content=[_FakeBlock("text", "   ")])
    backend = AnthropicLLM(client=_FakeClient(response))
    request = GenerationRequest(role="advocate", round_number=1, total_rounds=1, question="Q?")
    with pytest.raises(ValueError):
        backend.generate(request)


def test_anthropic_llm_render_user_content_without_history():
    rendered = AnthropicLLM._render_user_content(
        GenerationRequest(role="advocate", round_number=1, total_rounds=3, question="Q?")
    )
    assert "Debate transcript so far:" not in rendered
    assert "ADVOCATE" in rendered
    assert "round 1 of 3" in rendered
