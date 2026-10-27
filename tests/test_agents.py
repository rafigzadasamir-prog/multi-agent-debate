"""Tests for the Agent/Advocate/Critic/Judge wiring.

These use a recording fake backend rather than ScriptedLLM, so they verify
only that each role builds the right GenerationRequest and plumbs the
backend's reply straight through -- content-level heuristics are covered in
test_scripted_llm.py.
"""

from __future__ import annotations

from multi_agent_debate.agents import Advocate, Agent, Critic, Judge
from multi_agent_debate.llm import DebateTurn, GenerationRequest, LLMBackend


class _RecordingLLM(LLMBackend):
    def __init__(self, reply: str = "ok") -> None:
        self.reply = reply
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> str:
        self.requests.append(request)
        return self.reply


def test_advocate_role_and_system_prompt():
    advocate = Advocate(_RecordingLLM())
    assert advocate.role == "advocate"
    assert "Advocate" in advocate.system_prompt


def test_advocate_respond_builds_correct_request_and_returns_reply():
    llm = _RecordingLLM(reply="my considered answer")
    advocate = Advocate(llm)

    reply = advocate.respond("Q?", round_number=1, total_rounds=2, history=[])

    assert reply == "my considered answer"
    assert len(llm.requests) == 1
    request = llm.requests[0]
    assert request.role == "advocate"
    assert request.round_number == 1
    assert request.total_rounds == 2
    assert request.question == "Q?"
    assert request.system_prompt == advocate.system_prompt


def test_critic_role_and_custom_name():
    llm = _RecordingLLM()
    critic = Critic(llm, name="Skeptic")
    assert critic.role == "critic"
    assert critic.name == "Skeptic"

    critic.respond("Q?", 1, 1, [])

    assert llm.requests[0].role == "critic"


def test_judge_role_and_system_prompt_format_instructions():
    llm = _RecordingLLM()
    judge = Judge(llm)
    assert judge.role == "judge"
    assert "Final Answer:" in judge.system_prompt
    assert "Rationale:" in judge.system_prompt

    judge.respond("Q?", 2, 2, [])

    assert llm.requests[0].role == "judge"


def test_respond_passes_a_defensive_copy_of_history():
    llm = _RecordingLLM()
    advocate = Advocate(llm)
    history = [DebateTurn(role="critic", round_number=1, content="Issue 1: ...")]

    advocate.respond("Q?", 2, 2, history)

    captured = llm.requests[0].history
    assert captured == history
    assert captured is not history


def test_agent_base_class_has_generic_default_role():
    agent = Agent(_RecordingLLM(), "Generic", "You are generic.")
    assert agent.role == "agent"
    assert agent.name == "Generic"
    assert agent.system_prompt == "You are generic."
