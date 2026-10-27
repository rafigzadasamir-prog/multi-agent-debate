"""Tests for the deterministic, offline ScriptedLLM backend.

These exercise the worked example (the classic bat-and-ball puzzle) step by
step -- the Advocate's naive wrong answer, the Critic catching it, the
Advocate's revision, and the Judge's final correction -- plus the generic
fallback path used for arbitrary, non-arithmetic questions.
"""

from __future__ import annotations

import pytest

from multi_agent_debate.llm import DebateTurn, GenerationRequest
from multi_agent_debate.scripted_llm import (
    ScriptedLLM,
    _correct_answer,
    _detect_cost_puzzle,
    _naive_answer,
)

BAT_AND_BALL = (
    "A bat and a ball together cost $1.10. The bat costs $1.00 more than the "
    "ball. How much does the ball cost?"
)
GENERIC_QUESTION = "Should the team adopt a four-day work week?"


def _request(role: str, round_number: int, total_rounds: int, history=None, question=BAT_AND_BALL):
    return GenerationRequest(
        role=role,
        round_number=round_number,
        total_rounds=total_rounds,
        question=question,
        history=history or [],
    )


# --- puzzle detection and arithmetic helpers ---


def test_detect_cost_puzzle_matches_bat_and_ball():
    assert _detect_cost_puzzle(BAT_AND_BALL) == (1.10, 1.00)


def test_detect_cost_puzzle_rejects_unrelated_question():
    assert _detect_cost_puzzle(GENERIC_QUESTION) is None


def test_detect_cost_puzzle_rejects_questions_with_fewer_than_two_amounts():
    question = "A bat and a ball together cost some amount. The bat costs $1.00 more than the ball."
    assert _detect_cost_puzzle(question) is None


def test_naive_and_correct_answers_differ():
    naive = _naive_answer(1.10, 1.00)
    correct = _correct_answer(1.10, 1.00)
    assert naive == 0.10
    assert correct == 0.05
    assert naive != correct


# --- puzzle path: Advocate round 1 gives the classic wrong answer ---


def test_advocate_round1_gives_the_classic_wrong_answer():
    text = ScriptedLLM().generate(_request("advocate", 1, 2))
    assert "$0.10" in text
    assert "$0.05" not in text


# --- puzzle path: Critic catches the flaw ---


def test_critic_round1_catches_the_flaw():
    llm = ScriptedLLM()
    advocate_text = llm.generate(_request("advocate", 1, 2))
    history = [DebateTurn(role="advocate", round_number=1, content=advocate_text)]

    critic_text = llm.generate(_request("critic", 1, 2, history))

    assert critic_text.startswith("Issue 1:")
    assert "$1.20" in critic_text  # the implied, contradicting total
    assert "$0.05" in critic_text  # the corrected value it derives


def test_critic_accepts_an_already_consistent_answer():
    llm = ScriptedLLM()
    history = [DebateTurn(role="advocate", round_number=1, content="My answer: $0.05.")]
    critic_text = llm.generate(_request("critic", 1, 1, history))
    assert critic_text.startswith("No issues found")


# --- puzzle path: Advocate concedes and corrects in round 2 ---


def test_advocate_round2_concedes_and_corrects():
    llm = ScriptedLLM()
    advocate_r1 = llm.generate(_request("advocate", 1, 2))
    history = [DebateTurn(role="advocate", round_number=1, content=advocate_r1)]
    critic_r1 = llm.generate(_request("critic", 1, 2, history))
    history.append(DebateTurn(role="critic", round_number=1, content=critic_r1))

    advocate_r2 = llm.generate(_request("advocate", 2, 2, history))

    assert advocate_r2.startswith("Revised answer")
    assert "$0.05" in advocate_r2


def test_critic_round2_finds_no_further_issues_after_revision():
    llm = ScriptedLLM()
    advocate_r1 = llm.generate(_request("advocate", 1, 2))
    history = [DebateTurn(role="advocate", round_number=1, content=advocate_r1)]
    critic_r1 = llm.generate(_request("critic", 1, 2, history))
    history.append(DebateTurn(role="critic", round_number=1, content=critic_r1))
    advocate_r2 = llm.generate(_request("advocate", 2, 2, history))
    history.append(DebateTurn(role="advocate", round_number=2, content=advocate_r2))

    critic_r2 = llm.generate(_request("critic", 2, 2, history))

    assert critic_r2.startswith("No issues found")


def test_advocate_reaffirms_when_no_issue_was_raised():
    llm = ScriptedLLM()
    history = [
        DebateTurn(role="advocate", round_number=1, content="My answer: $0.05."),
        DebateTurn(role="critic", round_number=1, content="No issues found: consistent."),
    ]
    advocate_r2 = llm.generate(_request("advocate", 2, 2, history))
    assert advocate_r2.startswith("I'll stand by")
    assert "$0.05" in advocate_r2


# --- puzzle path: Judge synthesizes the corrected final answer ---


def test_judge_corrects_the_final_answer_and_cites_the_critique():
    llm = ScriptedLLM()
    advocate_r1 = llm.generate(_request("advocate", 1, 2))
    history = [DebateTurn(role="advocate", round_number=1, content=advocate_r1)]
    critic_r1 = llm.generate(_request("critic", 1, 2, history))
    history.append(DebateTurn(role="critic", round_number=1, content=critic_r1))
    advocate_r2 = llm.generate(_request("advocate", 2, 2, history))
    history.append(DebateTurn(role="advocate", round_number=2, content=advocate_r2))
    critic_r2 = llm.generate(_request("critic", 2, 2, history))
    history.append(DebateTurn(role="critic", round_number=2, content=critic_r2))

    judge_text = llm.generate(_request("judge", 2, 2, history))
    lines = judge_text.splitlines()

    assert lines[0] == "Final Answer: $0.05"
    assert lines[1].startswith("Rationale:")
    assert "round 1" in lines[1]


def test_judge_with_no_critique_reports_no_contradictions():
    llm = ScriptedLLM()
    history = [
        DebateTurn(role="advocate", round_number=1, content="My answer: $0.05."),
        DebateTurn(role="critic", round_number=1, content="No issues found: consistent."),
    ]
    judge_text = llm.generate(_request("judge", 1, 1, history))
    assert judge_text.startswith("Final Answer: $0.05")
    assert "no contradictions" in judge_text.lower()


def test_unsupported_role_raises_value_error():
    with pytest.raises(ValueError):
        ScriptedLLM().generate(_request("mediator", 1, 1))


# --- generic (non-arithmetic) fallback path ---


def test_generic_advocate_round1_states_initial_position():
    text = ScriptedLLM().generate(
        GenerationRequest(
            role="advocate", round_number=1, total_rounds=2, question=GENERIC_QUESTION
        )
    )
    assert "Initial position" in text


def test_generic_critic_round1_raises_two_issues():
    text = ScriptedLLM().generate(
        GenerationRequest(role="critic", round_number=1, total_rounds=2, question=GENERIC_QUESTION)
    )
    assert "Issue 1:" in text
    assert "Issue 2:" in text


def test_generic_advocate_round2_refines_after_issues():
    llm = ScriptedLLM()
    history = [
        DebateTurn(role="advocate", round_number=1, content="Initial position: yes."),
        DebateTurn(
            role="critic", round_number=1, content="Issue 1: check it.\nIssue 2: edge cases."
        ),
    ]
    text = llm.generate(
        GenerationRequest(
            role="advocate",
            round_number=2,
            total_rounds=2,
            question=GENERIC_QUESTION,
            history=history,
        )
    )
    assert "Refined position" in text


def test_generic_judge_produces_final_answer_and_rationale():
    history = [
        DebateTurn(role="advocate", round_number=1, content="Initial position: yes."),
        DebateTurn(
            role="critic", round_number=1, content="Issue 1: check it.\nIssue 2: edge cases."
        ),
        DebateTurn(role="advocate", round_number=2, content="Refined position: yes, re-derived."),
        DebateTurn(role="critic", round_number=2, content="No issues found: consistent."),
    ]
    judge_text = ScriptedLLM().generate(
        GenerationRequest(
            role="judge", round_number=2, total_rounds=2, question=GENERIC_QUESTION, history=history
        )
    )
    assert judge_text.startswith("Final Answer:")
    assert "Rationale:" in judge_text
    assert "round 1" in judge_text
