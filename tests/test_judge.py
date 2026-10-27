"""Focused tests for the Judge's synthesis logic: independent verification of
the puzzle's answer, citation of the Critic's issues by round, and graceful
behavior when the Advocate never got a chance to revise within the debate.
"""

from __future__ import annotations

from multi_agent_debate.agents import Judge
from multi_agent_debate.llm import DebateTurn, GenerationRequest
from multi_agent_debate.scripted_llm import ScriptedLLM

BAT_AND_BALL = (
    "A bat and a ball together cost $1.10. The bat costs $1.00 more than the "
    "ball. How much does the ball cost?"
)


def test_judge_system_prompt_requires_final_answer_and_rationale_format():
    judge = Judge(ScriptedLLM())
    assert "Final Answer:" in judge.system_prompt
    assert "Rationale:" in judge.system_prompt
    assert "by round" in judge.system_prompt


def test_judge_corrects_even_when_the_advocate_never_revised_within_the_debate():
    """With only 1 round, the Advocate has no chance to revise -- the Judge
    must still independently verify the arithmetic and correct the answer."""
    llm = ScriptedLLM()
    advocate_text = llm.generate(
        GenerationRequest(role="advocate", round_number=1, total_rounds=1, question=BAT_AND_BALL)
    )
    history = [DebateTurn(role="advocate", round_number=1, content=advocate_text)]
    critic_text = llm.generate(
        GenerationRequest(
            role="critic", round_number=1, total_rounds=1, question=BAT_AND_BALL, history=history
        )
    )
    history.append(DebateTurn(role="critic", round_number=1, content=critic_text))

    judge = Judge(llm)
    judge_text = judge.respond(BAT_AND_BALL, round_number=1, total_rounds=1, history=history)

    assert judge_text.startswith("Final Answer: $0.05")
    assert "did not reflect this correction" in judge_text


def test_judge_recognizes_when_the_advocate_already_corrected_the_answer():
    llm = ScriptedLLM()
    history = [
        DebateTurn(role="advocate", round_number=1, content="My answer: $0.10."),
        DebateTurn(role="critic", round_number=1, content="Issue 1: contradiction found."),
        DebateTurn(role="advocate", round_number=2, content="Revised answer: $0.05."),
        DebateTurn(role="critic", round_number=2, content="No issues found: consistent."),
    ]
    judge = Judge(llm)
    judge_text = judge.respond(BAT_AND_BALL, round_number=2, total_rounds=2, history=history)

    assert judge_text.startswith("Final Answer: $0.05")
    assert "already reflects this correction" in judge_text


def test_judge_rationale_cites_every_round_the_critic_raised_an_issue():
    llm = ScriptedLLM()
    history = [
        DebateTurn(role="critic", round_number=1, content="Issue 1: still wrong."),
        DebateTurn(role="critic", round_number=2, content="Issue 1: still wrong."),
        DebateTurn(role="critic", round_number=3, content="Issue 1: still wrong."),
        DebateTurn(role="advocate", round_number=3, content="My answer: $0.10."),
    ]
    judge = Judge(llm)
    judge_text = judge.respond(BAT_AND_BALL, round_number=3, total_rounds=3, history=history)

    assert "round 1" in judge_text
    assert "round 2" in judge_text
    assert "round 3" in judge_text


def test_judge_on_a_generic_question_echoes_the_advocates_final_claim():
    llm = ScriptedLLM()
    history = [
        DebateTurn(
            role="advocate", round_number=1, content="Initial position: the answer is yes."
        ),
        DebateTurn(role="critic", round_number=1, content="Issue 1: unchecked assumption."),
        DebateTurn(
            role="advocate", round_number=2, content="Refined position: still yes, re-derived."
        ),
        DebateTurn(role="critic", round_number=2, content="No issues found: consistent."),
    ]
    judge = Judge(llm)
    judge_text = judge.respond(
        "Should we ship it?", round_number=2, total_rounds=2, history=history
    )

    assert judge_text.startswith("Final Answer:")
    assert "Rationale:" in judge_text
    assert "round 1" in judge_text
