"""Full end-to-end debate runs using only the rule-based ScriptedLLM backend
(zero network calls). This is the integration-level counterpart to the
focused unit tests in test_scripted_llm.py, test_agents.py, and test_debate.py.
"""

from __future__ import annotations

import pytest

from multi_agent_debate.debate import Debate
from multi_agent_debate.scripted_llm import ScriptedLLM

BAT_AND_BALL = (
    "A bat and a ball together cost $1.10. The bat costs $1.00 more than the "
    "ball. How much does the ball cost?"
)


def test_end_to_end_debate_corrects_the_classic_wrong_answer(capsys):
    debate = Debate(ScriptedLLM(), rounds=2)

    result = debate.run(BAT_AND_BALL)

    captured = capsys.readouterr()
    assert "[round 1]" in captured.out

    assert result.question == BAT_AND_BALL
    assert result.rounds == 2
    assert len(result.transcript) == 5  # (advocate, critic) x 2 rounds, then judge

    advocate_r1 = result.transcript[0]
    assert advocate_r1.role == "advocate"
    assert "$0.10" in advocate_r1.content  # the classic wrong first instinct

    critic_r1 = result.transcript[1]
    assert critic_r1.role == "critic"
    assert critic_r1.content.startswith("Issue 1:")

    assert result.final_answer == "$0.05"  # the corrected, verified answer


def test_end_to_end_debate_writes_a_jsonl_transcript_file(tmp_path):
    debate = Debate(ScriptedLLM(), rounds=1)
    log_path = tmp_path / "out.jsonl"

    debate.run(BAT_AND_BALL, log_path=log_path, verbose=False)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3  # advocate, critic, judge


@pytest.mark.parametrize("rounds", [1, 2, 3, 5])
def test_end_to_end_corrects_the_answer_regardless_of_round_count(rounds):
    debate = Debate(ScriptedLLM(), rounds=rounds)
    result = debate.run(BAT_AND_BALL, verbose=False)
    assert result.rounds == rounds
    assert len(result.transcript) == 2 * rounds + 1
    assert result.final_answer == "$0.05"


def test_end_to_end_on_a_generic_non_arithmetic_question_has_sane_structure():
    debate = Debate(ScriptedLLM(), rounds=2)

    result = debate.run("Should the team adopt a four-day work week?", verbose=False)

    assert isinstance(result.final_answer, str)
    assert result.final_answer.strip() != ""
    judge_message = result.transcript[-1]
    assert judge_message.role == "judge"
    assert "Rationale:" in judge_message.content
