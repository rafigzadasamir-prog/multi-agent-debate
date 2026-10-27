"""Tests for the Debate orchestration loop: sequencing, round count, and
transcript structure. Uses a fake backend that records call order so these
tests are independent of ScriptedLLM's content heuristics.
"""

from __future__ import annotations

import pytest

from multi_agent_debate.debate import Debate, _extract_final_answer
from multi_agent_debate.llm import GenerationRequest, LLMBackend
from multi_agent_debate.transcript import load_transcript_jsonl


class _SequenceLLM(LLMBackend):
    """Records (role, round_number, total_rounds) for every call, in order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def generate(self, request: GenerationRequest) -> str:
        self.calls.append((request.role, request.round_number, request.total_rounds))
        if request.role == "judge":
            return "Final Answer: 42\nRationale: because reasons."
        return f"{request.role} says something in round {request.round_number}"


def test_debate_rejects_non_positive_round_counts():
    with pytest.raises(ValueError):
        Debate(_SequenceLLM(), rounds=0)
    with pytest.raises(ValueError):
        Debate(_SequenceLLM(), rounds=-1)


def test_debate_runs_rounds_in_the_correct_order():
    llm = _SequenceLLM()
    debate = Debate(llm, rounds=3)

    result = debate.run("Q?", verbose=False)

    assert debate.rounds == 3
    assert llm.calls == [
        ("advocate", 1, 3),
        ("critic", 1, 3),
        ("advocate", 2, 3),
        ("critic", 2, 3),
        ("advocate", 3, 3),
        ("critic", 3, 3),
        ("judge", 3, 3),
    ]
    assert len(result.transcript) == 7
    assert result.transcript[-1].role == "judge"
    assert result.transcript[-1].round == 3


@pytest.mark.parametrize("rounds", [1, 2, 4])
def test_transcript_length_matches_round_count(rounds):
    debate = Debate(_SequenceLLM(), rounds=rounds)
    result = debate.run("Q?", verbose=False)
    assert len(result.transcript) == 2 * rounds + 1


def test_transcript_message_structure_is_well_formed():
    debate = Debate(_SequenceLLM(), rounds=1)
    result = debate.run("Q?", verbose=False)

    assert [m.role for m in result.transcript] == ["advocate", "critic", "judge"]
    for message in result.transcript:
        assert message.round >= 1
        assert message.speaker
        assert message.content
        assert message.timestamp


def test_final_answer_is_extracted_from_the_judge_message():
    debate = Debate(_SequenceLLM(), rounds=1)
    result = debate.run("Q?", verbose=False)
    assert result.final_answer == "42"


def test_extract_final_answer_falls_back_when_format_is_missing():
    text = "just some prose, no final answer line"
    assert _extract_final_answer(text) == text


def test_debate_writes_a_jsonl_transcript_matching_the_result(tmp_path):
    debate = Debate(_SequenceLLM(), rounds=2)
    log_path = tmp_path / "transcript.jsonl"

    result = debate.run("Q?", log_path=log_path, verbose=False)

    loaded = load_transcript_jsonl(log_path)
    assert loaded == result.transcript


def test_log_path_is_truncated_at_the_start_of_each_run(tmp_path):
    debate = Debate(_SequenceLLM(), rounds=1)
    log_path = tmp_path / "transcript.jsonl"

    debate.run("Q?", log_path=log_path, verbose=False)
    debate.run("Q?", log_path=log_path, verbose=False)

    loaded = load_transcript_jsonl(log_path)
    assert len(loaded) == 3  # second run overwrites rather than appends


def test_custom_agent_names_appear_as_speakers():
    debate = Debate(
        _SequenceLLM(), rounds=1, advocate_name="Pro", critic_name="Con", judge_name="Ref"
    )
    result = debate.run("Q?", verbose=False)
    speakers = {m.role: m.speaker for m in result.transcript}
    assert speakers == {"advocate": "Pro", "critic": "Con", "judge": "Ref"}


def test_verbose_logging_prints_each_message(capsys):
    debate = Debate(_SequenceLLM(), rounds=1, advocate_name="Pro")
    debate.run("Q?", verbose=True)
    out = capsys.readouterr().out
    assert "Pro (advocate)" in out
    assert "[round 1]" in out
