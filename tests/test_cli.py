"""Tests for the CLI: argument parsing, backend selection, and a full
end-to-end run of python -m multi_agent_debate.run using the scripted backend.
"""

from __future__ import annotations

import json

from multi_agent_debate.llm import AnthropicLLM
from multi_agent_debate.run import build_llm, main, parse_args
from multi_agent_debate.scripted_llm import ScriptedLLM


def test_parse_args_defaults():
    args = parse_args(["What is 2+2?"])
    assert args.question == "What is 2+2?"
    assert args.rounds == 2
    assert args.backend == "auto"
    assert args.model is None
    assert args.transcript is None
    assert args.quiet is False


def test_parse_args_overrides():
    args = parse_args(
        ["Q?", "--rounds", "4", "--backend", "scripted", "--transcript", "out.jsonl", "--quiet"]
    )
    assert args.rounds == 4
    assert args.backend == "scripted"
    assert args.transcript == "out.jsonl"
    assert args.quiet is True


def test_build_llm_scripted_backend():
    assert isinstance(build_llm("scripted", None), ScriptedLLM)


def test_build_llm_auto_without_api_key_falls_back_to_scripted(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(build_llm("auto", None), ScriptedLLM)


def test_build_llm_auto_with_api_key_set_uses_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test-only")
    llm = build_llm("auto", None)
    assert isinstance(llm, AnthropicLLM)


def test_build_llm_explicit_anthropic_backend_with_model_override(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test-only")
    llm = build_llm("anthropic", "claude-sonnet-5")
    assert isinstance(llm, AnthropicLLM)
    assert llm.model == "claude-sonnet-5"


def test_main_runs_a_full_debate_with_the_scripted_backend(capsys, tmp_path):
    log_path = tmp_path / "transcript.jsonl"

    exit_code = main(
        [
            "A bat and a ball together cost $1.10. The bat costs $1.00 more than the ball. "
            "How much does the ball cost?",
            "--rounds",
            "2",
            "--backend",
            "scripted",
            "--transcript",
            str(log_path),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "FINAL ANSWER" in captured.out
    assert "$0.05" in captured.out
    assert log_path.exists()

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    for line in lines:
        json.loads(line)  # each line must be valid JSON


def test_main_quiet_flag_suppresses_per_message_log(capsys):
    exit_code = main(["What is 2+2?", "--backend", "scripted", "--rounds", "1", "--quiet"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "[round 1]" not in out
    assert "FINAL ANSWER" in out
