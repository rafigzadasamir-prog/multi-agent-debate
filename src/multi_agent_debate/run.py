"""CLI entry point.

Usage::

    python -m multi_agent_debate.run "question" --rounds 2
"""

from __future__ import annotations

import argparse
import os
import sys

from .debate import Debate
from .llm import AnthropicLLM, LLMBackend
from .scripted_llm import ScriptedLLM


def build_llm(backend: str, model: str | None) -> LLMBackend:
    """Choose a backend: 'scripted' always wins, 'anthropic' always calls the API,
    'auto' calls the API only if an ANTHROPIC_API_KEY is set, else falls back to
    the scripted backend.
    """
    use_anthropic = backend == "anthropic" or (
        backend == "auto" and bool(os.environ.get("ANTHROPIC_API_KEY"))
    )
    if use_anthropic:
        return AnthropicLLM(model=model) if model else AnthropicLLM()
    return ScriptedLLM()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m multi_agent_debate.run",
        description="Run a multi-agent (Advocate/Critic/Judge) LLM debate on a question.",
    )
    parser.add_argument("question", help="The question or problem to debate.")
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Number of Advocate/Critic rounds before the Judge's synthesis (default: 2).",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "scripted", "anthropic"],
        default="auto",
        help=(
            "LLM backend to use. 'auto' (default) uses the real Anthropic backend if "
            "ANTHROPIC_API_KEY is set, else the deterministic scripted backend."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Anthropic model name override (only relevant when calling the real API).",
    )
    parser.add_argument(
        "--transcript",
        default=None,
        help="Path to write the JSONL transcript to (optional).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the per-message stdout log; still prints the final answer.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    llm = build_llm(args.backend, args.model)
    debate = Debate(llm=llm, rounds=args.rounds)
    result = debate.run(args.question, log_path=args.transcript, verbose=not args.quiet)

    print()
    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(result.final_answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
