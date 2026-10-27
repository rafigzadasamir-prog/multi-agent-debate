"""Round-based orchestration of the Advocate / Critic / Judge debate.

The flow is:

    for round in 1..N:
        Advocate speaks (defends/refines its answer)
        Critic speaks (attacks the Advocate's latest answer)
    Judge speaks once, after the final round, synthesizing a final answer.

Every message is appended to an in-memory transcript (and optionally to a
``.jsonl`` file on disk) and printed to stdout as it is produced.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from .agents import Advocate, Agent, Critic, Judge
from .llm import DebateTurn, LLMBackend
from .transcript import DebateResult, Message, append_message_jsonl, start_transcript_file


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _extract_final_answer(judge_text: str) -> str:
    """Pull the answer out of the Judge's 'Final Answer: ...' line.

    Falls back to the full Judge message if that line is missing, so a Judge
    response that doesn't follow the requested format still yields something
    usable rather than raising.
    """
    for line in judge_text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("final answer:"):
            return stripped.split(":", 1)[1].strip()
    return judge_text.strip()


class Debate:
    """Orchestrates an N-round Advocate/Critic debate followed by Judge synthesis."""

    def __init__(
        self,
        llm: LLMBackend,
        rounds: int = 2,
        advocate_name: str = "Advocate",
        critic_name: str = "Critic",
        judge_name: str = "Judge",
    ) -> None:
        if rounds < 1:
            raise ValueError("rounds must be at least 1")
        self.rounds = rounds
        self.advocate = Advocate(llm, name=advocate_name)
        self.critic = Critic(llm, name=critic_name)
        self.judge = Judge(llm, name=judge_name)

    def run(
        self,
        question: str,
        *,
        log_path: str | Path | None = None,
        verbose: bool = True,
    ) -> DebateResult:
        """Run the full debate and return the transcript plus final answer.

        Args:
            question: The question or problem to debate.
            log_path: Optional path to a JSONL file to write the transcript to,
                incrementally, as each message is produced. The file is
                truncated at the start of the run.
            verbose: If True (default), print each message to stdout as it is
                produced.
        """
        if log_path is not None:
            start_transcript_file(log_path)

        transcript: list[Message] = []
        history: list[DebateTurn] = []

        def log(agent: Agent, text: str, round_number: int) -> None:
            message = Message(
                round=round_number,
                role=agent.role,
                speaker=agent.name,
                content=text,
                timestamp=_utc_now_iso(),
            )
            transcript.append(message)
            history.append(DebateTurn(role=agent.role, round_number=round_number, content=text))
            if verbose:
                print(f"[round {round_number}] {agent.name} ({agent.role}): {text}")
            if log_path is not None:
                append_message_jsonl(log_path, message)

        for round_number in range(1, self.rounds + 1):
            advocate_text = self.advocate.respond(question, round_number, self.rounds, history)
            log(self.advocate, advocate_text, round_number)

            critic_text = self.critic.respond(question, round_number, self.rounds, history)
            log(self.critic, critic_text, round_number)

        judge_text = self.judge.respond(question, self.rounds, self.rounds, history)
        log(self.judge, judge_text, self.rounds)

        return DebateResult(
            question=question,
            rounds=self.rounds,
            transcript=transcript,
            final_answer=_extract_final_answer(judge_text),
        )
