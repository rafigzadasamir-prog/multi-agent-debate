"""Agent roles for the debate: Advocate, Critic, and Judge.

Each role is a thin wrapper around an :class:`~multi_agent_debate.llm.LLMBackend`:
it carries a name and a role-specific system prompt, and turns the orchestrator's
call into a :class:`~multi_agent_debate.llm.GenerationRequest`.
"""

from __future__ import annotations

from .llm import DebateTurn, GenerationRequest, LLMBackend

ADVOCATE_SYSTEM_PROMPT = (
    "You are the Advocate in a structured multi-agent debate. Propose and defend the best "
    "answer to the question. In round 1, give your best initial answer with clear reasoning. "
    "In later rounds, directly address the Critic's most recent objections: concede and "
    "correct your answer if an objection reveals a genuine flaw, or explain why it does not "
    "change your answer if it does not. Be concise and concrete."
)

CRITIC_SYSTEM_PROMPT = (
    "You are the Critic (Skeptic) in a structured multi-agent debate. Your job is to find "
    "flaws in the Advocate's latest answer: check it against the stated facts and "
    "constraints, look for unstated assumptions, and look for contradictions. If you find a "
    "genuine flaw, state it clearly as 'Issue N: ...' describing the specific contradiction. "
    "If the latest answer is sound, say so clearly starting with 'No issues found:' and "
    "explain briefly why. Do not invent flaws that are not real."
)

JUDGE_SYSTEM_PROMPT = (
    "You are the Judge (Synthesizer) in a structured multi-agent debate. You speak once, "
    "after the final round. Review the full exchange between the Advocate and the Critic and "
    "produce the best final answer. Respond in exactly this format: a first line starting "
    "with 'Final Answer:' followed by the answer, then a line starting with 'Rationale:' that "
    "briefly explains the reasoning and explicitly names which of the Critic's issues (by "
    "round) were addressed and how."
)


class Agent:
    """Base class for a named debate participant backed by an LLM backend."""

    role: str = "agent"

    def __init__(self, llm: LLMBackend, name: str, system_prompt: str) -> None:
        self.llm = llm
        self.name = name
        self.system_prompt = system_prompt

    def respond(
        self,
        question: str,
        round_number: int,
        total_rounds: int,
        history: list[DebateTurn],
    ) -> str:
        """Ask the backing LLM for this agent's next message."""
        request = GenerationRequest(
            role=self.role,
            round_number=round_number,
            total_rounds=total_rounds,
            question=question,
            history=list(history),
            system_prompt=self.system_prompt,
        )
        return self.llm.generate(request)


class Advocate(Agent):
    """Proposes and defends an answer, revising it when a real flaw is shown."""

    role = "advocate"

    def __init__(self, llm: LLMBackend, name: str = "Advocate") -> None:
        super().__init__(llm, name, ADVOCATE_SYSTEM_PROMPT)


class Critic(Agent):
    """Challenges the Advocate's answer and surfaces concrete contradictions."""

    role = "critic"

    def __init__(self, llm: LLMBackend, name: str = "Critic") -> None:
        super().__init__(llm, name, CRITIC_SYSTEM_PROMPT)


class Judge(Agent):
    """Synthesizes the debate into one final answer with a cited rationale."""

    role = "judge"

    def __init__(self, llm: LLMBackend, name: str = "Judge") -> None:
        super().__init__(llm, name, JUDGE_SYSTEM_PROMPT)
