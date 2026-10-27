"""Deterministic, offline, rule-based LLM backend.

``ScriptedLLM`` makes zero network calls. It produces plausible,
role-appropriate text for the Advocate, Critic, and Judge from simple
heuristics, which is what makes the whole multi-round orchestration in
:mod:`multi_agent_debate.debate` unit-testable without any API key.

It has two response modes:

1. A word-problem solver for "two items, combined cost, stated difference"
   puzzles (the classic bat-and-ball shape). This is the one path that does
   real arithmetic: the Advocate's first instinct is the common *wrong*
   answer (naively subtracting the difference from the total), the Critic
   checks it against both constraints and catches the contradiction, and the
   Judge adopts the algebraically correct answer. This is what the README's
   worked example exercises.
2. A generic templated fallback for any other question, so the CLI and the
   orchestration logic also work end-to-end on arbitrary input.
"""

from __future__ import annotations

import re

from .llm import DebateTurn, GenerationRequest, LLMBackend

_DOLLAR_RE = re.compile(r"\$(\d+(?:\.\d+)?)")
_TOGETHER_RE = re.compile(r"together|total|combined", re.IGNORECASE)
_MORE_THAN_RE = re.compile(r"more than", re.IGNORECASE)


def _detect_cost_puzzle(question: str) -> tuple[float, float] | None:
    """Detect a "two items, combined cost, stated price difference" word problem.

    Returns ``(total, difference)`` if the question matches this shape
    (e.g. the classic bat-and-ball puzzle), else ``None``.
    """
    if not (_TOGETHER_RE.search(question) and _MORE_THAN_RE.search(question)):
        return None
    amounts = _DOLLAR_RE.findall(question)
    if len(amounts) < 2:
        return None
    total, difference = float(amounts[0]), float(amounts[1])
    if total <= 0 or difference < 0 or difference >= total:
        return None
    return total, difference


def _naive_answer(total: float, difference: float) -> float:
    """The common, intuitive, *wrong* answer: total minus difference."""
    return round(total - difference, 2)


def _correct_answer(total: float, difference: float) -> float:
    """The algebraically correct answer: solve x + (x + difference) = total."""
    return round((total - difference) / 2, 2)


def _extract_dollar_value(text: str) -> float | None:
    match = _DOLLAR_RE.search(text)
    return float(match.group(1)) if match else None


def _find_latest(
    history: list[DebateTurn], role: str, round_number: int | None = None
) -> DebateTurn | None:
    for turn in reversed(history):
        if turn.role == role and (round_number is None or turn.round_number == round_number):
            return turn
    return None


def _advocate_puzzle_round1(total: float, difference: float, naive: float) -> str:
    return (
        f"My answer: the cheaper item costs ${naive:.2f}. Reasoning: the two items "
        f"together cost ${total:.2f}, and since one costs ${difference:.2f} more than the "
        f"other, subtracting that extra ${difference:.2f} from the ${total:.2f} total leaves "
        f"${naive:.2f} for the cheaper item."
    )


def _advocate_puzzle_revision(correct: float, total: float, difference: float) -> str:
    return (
        f"Revised answer: the cheaper item costs ${correct:.2f}. You're right that my first "
        f"answer was inconsistent. Letting x be the cheaper item's price, the pricier item is "
        f"x + ${difference:.2f}, and together they must total ${total:.2f}: "
        f"x + (x + {difference:.2f}) = {total:.2f}, so x = ${correct:.2f}."
    )


def _advocate_generic_round1(question: str) -> str:
    topic = question.strip().rstrip("?")
    return (
        f'Initial position on "{topic}": the direct, common-sense answer holds. Supporting '
        "reasoning: (1) it aligns with the most straightforward reading of the premises, "
        "(2) nothing in the question suggests an exception, and (3) it matches the typical "
        "first-pass answer most people would give without double-checking it."
    )


def _advocate_generic_revision(question: str) -> str:
    topic = question.strip().rstrip("?")
    return (
        f'Refined position on "{topic}": after reviewing the Critic\'s objections, the same '
        "conclusion is retained, but it is now derived explicitly from the stated premises "
        "rather than from intuition alone, and alternative readings were checked and found "
        "not to change the outcome."
    )


def _advocate_generic_reaffirm(question: str) -> str:
    topic = question.strip().rstrip("?")
    return (
        f'Reaffirmed position on "{topic}": the refined conclusion stands; no new objections '
        "have been raised against it."
    )


def _critic_puzzle_response(request: GenerationRequest, total: float, difference: float) -> str:
    correct = _correct_answer(total, difference)
    advocate_turn = _find_latest(request.history, "advocate", request.round_number)
    claimed = _extract_dollar_value(advocate_turn.content) if advocate_turn else None

    if claimed is not None and abs(claimed - correct) > 0.005:
        implied_other = claimed + difference
        implied_total = claimed + implied_other
        return (
            f"Issue 1: the claimed value of ${claimed:.2f} does not satisfy the stated "
            f"constraints. If the cheaper item costs ${claimed:.2f}, the other item would cost "
            f"${implied_other:.2f} (${claimed:.2f} + ${difference:.2f}), for a combined total of "
            f"${implied_total:.2f} -- not the stated ${total:.2f}. Solving the two equations "
            f"simultaneously (x + (x + {difference:.2f}) = {total:.2f}) gives the consistent "
            f"value ${correct:.2f}."
        )
    return (
        f"No issues found: a value of ${claimed:.2f} satisfies both constraints (it sums to "
        f"${total:.2f} together with the stated ${difference:.2f} difference)."
    )


def _critic_generic_response(request: GenerationRequest) -> str:
    topic = request.question.strip().rstrip("?")
    if request.round_number == 1:
        return (
            f'Issue 1: the initial position on "{topic}" was reached by pattern-matching to the '
            "typical answer rather than by checking it against the explicit premises.\n"
            "Issue 2: no alternative interpretation or edge case was considered before "
            "committing to that answer."
        )
    return (
        "No issues found: the refined position explicitly derives from the stated premises "
        "and accounts for alternative readings; no further objections."
    )


def _judge_puzzle_response(request: GenerationRequest, total: float, difference: float) -> str:
    correct = _correct_answer(total, difference)
    naive = _naive_answer(total, difference)
    flagged_rounds = [
        turn.round_number
        for turn in request.history
        if turn.role == "critic" and turn.content.startswith("Issue")
    ]
    advocate_final = _find_latest(request.history, "advocate")
    final_claim = _extract_dollar_value(advocate_final.content) if advocate_final else None
    already_corrected = final_claim is not None and abs(final_claim - correct) < 0.005

    lines = [f"Final Answer: ${correct:.2f}"]
    if flagged_rounds:
        rounds_text = ", ".join(f"round {r}" for r in flagged_rounds)
        resolution = (
            "the Advocate's final answer already reflects this correction"
            if already_corrected
            else "the Advocate's final stated answer did not reflect this correction"
        )
        lines.append(
            f"Rationale: the Critic's objection in {rounds_text} -- that the naive answer of "
            f"${naive:.2f} (total minus difference) contradicts the stated total once checked -- "
            f"is correct and is addressed here. Solving x + (x + ${difference:.2f}) = "
            f"${total:.2f} simultaneously gives x = ${correct:.2f}; {resolution}."
        )
    else:
        lines.append(
            f"Rationale: no contradictions were raised during the debate; ${correct:.2f} "
            "satisfies both stated constraints."
        )
    return "\n".join(lines)


def _judge_generic_response(request: GenerationRequest) -> str:
    flagged_rounds = [
        turn.round_number
        for turn in request.history
        if turn.role == "critic" and turn.content.startswith("Issue")
    ]
    advocate_final = _find_latest(request.history, "advocate")
    claim_text = advocate_final.content if advocate_final else "No answer was proposed."
    if ":" in claim_text:
        final_claim = claim_text.split(":", 1)[-1].strip().split(".")[0]
    else:
        final_claim = claim_text
    cited = ", ".join(f"round {r}" for r in flagged_rounds) or "none"
    return (
        f"Final Answer: {final_claim}\n"
        f"Rationale: this synthesis reflects the Advocate's final position after the Critic's "
        f"objections (raised in {cited}) were checked against the stated premises and found not "
        "to change the conclusion; no unresolved contradictions remain."
    )


class ScriptedLLM(LLMBackend):
    """Rule-based backend; no network calls, fully deterministic given its input."""

    def generate(self, request: GenerationRequest) -> str:
        puzzle = _detect_cost_puzzle(request.question)
        if request.role == "advocate":
            return self._advocate(request, puzzle)
        if request.role == "critic":
            return self._critic(request, puzzle)
        if request.role == "judge":
            return self._judge(request, puzzle)
        raise ValueError(f"Unsupported role: {request.role!r}")

    @staticmethod
    def _advocate(request: GenerationRequest, puzzle: tuple[float, float] | None) -> str:
        if puzzle is None:
            if request.round_number == 1:
                return _advocate_generic_round1(request.question)
            prev_critic = _find_latest(request.history, "critic", request.round_number - 1)
            if prev_critic and prev_critic.content.startswith("Issue"):
                return _advocate_generic_revision(request.question)
            return _advocate_generic_reaffirm(request.question)

        total, difference = puzzle
        if request.round_number == 1:
            return _advocate_puzzle_round1(total, difference, _naive_answer(total, difference))

        prev_critic = _find_latest(request.history, "critic", request.round_number - 1)
        if prev_critic and prev_critic.content.startswith("Issue"):
            return _advocate_puzzle_revision(_correct_answer(total, difference), total, difference)

        prev_advocate = _find_latest(request.history, "advocate")
        value = _extract_dollar_value(prev_advocate.content) if prev_advocate else None
        value = value if value is not None else _correct_answer(total, difference)
        return f"I'll stand by my answer of ${value:.2f}, which satisfies both stated constraints."

    @staticmethod
    def _critic(request: GenerationRequest, puzzle: tuple[float, float] | None) -> str:
        if puzzle is None:
            return _critic_generic_response(request)
        total, difference = puzzle
        return _critic_puzzle_response(request, total, difference)

    @staticmethod
    def _judge(request: GenerationRequest, puzzle: tuple[float, float] | None) -> str:
        if puzzle is None:
            return _judge_generic_response(request)
        total, difference = puzzle
        return _judge_puzzle_response(request, total, difference)
