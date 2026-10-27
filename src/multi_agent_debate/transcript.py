"""Transcript data structures and JSONL persistence for debate runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Message:
    """One spoken turn in the debate."""

    round: int
    role: str
    speaker: str
    content: str
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DebateResult:
    """The full outcome of a :class:`~multi_agent_debate.debate.Debate` run."""

    question: str
    rounds: int
    transcript: list[Message] = field(default_factory=list)
    final_answer: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "rounds": self.rounds,
            "transcript": [message.to_dict() for message in self.transcript],
            "final_answer": self.final_answer,
        }


def start_transcript_file(path: str | Path) -> None:
    """Create (or truncate) the JSONL transcript file so a run starts clean."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def append_message_jsonl(path: str | Path, message: Message) -> None:
    """Append one message to a JSONL transcript file, creating it if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(message.to_dict(), ensure_ascii=False) + "\n")


def load_transcript_jsonl(path: str | Path) -> list[Message]:
    """Load a JSONL transcript file back into a list of :class:`Message`."""
    messages: list[Message] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            messages.append(Message(**json.loads(line)))
    return messages
