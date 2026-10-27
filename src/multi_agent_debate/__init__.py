"""Multi-agent LLM debate framework.

Three roles (Advocate, Critic, Judge) collaborate over N configurable rounds
to turn an initial, possibly-wrong answer into a better, checked one.
"""

from .agents import Advocate, Agent, Critic, Judge
from .debate import Debate
from .llm import AnthropicLLM, DebateTurn, GenerationRequest, LLMBackend
from .scripted_llm import ScriptedLLM
from .transcript import DebateResult, Message

__version__ = "0.1.0"

__all__ = [
    "Advocate",
    "Agent",
    "AnthropicLLM",
    "Critic",
    "Debate",
    "DebateResult",
    "DebateTurn",
    "GenerationRequest",
    "Judge",
    "LLMBackend",
    "Message",
    "ScriptedLLM",
]
