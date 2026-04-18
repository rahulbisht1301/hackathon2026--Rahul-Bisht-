import asyncio
import random
from collections import defaultdict

from agent.config import settings


class ToolFailureSimulator:
    """Stochastic but seedable failure injection for tool calls."""

    def __init__(self) -> None:
        self._rng = random.Random(settings.tool_failure_seed)
        self._call_counts: dict[str, int] = defaultdict(int)
        self._forced: dict[str, list[str]] = {}
        self._stats = {"injected_failures": 0, "recovered_failures": 0}

    def reset(self) -> None:
        self._rng = random.Random(settings.tool_failure_seed)
        self._call_counts.clear()
        self._forced.clear()
        self._stats = {"injected_failures": 0, "recovered_failures": 0}

    def force_failure_sequence(self, tool_name: str, sequence: list[str]) -> None:
        self._forced[tool_name] = sequence.copy()

    def mark_recovered_failure(self) -> None:
        self._stats["recovered_failures"] += 1

    def get_stats(self) -> dict[str, int]:
        return dict(self._stats)

    def should_fail(self, tool_name: str) -> str:
        self._call_counts[tool_name] += 1
        if tool_name in self._forced and self._forced[tool_name]:
            outcome = self._forced[tool_name].pop(0)
            if outcome != "none":
                self._stats["injected_failures"] += 1
            return outcome

        if tool_name == "issue_refund" and self._call_counts[tool_name] <= 1:
            return "none"

        roll = self._rng.random()
        if roll < settings.tool_failure_rate:
            self._stats["injected_failures"] += 1
            return self._rng.choice(["timeout", "malformed", "partial", "transient"])
        return "none"

    async def maybe_fail(self, tool_name: str) -> str:
        failure = self.should_fail(tool_name)
        if failure == "timeout":
            await asyncio.sleep(settings.tool_timeout_seconds + 0.01)
            raise TimeoutError(
                f"Tool '{tool_name}' timed out after {settings.tool_timeout_seconds} seconds."
            )
        if failure == "transient":
            raise RuntimeError(f"Tool '{tool_name}' transient error (503).")
        return failure


simulator = ToolFailureSimulator()

