"""Named random streams with exact capture/restore (frozen interface, 2026-10-09).

All randomness in a run comes from these streams. Kernels never draw; they receive
arrays drawn here. ``capture()`` returns a JSON-serialisable dict; ``restore()`` sets it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

STREAMS: tuple[str, ...] = (
    "population",  # initial heterogeneity draws
    "swarm",  # arrivals and behaviour of background agents
    "matching",  # tie-breaks, pro-rata rounding, speed-bump delays
    "batch",  # random batch lengths
    "llm",  # per-call sample seeds and MockLLM policies
    "interventions",  # shock delays, removal draws (re-spawned on branch)
    "view",  # pseudonyms and view reshuffles
)


class RNGRegistry:
    def __init__(self, seed: int, streams: tuple[str, ...] = STREAMS) -> None:
        self.seed = int(seed)
        self.streams = tuple(streams)
        root = np.random.SeedSequence(self.seed)
        children = root.spawn(len(self.streams))
        self._gens: dict[str, np.random.Generator] = {
            name: np.random.Generator(np.random.PCG64(child))
            for name, child in zip(self.streams, children, strict=True)
        }

    def __getitem__(self, name: str) -> np.random.Generator:
        return self._gens[name]

    def capture(self) -> dict[str, Any]:
        return {name: _to_plain(gen.bit_generator.state) for name, gen in self._gens.items()}

    def restore(self, state: dict[str, Any]) -> None:
        for name, st in state.items():
            self._gens[name].bit_generator.state = st

    def respawn(self, name: str, *salts: int) -> None:
        """Replace one stream deterministically (used for ``interventions`` on branch)."""
        ss = np.random.SeedSequence([self.seed, *[int(s) for s in salts]])
        self._gens[name] = np.random.Generator(np.random.PCG64(ss))


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, np.integer):
        return int(obj)
    return obj
