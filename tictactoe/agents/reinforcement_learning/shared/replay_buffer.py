"""Experience replay buffer.

Dependency chain position: types → state → board → game → agents → benchmark.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Experience:
    """One transition: (state, action, reward, next_state, done)."""
    state_tensor: list
    action: int
    reward: float
    next_state_tensor: list
    done: bool


class ReplayBuffer:
    """Fixed-size circular buffer for experience replay."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._buf: list[Experience] = []
        self._pos: int = 0

    def push(self, experience: Experience) -> None:
        if len(self._buf) < self._capacity:
            self._buf.append(experience)
        else:
            self._buf[self._pos] = experience
        self._pos = (self._pos + 1) % self._capacity

    def sample(self, batch_size: int) -> list[Experience]:
        if len(self._buf) < batch_size:
            raise ValueError(f"Buffer has {len(self._buf)} < {batch_size} experiences.")
        return random.sample(self._buf, batch_size)

    def __len__(self) -> int:
        return len(self._buf)

    def clear(self) -> None:
        self._buf = []
        self._pos = 0

    def is_ready(self, min_size: int) -> bool:
        return len(self._buf) >= min_size
