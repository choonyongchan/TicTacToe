"""Tests for ReplayBuffer."""
from tictactoe.agents.reinforcement_learning.shared.replay_buffer import ReplayBuffer, Experience
import pytest


def make_exp(r=0.0, done=False):
    return Experience([0.0]*3, 0, r, [0.0]*3, done)


def test_push_increases_length():
    buf = ReplayBuffer(10)
    buf.push(make_exp())
    assert len(buf) == 1


def test_circular_overwrite():
    buf = ReplayBuffer(3)
    for i in range(5):
        buf.push(Experience([float(i)]*3, i, float(i), [float(i)]*3, False))
    assert len(buf) == 3


def test_sample_returns_batch_size():
    buf = ReplayBuffer(20)
    for _ in range(10):
        buf.push(make_exp())
    batch = buf.sample(5)
    assert len(batch) == 5


def test_sample_raises_when_underfull():
    buf = ReplayBuffer(10)
    buf.push(make_exp())
    with pytest.raises(ValueError):
        buf.sample(5)


def test_clear_resets():
    buf = ReplayBuffer(10)
    buf.push(make_exp())
    buf.clear()
    assert len(buf) == 0


def test_is_ready():
    buf = ReplayBuffer(10)
    assert not buf.is_ready(5)
    for _ in range(5):
        buf.push(make_exp())
    assert buf.is_ready(5)
    assert not buf.is_ready(6)
