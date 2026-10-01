import pytest

from rl.agent import StateActionTransition, StateActionTransitionException


def test_new_transition_is_empty():
    assert StateActionTransition(1.0, 0.2).getSize() == 0


def test_frequent_rewarding_states_chosen_more():
    sat = StateActionTransition(1.0, 0.2)
    sat.update(1, 10)
    assert sat.getSize() == 1
    assert sat.getNextState() == 1
    sat.update(2, 10)
    sat.update(2, 10)
    assert sat.getSize() == 2
    sat.update(3, 10)
    sat.update(3, 10)
    sat.update(3, 10)
    assert sat.getSize() == 3

    counts = {1: 0, 2: 0, 3: 0}
    for _ in range(1000):
        counts[sat.getNextState()] += 1
    assert counts[1] < counts[2] < counts[3]


def test_empty_transition_reward_raises():
    with pytest.raises(StateActionTransitionException):
        StateActionTransition(1.0, 0.2).getReward(23)


def test_empty_transition_next_state_raises():
    with pytest.raises(StateActionTransitionException):
        StateActionTransition(1.0, 0.2).getNextState()


def test_unknown_state_reward_raises():
    sat = StateActionTransition(1.0, 0.2)
    sat.update(10, 100)
    with pytest.raises(StateActionTransitionException):
        sat.getReward(23)
