# rl - Reinforcement Learning
# Copyright (C) 2016  Joey Andres<yeojserdna@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import pytest
from rl.agent import StateActionTransition, StateActionTransitionException

# Ensure compatibility with both camelCase and snake_case method names
for _attr, _alias in [
    ("getSize", "get_size"),
    ("getNextState", "get_next_state"),
    ("getReward", "get_reward"),
]:
    if hasattr(StateActionTransition, _attr) and not hasattr(StateActionTransition, _alias):
        setattr(StateActionTransition, _alias, getattr(StateActionTransition, _attr))
    elif hasattr(StateActionTransition, _alias) and not hasattr(StateActionTransition, _attr):
        setattr(StateActionTransition, _attr, getattr(StateActionTransition, _alias))


def test_state_action_transition_initial_size():
    """
    Scenario: StateActionTransition have the ability to represent offline model for offline calculation
    Given: StateActionTransition instance
    When: Size is acquired
    Then: Returns 0
    """
    sat = StateActionTransition(1.0, 0.2)
    assert sat.getSize() == 0


def test_state_action_transition_frequency_selection():
    """
    Scenario: StateActionTransition have the ability to represent offline model for offline calculation
    Given: StateActionTransition instance and a set of states
    When: Update all of them, one more than the other
    Then: The one exposed to most reward and often will likely be chosen more
    """
    sat = StateActionTransition(1.0, 0.2)

    state01 = 1
    state02 = 2
    state03 = 3

    # Update all of them. One more than the other.
    sat.update(state01, 10.0)
    assert sat.getSize() == 1
    assert sat.getNextState() == state01  # 1 state stored.

    sat.update(state02, 10.0)
    sat.update(state02, 10.0)
    assert sat.getSize() == 2  # 2 states stored.

    sat.update(state03, 10.0)
    sat.update(state03, 10.0)
    sat.update(state03, 10.0)
    assert sat.getSize() == 3  # 3 states stored.

    # The one exposed to most reward and often will likely be chosen more
    state01_occurrence_count = 0
    state02_occurrence_count = 0
    state03_occurrence_count = 0

    for _ in range(1000):
        state = sat.getNextState()
        if state == state01:
            state01_occurrence_count += 1
        elif state == state02:
            state02_occurrence_count += 1
        elif state == state03:
            state03_occurrence_count += 1
        else:
            pytest.fail("Not supposed to happen.")

    assert state01_occurrence_count < state02_occurrence_count
    assert state02_occurrence_count < state03_occurrence_count


def test_empty_state_action_transition_get_reward_raises():
    """
    Scenario: StateActionTransition have the ability to represent offline model for offline calculation
    Given: Empty StateActionTransition (No states)
    When: I try to access the reward
    Then: I get an exception
    """
    sat = StateActionTransition(1.0, 0.2)
    with pytest.raises(StateActionTransitionException):
        sat.getReward(23)


def test_empty_state_action_transition_get_next_state_raises():
    """
    Scenario: StateActionTransition have the ability to represent offline model for offline calculation
    Given: Empty StateActionTransition (No states)
    When: I try to access next state
    Then: I get an exception
    """
    sat = StateActionTransition(1.0, 0.2)
    with pytest.raises(StateActionTransitionException):
        sat.getNextState()


def test_state_action_transition_unknown_state_get_reward_raises():
    """
    Scenario: StateActionTransition have the ability to represent offline model for offline calculation
    Given: StateActionTransition with updated state
    When: I try to access a reward for a state that was not known to this StateActionTransition
    Then: I get an exception
    """
    sat = StateActionTransition(1.0, 0.2)
    state10 = 10
    state23 = 23
    sat.update(state10, 100.0)
    with pytest.raises(StateActionTransitionException):
        sat.getReward(state23)