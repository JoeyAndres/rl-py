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
from rl.agent import StateAction


@pytest.fixture
def state01():
    return 10


@pytest.fixture
def action01():
    return 10


@pytest.fixture
def state02():
    return 20


@pytest.fixture
def action02():
    return 20


def test_state_action_accessors(state01, action01):
    """
    Scenario: StateAction data type
    Given: StateAction instance
    When: Accessor method are invoked
    Then: Retrieves appropriate value
    """
    state_action = StateAction(state01, action01)
    assert state_action.getState() == state01
    assert state_action.getAction() == action01


def test_state_action_equality(state01, action01, state02):
    """
    Scenario: StateAction data type
    Given: Multiple StateAction instance
    When: Equality operator is invoked
    Then: Returns true
    """
    state_action01 = StateAction(state01, action01)
    state_action02 = StateAction(state01, action01)
    state_action03 = StateAction(state02, action01)

    assert state_action01 == state_action02
    assert state_action01 != state_action03
    assert state_action02 != state_action03


def test_state_action_less_operator(state01, action01, state02, action02):
    """
    Scenario: StateAction data type
    Given: Multiple StateAction instance
    When: Less operator is invoked
    Then: Compares lexically <S, A>
    """
    state_action01 = StateAction(state01, action01)
    state_action02 = StateAction(state01, action02)
    state_action03 = StateAction(state01, action02)
    state_action04 = StateAction(state02, action01)

    assert state_action01 != state_action02
    assert state_action01 < state_action02
    assert state_action03 != state_action04
    assert state_action03 < state_action04