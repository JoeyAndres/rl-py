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

from rl.agent import ActuatorFactory, Agent, SensorDiscreteFactory
from rl.algorithm import DynaQFactory
from rl.policy import EpsilonGreedyFactory
from tests.helpers import (
    A,
    B,
    L,
    R,
    RandomWalkEnvironmentFactory,
    T,
)


def test_agent_interacts_with_the_environment_as_expected():
    """Agent interacts with the environment as expected."""
    # Given: A binary environment in which 1 is good and 0 is bad.
    # Setup actuator with actions.
    arw = ActuatorFactory([L, R]).get()
    # Setup sensor.
    srw = SensorDiscreteFactory(B).get()
    srw.add_terminal_state(T)  # Setup terminal state.

    # Setup environment.
    rwe = RandomWalkEnvironmentFactory(arw, srw).get()

    policy = EpsilonGreedyFactory(1.0).get()
    dyna_q_algorithm = DynaQFactory(0.1, 0.9, policy, 100, 1.0, 1.0).get()
    agent = Agent(rwe, dyna_q_algorithm)

    # When: When I move left.
    assert agent.get_last_observed_state() == B
    agent.apply_action(L)

    # Then: I move from B to A.
    assert agent.get_last_observed_state() == A