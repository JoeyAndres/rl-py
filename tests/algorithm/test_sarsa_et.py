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
from rl.algorithm import SarsaETFactory
from rl.policy import EpsilonGreedyFactory
from tests.helpers import (
    B,
    L,
    R,
    RandomWalkEnvironmentFactory,
    T,
)


def test_sarsa_et_converge_to_a_solution():
    # Setup actuator with actions.
    arw = ActuatorFactory([L, R]).get()
    # Setup sensor.
    srw = SensorDiscreteFactory(B).get()
    srw.addTerminalState(T)  # Setup terminal state.

    # Setup environment.
    rwe = RandomWalkEnvironmentFactory