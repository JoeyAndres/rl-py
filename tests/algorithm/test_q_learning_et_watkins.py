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
from rl.algorithm import QLearningETFactory
from rl.policy import EpsilonGreedyFactory
from tests.helpers import B, L, R, T, RandomWalkEnvironmentFactory, SensorRandomWalk


def test_qlearning_et_watkins_converge_to_a_solution():
    # GIVEN: A random walk environment
    # Setup actuator with actions.
    arw = ActuatorFactory([L, R]).get()
    # Setup sensor.
    srw = SensorDiscreteFactory(B).get()
    srw.add_terminal_state(T)  # Setup terminal state.

    # Setup environment.
    rwe = RandomWalkEnvironmentFactory(arw, srw).get()

    policy = EpsilonGreedyFactory(1.0).get()
    q_learning_et_watkins = QLearningETFactory(0.1, 0.9, policy, 0.5).get()

    agent = Agent(rwe, q_learning_et_watkins)

    # WHEN: We do multiple episodes
    iteration_count = 0
    for _ in range(100):
        agent.reset()  # Overloaded to go back to set b.

        iteration_count = agent.execute_episode()

        # THEN: At the end, we solve the random walk environment in 2 iteration
        assert iteration_count <= 2