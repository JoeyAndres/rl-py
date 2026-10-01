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

from rl.agent import ActuatorFactory, AgentGD
from rl.algorithm import SarsaETGDFactory
from rl.coding import DimensionInfo, TileCodeCorrectFactory
from rl.policy import EpsilonGreedyFactory
from tests.helpers import MountainCarEnvironmentFactory, SensorMountainCarFactory


@pytest.mark.rl_sarsa_etgd
def test_sarsa_eligibility_traces_and_gradient_descent_converge_to_a_solution():
    """Scenario: Sarsa Eligibility Traces and Gradient Descent converge to a solution."""
    # Given: A Mountain Car environment
    reverse = [0.0]
    neutral = [1.0]
    forward = [2.0]

    amc = ActuatorFactory([reverse, neutral, forward]).get()
    # Setup sensor.
    smc = SensorMountainCarFactory.create()
    # Setup environment.
    mce = MountainCarEnvironmentFactory(amc, smc).get()

    policy = EpsilonGreedyFactory(1.0).get()

    # Setup tile coding.
    dimensional_info_vector = [
        DimensionInfo(-1.2, 0.5, 10),  # Velocity.
        DimensionInfo(-0.07, 0.07, 10),  # Position.
        # Action dimension.
        DimensionInfo(0.0, 2.0, 3, 0.0),
    ]

    # When: We do multiple episodes with Default weight container (vector)
    # Setup tile coding with 8 offsets.
    tile_code = TileCodeCorrectFactory(dimensional_info_vector, 8).get()
    sarsa = SarsaETGDFactory(tile_code, 0.1, 1.0, 0.9, policy).get()
    agent = AgentGD(mce, sarsa)

    iteration_count = 0
    for _ in range(1