"""
rl - Reinforcement Learning
Copyright (C) 2016  Joey Andres<yeojserdna@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import math
from typing import Tuple

from rl.agent import Actuator, Environment, Sensor, StateAction
from rl.utility import FactoryAbstract

POS = 0
VEL = 1


class MountainCarEnvironment(Environment[Tuple[float, ...], Tuple[float, ...]]):
    """Classic mountain car. State is (position, velocity); action is (0|1|2,)."""

    def __init__(
        self, actuator: Actuator[Tuple[float, ...]], sensor: Sensor[Tuple[float, ...]]
    ) -> None:
        super().__init__(actuator, sensor)

    def getNextStateAndReward(
        self, stateAction: StateAction[Tuple[float, ...], Tuple[float, ...]]
    ) -> Tuple[Tuple[float, ...], float]:
        act = list(stateAction.getAction())
        nextState = list(stateAction.getState())

        copyAct = int(act[0]) - 1
        nextReward = -1.0 if copyAct == 0 else -2.0

        nextState[VEL] += 0.001 * copyAct - 0.0025 * math.cos(3.0 * nextState[POS])
        if nextState[VEL] < -0.07:
            nextState[VEL] = -0.07
        elif nextState[VEL] >= 0.07:
            nextState[VEL] = 0.06999999

        nextState[POS] += nextState[VEL]
        if nextState[POS] >= 0.5:
            nextState[POS] = 0.5
            nextReward = 0.0
        elif nextState[POS] < -1.2:
            nextState[POS] = -1.2
            nextState[VEL] = 0.0

        return tuple(nextState), nextReward


class MountainCarEnvironmentFactory(FactoryAbstract[Environment[Tuple[float, ...], Tuple[float, ...]]]):
    def __init__(
        self, actuator: Actuator[Tuple[float, ...]], sensor: Sensor[Tuple[float, ...]]
    ) -> None:
        super().__init__()
        self._instance = MountainCarEnvironment(actuator, sensor)
