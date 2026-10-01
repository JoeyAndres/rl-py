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

from enum import IntEnum
from typing import Dict, Tuple

from rl.agent import Actuator, Environment, Sensor, StateAction
from rl.utility import FactoryAbstract


class State(IntEnum):
    A = 0
    B = 1
    C = 2
    D = 3
    T = 4


class Action(IntEnum):
    L = 0
    R = 1


A, B, C, D, T = State.A, State.B, State.C, State.D, State.T
L, R = Action.L, Action.R


class RandomWalkEnvironment(Environment[int, int]):
    """5-state random walk: A <-> B <-> C <-> D, with T terminal off either end."""

    State = State
    Action = Action

    def __init__(self, actuator: Actuator[int], sensor: Sensor[int]) -> None:
        super().__init__(actuator, sensor)
        self._env: Dict[StateAction[int, int], int] = {
            StateAction(A, L): T,
            StateAction(A, R): B,
            StateAction(B, L): A,
            StateAction(B, R): C,
            StateAction(C, L): B,
            StateAction(C, R): D,
            StateAction(D, L): C,
            StateAction(D, R): T,
        }

    def getNextStateAndReward(
        self, stateAction: StateAction[int, int]
    ) -> Tuple[int, float]:
        nextState = self._env[stateAction]
        nextReward = 0.0 if nextState == T else -1.0
        return nextState, nextReward


class RandomWalkEnvironmentFactory(FactoryAbstract[Environment[int, int]]):
    def __init__(self, actuator: Actuator[int], sensor: Sensor[int]) -> None:
        super().__init__()
        self._instance = RandomWalkEnvironment(actuator, sensor)
