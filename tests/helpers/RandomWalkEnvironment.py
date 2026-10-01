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
from typing import Any, Dict, Tuple

from rl.agent import Actuator, Environment, Sensor, StateAction
from rl.utility import FactoryAbstract


class RandomWalkEnvironment(Environment):
    class State(IntEnum):
        A = 0
        B = 1
        C = 2
        D = 3
        T = 4

    class Action(IntEnum):
        L = 0
        R = 1

    def __init__(self, actuator: Actuator, sensor: Sensor) -> None:
        super().__init__(actuator, sensor)
        self._env: Dict[StateAction, int] = {
            StateAction(A, L): T,
            StateAction(A, R): B,
            StateAction(B, L): A,
            StateAction(B, R): C,
            StateAction(C, L): B,
            StateAction(C, R): D,
            StateAction(D, L): C,
            StateAction(D, R): T,
        }

    def getNextStateAndReward(self, stateAction: StateAction) -> Tuple[int, float]:
        nextState = self._env[stateAction]
        nextReward = -1.0
        if nextState == T:
            nextReward = 0.0
        return nextState, nextReward

    def get_next_state_and_reward(self, state_action: StateAction) -> Tuple[int, float]:
        return self.getNextStateAndReward(state_action)


State = RandomWalkEnvironment.State
Action = RandomWalkEnvironment.Action

A = RandomWalkEnvironment.State.A
B = RandomWalkEnvironment.State.B
C = RandomWalkEnvironment.State.C
D = RandomWalkEnvironment.State.D
T = RandomWalkEnvironment.State.T

L = RandomWalkEnvironment.Action.L
R = RandomWalkEnvironment.Action.R


class RandomWalkEnvironmentFactory(FactoryAbstract):
    def __init__(self, actuator: Actuator, sensor: Sensor) -> None:
        super().__init__()
        self._instance = RandomWalkEnvironment(actuator, sensor)