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

import math

try:
    from rl.agent import Environment, StateAction
except ImportError:
    from rl import Environment, StateAction

try:
    from rl import FactoryAbstract
except ImportError:
    from rl.utility import FactoryAbstract

POS = 0
VEL = 1


class MountainCarEnvironment(Environment):
    POS = POS
    VEL = VEL
    SA = StateAction

    def __init__(self, actuator, sensor):
        super().__init__(actuator, sensor)

    def getNextStateAndReward(self, stateAction):
        if hasattr(stateAction, "getAction"):
            act = stateAction.getAction()
        elif hasattr(stateAction, "get_action"):
            act = stateAction.get_action()
        elif hasattr(stateAction, "action"):
            act = stateAction.action
        else:
            act = stateAction[1]

        if hasattr(act, "at"):
            act_val = act.at(0)
        elif hasattr(act, "__getitem__"):
            act_val = act[0]
        else:
            act_val = act

        copyAct = int(act_val) - 1

        if hasattr(stateAction, "getState"):
            state = stateAction.getState()
        elif hasattr(stateAction, "get_state"):
            state = stateAction.get_state()
        elif hasattr(stateAction, "state"):
            state = stateAction.state
        else:
            state = stateAction[0]

        if hasattr(state, "copy"):
            nextState = state.copy()
        elif hasattr(state, "__iter__"):
            nextState = list(state)
        else:
            nextState = [state[POS], state[VEL]]

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

        return nextState, nextReward

    get_next_state_and_reward = getNextStateAndReward


class MountainCarEnvironmentFactory(FactoryAbstract):
    def __init__(self, actuator, sensor):
        super().__init__()
        self._instance = MountainCarEnvironment(actuator, sensor)

    def get(self):
        return self._instance