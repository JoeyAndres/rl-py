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

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Sequence, Set, Tuple

from ...coding.CourseCode import CourseCode
from ...declares import FLOAT

#: State and action parameters are tuples of floats so they can key dicts
#: (C++: floatArray<STATE_DIM> / floatArray<ACTION_DIM>).
StateParam = Tuple[FLOAT, ...]
ActionParam = Tuple[FLOAT, ...]


class GradientDescentAbstract(ABC):
    """Gradient Descent implementation. Abstract class.

    The C++ D template parameter is the course code's dimension and
    STATE_DIM is the length of the state parameters; ACTION_DIM is
    implied as D - STATE_DIM.
    """

    def __init__(
        self,
        courseCode: CourseCode,
        stepSize: FLOAT,
        discountRate: FLOAT,
        lambda_: FLOAT,
    ) -> None:
        """
        :param courseCode: Course code (e.g. a TileCode).
        :param stepSize: Step size of the weight update.
        :param discountRate: Mixed with lambda_ on how past states influence current.
        :param lambda_: Mixed with discountRate on how past states influence current.
        """
        self._courseCode = courseCode
        self._stepSize = stepSize
        self._discountRate = discountRate
        self._lambda = lambda_
        self._discountRateTimesLambda = discountRate * lambda_

    def getSize(self) -> int:
        return self._courseCode.getSize()

    def getValueFromParameters(self, parameters: Sequence[FLOAT]) -> FLOAT:
        return self._courseCode.getValueFromParameters(parameters)

    @abstractmethod
    def updateWeights(
        self,
        currentStateVector: StateParam,
        currentActionVector: ActionParam,
        nextStateVector: StateParam,
        nextActionValue: FLOAT,
        reward: FLOAT,
    ) -> None:
        """Update weights with tdError.

        :param currentStateVector: State vector.
        :param currentActionVector: Action vector.
        :param nextStateVector: Next state vector.
        :param nextActionValue: Value of the next action.
        :param reward: Reward for taking the current action from the current state.
        """

    def buildActionValues(
        self, actionSet: Set[ActionParam], nextState: StateParam
    ) -> Tuple[Dict[ActionParam, FLOAT], ActionParam]:
        """
        :param actionSet: Set of actions.
        :param nextState: State to evaluate each action from.
        :return: Mapping of action to value, and the action with the highest value.
        """
        actionValueMap: Dict[ActionParam, FLOAT] = {}
        maxAction = None
        maxVal = 0.0
        for action in actionSet:
            value = self.getValueFromParameters(tuple(nextState) + tuple(action))
            actionValueMap[action] = value
            if maxAction is None or maxVal < value:
                maxVal = value
                maxAction = action
        return actionValueMap, maxAction

    def getMaxValue(self, actionValueMap: Dict[ActionParam, FLOAT]) -> FLOAT:
        """
        :param actionValueMap: Mapping of action to value.
        :return: The highest value.
        """
        return max(actionValueMap.values())


spGradientDescentAbstract = GradientDescentAbstract
