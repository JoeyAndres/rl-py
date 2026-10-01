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

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

D = TypeVar("D", bound=int)
STATE_DIM = TypeVar("STATE_DIM", bound=int)

StateParam = Tuple[float, ...]
spStateParam = Optional[StateParam]
ActionParam = Tuple[float, ...]
spActionParam = Optional[ActionParam]
spActionValueMap = Dict[ActionParam, float]
spActionSet = Iterable[ActionParam]


class GradientDescentAbstract(ABC, Generic[D, STATE_DIM]):
    """Gradient Descent implementation. Abstract class.

    tparam D: Number of dimensions.
    tparam STATE_DIM: Number of dimensions in State.
                      This also implies ACTION_DIM = D - STATE_DIM.
    """

    ACTION_DIM: int

    def __init__(
        self,
        courseCode: Any,
        stepSize: float,
        discountRate: float,
        lambda_: float,
    ) -> None:
        """
        :param courseCode: Type of course coding.
        :param stepSize: Step size for gradient descent.
        :param discountRate: Discount rate for gradient descent.
        :param lambda_: How influential current state-action is to other state-actions.
        """
        self._courseCode = courseCode
        self._stepSize: float = float(stepSize)
        self._discountRate: float = float(discountRate)
        self._lambda: float = float(lambda_)
        self._discountRateTimesLambda: float = self._discountRate * self._lambda

    def getSize(self) -> int:
        """Size of both weight vector and traces vector."""
        if hasattr(self._courseCode, "getSize"):
            return self._courseCode.getSize()
        return len(self._courseCode)

    def getValueFromParameters(self, parameters: Sequence[float]) -> float:
        """Get the value of the parameters in the real space.

        :param parameters: array of parameters.
        :return: corresponding value.
        """
        return float(self._courseCode.getValueFromParameters(parameters))

    @abstractmethod
    def updateWeights(
        self,
        currentStateVector: StateParam,
        currentActionVector: ActionParam,
        nextStateVector: StateParam,
        nextActionValue: float,
        reward: float,
    ) -> None:
        """
        :param currentStateVector: array of current states.
        :param currentActionVector: action taken to get to nextStateVector.
        :param nextStateVector: array of next states.
        :param nextActionValue: value of next state-action.
        :param reward: reward for taking nextAction.
        """
        pass

    def buildActionValues(
        self,
        actionSet: spActionSet,
        nextState: StateParam,
    ) -> Tuple[spActionValueMap, spActionParam]:
        """Acquires the action-value map as well as the action with max action value.

        :param actionSet: Set/collection of actions.
        :param nextState: Next state.
        :return: (actionValueMap, maxAction)
        """
        actionVectorValueMap: spActionValueMap = {}
        maxAction: spActionParam = None

        actionIter = iter(actionSet)
        try:
            firstAction = next(actionIter)
        except StopIteration:
            return actionVectorValueMap, maxAction

        firstActionTuple = tuple(firstAction)
        paramCopy = tuple(nextState) + firstActionTuple
        maxVal = self.getValueFromParameters(paramCopy)
        actionVectorValueMap[firstActionTuple] = maxVal
        maxAction = firstActionTuple

        for action in actionIter:
            actionTuple = tuple(action)
            paramCopy = tuple(nextState) + actionTuple
            value = self.getValueFromParameters(paramCopy)
            actionVectorValueMap[actionTuple] = value
            if maxVal < value:
                maxVal = value
                maxAction = actionTuple

        return actionVectorValueMap, maxAction

    def getMaxValue(self, actionValueMap: spActionValueMap) -> float:
        """
        :param actionValueMap: state-action to value mapping.
        :return: max value.
        """
        if not actionValueMap:
            return 0.0
        return max(actionValueMap.values())

    # Pythonic snake_case aliases
    get_size = getSize
    get_value_from_parameters = getValueFromParameters
    update_weights = updateWeights
    build_action_values = buildActionValues
    get_max_value = getMaxValue