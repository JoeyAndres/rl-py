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

from typing import Dict, Optional, Set, Tuple

from ...agent.StateAction import StateAction
from ...coding.CourseCode import CourseCode
from ...declares import FLOAT
from ...policy.Policy import Policy
from ..LearningAlgorithm import LearningAlgorithm
from .GradientDescentAbstract import ActionParam, GradientDescentAbstract, StateParam


class ReinforcementLearningGDAbstract(LearningAlgorithm[StateParam, ActionParam]):
    """Base class for reinforcement learning with gradient descent.

    Children assign _gradientDescent to pick the weight update rule.
    """

    def __init__(
        self,
        tileCode: CourseCode,
        stepSize: FLOAT,
        discountRate: FLOAT,
        lambda_: FLOAT,
        policy: Policy[StateParam, ActionParam],
    ) -> None:
        """
        :param tileCode: Course code used for function approximation.
        :param stepSize: Step size of the weight update.
        :param discountRate: How much future rewards are considered.
        :param lambda_: Eligibility trace decay.
        :param policy: Control policy.
        """
        super().__init__(policy)
        self._gradientDescent: Optional[GradientDescentAbstract] = None

    def update(
        self,
        currentStateAction: StateAction[StateParam, ActionParam],
        nextState: StateParam,
        reward: FLOAT,
        actionSet: Set[ActionParam],
    ) -> None:
        actionValueMap, maxAction = self._buildActionValues(actionSet, nextState)
        nextAction = self._getLearningPolicyAction(actionValueMap, actionSet, maxAction)
        self._gradientDescent.updateWeights(
            currentStateAction.getState(),
            currentStateAction.getAction(),
            nextState,
            actionValueMap[nextAction],
            reward,
        )

    def getAction(
        self, state: StateParam, actionSet: Set[ActionParam]
    ) -> ActionParam:
        actionValueMap, maxAction = self._buildActionValues(actionSet, state)
        return self._learningPolicy.getAction(actionValueMap, actionSet, maxAction)

    def getStateActionValue(
        self, stateAction: StateAction[StateParam, ActionParam]
    ) -> FLOAT:
        return self._getStateActionValue(stateAction)

    def _getStateActionValue(
        self, stateAction: StateAction[StateParam, ActionParam]
    ) -> FLOAT:
        params = tuple(stateAction.getState()) + tuple(stateAction.getAction())
        return self._gradientDescent.getValueFromParameters(params)

    def _buildActionValues(
        self, actionSet: Set[ActionParam], nextState: StateParam
    ) -> Tuple[Dict[ActionParam, FLOAT], ActionParam]:
        return self._gradientDescent.buildActionValues(actionSet, nextState)
