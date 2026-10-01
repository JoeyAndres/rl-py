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

from typing import Generic, Set, TypeVar

from ..agent.StateAction import StateAction
from ..policy.Policy import Policy
from .DynaQRLMP import DynaQRLMP

S = TypeVar("S")
A = TypeVar("A")


class DynaQ(DynaQRLMP[S, A], Generic[S, A]):
    """DynaQ algorithm implementation.

    DynaQ algorithm is a QLearning algorithm that employs a model to "model"
    the environment. Thus, in every update, the algorithm can do n
    simulations, enabling faster convergence to "optimal" policy.
    """

    def __init__(
        self,
        stepSize: float,
        discountRate: float,
        policy: Policy[S, A],
        simulationIterationCount: int,
        stateTransitionGreediness: float,
        stateTransitionStepSize: float,
    ) -> None:
        """
        :param stepSize: range [0.0, 1.0]. High step size means faster
                         learning, but less precise convergence.
        :param discountRate: range [0.0, 1.0]. High discount rate means more
                             consideration of future events.
        :param policy: online policy, that is policy used for action selection.
        :param simulationIterationCount: number of simulations per update/backup.
        :param stateTransitionGreediness: greediness in selecting highest value model.
        :param stateTransitionStepSize: how fast does a model update a value of
                                        a state-action pair.
        """
        super().__init__(
            stepSize,
            discountRate,
            policy,
            simulationIterationCount,
            stateTransitionGreediness,
            stateTransitionStepSize,
        )

    def update(
        self,
        currentStateAction: StateAction[S, A],
        nextState: S,
        reward: float,
        actionSet: Set[A],
    ) -> None:
        nextAction = self.getLearningAction(nextState, actionSet)
        self.updateStateAction(
            currentStateAction, StateAction(nextState, nextAction), reward
        )
        self.backUpStateActionPair(
            currentStateAction, reward, StateAction(nextState, nextAction)
        )
        # Update model.
        self._updateModel(currentStateAction, nextState, reward)
        # Simulation.
        self._simulate(actionSet)
