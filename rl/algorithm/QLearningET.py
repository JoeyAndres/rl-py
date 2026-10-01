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

from typing import Any, Generic, Optional, Set, TypeVar, final

try:
    from ..agent.StateAction import StateAction
    from ..policy.Policy import Policy
    from .EligibilityTraces import EligibilityTraces
    from .QLearning import QLearning
    from .ReinforcementLearning import ReinforcementLearning
except (ImportError, ValueError):
    try:
        from ..agent.state_action import StateAction
        from ..policy.policy import Policy
        from .eligibility_traces import EligibilityTraces
        from .q_learning import QLearning
        from .reinforcement_learning import ReinforcementLearning
    except (ImportError, ValueError):
        try:
            from StateAction import StateAction  # type: ignore
            from Policy import Policy  # type: ignore
            from EligibilityTraces import EligibilityTraces  # type: ignore
            from QLearning import QLearning  # type: ignore
            from ReinforcementLearning import ReinforcementLearning  # type: ignore
        except (ImportError, ValueError):
            from rl.agent.StateAction import StateAction  # type: ignore
            from rl.policy.Policy import Policy  # type: ignore
            from rl.algorithm.EligibilityTraces import EligibilityTraces  # type: ignore
            from rl.algorithm.QLearning import QLearning  # type: ignore
            from rl.algorithm.ReinforcementLearning import ReinforcementLearning  # type: ignore

S = TypeVar("S")
A = TypeVar("A")


@final
class QLearningET(EligibilityTraces[S, A], QLearning[S, A], Generic[S, A]):
    """Like QLearning in that it uses different policy for learning and action
    selection. Each iteration, it updates all other states via eligibility
    traces, making for faster convergence.
    """

    def __init__(
        self,
        stepSize: float,
        discountRate: float,
        policy: Policy[S, A],
        lambda_: float,
        **kwargs: Any,
    ) -> None:
        stepSize = kwargs.get("step_size", stepSize)
        discountRate = kwargs.get("discount_rate", discountRate)
        lambda_ = kwargs.get("lambda", lambda_)

        EligibilityTraces.__init__(self, lambda_)
        QLearning.__init__(self, stepSize, discountRate, policy)

    def update(
        self,
        currentStateAction: StateAction[S, A],
        nextState: S,
        reward: float,
        actionSet: Set[A],
        **kwargs: Any,
    ) -> None:
        currentStateAction = kwargs.get("current_state_action", currentStateAction)
        nextState = kwargs.get("next_state", nextState)
        reward = kwargs.get("currentStateActionValue", kwargs.get("reward", reward))
        actionSet = kwargs.get("action_set", actionSet)

        if hasattr(self, "getLearningAction"):
            nextAction: A = self.getLearningAction(nextState, actionSet)
        else:
            nextAction = self.get_learning_action(nextState, actionSet)

        nextStateAction = StateAction(nextState, nextAction)

        if hasattr(ReinforcementLearning, "updateStateAction"):
            ReinforcementLearning.updateStateAction(
                self, currentStateAction, nextStateAction, reward
            )
        elif hasattr(ReinforcementLearning, "update_state_action"):
            ReinforcementLearning.update_state_action(
                self, currentStateAction, nextStateAction, reward
            )

        if not hasattr(self, "_eligibilityTraces") and not hasattr(self, "_eligibility_traces"):
            self._eligibilityTraces: dict[StateAction[S, A], float] = {}

        if hasattr(self, "_eligibilityTraces"):
            self._eligibilityTraces.setdefault(currentStateAction, 0.0)
        if hasattr(self, "_eligibility_traces"):
            self._eligibility_traces.setdefault(currentStateAction, 0.0)

        container = getattr(
            self,
            "_stateActionPairContainer",
            getattr(self, "_state_action_pair_container", None),
        )
        step_size = getattr(
            self, "_stepSize", getattr(self, "_step_size", None)
        )
        discount_rate = getattr(
            self, "_discountRate", getattr(self, "_discount_rate", None)
        )

        if hasattr(self, "_updateEligibilityTraces"):
            self._updateEligibilityTraces(
                currentStateAction,
                nextStateAction,
                reward,
                container,
                step_size,
                discount_rate,
            )
        elif hasattr(self, "_update_eligibility_traces"):
            self._update_eligibility_traces(
                currentStateAction,
                nextStateAction,
                reward,
                container,
                step_size,
                discount_rate,
            )