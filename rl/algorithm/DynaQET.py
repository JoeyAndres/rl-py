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

from typing import Any, Dict, Generic, Optional, Set, TypeVar, final

S = TypeVar("S")
A = TypeVar("A")

from .DynaQ import DynaQ
from .EligibilityTraces import EligibilityTraces
from ..agent.StateAction import StateAction
from ..policy.Policy import Policy


@final
class DynaQET(DynaQ[S, A], EligibilityTraces[S, A], Generic[S, A]):
    """
    Employs the simulation of DynaQ and eligibility traces.

    Like DynaQ, it does n simulations per update, but after that performs eligibility traces.

    See Also:
        DynaQ
        EligibilityTraces
    """

    def __init__(
        self,
        step_size: float = 0.0,
        discount_rate: float = 0.0,
        policy: Optional[Policy[S, A]] = None,
        simulation_iteration_count: int = 0,
        state_transition_greediness: float = 0.0,
        state_transition_step_size: float = 0.0,
        lambda_: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """
        :param step_size: range [0.0, 1.0]. High step size means faster learning, but
                          less precise convergence.
        :param discount_rate: range [0.0, 1.0]. High discount rate means more
                              consideration of future events.
        :param policy: online policy, that is policy used for action selection.
        :param simulation_iteration_count: number of simulations per update/backup.
        :param state_transition_greediness: greediness in selecting highest value model.
        :param state_transition_step_size: how fast does a model update a value of a
                                           state-action pair.
        :param lambda_: huge lambda means basically back up ranging from terminal state
                        to initial state. Small lambda converges to TD(0).
        """
        step_size = kwargs.get("stepSize", step_size)
        discount_rate = kwargs.get("discountRate", discount_rate)
        if "policy" in kwargs and kwargs["policy"] is not None:
            policy = kwargs["policy"]
        simulation_iteration_count = kwargs.get(
            "simulationIterationCount", simulation_iteration_count
        )
        state_transition_greediness = kwargs.get(
            "stateTransitionGreediness", state_transition_greediness
        )
        state_transition_step_size = kwargs.get(
            "stateTransitionStepSize", state_transition_step_size
        )
        lambda_ = kwargs.get("lambda", kwargs.get("lambda_", lambda_))

        DynaQ.__init__(
            self,
            step_size,
            discount_rate,
            policy,
            simulation_iteration_count,
            state_transition_greediness,
            state_transition_step_size,
        )
        EligibilityTraces.__init__(self, lambda_)

        self._step_size = step_size
        self._stepSize = step_size
        self._discount_rate = discount_rate
        self._discountRate = discount_rate
        self._lambda = lambda_

        if not hasattr(self, "_eligibility_traces") and not hasattr(self, "_eligibilityTraces"):
            self._eligibility_traces = {}
            self._eligibilityTraces = self._eligibility_traces
        elif hasattr(self, "_eligibility_traces") and not hasattr(self, "_eligibilityTraces"):
            self._eligibilityTraces = self._eligibility_traces
        elif hasattr(self, "_eligibilityTraces") and not hasattr(self, "_eligibility_traces"):
            self._eligibility_traces = self._eligibilityTraces

    def update(
        self,
        current_state_action: Optional[StateAction[S, A]] = None,
        next_state: Optional[S] = None,
        reward: float = 0.0,
        action_set: Optional[Set[A]] = None,
        **kwargs: Any,
    ) -> None:
        if current_state_action is None:
            current_state_action = kwargs.get("currentStateAction")
        if next_state is None:
            next_state = kwargs.get("nextState")
        if "reward" in kwargs:
            reward = kwargs["reward"]
        elif "currentStateActionValue" in kwargs:
            reward = kwargs["currentStateActionValue"]
        if action_set is None:
            action_set = kwargs.get("actionSet")

        if hasattr(self, "get_learning_action"):
            next_action = self.get_learning_action(next_state, action_set)
        elif hasattr(self, "getLearningAction"):
            next_action = self.getLearningAction(next_state, action_set)
        else:
            next_action = None

        next_state_action = StateAction(next_state, next_action)

        # For some reason I need to call the
        # grand parent class ReinforcementLearning.
        if hasattr(DynaQ, "update_state_action"):
            DynaQ.update_state_action(
                self, current_state_action, next_state_action, reward
            )
        elif hasattr(DynaQ, "updateStateAction"):
            DynaQ.updateStateAction(
                self, current_state_action, next_state_action, reward
            )
        elif hasattr(self, "update_state_action"):
            self.update_state_action(
                current_state_action, next_state_action, reward
            )
        elif hasattr(self, "updateStateAction"):
            self.updateStateAction(
                current_state_action, next_state_action, reward
            )

        traces = getattr(self, "_eligibility_traces", None)
        if traces is None:
            traces = getattr(self, "_eligibilityTraces", None)
        if traces is not None and current_state_action is not None:
            traces.setdefault(current_state_action, 0.0)

        container = getattr(
            self,
            "_state_action_pair_container",
            getattr(self, "_stateActionPairContainer", None),
        )
        step_size = getattr(
            self,
            "_step_size",
            getattr(self, "_stepSize", 0.0),
        )
        discount_rate = getattr(
            self,
            "_discount_rate",
            getattr(self, "_discountRate", 0.0),
        )

        if hasattr(self, "_update_eligibility_traces"):
            self._update_eligibility_traces(
                current_state_action,
                next_state_action,
                reward,
                container,
                step_size,
                discount_rate,
            )
        elif hasattr(self, "_updateEligibilityTraces"):
            self._updateEligibilityTraces(
                current_state_action,
                next_state_action,
                reward,
                container,
                step_size,
                discount_rate,
            )

        # Update model.
        if hasattr(self, "_update_model"):
            self._update_model(current_state_action, next_state, reward)
        elif hasattr(self, "_updateModel"):
            self._updateModel(current_state_action, next_state, reward)

        # Simulation.
        if hasattr(self, "_simulate"):
            self._simulate(action_set)