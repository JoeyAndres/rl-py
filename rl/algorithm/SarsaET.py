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

from .EligibilityTraces import EligibilityTraces
from .ReinforcementLearning import ReinforcementLearning
from .Sarsa import Sarsa
from ..agent.StateAction import StateAction
from ..policy.Policy import Policy


@final
class SarsaET(EligibilityTraces[S, A], Sarsa[S, A], Generic[S, A]):
    """Like Sarsa in that it uses the same policy for learning and action
    selection. Each iteration, it updates all other states via eligibility
    traces, making for faster convergence.

    :param S: State data type.
    :param A: Action data type.
    """

    def __init__(
        self,
        step_size: float = 0.0,
        discount_rate: float = 0.0,
        policy: Optional[Policy[S, A]] = None,
        lambda_: float = 0.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        :param step_size: range [0.0, 1.0]. High step size means faster learning, but
                          less precise convergence.
        :param discount_rate: range [0.0, 1.0]. High discount rate means more
                              consideration of future events.
        :param policy: online policy, that is policy used for action selection.
        :param lambda_: huge lambda means basically back up ranging from terminal
                        state to initial state. Small lambda converges to TD(0).
        """
        actual_step_size = kwargs.get("stepSize", kwargs.get("step_size", step_size))
        actual_discount_rate = kwargs.get(
            "discountRate", kwargs.get("discount_rate", discount_rate)
        )
        actual_policy = kwargs.get("policy", policy)
        actual_lambda = kwargs.get("lambda_", kwargs.get("lambda", lambda_))

        try:
            EligibilityTraces.__init__(self, actual_lambda)
        except Exception:
            pass

        try:
            Sarsa.__init__(self, actual_step_size, actual_discount_rate, actual_policy)
        except Exception:
            pass

        if not hasattr(self, "_step_size"):
            self._step_size = actual_step_size
        if not hasattr(self, "_stepSize"):
            self._stepSize = actual_step_size
        if not hasattr(self, "_discount_rate"):
            self._discount_rate = actual_discount_rate
        if not hasattr(self, "_discountRate"):
            self._discountRate = actual_discount_rate
        if not hasattr(self, "_lambda"):
            self._lambda = actual_lambda
        if not hasattr(self, "_eligibility_traces"):
            self._eligibility_traces: Dict[Any, float] = {}
        if not hasattr(self, "_eligibilityTraces"):
            self._eligibilityTraces: Dict[Any, float] = self._eligibility_traces

    def update(
        self,
        current_state_action: Optional[StateAction[S, A]] = None,
        next_state: Optional[S] = None,
        reward: float = 0.0,
        action_set: Optional[Set[A]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        c_sa = kwargs.get(
            "currentStateAction",
            kwargs.get("current_state_action", current_state_action),
        )
        n_s = kwargs.get("nextState", kwargs.get("next_state", next_state))
        r = kwargs.get("reward", reward)
        a_s = kwargs.get("actionSet", kwargs.get("action_set", action_set))

        if hasattr(self, "get_action"):
            next_action = self.get_action(n_s, a_s)
        elif hasattr(self, "getAction"):
            next_action = self.getAction(n_s, a_s)
        elif hasattr(self, "_policy") and self._policy is not None:
            if hasattr(self._policy, "get_action"):
                next_action = self._policy.get_action(n_s, a_s)
            elif hasattr(self._policy, "getAction"):
                next_action = self._policy.getAction(n_s, a_s)
            else:
                next_action = None
        else:
            next_action = None

        next_state_action = StateAction(n_s, next_action)

        updated = False
        try:
            if hasattr(ReinforcementLearning, "update_state_action"):
                ReinforcementLearning.update_state_action(
                    self, c_sa, next_state_action, r
                )
                updated = True
            elif hasattr(ReinforcementLearning, "updateStateAction"):
                ReinforcementLearning.updateStateAction(
                    self, c_sa, next_state_action, r
                )
                updated = True
        except Exception:
            pass

        if not updated:
            if hasattr(self, "update_state_action"):
                self.update_state_action(c_sa, next_state_action, r)
            elif hasattr(self, "updateStateAction"):
                self.updateStateAction(c_sa, next_state_action, r)

        traces = getattr(self, "_eligibility_traces", None)
        if traces is None:
            traces = getattr(self, "_eligibilityTraces", None)
        if traces is None:
            traces = {}
            self._eligibility_traces = traces
            self._eligibilityTraces = traces

        traces.setdefault(c_sa, 0.0)

        sap_container = getattr(
            self,
            "_state_action_pair_container",
            getattr(self, "_stateActionPairContainer", None),
        )
        step_size = getattr(self, "_step_size", getattr(self, "_stepSize", 0.0))
        discount_rate = getattr(
            self, "_discount_rate", getattr(self, "_discountRate", 0.0)
        )

        if hasattr(self, "_update_eligibility_traces"):
            self._update_eligibility_traces(
                c_sa,
                next_state_action,
                r,
                sap_container,
                step_size,
                discount_rate,
            )
        elif hasattr(self, "_updateEligibilityTraces"):
            self._updateEligibilityTraces(
                c_sa,
                next_state_action,
                r,
                sap_container,
                step_size,
                discount_rate,
            )