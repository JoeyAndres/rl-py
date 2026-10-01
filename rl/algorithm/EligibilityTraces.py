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

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Generic, TypeVar

if TYPE_CHECKING:
    from ..agent.StateAction import StateAction
    from ..agent.StateActionPairContainer import StateActionPairContainer
else:
    try:
        from ..agent.StateAction import StateAction
    except Exception:
        try:
            from rl.agent.StateAction import StateAction
        except Exception:
            try:
                from agent.StateAction import StateAction
            except Exception:
                StateAction = None

    try:
        from ..agent.StateActionPairContainer import StateActionPairContainer
    except Exception:
        try:
            from rl.agent.StateActionPairContainer import StateActionPairContainer
        except Exception:
            try:
                from agent.StateActionPairContainer import StateActionPairContainer
            except Exception:
                StateActionPairContainer = None

S = TypeVar("S")
A = TypeVar("A")


class EligibilityTraces(Generic[S, A]):
    """
    Uses eligibility traces to update state-value pairs. This enables for
    fast convergence.

    As opposed to ordinary Temporal Difference or what's called in the
    literature as TD(0), eligibility traces TD(lambda) bridges the
    gap between TD(0) and Monte Carlo methods. A huge lambda means
    basically back up ranging from terminal state to initial state.
    Small lambda converges to TD(0).

    Note that there is a performance penalty as result of having
    to iterate all of state space.

    TODO: Make another parent class for all other eligibility trace algorithm that contains
    a merge of rl and Eligibility traces. This way, we can sync some shit.

    TODO: Connect the reset() to resetEligibilityTraces() function that have yet to be made.
    """

    def __init__(self, lambda_: float) -> None:
        """
        :param lambda_: A huge lambda means basically back up ranging from
                        terminal state to initial state. Small lambda converges to TD(0).
        """
        self._lambda: float = float(lambda_)
        self._eligibility_traces: Dict[StateAction[S, A], float] = defaultdict(float)
        self._eligibilityTraces = self._eligibility_traces

    def set_lambda(self, lambda_: float) -> None:
        """
        :param lambda_: A huge lambda means basically back up ranging from
                        terminal state to initial state. Small lambda converges to TD(0).
        """
        self._lambda = float(lambda_)

    def get_lambda(self) -> float:
        """
        :return: current lambda.
        """
        return self._lambda

    # C++ compatibility aliases
    setLambda = set_lambda
    getLambda = get_lambda

    def _update_eligibility_traces(
        self,
        current_state_action: StateAction[S, A],
        next_state_action: StateAction[S, A],
        reward: float,
        state_action_pair_value: StateActionPairContainer[S, A],
        step_size: float,
        discount_rate: float,
    ) -> None:
        """
        Iterate through all states and update each with respect to their
        eligibility Trace.

        :param current_state_action: Current state-action pair.
        :param next_state_action: Next state-action pair.
        :param reward: Reward for taking state-action pair.
        :param state_action_pair_value: A state-action pair container.
        :param step_size: step size for the update.
        :param discount_rate: discount rate for the update.
        """
        if hasattr(state_action_pair_value, "__getitem__"):
            next_val = state_action_pair_value[next_state_action]
            curr_val = state_action_pair_value[current_state_action]
        elif hasattr(state_action_pair_value, "getStateActionValue"):
            next_val = state_action_pair_value.getStateActionValue(next_state_action)
            curr_val = state_action_pair_value.getStateActionValue(current_state_action)
        elif hasattr(state_action_pair_value, "get_state_action_value"):
            next_val = state_action_pair_value.get_state_action_value(next_state_action)
            curr_val = state_action_pair_value.get_state_action_value(current_state_action)
        else:
            raise TypeError(
                "state_action_pair_value must support indexing or getStateActionValue"
            )

        td_error = reward + discount_rate * (
            reward + discount_rate * next_val - curr_val
        )

        self._eligibility_traces[current_state_action] = (
            self._eligibility_traces[current_state_action] + 1.0
        )

        iterator = (
            state_action_pair_value.items()
            if hasattr(state_action_pair_value, "items")
            else state_action_pair_value
        )

        for entry in iterator:
            if isinstance(entry, tuple) and len(entry) == 2:
                key, value = entry
            elif hasattr(entry, "first") and hasattr(entry, "second"):
                key, value = entry.first, entry.second
            else:
                key = entry
                value = (
                    state_action_pair_value[key]
                    if hasattr(state_action_pair_value, "__getitem__")
                    else state_action_pair_value.getStateActionValue(key)
                )

            if (
                StateAction is not None
                and hasattr(key, "getState")
                and hasattr(key, "getAction")
            ):
                sa = StateAction(key.getState(), key.getAction())
            elif (
                StateAction is not None
                and hasattr(key, "get_state")
                and hasattr(key, "get_action")
            ):
                sa = StateAction(key.get_state(), key.get_action())
            else:
                sa = key

            new_value = value + step_size * td_error * self._eligibility_traces[sa]

            if hasattr(state_action_pair_value, "setStateActionValue"):
                state_action_pair_value.setStateActionValue(sa, new_value)
            elif hasattr(state_action_pair_value, "set_state_action_value"):
                state_action_pair_value.set_state_action_value(sa, new_value)
            else:
                state_action_pair_value[sa] = new_value

            self._eligibility_traces[sa] = (
                discount_rate * self._lambda * self._eligibility_traces[sa]
            )

    # C++ compatibility alias
    _updateEligibilityTraces = _update_eligibility_traces