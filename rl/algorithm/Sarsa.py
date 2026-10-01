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

from typing import Any, Generic, Optional, Set, TypeVar

try:
    from rl.agent.state_action import StateAction
    from rl.policy.policy import Policy
    from rl.algorithm.reinforcement_learning import ReinforcementLearning
    from rl.agent.state_action_pair_container import StateActionPairContainerSimple
except ImportError:
    try:
        from ..agent.state_action import StateAction
        from ..policy.policy import Policy
        from .reinforcement_learning import ReinforcementLearning
        from ..agent.state_action_pair_container import StateActionPairContainerSimple
    except ImportError:
        from agent.StateAction import StateAction  # type: ignore
        from policy.Policy import Policy  # type: ignore
        from algorithm.ReinforcementLearning import ReinforcementLearning  # type: ignore
        StateActionPairContainerSimple = Any  # type: ignore

S = TypeVar("S")
A = TypeVar("A")
try:
    SAP_CONTAINER = TypeVar("SAP_CONTAINER", default=StateActionPairContainerSimple)
except TypeError:
    SAP_CONTAINER = TypeVar("SAP_CONTAINER")


class Sarsa(ReinforcementLearning[S, A, SAP_CONTAINER], Generic[S, A, SAP_CONTAINER]):
    """Basic on-policy reinforcement learning algorithm.

    The same policy is used for both learning and taking action given state.
    """

    SA = StateAction

    def __init__(
        self,
        step_size: float,
        discount_rate: float,
        policy: Policy[S, A],
    ) -> None:
        """
        :param step_size: range [0.0, 1.0]. High step size means faster learning, but
            less precise convergence.
        :param discount_rate: range [0.0, 1.0]. High discount rate means more
            consideration of future events.
        :param policy: online policy, that is policy used for action selection.
        """
        super().__init__(step_size, discount_rate, policy)
        if hasattr(self, "set_learning_policy"):
            self.set_learning_policy(policy)
        else:
            self.setLearningPolicy(policy)

    # TODO(jandres): Make setLearningPolicy and setPolicy the same.
    def update(
        self,
        current_state_action: StateAction[S, A],
        next_state: S,
        reward: float,
        action_set: Set[A],
    ) -> None:
        if hasattr(self, "get_action"):
            next_action = self.get_action(next_state, action_set)
        else:
            next_action = self.getAction(next_state, action_set)

        next_state_action = StateAction(next_state, next_action)

        if hasattr(super(), "update_state_action"):
            super().update_state_action(
                current_state_action,
                next_state_action,
                reward,
            )
        elif hasattr(super(), "updateStateAction"):
            super().updateStateAction(
                current_state_action,
                next_state_action,
                reward,
            )
        elif hasattr(self, "update_state_action"):
            self.update_state_action(
                current_state_action,
                next_state_action,
                reward,
            )
        else:
            self.updateStateAction(
                current_state_action,
                next_state_action,
                reward,
            )

        if hasattr(self, "back_up_state_action_pair"):
            self.back_up_state_action_pair(
                current_state_action,
                reward,
                next_state_action,
            )
        else:
            self.backUpStateActionPair(
                current_state_action,
                reward,
                next_state_action,
            )


spSarsa = Sarsa
