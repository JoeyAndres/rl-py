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
from .ReinforcementLearning import ReinforcementLearning

S = TypeVar("S")
A = TypeVar("A")


class QLearning(ReinforcementLearning[S, A], Generic[S, A]):
    """
    Basic off-policy reinforcement learning algorithm.

    Different policy is used for learning and action selection.
    Note that default offline/learning policy is EpsilonGreedy
    """

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

    def update(
        self,
        current_state_action: StateAction[S, A],
        next_state: S,
        reward: float,
        action_set: Set[A],
    ) -> None:
        # Note: this algorithm is in pg. 145 of Sutton Barto 2nd edition.
        # Q(S, A) <- Q(S, A) + α[ R + γ max a Q(S' , a) − Q(S, A)]
        next_action = self.get_learning_action(next_state, action_set)

        super().update_state_action(
            current_state_action,
            StateAction(next_state, next_action),
            reward,
        )

        self.back_up_state_action_pair(
            current_state_action,
            reward,
            StateAction(next_state, next_action),
        )