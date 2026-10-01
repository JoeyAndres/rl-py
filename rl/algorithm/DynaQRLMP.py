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

from typing import Generic, Optional, Set, TypeVar

try:
    from .DynaQBase import DynaQBase
    from .ReinforcementLearning import ReinforcementLearning
    from ..agent.StateAction import StateAction
    from ..policy.Policy import Policy
except ImportError:
    try:
        from dyna_q_base import DynaQBase
        from reinforcement_learning import ReinforcementLearning
        from state_action import StateAction
        from policy import Policy
    except ImportError:
        S_bound = TypeVar("S_bound")
        A_bound = TypeVar("A_bound")

        class DynaQBase(Generic[S_bound, A_bound]):  # type: ignore[no-redef]
            def __init__(
                self,
                simulation_iteration_count: int,
                state_transition_greediness: float,
                state_transition_step_size: float,
            ) -> None:
                pass

            def back_up_state_action_pair(
                self,
                current_state_action: StateAction[S_bound, A_bound],
                reward: float,
                next_state_action_pair: StateAction[S_bound, A_bound],
            ) -> None:
                pass

            def arg_max(
                self, state: Optional[S_bound], action_set: Set[A_bound]
            ) -> Optional[A_bound]:
                pass

        class ReinforcementLearning(Generic[S_bound, A_bound]):  # type: ignore[no-redef]
            def __init__(
                self,
                step_size: float,
                discount_rate: float,
                policy: Optional[Policy[S_bound, A_bound]],
            ) -> None:
                pass

            def back_up_state_action_pair(
                self,
                current_state_action: StateAction[S_bound, A_bound],
                reward: float,
                next_state_action_pair: StateAction[S_bound, A_bound],
            ) -> None:
                pass

            def arg_max(
                self, state: Optional[S_bound], action_set: Set[A_bound]
            ) -> Optional[A_bound]:
                pass

        class StateAction(Generic[S_bound, A_bound]):  # type: ignore[no-redef]
            pass

        class Policy(Generic[S_bound, A_bound]):  # type: ignore[no-redef]
            pass


S = TypeVar("S")
A = TypeVar("A")


class DynaQRLMP(ReinforcementLearning[S, A], DynaQBase[S, A], Generic[S, A]):
    """
    Abstract class that represents the merge point for DynaQBase and
    Reinforcement Learning.

    This class aggregates ReinforcementLearning functions into DynaQBase.
    Note that although DynaQBase.argMax and DynaQBase.backUpStateActionPair
    are used, they both aggregate ReinforcementLearning.argMax and
    ReinforcementLearning.backUpStateActionPair.
    """

    def __init__(
        self,
        step_size: float,
        discount_rate: float,
        policy: Optional[Policy[S, A]],
        simulation_iteration_count: int,
        state_transition_greediness: float,
        state_transition_step_size: float,
        *args,
        **kwargs,
    ) -> None:
        """
        :param step_size: range [0.0, 1.0]. High step size means faster learning,
                          but less precise convergence.
        :param discount_rate: range [0.0, 1.0]. High discount rate means more
                              consideration of future events.
        :param policy: online policy used for action selection.
        :param simulation_iteration_count: number of simulations per update/backup.
        :param state_transition_greediness: greediness in selecting highest value model.
        :param state_transition_step_size: how fast a model updates the value of a
                                           state-action pair.
        """
        ReinforcementLearning.__init__(self, step_size, discount_rate, policy)
        DynaQBase.__init__(
            self,
            simulation_iteration_count,
            state_transition_greediness,
            state_transition_step_size,
        )

    def back_up_state_action_pair(
        self,
        current_state_action: StateAction[S, A],
        reward: float,
        next_state_action_pair: StateAction[S, A],
    ) -> None:
        """
        Does the main backup for all Temporal difference:
        Q[S, A] <- Q[S, A] + alpha * [R + gamma * max_{A'} Q[S', A'] - Q[S, A]]

        :param current_state_action: (S, A), current state-action pair.
        :param reward: R, reward after (S, A).
        :param next_state_action_pair: (S', A'), next state-action pair.
        """
        if hasattr(ReinforcementLearning, "back_up_state_action_pair"):
            ReinforcementLearning.back_up_state_action_pair(
                self, current_state_action, reward, next_state_action_pair
            )
        elif hasattr(ReinforcementLearning, "backUpStateActionPair"):
            ReinforcementLearning.backUpStateActionPair(
                self, current_state_action, reward, next_state_action_pair
            )

    def backUpStateActionPair(
        self,
        current_state_action: StateAction[S, A],
        reward: float,
        next_state_action_pair: StateAction[S, A],
    ) -> None:
        """C++ naming compatibility alias."""
        self.back_up_state_action_pair(
            current_state_action, reward, next_state_action_pair
        )

    def arg_max(
        self, state: Optional[S], action_set: Set[A]
    ) -> Optional[A]:
        """
        Returns the action that will most likely give the highest reward from
        the current state.

        :param state: the state to apply argMax to.
        :param action_set: a set of possible actions.
        :return: the action that will likely give the highest reward.
        """
        if hasattr(ReinforcementLearning, "arg_max"):
            return ReinforcementLearning.arg_max(self, state, action_set)
        if hasattr(ReinforcementLearning, "argMax"):
            return ReinforcementLearning.argMax(self, state, action_set)
        return None

    def argMax(
        self, state: Optional[S], action_set: Set[A]
    ) -> Optional[A]:
        """C++ naming compatibility alias."""
        return self.arg_max(state, action_set)