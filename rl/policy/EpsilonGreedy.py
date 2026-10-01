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

import random
from abc import ABC, abstractmethod
from typing import (
    Collection,
    Generic,
    Iterable,
    Mapping,
    Optional,
    TypeVar,
    Union,
    cast,
    overload,
)

FLOAT = float

S = TypeVar("S")
A = TypeVar("A")

_SENTINEL = object()


from .Policy import Policy


spPolicy = Policy


class EpsilonGreedy(Policy[S, A], Generic[S, A]):
    """Selects greedy action with probability greediness.

    Let 0 <= greediness < 1, which is the probability of selecting optimal
    action. During the selection of non-optimal, they are selected in equiprobable
    manner.
    """

    def __init__(self, greediness: float = 1.0) -> None:
        """Initialize EpsilonGreedy.

        :param greediness: probability of selecting greedy action.
        """
        super().__init__()
        self._greediness: float = float(greediness)
        self._distribution: tuple[float, float] = (0.0, 1.0)

    def argMax(
        self,
        actionValues: Mapping[A, float],
        actionSet: Iterable[A],
    ) -> A:
        """Returns the action that will likely give the highest reward from the current state.

        :param actionValues: map of action to value.
        :param actionSet: a set/collection of possible actions.
        :return: the action that will likely give the highest reward.
        """
        iterator = iter(actionSet)
        try:
            max_action = next(iterator)
        except StopIteration:
            raise ValueError("actionSet must not be empty.")

        max_val = actionValues[max_action]
        for action in iterator:
            val = actionValues[action]
            if val > max_val:
                max_action = action
                max_val = val

        return max_action

    def arg_max(
        self,
        action_values: Mapping[A, float],
        action_set: Iterable[A],
    ) -> A:
        return self.argMax(action_values, action_set)

    @overload
    def getAction(
        self,
        actionValues: Mapping[A, float],
        actionSet: Collection[A],
    ) -> A:
        ...

    @overload
    def getAction(
        self,
        actionValues: Mapping[A, float],
        actionSet: Collection[A],
        maxAction: A,
    ) -> A:
        ...

    def getAction(
        self,
        actionValues: Mapping[A, float],
        actionSet: Collection[A],
        maxAction: Union[A, object] = _SENTINEL,
    ) -> A:
        """Selects an action according to epsilon-greedy policy.

        :param actionValues: a mapping of actions to their corresponding value.
        :param actionSet: collection of possible actions.
        :param maxAction: optional precomputed action with maximum value.
        :return: the selected action.
        """
        if not actionSet:
            raise ValueError("actionSet must not be empty.")

        if maxAction is _SENTINEL:
            if self._greediness == 1.0:
                return self.argMax(actionValues, actionSet)
            r = random.random()
            if r > self._greediness:
                actions = list(actionSet)
                random_index = random.randint(0, len(actions) - 1)
                return actions[random_index]
            else:
                return self.argMax(actionValues, actionSet)
        else:
            resolved_max_action = cast(A, maxAction)
            if self._greediness == 1.0:
                return resolved_max_action
            r = random.random()
            if r > self._greediness:
                actions = list(actionSet)
                random_index = random.randint(0, len(actions) - 1)
                return actions[random_index]
            else:
                return resolved_max_action

    def get_action(
        self,
        action_values: Mapping[A, float],
        action_set: Collection[A],
        max_action: Union[A, object] = _SENTINEL,
    ) -> A:
        return self.getAction(action_values, action_set, max_action)

    def setGreediness(self, greediness: float) -> None:
        """Set greediness probability."""
        self._greediness = float(greediness)

    def set_greediness(self, greediness: float) -> None:
        self.setGreediness(greediness)

    def getGreediness(self) -> float:
        """Return current greediness."""
        return self._greediness

    def get_greediness(self) -> float:
        return self.getGreediness()

    @property
    def greediness(self) -> float:
        """Probability of selecting a greedy action."""
        return self._greediness

    @greediness.setter
    def greediness(self, greediness: float) -> None:
        self._greediness = float(greediness)


spEpsilonGreedy = EpsilonGreedy

__all__ = [
    "FLOAT",
    "Policy",
    "spPolicy",
    "EpsilonGreedy",
    "spEpsilonGreedy",
]