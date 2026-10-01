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

from abc import ABC, abstractmethod
from typing import Generic, Mapping, Optional, Set, TypeVar, overload

S = TypeVar("S")
A = TypeVar("A")

# Type aliases corresponding to rl declares
spAction = A
spActionSet = Set[A]
spActionValueMap = Mapping[A, float]


class Policy(ABC, Generic[S, A]):
    """
    Base class for all Policy.

    Policy sets the rule for action selection given a mapping of action to
    action values and a set of action.

    :tparam S: State data type.
    :tparam A: Action data type.
    """

    @overload
    @abstractmethod
    def getAction(
        self,
        actionValues: Mapping[A, float],
        actionSet: Set[A],
    ) -> A:
        ...

    @overload
    @abstractmethod
    def getAction(
        self,
        actionValues: Mapping[A, float],
        actionSet: Set[A],
        maxAction: A,
    ) -> A:
        ...

    @abstractmethod
    def getAction(
        self,
        actionValues: Mapping[A, float],
        actionSet: Set[A],
        maxAction: Optional[A] = None,
    ) -> A:
        """
        Returns action given a mapping of actions and their value and a
        set of actions.

        :param actionValues: A mapping of actions to their corresponding value.
        :param actionSet: Set of actions.
        :param maxAction: Action with maximum value.
        :return: Action given a mapping of actions and their value and a
                 set of actions.
        """
        raise NotImplementedError

    def get_action(
        self,
        action_values: Mapping[A, float],
        action_set: Set[A],
        max_action: Optional[A] = None,
    ) -> A:
        """
        Returns action given a mapping of actions and their value and a
        set of actions. Pythonic alias for getAction.
        """
        return self.getAction(action_values, action_set, max_action)


# Policy wrapped in smart pointer (in Python, reference-counted object)
spPolicy = Policy