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

from typing import Generic, Optional, Set, TypeVar

A = TypeVar("A")


class ActionContainer(Generic[A]):
    """Utility class for storing Action.

    Attributes:
        _actionData (Set[A]): Storage for action elements.
    """

    def __init__(self, actionSet: Optional[Set[A]] = None) -> None:
        """Create an ActionContainer.

        Args:
            actionSet: Initial set of actions. Defaults to an empty set.
        """
        if actionSet is None:
            self._actionData: Set[A] = set()
        else:
            self._actionData: Set[A] = set(actionSet)

    def getActionSet(self) -> Set[A]:
        """Return the set of actions."""
        return self._actionData

    def addAction(self, data: A) -> None:
        """Add an action to the container.

        Args:
            data: Action to be added.
        """
        self._actionData.add(data)

    def setActionSet(self, dataSet: Set[A]) -> None:
        """Replace the action set with a new one.

        Args:
            dataSet: New set of actions.
        """
        self._actionData = set(dataSet)

    # Pythonic aliases
    get_action_set = getActionSet
    add_action = addAction
    set_action_set = setActionSet


# Type alias corresponding to spActionContainer<A>
spActionContainer = ActionContainer