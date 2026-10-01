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

from typing import Generic, Set, TypeVar

from ..utility.FactoryAbstract import FactoryAbstract
from .ActionContainer import ActionContainer

A = TypeVar("A")

spActionSet = Set[A]
spActionContainer = ActionContainer[A]


class ActionContainerFactory(FactoryAbstract[ActionContainer[A]], Generic[A]):
    """Factory object for ActionContainer.

    See ActionContainer.
    """

    def __init__(self, action_set: Set[A]) -> None:
        """Factory creator method.

        :param action_set: Initial set of actions.
        """
        super().__init__()
        self._instance: ActionContainer[A] = ActionContainer(action_set)