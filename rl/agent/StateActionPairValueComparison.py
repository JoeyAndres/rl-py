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

from typing import Generic, Tuple, TypeVar

from .StateAction import StateAction

S = TypeVar("S")
A = TypeVar("A")


class StateActionPairValueComparison(Generic[S, A]):
    """
    Comparison object for std::map and other containers in need of comparison.

    Created as a comparison object for StateAction[S, A] and their corresponding
    Value/Priority. This comparison object is mainly used in std::priority_queue.
    """

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        lhs: Tuple[StateAction[S, A], float],
        rhs: Tuple[StateAction[S, A], float],
    ) -> bool:
        if lhs[0] < rhs[0]:
            return True
        elif lhs[0] > rhs[0]:
            return False

        if lhs[1] < rhs[1]:
            return True
        return False