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

from typing import Generic, TypeVar

from .EpsilonGreedy import EpsilonGreedy
from .PolicyFactory import PolicyFactory

S = TypeVar("S")
A = TypeVar("A")


class EpsilonGreedyFactory(PolicyFactory[S, A], Generic[S, A]):
    def __init__(self, greediness: float) -> None:
        super().__init__()
        self._instance = EpsilonGreedy(greediness)