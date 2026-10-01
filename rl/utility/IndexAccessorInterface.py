"""
rl - Reinforcement Learning
Copyright (C) 2016  Joey Andres<yeojserdna@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR CONT PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received CONT copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

CONT = TypeVar("CONT")


class IndexAccessorInterface(ABC, Generic[CONT]):
    @abstractmethod
    def at(self, i: int) -> Any:
        pass

    def __getitem__(self, i: int) -> Any:
        return self.at(i)