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

from abc import ABC
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class FactoryAbstract(Generic[T], ABC):
    """The template for all factory method classes.

    Type parameter T represents the type created by this factory method.
    """

    def __init__(self) -> None:
        if type(self) is FactoryAbstract:
            raise TypeError("FactoryAbstract cannot be instantiated directly.")
        self._instance: Optional[T] = None

    def get(self) -> Optional[T]:
        return self._instance