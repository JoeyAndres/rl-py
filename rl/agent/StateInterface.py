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

from typing import Any, Generic, TypeVar

S = TypeVar("S")


class StateInterface(Generic[S]):
    """
    An interface class for States.
    """

    def __init__(self, value: S) -> None:
        self._value: S = value

    def __lt__(self, rhs: Any) -> bool:
        if isinstance(rhs, StateInterface):
            return self._value < rhs._value
        return NotImplemented

    def __gt__(self, rhs: Any) -> bool:
        if isinstance(rhs, StateInterface):
            return self._value > rhs._value
        return NotImplemented

    def __le__(self, rhs: Any) -> bool:
        if isinstance(rhs, StateInterface):
            return self._value <= rhs._value
        return NotImplemented

    def __ge__(self, rhs: Any) -> bool:
        if isinstance(rhs, StateInterface):
            return self._value >= rhs._value
        return NotImplemented

    def __eq__(self, rhs: Any) -> bool:
        if isinstance(rhs, StateInterface):
            return self._value == rhs._value
        return NotImplemented

    def __ne__(self, rhs: Any) -> bool:
        if isinstance(rhs, StateInterface):
            return self._value != rhs._value
        return NotImplemented

    def assign(self, rhs: StateInterface[S]) -> StateInterface[S]:
        if self is rhs:
            return self
        self._value = rhs._value
        return self

    def getValue(self) -> S:
        return self._value

    def get_value(self) -> S:
        return self._value

    def __hash__(self) -> int:
        return hash(self._value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._value!r})"