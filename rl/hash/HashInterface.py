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

from abc import ABC, abstractmethod
from typing import Generic, Optional, Sequence, TypeVar, Union, overload

BYTE = int
RET = TypeVar("RET")


class HashInterface(ABC, Generic[RET]):
    """Interface for all hash encapsulation."""

    @overload
    @abstractmethod
    def hash(self, byte_array: Sequence[BYTE], len: int) -> RET:
        ...

    @overload
    @abstractmethod
    def hash(self, byte_array: Union[bytes, bytearray, Sequence[BYTE]]) -> RET:
        ...

    @abstractmethod
    def hash(
        self,
        byte_array: Union[bytes, bytearray, Sequence[BYTE]],
        len: Optional[int] = None,
    ) -> RET:
        """
        Compute hash of the given byte array.

        :param byte_array: The byte array to hash.
        :param len: The length of the byte array.
        :return: Computed hash value.
        """
        pass