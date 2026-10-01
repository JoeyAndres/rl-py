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

import struct
from abc import ABC, abstractmethod
from typing import Any, Generic, List, Optional, Sequence, TypeVar, Union, overload

# Type aliases representing rl types
INT = int
BYTE = int
RET = TypeVar("RET")

SIZEOF_INT: int = 8
DEFAULT_MEMORY_SIZE: int = 4096

_rndseq: Optional[List[int]] = None


def hashUNH(
    coordinates: Sequence[int],
    num_indices: int,
    memory_size: int = DEFAULT_MEMORY_SIZE,
) -> int:
    """Hash function from University of New Hampshire CMAC program.

    Args:
        coordinates: Coordinates array (int64 values).
        num_indices: Number of indices to hash.
        memory_size: Size of hash table / memory range (default: 4096).

    Returns:
        Computed hash value in the range [0, memory_size - 1].
    """
    global _rndseq
    if _rndseq is None:
        _rndseq = []
        # Linear congruential generator equivalent to std::minstd_rand0
        state = 1
        for _ in range(2048):
            val = 0
            for _ in range(SIZEOF_INT):
                state = (state * 16807) % 2147483647
                val = (val << 8) | (state & 0xFF)
            if val >= 0x8000000000000000:
                val -= 0x10000000000000000
            _rndseq.append(val)

    coords_len = len(coordinates)
    sum_val = 0
    for i in range(num_indices):
        coord = coordinates[i] if i < coords_len else 0
        idx = (coord + 449 * i) % 2048
        if idx < 0:
            idx += 2048
        sum_val += _rndseq[idx]

    idx = sum_val % memory_size
    if idx < 0:
        idx += memory_size

    return idx


def _to_int64_coordinates(data: Union[bytes, bytearray, Sequence[int]]) -> List[int]:
    """Convert byte array or integer sequence to list of 64-bit integers."""
    if isinstance(data, (bytes, bytearray)):
        num_ints = len(data) // SIZEOF_INT
        if num_ints > 0:
            return list(struct.unpack(f"<{num_ints}q", data[: num_ints * SIZEOF_INT]))
        return [int(b) for b in data]
    return [int(x) for x in data]


class HashInterface(ABC, Generic[RET]):
    """Interface for all hash encapsulation.

    Equivalent to C++ template<class RET> class HashInterface.
    """

    @overload
    @abstractmethod
    def hash(self, byte_array: Sequence[int], length: int) -> RET:
        ...

    @overload
    @abstractmethod
    def hash(self, byte_array: Union[bytes, bytearray, Sequence[int]]) -> RET:
        ...

    @abstractmethod
    def hash(
        self,
        byte_array: Union[bytes, bytearray, Sequence[int]],
        length: Optional[int] = None,
        **kwargs: Any,
    ) -> RET:
        """Compute the hash of the byte array.

        Args:
            byte_array: Byte array or sequence of values to hash.
            length: Optional length of the array to hash.
            **kwargs: Additional keyword arguments (e.g. len).

        Returns:
            The computed hash value of type RET.
        """
        pass


class UNH(HashInterface[int]):
    """UNH implementation encapsulation.

    Inherits from HashInterface[rl::INT].
    """

    @overload
    def hash(self, byte_array: Sequence[int], length: int) -> int:
        ...

    @overload
    def hash(self, byte_array: Union[bytes, bytearray, Sequence[int]]) -> int:
        ...

    def hash(
        self,
        byte_array: Union[bytes, bytearray, Sequence[int]],
        length: Optional[int] = None,
        **kwargs: Any,
    ) -> int:
        """Compute UNH hash for the given byte array.

        Equivalent to C++:
        - INT UNH::hash(const rl::BYTE * const byteArray, size_t len)
        - INT UNH::hash(const vector<rl::BYTE>& byteArray)

        Args:
            byte_array: Byte array or sequence of values.
            length: Optional length of the array.
            **kwargs: Allows passing 'len' as keyword argument.

        Returns:
            Computed 64-bit integer hash value.
        """
        if length is None and "len" in kwargs:
            length = kwargs["len"]

        coordinates = _to_int64_coordinates(byte_array)
        num_indices = (length if length is not None else len(byte_array)) * SIZEOF_INT
        return hashUNH(coordinates, num_indices, DEFAULT_MEMORY_SIZE)


# Namespace simulation matching C++ namespace hierarchy
class rl:
    INT = INT
    BYTE = BYTE

    class hash:
        HashInterface = HashInterface
        UNH = UNH
        hashUNH = staticmethod(hashUNH)