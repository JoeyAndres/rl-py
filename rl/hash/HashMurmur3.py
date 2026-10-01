"""rl - Reinforcement Learning
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
from typing import Generic, List, Optional, Sequence, TypeVar, Union

MURMUR_HASH_SEED: int = 0x666
_MASK64: int = 0xFFFFFFFFFFFFFFFF


def _rotl64(x: int, r: int) -> int:
    return ((x << r) & _MASK64) | (x >> (64 - r))


def _fmix64(k: int) -> int:
    k = (k ^ (k >> 33)) & _MASK64
    k = (k * 0xFF51AFD7ED558CCD) & _MASK64
    k = (k ^ (k >> 33)) & _MASK64
    k = (k * 0xC4CEB9FE1A85EC53) & _MASK64
    k = (k ^ (k >> 33)) & _MASK64
    return k


class HashMurmur3Out:
    """Encapsulates the output of the murmur hash.

    uint64_t hashVal[2] is the array being encapsulated since output of
    Murmur3 hash is 128 bit.
    """

    def __init__(self, val0: int = 0, val1: int = 0) -> None:
        self.hashVal: List[int] = [val0 & _MASK64, val1 & _MASK64]

    def __getitem__(self, index: int) -> int:
        return self.hashVal[index]

    def __setitem__(self, index: int, value: int) -> None:
        self.hashVal[index] = value & _MASK64

    def __eq__(self, other: object) -> bool:
        if isinstance(other, HashMurmur3Out):
            return self.hashVal == other.hashVal
        return False

    def __repr__(self) -> str:
        return f"HashMurmur3Out(hashVal={self.hashVal})"


def MurmurHash3_x64_128(
    key: Union[bytes, bytearray, Sequence[int]],
    length: int,
    seed: int,
    out: HashMurmur3Out,
) -> None:
    """MurmurHash3 128-bit hash calculation for x64 architecture."""
    if isinstance(key, (bytes, bytearray)):
        data = bytes(key[:length])
    else:
        data = bytes(key[:length])

    nblocks = length // 16

    h1 = seed & _MASK64
    h2 = seed & _MASK64

    c1 = 0x87C37B91114253D5
    c2 = 0x4CF5AD432745937F

    # body
    for i in range(nblocks):
        offset = i * 16
        k1 = int.from_bytes(data[offset : offset + 8], "little")
        k2 = int.from_bytes(data[offset + 8 : offset + 16], "little")

        k1 = (k1 * c1) & _MASK64
        k1 = _rotl64(k1, 31)
        k1 = (k1 * c2) & _MASK64
        h1 ^= k1

        h1 = _rotl64(h1, 27)
        h1 = (h1 + h2) & _MASK64
        h1 = (h1 * 5 + 0x52DCE729) & _MASK64

        k2 = (k2 * c2) & _MASK64
        k2 = _rotl64(k2, 33)
        k2 = (k2 * c1) & _MASK64
        h2 ^= k2

        h2 = _rotl64(h2, 31)
        h2 = (h2 + h1) & _MASK64
        h2 = (h2 * 5 + 0x38495AB5) & _MASK64

    # tail
    tail = data[nblocks * 16 : length]
    tail_len = len(tail)

    k1 = 0
    k2 = 0

    if tail_len >= 15:
        k2 ^= tail[14] << 48
    if tail_len >= 14:
        k2 ^= tail[13] << 40
    if tail_len >= 13:
        k2 ^= tail[12] << 32
    if tail_len >= 12:
        k2 ^= tail[11] << 24
    if tail_len >= 11:
        k2 ^= tail[10] << 16
    if tail_len >= 10:
        k2 ^= tail[9] << 8
    if tail_len >= 9:
        k2 ^= tail[8]
        k2 = (k2 * c2) & _MASK64
        k2 = _rotl64(k2, 33)
        k2 = (k2 * c1) & _MASK64
        h2 ^= k2

    if tail_len >= 8:
        k1 ^= tail[7] << 56
    if tail_len >= 7:
        k1 ^= tail[6] << 48
    if tail_len >= 6:
        k1 ^= tail[5] << 40
    if tail_len >= 5:
        k1 ^= tail[4] << 32
    if tail_len >= 4:
        k1 ^= tail[3] << 24
    if tail_len >= 3:
        k1 ^= tail[2] << 16
    if tail_len >= 2:
        k1 ^= tail[1] << 8
    if tail_len >= 1:
        k1 ^= tail[0]
        k1 = (k1 * c1) & _MASK64
        k1 = _rotl64(k1, 31)
        k1 = (k1 * c2) & _MASK64
        h1 ^= k1

    # finalization
    h1 ^= length
    h2 ^= length

    h1 = (h1 + h2) & _MASK64
    h2 = (h2 + h1) & _MASK64

    h1 = _fmix64(h1)
    h2 = _fmix64(h2)

    h1 = (h1 + h2) & _MASK64
    h2 = (h2 + h1) & _MASK64

    out.hashVal[0] = h1
    out.hashVal[1] = h2


T = TypeVar("T")


class HashInterface(ABC, Generic[T]):
    """Interface for all hash encapsulation."""

    @abstractmethod
    def hash(
        self,
        byte_array: Union[bytes, bytearray, Sequence[int]],
        length: Optional[int] = None,
    ) -> T:
        """Calculate hash of byte_array."""
        pass


class Murmur3(HashInterface[HashMurmur3Out]):
    """Murmur3 hash encapsulated."""

    def __init__(self, seed: int = MURMUR_HASH_SEED) -> None:
        self._seed: int = seed

    def hash(
        self,
        byte_array: Union[bytes, bytearray, Sequence[int]],
        length: Optional[int] = None,
    ) -> HashMurmur3Out:
        """Hash the given byte sequence or byte vector."""
        if length is None:
            length = len(byte_array)
        out = HashMurmur3Out()
        MurmurHash3_x64_128(byte_array, length, self._seed, out)
        return out