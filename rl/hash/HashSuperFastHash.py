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

from abc import ABC, abstractmethod
from typing import Generic, Optional, Sequence, TypeVar, Union, overload

# Type aliases representing rl::INT and rl::BYTE
INT = int
BYTE = int

RET = TypeVar("RET")


class HashInterface(ABC, Generic[RET]):
    """Interface for all hash encapsulation."""

    @overload
    @abstractmethod
    def hash(self, byte_array: Union[bytes, bytearray, Sequence[int]]) -> RET:
        ...

    @overload
    @abstractmethod
    def hash(self, byte_array: Union[bytes, bytearray, Sequence[int]], length: int) -> RET:
        ...

    @abstractmethod
    def hash(
        self,
        byte_array: Union[bytes, bytearray, Sequence[int]],
        length: Optional[int] = None,
    ) -> RET:
        """Compute the hash of the given byte array.

        :param byte_array: Bytes or sequence of byte values to hash.
        :param length: Optional number of bytes to hash.
        :return: Hash value of type RET.
        """
        pass


def super_fast_hash(
    data: Optional[Union[bytes, bytearray, Sequence[int]]],
    length: Optional[int] = None,
) -> int:
    """SuperFastHash algorithm implementation by Paul Hsieh."""
    if data is None:
        return 0

    byte_data = bytes(data)
    if length is None:
        length = len(byte_data)

    if length <= 0:
        return 0

    data_bytes = byte_data[:length]
    data_len = len(data_bytes)
    hash_val = length & 0xFFFFFFFF
    rem = data_len & 3
    num_blocks = data_len >> 2

    def get16bits(d: bytes, offset: int) -> int:
        return d[offset] | (d[offset + 1] << 8)

    offset = 0
    for _ in range(num_blocks):
        hash_val = (hash_val + get16bits(data_bytes, offset)) & 0xFFFFFFFF
        tmp = ((get16bits(data_bytes, offset + 2) << 11) ^ hash_val) & 0xFFFFFFFF
        hash_val = ((hash_val << 16) ^ tmp) & 0xFFFFFFFF
        offset += 4
        hash_val = (hash_val + (hash_val >> 11)) & 0xFFFFFFFF

    if rem == 3:
        hash_val = (hash_val + get16bits(data_bytes, offset)) & 0xFFFFFFFF
        hash_val = (hash_val ^ (hash_val << 16)) & 0xFFFFFFFF
        byte_val = data_bytes[offset + 2]
        signed_byte = byte_val if byte_val < 128 else byte_val - 256
        hash_val = (hash_val ^ ((signed_byte << 18) & 0xFFFFFFFF)) & 0xFFFFFFFF
        hash_val = (hash_val + (hash_val >> 11)) & 0xFFFFFFFF
    elif rem == 2:
        hash_val = (hash_val + get16bits(data_bytes, offset)) & 0xFFFFFFFF
        hash_val = (hash_val ^ (hash_val << 11)) & 0xFFFFFFFF
        hash_val = (hash_val + (hash_val >> 17)) & 0xFFFFFFFF
    elif rem == 1:
        byte_val = data_bytes[offset]
        signed_byte = byte_val if byte_val < 128 else byte_val - 256
        hash_val = (hash_val + signed_byte) & 0xFFFFFFFF
        hash_val = (hash_val ^ (hash_val << 10)) & 0xFFFFFFFF
        hash_val = (hash_val + (hash_val >> 1)) & 0xFFFFFFFF

    # Force "avalanching" of final 127 bits
    hash_val = (hash_val ^ (hash_val << 3)) & 0xFFFFFFFF
    hash_val = (hash_val + (hash_val >> 5)) & 0xFFFFFFFF
    hash_val = (hash_val ^ (hash_val << 4)) & 0xFFFFFFFF
    hash_val = (hash_val + (hash_val >> 17)) & 0xFFFFFFFF
    hash_val = (hash_val ^ (hash_val << 25)) & 0xFFFFFFFF
    hash_val = (hash_val + (hash_val >> 6)) & 0xFFFFFFFF

    return hash_val


class SuperFastHash(HashInterface[INT]):
    """SuperFastHash implementation encapsulated."""

    @overload
    def hash(self, byte_array: Union[bytes, bytearray, Sequence[int]]) -> INT:
        ...

    @overload
    def hash(self, byte_array: Union[bytes, bytearray, Sequence[int]], length: int) -> INT:
        ...

    def hash(
        self,
        byte_array: Union[bytes, bytearray, Sequence[int]],
        length: Optional[int] = None,
    ) -> INT:
        """Compute SuperFastHash of the given byte array.

        :param byte_array: Byte array or vector of bytes.
        :param length: Optional number of bytes to hash.
        :return: 32-bit integer hash value.
        """
        if length is not None:
            return super_fast_hash(byte_array, length)
        return super_fast_hash(byte_array, len(byte_array))


# Namespace encapsulation mirroring rl::hash
class rl:
    class hash:
        HashInterface = HashInterface
        SuperFastHash = SuperFastHash


__all__ = ["BYTE", "INT", "HashInterface", "SuperFastHash", "super_fast_hash", "rl"]