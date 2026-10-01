from __future__ import annotations

import collections.abc
import struct
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

# Constants corresponding to declares.h
BYTE_BIT_SIZE: int = 8
FLOAT = float
FLOAT_SIZE: int = 8 * BYTE_BIT_SIZE  # 64 bits (sizeof(double) * 8)
UINT = int
INT = int
BYTE = int

floatVector = List[FLOAT]
floatArray = List[FLOAT]

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T")
KeyType = TypeVar("KeyType")
ValueType = TypeVar("ValueType")


class Converter:
    """
    Emulates the C++ union Converter { FLOAT f; uint64_t i; }.
    Converts between 64-bit IEEE-754 double precision float and 64-bit unsigned integer.
    """

    def __init__(self, f: float = 0.0, i: Optional[int] = None) -> None:
        self._f: float = 0.0
        self._i: int = 0
        if i is not None:
            self.i = i
        else:
            self.f = f

    @property
    def f(self) -> float:
        return self._f

    @f.setter
    def f(self, val: float) -> None:
        self._f = float(val)
        self._i = struct.unpack("=Q", struct.pack("=d", self._f))[0]

    @property
    def i(self) -> int:
        return self._i

    @i.setter
    def i(self, val: int) -> None:
        self._i = int(val) & 0xFFFFFFFFFFFFFFFF
        self._f = struct.unpack("=d", struct.pack("=Q", self._i))[0]


class bitset:
    """
    Python implementation of std::bitset.
    Stores a fixed-size sequence of N bits.
    """

    def __init__(self, size: int, value: Union[int, str] = 0) -> None:
        self._size: int = size
        self._bits: List[bool] = [False] * size
        if isinstance(value, str):
            for i, ch in enumerate(reversed(value)):
                if i < size:
                    self._bits[i] = ch == "1"
        elif isinstance(value, int):
            for i in range(min(size, value.bit_length())):
                self._bits[i] = bool((value >> i) & 1)

    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> bool:
        if not (0 <= index < self._size):
            raise IndexError(f"bitset index {index} out of range [0, {self._size})")
        return self._bits[index]

    def __setitem__(self, index: int, val: bool) -> None:
        if not (0 <= index < self._size):
            raise IndexError(f"bitset index {index} out of range [0, {self._size})")
        self._bits[index] = bool(val)

    def set(self, index: Optional[int] = None, val: bool = True) -> bitset:
        if index is None:
            self._bits = [True] * self._size
        else:
            self[index] = val
        return self

    def reset(self, index: Optional[int] = None) -> bitset:
        if index is None:
            self._bits = [False] * self._size
        else:
            self[index] = False
        return self

    def flip(self, index: Optional[int] = None) -> bitset:
        if index is None:
            self._bits = [not b for b in self._bits]
        else:
            self[index] = not self[index]
        return self

    def test(self, index: int) -> bool:
        return self[index]

    def any(self) -> bool:
        return any(self._bits)

    def none(self) -> bool:
        return not any(self._bits)

    def all(self) -> bool:
        return all(self._bits)

    def count(self) -> int:
        return sum(1 for b in self._bits if b)

    def to_string(self, zero: str = "0", one: str = "1") -> str:
        return "".join(
            one if self._bits[i] else zero for i in range(self._size - 1, -1, -1)
        )

    def to_ulong(self) -> int:
        res = 0
        for i in range(self._size):
            if self._bits[i]:
                res |= 1 << i
        return res

    def to_ullong(self) -> int:
        return self.to_ulong()

    def __invert__(self) -> bitset:
        res = bitset(self._size)
        res._bits = [not b for b in self._bits]
        return res

    def __and__(self, other: bitset) -> bitset:
        res = bitset(self._size)
        res._bits = [a and b for a, b in zip(self._bits, other._bits)]
        return res

    def __or__(self, other: bitset) -> bitset:
        res = bitset(self._size)
        res._bits = [a or b for a, b in zip(self._bits, other._bits)]
        return res

    def __xor__(self, other: bitset) -> bitset:
        res = bitset(self._size)
        res._bits = [a ^ b for a, b in zip(self._bits, other._bits)]
        return res

    def __eq__(self, other: object) -> bool:
        if isinstance(other, bitset):
            return self._size == other._size and self._bits == other._bits
        return False

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return f"bitset<{self._size}>('{self.to_string()}')"


BitSet = bitset


class MultiMap(Generic[KeyType, ValueType]):
    """
    Associative container that contains key-value pairs while permitting
    multiple entries with the same key, emulating std::multimap.
    """

    def __init__(
        self, items: Optional[Iterable[Tuple[KeyType, ValueType]]] = None
    ) -> None:
        self._items: List[Tuple[KeyType, ValueType]] = []
        if items is not None:
            for item in items:
                self.insert(item)

    def insert(
        self,
        key_or_pair: Union[Tuple[KeyType, ValueType], KeyType],
        value: Optional[ValueType] = None,
    ) -> None:
        if value is None and isinstance(key_or_pair, tuple) and len(key_or_pair) == 2:
            self._items.append((key_or_pair[0], key_or_pair[1]))
        else:
            self._items.append((key_or_pair, value))  # type: ignore[arg-type]

    def __getitem__(self, key: KeyType) -> List[ValueType]:
        return [v for k, v in self._items if k == key]

    def __iter__(self) -> Iterator[Tuple[KeyType, ValueType]]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: object) -> bool:
        return any(k == key for k, _ in self._items)

    def count(self, key: KeyType) -> int:
        return sum(1 for k, _ in self._items if k == key)

    def items(self) -> List[Tuple[KeyType, ValueType]]:
        return list(self._items)

    def keys(self) -> List[KeyType]:
        return [k for k, _ in self._items]

    def values(self) -> List[ValueType]:
        return [v for _, v in self._items]

    def to_dict(self) -> Dict[KeyType, List[ValueType]]:
        res: Dict[KeyType, List[ValueType]] = {}
        for k, v in self._items:
            res.setdefault(k, []).append(v)
        return res

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MultiMap):
            return self._items == other._items
        elif isinstance(other, list):
            return self._items == other
        return False

    def __repr__(self) -> str:
        return f"MultiMap({self._items!r})"


def flip_pair(p: Tuple[A, B]) -> Tuple[B, A]:
    """
    Flips a pair (A, B) to (B, A).

    Args:
        p: Pair of elements (first, second).

    Returns:
        Flipped pair (second, first).
    """
    return (p[1], p[0])


def flipMap(
    src: Union[Mapping[A, B], Iterable[Tuple[A, B]]]
) -> MultiMap[B, A]:
    """
    Flips an associative container of A, B pairs to B, A pairs.

    Args:
        src: An associative container (e.g. dict, MultiMap) or iterable of pairs.

    Returns:
        A MultiMap with flipped pairs (B, A).
    """
    dst: MultiMap[B, A] = MultiMap()
    if isinstance(src, collections.abc.Mapping):
        items: Iterable[Tuple[A, B]] = src.items()
    else:
        items = src

    for item in items:
        dst.insert(flip_pair((item[0], item[1])))

    return dst


flip_map = flipMap


def floatArrayToBitset(fv: Sequence[float]) -> bitset:
    """
    Converts an array/sequence of float to a bitset.

    Args:
        fv: Feature vector (sequence of floats).

    Returns:
        bitset representing the concatenated bit representation.
    """
    d_count = len(fv)
    rv = bitset(d_count * FLOAT_SIZE)

    for d in range(d_count):
        d_rev = (d_count - 1) - d
        converter = Converter(f=fv[d])
        temp_bitset = bitset(FLOAT_SIZE, converter.i)

        for b in range(temp_bitset.size()):
            rv[FLOAT_SIZE * d_rev + b] = temp_bitset[b]

    return rv


float_array_to_bitset = floatArrayToBitset


def concatArray(a1: Sequence[T], a2: Sequence[T]) -> List[T]:
    """
    Concatenates two arrays/sequences.

    Args:
        a1: First array with dimension D1.
        a2: Second array with dimension D2.

    Returns:
        Concatenated list of elements from a1 and a2 with dimension D1 + D2.
    """
    return list(a1) + list(a2)


concat_array = concatArray