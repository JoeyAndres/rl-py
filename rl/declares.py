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

from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

BYTE_BIT_SIZE: int = 8

FLOAT = float
FLOAT_SIZE: int = 8 * BYTE_BIT_SIZE

UINT = int
INT = int
BYTE = int

DEFAULT_GREEDINESS_LEARNING_POLICY: FLOAT = 1.0

# Hashing constants.
MURMUR_HASH_SEED: UINT = 0x666

# TODO(jandres): Add this to some factory generated module that sets settings.
MAX_EPISODES: UINT = 100000

FEATURE_VECTOR = List[UINT]

floatVector = List[FLOAT]

N = TypeVar("N")


class floatArray(list, Generic[N]):
    """An array of float."""
    pass


spFloatVector = List[FLOAT]


class spFloatArray(list, Generic[N]):
    """floatArray wrapped in shared_ptr."""
    pass


S = TypeVar("S")
A = TypeVar("A")
X = TypeVar("X")

StateAndReward = Tuple[S, FLOAT]


class spState(Generic[S]):
    """Wraps states in shared_ptr."""

    def __init__(self, value: Optional[S] = None) -> None:
        self._value: Optional[S] = value

    @property
    def value(self) -> Optional[S]:
        return self._value

    @value.setter
    def value(self, val: Optional[S]) -> None:
        self._value = val

    def get(self) -> Optional[S]:
        return self._value

    def __getattr__(self, name: str) -> Any:
        if self._value is not None:
            return getattr(self._value, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __lt__(self, other: Any) -> bool:
        other_val = other._value if isinstance(other, spState) else other
        return self._value < other_val

    def __le__(self, other: Any) -> bool:
        other_val = other._value if isinstance(other, spState) else other
        return self._value <= other_val

    def __gt__(self, other: Any) -> bool:
        other_val = other._value if isinstance(other, spState) else other
        return self._value > other_val

    def __ge__(self, other: Any) -> bool:
        other_val = other._value if isinstance(other, spState) else other
        return self._value >= other_val

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, spState):
            return self._value == other._value
        return self._value == other

    def __ne__(self, other: Any) -> bool:
        return not (self == other)

    def __hash__(self) -> int:
        return hash(self._value)

    def __bool__(self) -> bool:
        return self._value is not None

    def __repr__(self) -> str:
        return f"spState({self._value!r})"


class spStateComp(Generic[S]):
    """Comparison for spState should be by the value they dereference,
    not their pointer address value."""

    def __call__(self, x: Union[spState[S], S], y: Union[spState[S], S]) -> bool:
        val_x = x._value if isinstance(x, spState) else x
        val_y = y._value if isinstance(y, spState) else y
        return val_x < val_y


class spAction(Generic[A]):
    """Wraps action in shared_ptr."""

    def __init__(self, value: Optional[A] = None) -> None:
        self._value: Optional[A] = value

    @property
    def value(self) -> Optional[A]:
        return self._value

    @value.setter
    def value(self, val: Optional[A]) -> None:
        self._value = val

    def get(self) -> Optional[A]:
        return self._value

    def __getattr__(self, name: str) -> Any:
        if self._value is not None:
            return getattr(self._value, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __lt__(self, other: Any) -> bool:
        other_val = other._value if isinstance(other, spAction) else other
        return self._value < other_val

    def __le__(self, other: Any) -> bool:
        other_val = other._value if isinstance(other, spAction) else other
        return self._value <= other_val

    def __gt__(self, other: Any) -> bool:
        other_val = other._value if isinstance(other, spAction) else other
        return self._value > other_val

    def __ge__(self, other: Any) -> bool:
        other_val = other._value if isinstance(other, spAction) else other
        return self._value >= other_val

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, spAction):
            return self._value == other._value
        return self._value == other

    def __ne__(self, other: Any) -> bool:
        return not (self == other)

    def __hash__(self) -> int:
        return hash(self._value)

    def __bool__(self) -> bool:
        return self._value is not None

    def __repr__(self) -> str:
        return f"spAction({self._value!r})"


class spActionComp(Generic[A]):
    """Comparison for spAction should be by the value they dereference,
    not their pointer address value."""

    def __call__(self, x: Union[spAction[A], A], y: Union[spAction[A], A]) -> bool:
        val_x = x._value if isinstance(x, spAction) else x
        val_y = y._value if isinstance(y, spAction) else y
        return val_x < val_y


spStateAndReward = Tuple[spState[S], FLOAT]
spStateXMap = Dict[spState[S], X]
spStateSet = Set[spState[S]]
spActionSet = Set[spAction[A]]
spActionValueMap = Dict[spAction[A], FLOAT]

SEGMENT_SIZE: int = 100