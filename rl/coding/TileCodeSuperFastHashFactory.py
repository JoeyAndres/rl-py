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

from typing import Any, Generic, List, Optional, Sequence, TypeVar, overload

from ..declares import FLOAT
from .DimensionInfo import DimensionInfo
from .TileCode import DEFAULT_TILE_CONT
from .TileCodeHashedFactory import TileCodeHashedFactory
from .TileCodeSuperFastHash import TileCodeSuperFastHash

D = TypeVar("D")
NUM_TILINGS = TypeVar("NUM_TILINGS")
WEIGHT_CONT = TypeVar("WEIGHT_CONT")


class TileCodeSuperFastHashFactory(
    TileCodeHashedFactory[D, NUM_TILINGS, WEIGHT_CONT, TileCodeSuperFastHash],
    Generic[D, NUM_TILINGS, WEIGHT_CONT],
):
    """
    Factory method for TileCodeSuperFastHash.

    :tparam D: Number of dimension.
    :tparam NUM_TILINGS: Number of tilings.
    :tparam WEIGHT_CONT: The container object to store the weights.
    """

    tile_code_class = TileCodeSuperFastHash

    @overload
    def __init__(
        self,
        dimensional_infos: Sequence[DimensionInfo[FLOAT]],
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        dimensional_infos: Sequence[DimensionInfo[FLOAT]],
        size_hint: int,
    ) -> None:
        ...

    def __init__(
        self,
        dimensional_infos: Sequence[DimensionInfo[FLOAT]],
        size_hint: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if size_hint is not None:
            super().__init__(dimensional_infos, size_hint, *args, **kwargs)
        else:
            super().__init__(dimensional_infos, *args, **kwargs)