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

from __future__ import annotations

from typing import Any, Generic, Optional, Sequence, Type, TypeVar

try:
    from .TileCodeFactory import TileCodeFactory
except (ImportError, ValueError):
    try:
        from TileCodeFactory import TileCodeFactory
    except ImportError:
        TileCodeFactory = Generic  # type: ignore

try:
    from .DimensionInfo import DimensionInfo
except (ImportError, ValueError):
    try:
        from DimensionInfo import DimensionInfo
    except ImportError:
        try:
            from .TileCode import DimensionInfo
        except (ImportError, ValueError):
            try:
                from TileCode import DimensionInfo
            except ImportError:
                DimensionInfo = Any  # type: ignore

try:
    from .TileCode import TileCode
except (ImportError, ValueError):
    try:
        from TileCode import TileCode
    except ImportError:
        TileCode = Any  # type: ignore

FLOAT = float

D = TypeVar("D")
NUM_TILINGS = TypeVar("NUM_TILINGS")
WEIGHT_CONT = TypeVar("WEIGHT_CONT")
TILE_CODE_CLASS = TypeVar("TILE_CODE_CLASS")


class TileCodeHashedFactory(
    TileCodeFactory[D, NUM_TILINGS, WEIGHT_CONT, TILE_CODE_CLASS],
    Generic[D, NUM_TILINGS, WEIGHT_CONT, TILE_CODE_CLASS],
):
    """Factory method for TileCode that utilizes hashing. Abstract class.

    :param D: Number of dimensions.
    :param NUM_TILINGS: Number of tilings.
    :param WEIGHT_CONT: The container object to store the weights.
    :param TILE_CODE_CLASS: The TileCode child that utilizes hashing.
    """

    tile_code_class: Optional[Type[Any]] = None

    def __init__(
        self,
        dimensional_infos: Optional[Sequence[DimensionInfo[FLOAT]]] = None,
        size_hint: Optional[int] = None,
        tile_code_class: Optional[Type[TILE_CODE_CLASS]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize TileCodeHashedFactory.

        :param dimensional_infos: Sequence of DimensionInfo.
        :param size_hint: Size hint for hashing.
        :param tile_code_class: The TileCode child class that utilizes hashing.
        """
        if dimensional_infos is None and args:
            dimensional_infos = args[0]
            if len(args) > 1 and size_hint is None:
                size_hint = args[1]
            if len(args) > 2 and tile_code_class is None:
                tile_code_class = args[2]

        if dimensional_infos is None:
            dimensional_infos = kwargs.pop("dimensionalInfos", None) or kwargs.pop("dimensional_infos", None)
        if size_hint is None:
            size_hint = kwargs.pop("sizeHint", None) or kwargs.pop("size_hint", None)
        if tile_code_class is None:
            tile_code_class = kwargs.pop("tileCodeClass", None) or kwargs.pop("tile_code_class", None)

        self._dimensional_infos = dimensional_infos
        self._size_hint = size_hint

        cls = (
            tile_code_class
            or getattr(self, "tile_code_class", None)
            or getattr(self, "_tile_code_class", None)
        )
        if cls is None:
            for base in getattr(self.__class__, "__orig_bases__", []):
                base_args = getattr(base, "__args__", None)
                if base_args and len(base_args) >= 4 and isinstance(base_args[3], type):
                    cls = base_args[3]
                    break

        if cls is not None and dimensional_infos is not None and size_hint is not None:
            self._instance: Optional[TileCode] = cls(dimensional_infos, size_hint)
        else:
            self._instance = None

    def get(self) -> Optional[Any]:
        """Return the created TileCode instance."""
        return self._instance


__all__ = ["TileCodeHashedFactory"]