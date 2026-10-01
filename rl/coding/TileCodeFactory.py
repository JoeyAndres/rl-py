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

from typing import ClassVar, Optional, Sequence, Type

from ..declares import FLOAT
from ..utility.FactoryAbstract import FactoryAbstract
from .DimensionInfo import DimensionInfo
from .TileCode import TileCode


class TileCodeFactory(FactoryAbstract[TileCode]):
    """Factory method for TileCode. Abstract: subclasses set TILE_CODE_CLASS."""

    TILE_CODE_CLASS: ClassVar[Optional[Type[TileCode]]] = None

    def __init__(
        self,
        dimensionalInfos: Sequence[DimensionInfo[FLOAT]],
        numTilings: int,
        weightCont=None,
    ) -> None:
        super().__init__()
        self._instance = self._tileCodeClass()(
            dimensionalInfos, numTilings, weightCont=weightCont
        )

    @classmethod
    def _tileCodeClass(cls) -> Type[TileCode]:
        if cls.TILE_CODE_CLASS is None:
            raise TypeError(f"{cls.__name__} must define TILE_CODE_CLASS.")
        return cls.TILE_CODE_CLASS
