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

from typing import Generic, Optional, Sequence, Type, TypeVar

from ..declares import FLOAT
from ..utility.FactoryAbstract import FactoryAbstract
from .DimensionInfo import DimensionInfo
from .TileCode import TileCode

D = TypeVar("D")
NUM_TILINGS = TypeVar("NUM_TILINGS")
WEIGHT_CONT = TypeVar("WEIGHT_CONT")
TILE_CODE_CLASS = TypeVar("TILE_CODE_CLASS", bound=TileCode)


class TileCodeFactory(
    FactoryAbstract[TileCode[D, NUM_TILINGS, WEIGHT_CONT]],
    Generic[D, NUM_TILINGS, WEIGHT_CONT, TILE_CODE_CLASS],
):
    """Factory method for TileCode. Abstract class.

    :tparam D: Number of dimension.
    :tparam NUM_TILINGS: Number of tilings.
    :tparam WEIGHT_CONT: The container object to store the weights.
    :tparam TILE_CODE_CLASS: TileCode child that don't utilizes hashing.
    """

    _tile_code_class: Optional[Type[TILE_CODE_CLASS]] = None

    def __init__(
        self,
        dimensional_infos: Sequence[DimensionInfo[FLOAT]],
        tile_code_class: Optional[Type[TILE_CODE_CLASS]] = None,
    ) -> None:
        super().__init__()
        cls = tile_code_class or self._tile_code_class
        if cls is None:
            raise TypeError(
                "TileCodeFactory is an abstract factory; "
                "TILE_CODE_CLASS must be specified."
            )
        self._instance: TileCode[D, NUM_TILINGS, WEIGHT_CONT] = cls(
            dimensional_infos
        )