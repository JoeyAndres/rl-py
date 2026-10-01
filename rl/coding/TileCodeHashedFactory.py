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

from typing import Optional, Sequence

from ..declares import FLOAT
from .DimensionInfo import DimensionInfo
from .TileCodeFactory import TileCodeFactory


class TileCodeHashedFactory(TileCodeFactory):
    """Factory method for TileCode children that utilize hashing.

    Accepts an optional sizeHint; the bigger it is, the less likely collisions are.
    """

    def __init__(
        self,
        dimensionalInfos: Sequence[DimensionInfo[FLOAT]],
        numTilings: int,
        sizeHint: Optional[int] = None,
        weightCont=None,
    ) -> None:
        super(TileCodeFactory, self).__init__()
        self._instance = self._tileCodeClass()(
            dimensionalInfos, numTilings, sizeHint, weightCont
        )
