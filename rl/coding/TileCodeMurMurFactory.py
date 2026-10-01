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

from typing import Generic, List, Optional, Sequence, TypeVar

try:
    from .DimensionInfo import DimensionInfo
    from .TileCodeHashedFactory import TileCodeHashedFactory
    from .TileCodeMurMur import TileCodeMurMur
except ImportError:
    from DimensionInfo import DimensionInfo
    from TileCodeHashedFactory import TileCodeHashedFactory
    from TileCodeMurMur import TileCodeMurMur

DEFAULT_TILE_CONT = List[float]

D = TypeVar("D", bound=int)
NUM_TILINGS = TypeVar("NUM_TILINGS", bound=int)
WEIGHT_CONT = TypeVar("WEIGHT_CONT")


class TileCodeMurMurFactory(
    TileCodeHashedFactory[D, NUM_TILINGS, WEIGHT_CONT, TileCodeMurMur],
    Generic[D, NUM_TILINGS, WEIGHT_CONT],
):
    """
    Factory method for TileCodeMurMur.

    :tparam D: Number of dimensions.
    :tparam NUM_TILINGS: Number of tilings.
    :tparam WEIGHT_CONT: The container object to store the weights.
    """

    def __init__(
        self,
        dimensional_infos: Sequence[DimensionInfo[float]],
        size_hint: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize TileCodeMurMurFactory.

        Inherits constructors from TileCodeFactory and TileCodeHashedFactory.

        :param dimensional_infos: Dimensional information for each dimension.
        :param size_hint: Optional size hint for hashing.
        """
        if size_hint is not None:
            super().__init__(dimensional_infos, size_hint, *args, **kwargs)
        else:
            super().__init__(dimensional_infos, *args, **kwargs)