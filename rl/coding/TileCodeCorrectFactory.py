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

from typing import TYPE_CHECKING, Any, Generic, Sequence, TypeVar

from ..declares import DEFAULT_TILE_CONT
from .TileCodeCorrect import TileCodeCorrect
from .TileCodeFactory import TileCodeFactory

if TYPE_CHECKING:
    from .DimensionInfo import DimensionInfo

D = TypeVar("D", bound=int)
NUM_TILINGS = TypeVar("NUM_TILINGS", bound=int)
WEIGHT_CONT = TypeVar("WEIGHT_CONT")


class TileCodeCorrectFactory(
    TileCodeFactory[D, NUM_TILINGS, WEIGHT_CONT, TileCodeCorrect],
    Generic[D, NUM_TILINGS, WEIGHT_CONT],
):
    """Factory method for TileCodeCorrect.

    :tparam D: Number of dimension.
    :tparam NUM_TILINGS: Number of tilings.
    :tparam WEIGHT_CONT: The container object to store the weights.
    """

    tile_code_class: type[TileCodeCorrect] = TileCodeCorrect

    def __init__(
        self,
        dimensional_infos: Sequence[DimensionInfo] | Sequence[Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(
                dimensional_infos,
                tile_code_class=TileCodeCorrect,
                *args,
                **kwargs,
            )
        except TypeError:
            try:
                super().__init__(dimensional_infos, *args, **kwargs)
            except TypeError:
                super().__init__()
        if getattr(self, "_instance", None) is None:
            self._instance = TileCodeCorrect(dimensional_infos)