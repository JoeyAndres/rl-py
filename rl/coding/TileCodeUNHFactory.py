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

from typing import Generic, List, TypeVar

from .TileCodeHashedFactory import TileCodeHashedFactory
from .TileCodeUNH import TileCodeUNH

DEFAULT_TILE_CONT = List[float]

D = TypeVar("D")
NUM_TILINGS = TypeVar("NUM_TILINGS")
WEIGHT_CONT = TypeVar("WEIGHT_CONT")


class TileCodeUNHFactory(
    TileCodeHashedFactory[D, NUM_TILINGS, WEIGHT_CONT, TileCodeUNH],
    Generic[D, NUM_TILINGS, WEIGHT_CONT],
):
    """Factory method for TileCodeUNH.

    :tparam D: Number of dimension.
    :tparam NUM_TILINGS: Number of tilings.
    :tparam WEIGHT_CONT: The container object to store the weights.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


__all__ = ["TileCodeUNHFactory"]