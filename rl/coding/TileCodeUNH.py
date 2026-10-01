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

import math
from typing import (
    Any,
    Generic,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

from ..hash.HashUNH import UNH
from .TileCode import TileCode, DEFAULT_TILE_CONT
from .DimensionInfo import DimensionInfo
from ..declares import FEATURE_VECTOR, FLOAT, INT, UINT, floatArray


D = TypeVar("D")
NUM_TILINGS = TypeVar("NUM_TILINGS")
WEIGHT_CONT = TypeVar("WEIGHT_CONT")


class TileCodeUNH(TileCode[D, NUM_TILINGS, WEIGHT_CONT], Generic[D, NUM_TILINGS, WEIGHT_CONT]):
    """Tile Code using University New Hampshire hash, or UNH."""

    def __init__(
        self,
        dimensionalInfos: Sequence[DimensionInfo],
        sizeHint: Optional[int] = None,
        numTilings: Optional[int] = None,
    ) -> None:
        if sizeHint is not None:
            super().__init__(dimensionalInfos, sizeHint)
        else:
            super().__init__(dimensionalInfos)

        if numTilings is not None:
            self._numTilings = numTilings
        elif hasattr(self, "getNumTilings") and callable(self.getNumTilings):
            self._numTilings = self.getNumTilings()
        else:
            self._numTilings = 0

        self._normalization: List[FLOAT] = [0.0] * self.getDimension()

        dim_count = (
            self._dimensionalInfos.size()
            if hasattr(self._dimensionalInfos, "size") and callable(self._dimensionalInfos.size)
            else len(self._dimensionalInfos)
        )

        for i in range(dim_count):
            dim_info = (
                self._dimensionalInfos.at(i)
                if hasattr(self._dimensionalInfos, "at") and callable(self._dimensionalInfos.at)
                else self._dimensionalInfos[i]
            )
            range_diff = (
                dim_info.GetRangeDifference()
                if hasattr(dim_info, "GetRangeDifference")
                else dim_info.getRangeDifference()
            )
            self._normalization[i] = self._numTilings / range_diff

    def mod(self, n: int, k: int) -> int:
        return n % k if n >= 0 else k - 1 - ((-n - 1) % k)

    def getFeatureVector(self, parameters: Union[floatArray, Sequence[float]]) -> FEATURE_VECTOR:
        fv: FEATURE_VECTOR = []

        dim = self.getDimension()
        num_tilings = (
            self.getNumTilings()
            if hasattr(self, "getNumTilings") and callable(self.getNumTilings)
            else self._numTilings
        )

        base: List[int] = [0] * dim
        q_states: List[int] = [0] * dim

        for i in range(dim):
            param_val = (
                parameters.at(i)
                if hasattr(parameters, "at") and callable(parameters.at)
                else parameters[i]
            )
            # Note to floor since casting to integer is not consistent
            # with negative number. Casting is always a number toward zero.
            q_states[i] = math.floor(param_val * self._normalization[i])

        for i in range(num_tilings):
            tile_components: List[int] = [0] * (dim + 1)

            for j in range(dim):
                tile_components[j] = q_states[j] - self.mod(
                    q_states[j] - base[j], num_tilings
                )
                base[j] += 1 + (j * 2)

            tile_components[dim] = i
            hash_alg = UNH()
            fv.append(hash_alg.hash(tile_components, self._sizeCache))

        return fv