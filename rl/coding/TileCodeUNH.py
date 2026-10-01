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

import math
from typing import List, Sequence

from ..declares import FEATURE_VECTOR, FLOAT
from ..hash.HashUNH import hashUNH
from .TileCode import TileCode


class TileCodeUNH(TileCode):
    """Tile Code using University of New Hampshire hash, or UNH."""

    def __init__(self, dimensionalInfos, numTilings, sizeHint=None, weightCont=None) -> None:
        super().__init__(dimensionalInfos, numTilings, sizeHint, weightCont)
        self._normalization: List[FLOAT] = [
            numTilings / di.GetRangeDifference() for di in self._dimensionalInfos
        ]

    @staticmethod
    def mod(n: int, k: int) -> int:
        return n % k if n >= 0 else k - 1 - ((-n - 1) % k)

    def getFeatureVector(self, parameters: Sequence[FLOAT]) -> FEATURE_VECTOR:
        dim = self.getDimension()
        numTilings = self.getNumTilings()
        fv: FEATURE_VECTOR = []
        base: List[int] = [0] * dim
        # floor, since truncation is inconsistent for negative numbers.
        qStates = [math.floor(parameters[i] * self._normalization[i]) for i in range(dim)]
        for i in range(numTilings):
            tileComponents: List[int] = [0] * (dim + 1)
            for j in range(dim):
                tileComponents[j] = qStates[j] - self.mod(qStates[j] - base[j], numTilings)
                base[j] += 1 + (j * 2)
            tileComponents[dim] = i
            # The C++ called UNH::hash(components, sizeCache), which reads
            # sizeCache*8 coordinates past the array and hashes into a fixed
            # 4096 slots (possibly beyond the weight vector). Hash the actual
            # components into [0, size) as the UNH CMAC code intends.
            fv.append(hashUNH(tileComponents, len(tileComponents), self._sizeCache))
        return fv
