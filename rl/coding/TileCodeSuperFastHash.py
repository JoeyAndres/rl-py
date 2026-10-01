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

import struct
from typing import List, Sequence

from ..declares import FEATURE_VECTOR, FLOAT
from ..hash.HashSuperFastHash import SuperFastHash
from .TileCode import TileCode


class TileCodeSuperFastHash(TileCode):
    """Tile Code using SuperFastHash."""

    def getFeatureVector(self, parameters: Sequence[FLOAT]) -> FEATURE_VECTOR:
        fv: FEATURE_VECTOR = []
        tileComponents: List[int] = [0] * (self.getDimension() + 1)
        hashAlg = SuperFastHash()
        for i in range(self.getNumTilings()):
            for j in range(self.getDimension()):
                tileComponents[j] = self.paramToGridValue(parameters[j], i, j)
            # Add a unique number_tiling identifier.
            tileComponents[self.getDimension()] = i
            data = struct.pack(f"<{len(tileComponents)}q", *tileComponents)
            fv.append(hashAlg.hash(data, len(data)) % self._sizeCache)
        return fv
