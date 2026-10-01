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

import random
from abc import abstractmethod
from typing import Generic, List, MutableSequence, Optional, Sequence, TypeVar

from ..declares import FEATURE_VECTOR, FLOAT
from ..utility.IndexAccessorInterface import IndexAccessorInterface
from .CourseCode import CourseCode
from .DimensionInfo import DimensionInfo

#: Default container for tile code weights (C++: std::vector<FLOAT>).
DEFAULT_TILE_CONT = list

WEIGHT_CONT = TypeVar("WEIGHT_CONT", bound=MutableSequence[float])


class TileCode(CourseCode, IndexAccessorInterface[WEIGHT_CONT], Generic[WEIGHT_CONT]):
    """Base object encapsulate tile coding.

    The C++ template parameters map as follows:

    * D (number of dimensions) is len(dimensionalInfos).
    * NUM_TILINGS is the numTilings constructor argument.
    * WEIGHT_CONT is the weightCont constructor argument: a callable
      (size, initialValue) -> MutableSequence[float].
    """

    def __init__(
        self,
        dimensionalInfos: Sequence[DimensionInfo[FLOAT]],
        numTilings: int,
        sizeHint: Optional[int] = None,
        weightCont=None,
    ) -> None:
        """
        :param dimensionalInfos: Information for each dimension.
        :param numTilings: Number of tilings.
        :param sizeHint: Size of the weight vector. Defaults to the number of
                         distinct grid points across all tilings.
        :param weightCont: Factory for the weight container. Defaults to list.
        """
        super().__init__(len(dimensionalInfos))
        self._dimensionalInfos: List[DimensionInfo[FLOAT]] = list(dimensionalInfos)
        self._numTilings: int = numTilings
        self._sizeCache: int = (
            sizeHint
            if sizeHint is not None
            else self._calculateSize(self._dimensionalInfos, numTilings)
        )
        if weightCont is None:
            self._w: WEIGHT_CONT = [0.0] * self._sizeCache  # type: ignore[assignment]
        else:
            self._w = weightCont(self._sizeCache, 0.0)

        # Pre-computing the random offsets keeps samples consistent; recomputing
        # them every call would need many more tilings to generalize as well.
        self._pseudoRNG = random.Random()
        self._randomOffsets: List[List[FLOAT]] = [
            [
                self._pseudoRNG.uniform(0.0, 1.0)
                * di.GetOffsets()
                * di.getGeneralizationScale()
                for di in self._dimensionalInfos
            ]
            for _ in range(numTilings)
        ]

    def at(self, i: int) -> float:
        return self._w[i]

    def __setitem__(self, i: int, value: float) -> None:
        self._w[i] = value

    @abstractmethod
    def getFeatureVector(self, parameters: Sequence[FLOAT]) -> FEATURE_VECTOR:
        """Hash the parameters in real space to natural space [0, size).

        :param parameters: Point in real space, one value per dimension.
        :return: Feature vector, one index per tiling.
        """

    def getValueFromFeatureVector(self, fv: FEATURE_VECTOR) -> FLOAT:
        """
        :param fv: Feature vector.
        :return: Sum of the weights indexed by the feature vector.
        """
        return sum(self._w[f] for f in fv)

    def getSize(self) -> int:
        return self._sizeCache

    def getNumTilings(self) -> int:
        return self._numTilings

    def getValueFromParameters(self, parameters: Sequence[FLOAT]) -> FLOAT:
        return self.getValueFromFeatureVector(self.getFeatureVector(parameters))

    def paramToGridValue(
        self, param: FLOAT, tilingIndex: int, dimensionIndex: int
    ) -> int:
        di = self._dimensionalInfos[dimensionIndex]
        randomOffset = self._randomOffsets[tilingIndex][dimensionIndex]
        # C++ returns size_t, so truncate toward zero.
        return int(
            (param + randomOffset * di.getGeneralizationScale() - di.getLowerBound())
            * di.GetGridCountIdeal()
            / di.GetRangeDifference()
        )

    def _calculateSizeCache(self) -> int:
        return self._calculateSize(self._dimensionalInfos, self._numTilings)

    @staticmethod
    def _calculateSize(
        dims: Sequence[DimensionInfo[FLOAT]], numTilings: int
    ) -> int:
        size = 1
        for di in dims:
            size *= di.GetGridCountReal()
        return size * numTilings


spTileCode = TileCode
