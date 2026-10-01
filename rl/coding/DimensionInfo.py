from __future__ import annotations

import math
from typing import Generic, Tuple, TypeVar

D = TypeVar("D", float, int)


class DimensionInfo(Generic[D]):
    """
    Encapsulate tile code dimension information.
    """

    def __init__(
        self,
        lowerRange: D,
        higherRange: D,
        gridCount: int,
        generalizationScale: float = 1.0,
    ) -> None:
        """
        DimensionInfo consists of range [a, b] and gridCount. Higher
        grid count means more precision, less grid count means more
        generalization.

        :param lowerRange: a in [a, b]
        :param higherRange: b in [a, b]
        :param gridCount: High value means more precision, less means more generalization.
        :param generalizationScale: scales the generalization. Greater than 1 means exaggerated,
                                    [0, 1], otherwise, reduced influence.
        """
        self._range: Tuple[D, D] = (lowerRange, higherRange)
        self._gridCountIdeal: int = gridCount
        self._generalizationScale: float = float(generalizationScale)

    def GetOffsets(self) -> D:
        """
        :return: range/gridCount. Divide this with number of tilings
                 when used for acquiring feature vector to get the real increment.
        """
        return self.GetRangeDifference() / self._gridCountIdeal  # type: ignore[return-value]

    def GetGridCountReal(self) -> int:
        """
        :return: usually gridCount + 1.
        """
        return self._gridCountIdeal + 1

    def SetGridCountIdeal(self, gridCountIdeal: int) -> None:
        """
        Changes the gridCount.

        :param gridCountIdeal: new gridCount value.
        """
        self._gridCountIdeal = gridCountIdeal

    def GetGridCountIdeal(self) -> int:
        """
        :return: gridCount.
        """
        return self._gridCountIdeal

    def GetRangeDifference(self) -> float:
        """
        :return: Acquire the difference in range [a, b]. Or |b - a|.
        """
        return float(abs(self._range[0] - self._range[1]))

    def setUpperBound(self, upperBound: D) -> None:
        """
        :param upperBound: to set b in range [a, b].
        """
        self._range = (self._range[0], upperBound)

    def setLowerBound(self, lowerBound: D) -> None:
        """
        :param lowerBound: to set a in range [a, b].
        """
        self._range = (lowerBound, self._range[1])

    def getUpperBound(self) -> D:
        """
        :return: b in range [a, b]
        """
        return self._range[1]

    def getLowerBound(self) -> D:
        """
        :return: a in range [a, b].
        """
        return self._range[0]

    def getGeneralizationScale(self) -> float:
        """
        :return: the generalization amount.
        """
        return self._generalizationScale

    def setGeneralizationScale(self, generalization: float) -> None:
        """
        :param generalization: scales the generalization. Greater than 1 means
                               exaggerated, (0, 1), otherwise, reduced influence.
        """
        self._generalizationScale = generalization


DimensionInfoF = DimensionInfo[float]