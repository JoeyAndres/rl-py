from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import (
    Any,
    Generic,
    List,
    MutableSequence,
    Optional,
    Sequence,
    TypeVar,
)

# Type declarations corresponding to rl/declares.h
FLOAT = float
UINT = int
FEATURE_VECTOR = List[int]
DEFAULT_TILE_CONT = List[float]

CONT = TypeVar("CONT", bound=MutableSequence[Any])
D = TypeVar("D")
NUM_TILINGS_TYPE = TypeVar("NUM_TILINGS_TYPE")
WEIGHT_CONT = TypeVar("WEIGHT_CONT", bound=MutableSequence[float])


# Base interface corresponding to rl::utility::IndexAccessorInterface
try:
    from rl.utility.IndexAccessorInterface import IndexAccessorInterface
except ImportError:
    try:
        from IndexAccessorInterface import IndexAccessorInterface
    except ImportError:

        class IndexAccessorInterface(ABC, Generic[CONT]):
            @abstractmethod
            def at(self, i: int) -> Any:
                pass

            def __getitem__(self, i: int) -> Any:
                return self.at(i)

            def __setitem__(self, i: int, value: Any) -> None:
                pass


# Base class corresponding to rl::coding::CourseCode
try:
    from rl.coding.CourseCode import CourseCode
except ImportError:
    try:
        from CourseCode import CourseCode
    except ImportError:

        class CourseCode(ABC, Generic[D]):
            def __init__(self, dimension: int = 0) -> None:
                self._dimension = dimension

            def getDimension(self) -> int:
                return self._dimension

            def get_dimension(self) -> int:
                return self.getDimension()

            @abstractmethod
            def getValueFromParameters(self, parameters: Sequence[float]) -> float:
                pass

            @abstractmethod
            def getSize(self) -> int:
                pass


# Class corresponding to rl::coding::DimensionInfo
try:
    from rl.coding.DimensionInfo import DimensionInfo
except ImportError:
    try:
        from DimensionInfo import DimensionInfo
    except ImportError:

        class DimensionInfo:
            def __init__(
                self,
                lowerRange: float = 0.0,
                higherRange: float = 0.0,
                gridCount: int = 0,
                generalizationScale: float = 1.0,
            ) -> None:
                self._lowerRange = lowerRange
                self._higherRange = higherRange
                self._gridCountIdeal = gridCount
                self._generalizationScale = generalizationScale

            def GetOffsets(self) -> float:
                if self._gridCountIdeal == 0:
                    return 0.0
                return (self._higherRange - self._lowerRange) / self._gridCountIdeal

            def GetGridCountReal(self) -> int:
                return self._gridCountIdeal + 1

            def SetGridCountIdeal(self, gridCountIdeal: int) -> None:
                self._gridCountIdeal = gridCountIdeal

            def GetGridCountIdeal(self) -> int:
                return self._gridCountIdeal

            def GetRangeDifference(self) -> float:
                return abs(self._higherRange - self._lowerRange)

            def setUpperBound(self, upperBound: float) -> None:
                self._higherRange = upperBound

            def setLowerBound(self, lowerBound: float) -> None:
                self._lowerRange = lowerBound

            def getUpperBound(self) -> float:
                return self._higherRange

            def getLowerBound(self) -> float:
                return self._lowerRange

            def getGeneralizationScale(self) -> float:
                return self._generalizationScale

            def setGeneralizationScale(self, generalization: float) -> None:
                self._generalizationScale = generalization

            get_offsets = GetOffsets
            get_grid_count_real = GetGridCountReal
            set_grid_count_ideal = SetGridCountIdeal
            get_grid_count_ideal = GetGridCountIdeal
            get_range_difference = GetRangeDifference
            set_upper_bound = setUpperBound
            set_lower_bound = setLowerBound
            get_upper_bound = getUpperBound
            get_lower_bound = getLowerBound
            get_generalization_scale = getGeneralizationScale
            set_generalization_scale = setGeneralizationScale


class TileCode(
    CourseCode[D],
    IndexAccessorInterface[WEIGHT_CONT],
    Generic[D, NUM_TILINGS_TYPE, WEIGHT_CONT],
):
    """
    Base object encapsulate tile coding.

    For an in-depth explanation of Tile Coding, see Tile Coding documentation.
    """

    NUM_TILINGS: int = 1

    def __init__(
        self,
        dimensionalInfos: Sequence[DimensionInfo],
        sizeHint: Optional[int] = None,
        numTilings: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        :param dimensionalInfos: Sequence of DimensionInfo instances.
        :param sizeHint: Size of weight container (optional).
        :param numTilings: Number of tilings (optional).
        """
        if numTilings is not None:
            self._num_tilings: int = numTilings
        elif "num_tilings" in kwargs:
            self._num_tilings = kwargs["num_tilings"]
        elif hasattr(self, "NUM_TILINGS") and self.NUM_TILINGS:
            self._num_tilings = self.NUM_TILINGS
        else:
            self._num_tilings = 1

        super().__init__(len(dimensionalInfos))
        self._dimensionalInfos: List[DimensionInfo] = list(dimensionalInfos)
        self._pseudoRNG = random.Random()

        if sizeHint is None:
            self._sizeCache: int = self._calculateSize(
                self._dimensionalInfos, self._num_tilings
            )
        else:
            self._sizeCache = sizeHint

        self._w: List[float] = [0.0] * self._sizeCache

        # Calculate random offsets.
        self._randomOffsets: List[List[float]] = []
        for i in range(self._num_tilings):
            offsets: List[float] = []
            for j in range(self.getDimension()):
                offsets.append(
                    self._pseudoRNG.uniform(0.0, 1.0)
                    * self._dimensionalInfos[j].GetOffsets()
                    * self._dimensionalInfos[j].getGeneralizationScale()
                )
            self._randomOffsets.append(offsets)

    def at(self, i: int) -> float:
        return self._w[i]

    def __getitem__(self, i: int) -> float:
        return self.at(i)

    def __setitem__(self, i: int, value: float) -> None:
        self._w[i] = value

    @abstractmethod
    def getFeatureVector(
        self, parameters: Sequence[float]
    ) -> FEATURE_VECTOR:
        """
        Hashed the parameter in Real space to a Natural space [0, infinity).
        :param parameters:
        :return: Vector of "discretize" index.
        """
        pass

    def getValueFromFeatureVector(self, fv: Sequence[int]) -> float:
        """
        Get the value of the parameters in the real space.
        :param fv: corresponding feature vector.
        :return: corresponding value.
        """
        total: float = 0.0
        for f in fv:
            total += self._w[f]
        return total

    def getSize(self) -> int:
        return self._sizeCache

    def getNumTilings(self) -> int:
        """
        :return: number of tilings.
        """
        return self._num_tilings

    def getValueFromParameters(self, parameters: Sequence[float]) -> float:
        fv = self.getFeatureVector(parameters)
        return self.getValueFromFeatureVector(fv)

    def paramToGridValue(
        self, param: float, tilingIndex: int, dimensionIndex: int
    ) -> int:
        """
        :param param:
        :param tilingIndex:
        :param dimensionIndex:
        :return:
        """
        randomOffset = self._randomOffsets[tilingIndex][dimensionIndex]
        dimGeneralizationScale = (
            self._dimensionalInfos[dimensionIndex].getGeneralizationScale()
        )
        dimLowerBound = self._dimensionalInfos[dimensionIndex].getLowerBound()
        dimGridCountIdeal = (
            self._dimensionalInfos[dimensionIndex].GetGridCountIdeal()
        )
        dimRangeMagnitude = (
            self._dimensionalInfos[dimensionIndex].GetRangeDifference()
        )

        return int(
            (
                (
                    param
                    + randomOffset * dimGeneralizationScale
                    - dimLowerBound
                )
                * dimGridCountIdeal
            )
            / dimRangeMagnitude
        )

    def _calculateSizeCache(self) -> int:
        """
        :return: Number of possible grid points.
        """
        return self._calculateSize(self._dimensionalInfos, self._num_tilings)

    @classmethod
    def _calculateSize(
        cls,
        dims: Sequence[DimensionInfo],
        numTilings: Optional[int] = None,
    ) -> int:
        if numTilings is None:
            numTilings = getattr(cls, "NUM_TILINGS", 1)
        size = 1
        for di in dims:
            size *= di.GetGridCountReal()
        size *= numTilings
        return size

    @classmethod
    def __class_getitem__(cls, params: Any) -> Any:
        try:
            if not isinstance(params, tuple):
                params = (params,)
            if len(params) == 1:
                params = (Any, Any, params[0])
            elif len(params) == 2:
                params = (params[0], params[1], DEFAULT_TILE_CONT)
            return super().__class_getitem__(params)
        except Exception:
            return cls

    def __len__(self) -> int:
        return self.getSize()

    # Pythonic snake_case aliases
    get_feature_vector = getFeatureVector
    get_value_from_feature_vector = getValueFromFeatureVector
    get_size = getSize
    get_num_tilings = getNumTilings
    get_value_from_parameters = getValueFromParameters
    param_to_grid_value = paramToGridValue
    _calculate_size_cache = _calculateSizeCache
    _calculate_size = _calculateSize


spTileCode = TileCode