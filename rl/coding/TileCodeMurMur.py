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
from typing import (
    Any,
    Generic,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

try:
    from typing import override
except ImportError:
    def override(func: Any) -> Any:
        return func

try:
    from ..hash.HashMurmur3 import HashMurmur3Out, Murmur3
    from .TileCode import DEFAULT_TILE_CONT, TileCode
    from .DimensionInfo import DimensionInfo
    from ..declares import BYTE, FEATURE_VECTOR, FLOAT, INT, UINT, floatArray
except (ImportError, ValueError):
    try:
        from rl.hash.HashMurmur3 import HashMurmur3Out, Murmur3
        from rl.coding.TileCode import DEFAULT_TILE_CONT, TileCode
        from rl.coding.DimensionInfo import DimensionInfo
        from rl.declares import BYTE, FEATURE_VECTOR, FLOAT, INT, UINT, floatArray
    except ImportError:
        FLOAT = float
        INT = int
        UINT = int
        BYTE = int
        FEATURE_VECTOR = List[int]
        floatArray = Sequence[float]
        DEFAULT_TILE_CONT = List[float]

        _T = TypeVar("_T")

        class DimensionInfo(Generic[_T]):
            pass

        class HashMurmur3Out:
            hashVal: List[int]

            def __init__(self, hash_val: Optional[List[int]] = None) -> None:
                self.hashVal = hash_val if hash_val is not None else [0, 0]

        class Murmur3:
            def hash(
                self, byte_array: bytes, length: Optional[int] = None
            ) -> HashMurmur3Out:
                raise NotImplementedError

        _D_dummy = TypeVar("_D_dummy")
        _NUM_TILINGS_dummy = TypeVar("_NUM_TILINGS_dummy")
        _WEIGHT_CONT_dummy = TypeVar("_WEIGHT_CONT_dummy")

        class TileCode(Generic[_D_dummy, _NUM_TILINGS_dummy, _WEIGHT_CONT_dummy]):
            def __init__(
                self,
                dimensional_infos: Sequence[DimensionInfo[FLOAT]],
                size_hint: Optional[int] = None,
                *args: Any,
                **kwargs: Any,
            ) -> None:
                self._dimensionalInfos = dimensional_infos
                self._sizeCache = size_hint if size_hint is not None else 0

            def getDimension(self) -> int:
                return len(self._dimensionalInfos)

            def getNumTilings(self) -> int:
                return 0

            def paramToGridValue(
                self, param: float, tiling_index: int, dimension_index: int
            ) -> int:
                return 0


D = TypeVar("D")
NUM_TILINGS = TypeVar("NUM_TILINGS")
try:
    WEIGHT_CONT = TypeVar("WEIGHT_CONT", default=DEFAULT_TILE_CONT)  # type: ignore
except TypeError:
    WEIGHT_CONT = TypeVar("WEIGHT_CONT")


class TileCodeMurMur(
    TileCode[D, NUM_TILINGS, WEIGHT_CONT],
    Generic[D, NUM_TILINGS, WEIGHT_CONT],
):
    """
    Tile Code using MurMur3 hash.

    :tparam D: Number of dimensions.
    :tparam NUM_TILINGS: Number of tilings.
    :tparam WEIGHT_CONT: The container object to store the weights.
    """

    def __init__(
        self,
        dimensional_infos: Sequence[DimensionInfo[FLOAT]],
        size_hint: Optional[int] = None,
        num_tilings: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize TileCodeMurMur.

        :param dimensional_infos: Sequence of DimensionInfo for each dimension.
        :param size_hint: Size hint for cache/hash table size.
        :param num_tilings: Optional number of tilings.
        """
        if "dimensionalInfos" in kwargs:
            dimensional_infos = kwargs.pop("dimensionalInfos")
        if "sizeHint" in kwargs:
            size_hint = kwargs.pop("sizeHint")
        if "numTilings" in kwargs:
            num_tilings = kwargs.pop("numTilings")

        if num_tilings is not None:
            self._num_tilings: int = num_tilings
        if size_hint is not None:
            self._sizeCache: int = size_hint

        if size_hint is not None and num_tilings is not None:
            try:
                super().__init__(dimensional_infos, size_hint, num_tilings, **kwargs)
            except TypeError:
                try:
                    super().__init__(dimensional_infos, size_hint, **kwargs)
                except TypeError:
                    super().__init__(dimensional_infos, **kwargs)
        elif size_hint is not None:
            try:
                super().__init__(dimensional_infos, size_hint, **kwargs)
            except TypeError:
                super().__init__(dimensional_infos, **kwargs)
        else:
            try:
                super().__init__(dimensional_infos, **kwargs)
            except TypeError:
                pass

    @override
    def getFeatureVector(
        self, parameters: Union[floatArray, Sequence[FLOAT]]
    ) -> FEATURE_VECTOR:
        """
        Hash the parameters in Real space to a Natural space [0, sizeCache).

        :param parameters: Float array/sequence of dimension D.
        :return: Feature vector of discretized indices.
        """
        fv: FEATURE_VECTOR = []

        dimension: int = (
            self.getDimension()
            if callable(getattr(self, "getDimension", None))
            else getattr(
                self,
                "dimension",
                getattr(self, "D", len(parameters)),
            )
        )
        num_tilings: int = (
            self.getNumTilings()
            if callable(getattr(self, "getNumTilings", None))
            else getattr(
                self,
                "_num_tilings",
                getattr(self, "num_tilings", getattr(self, "NUM_TILINGS", 0)),
            )
        )
        size_cache: int = getattr(
            self,
            "_sizeCache",
            getattr(
                self,
                "_size_cache",
                self.getSize() if callable(getattr(self, "getSize", None)) else 1,
            ),
        )

        tile_components: List[int] = [0] * (dimension + 1)
        for i in range(num_tilings):
            for j in range(dimension):
                tile_components[j] = self.paramToGridValue(parameters[j], i, j)

            # Add a unique number_tiling identifier.
            tile_components[dimension] = i

            _hash_alg = Murmur3()
            raw_bytes = struct.pack(f"={len(tile_components)}q", *tile_components)
            try:
                hash_out = _hash_alg.hash(raw_bytes, len(raw_bytes))
            except TypeError:
                hash_out = _hash_alg.hash(raw_bytes)

            if hasattr(hash_out, "hashVal"):
                hash_val = hash_out.hashVal[0]
            elif hasattr(hash_out, "hash_val"):
                hash_val = hash_out.hash_val[0]
            elif isinstance(hash_out, (list, tuple)):
                hash_val = hash_out[0]
            else:
                hash_val = int(hash_out)

            hash_val_u64 = int(hash_val) & 0xFFFFFFFFFFFFFFFF
            fv.append(hash_val_u64 % size_cache)

        return fv

    get_feature_vector = getFeatureVector


__all__ = ["TileCodeMurMur"]