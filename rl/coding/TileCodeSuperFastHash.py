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

FLOAT = float
UINT = int
INT = int
BYTE = int
FEATURE_VECTOR = List[int]
DEFAULT_TILE_CONT = List[float]

try:
    from ..hash.HashInterface import HashInterface
except (ImportError, ValueError):
    try:
        from rl.hash.HashInterface import HashInterface
    except (ImportError, ValueError):
        try:
            from HashInterface import HashInterface
        except (ImportError, ValueError):
            RET = TypeVar("RET")

            class HashInterface(Generic[RET]):  # type: ignore
                def hash(
                    self,
                    byte_array: Union[bytes, bytearray, Sequence[int]],
                    length: Optional[int] = None,
                ) -> RET:
                    raise NotImplementedError


def _super_fast_hash(data: bytes, length: Optional[int] = None) -> int:
    if length is None:
        length = len(data)
    if length <= 0 or not data:
        return 0

    hash_val = length
    rem = length & 3
    blocks = length >> 2
    offset = 0

    for _ in range(blocks):
        hash_val = (hash_val + (data[offset] | (data[offset + 1] << 8))) & 0xFFFFFFFF
        tmp = (((data[offset + 2] | (data[offset + 3] << 8)) << 11) ^ hash_val) & 0xFFFFFFFF
        hash_val = ((hash_val << 16) ^ tmp) & 0xFFFFFFFF
        offset += 4
        hash_val = (hash_val + (hash_val >> 11)) & 0xFFFFFFFF

    if rem == 3:
        hash_val = (hash_val + (data[offset] | (data[offset + 1] << 8))) & 0xFFFFFFFF
        hash_val = (hash_val ^ (hash_val << 16)) & 0xFFFFFFFF
        signed_char = data[offset + 2] if data[offset + 2] < 128 else data[offset + 2] - 256
        hash_val = (hash_val ^ ((signed_char << 18) & 0xFFFFFFFF)) & 0xFFFFFFFF
        hash_val = (hash_val + (hash_val >> 11)) & 0xFFFFFFFF
    elif rem == 2:
        hash_val = (hash_val + (data[offset] | (data[offset + 1] << 8))) & 0xFFFFFFFF
        hash_val = (hash_val ^ (hash_val << 11)) & 0xFFFFFFFF
        hash_val = (hash_val + (hash_val >> 17)) & 0xFFFFFFFF
    elif rem == 1:
        signed_char = data[offset] if data[offset] < 128 else data[offset] - 256
        hash_val = (hash_val + signed_char) & 0xFFFFFFFF
        hash_val = (hash_val ^ (hash_val << 10)) & 0xFFFFFFFF
        hash_val = (hash_val + (hash_val >> 1)) & 0xFFFFFFFF

    hash_val = (hash_val ^ ((hash_val << 3) & 0xFFFFFFFF)) & 0xFFFFFFFF
    hash_val = (hash_val + (hash_val >> 5)) & 0xFFFFFFFF
    hash_val = (hash_val ^ ((hash_val << 4) & 0xFFFFFFFF)) & 0xFFFFFFFF
    hash_val = (hash_val + (hash_val >> 17)) & 0xFFFFFFFF
    hash_val = (hash_val ^ ((hash_val << 25) & 0xFFFFFFFF)) & 0xFFFFFFFF
    hash_val = (hash_val + (hash_val >> 6)) & 0xFFFFFFFF
    return hash_val


try:
    from ..hash.HashSuperFastHash import SuperFastHash
except (ImportError, ValueError):
    try:
        from rl.hash.HashSuperFastHash import SuperFastHash
    except (ImportError, ValueError):
        try:
            from HashSuperFastHash import SuperFastHash
        except (ImportError, ValueError):

            class SuperFastHash(HashInterface[int]):  # type: ignore
                def hash(
                    self,
                    byte_array: Union[bytes, bytearray, Sequence[int]],
                    length: Optional[int] = None,
                ) -> int:
                    if isinstance(byte_array, (bytes, bytearray)):
                        b = bytes(byte_array)
                    else:
                        b = bytes(byte_array)
                    l = len(b) if length is None else length
                    return _super_fast_hash(b, l)


try:
    from .DimensionInfo import DimensionInfo
except (ImportError, ValueError):
    try:
        from rl.coding.DimensionInfo import DimensionInfo
    except (ImportError, ValueError):
        try:
            from DimensionInfo import DimensionInfo
        except (ImportError, ValueError):
            T = TypeVar("T")

            class DimensionInfo(Generic[T]):  # type: ignore
                def __init__(self, min_val: T, max_val: T, partitions: int, offsets: int = 0) -> None:
                    self._min = min_val
                    self._max = max_val
                    self._partitions = partitions
                    self._offsets = offsets

                def getMin(self) -> T:
                    return self._min

                def getMax(self) -> T:
                    return self._max

                def getOffsets(self) -> int:
                    return self._offsets

                def getGeneralizationScale(self) -> float:
                    return (float(self._max) - float(self._min)) / float(self._partitions)


try:
    from .TileCode import TileCode, DEFAULT_TILE_CONT
except (ImportError, ValueError):
    try:
        from rl.coding.TileCode import TileCode, DEFAULT_TILE_CONT
    except (ImportError, ValueError):
        try:
            from TileCode import TileCode, DEFAULT_TILE_CONT
        except (ImportError, ValueError):
            TC_D = TypeVar("TC_D")
            TC_NUM = TypeVar("TC_NUM")
            TC_WEIGHT = TypeVar("TC_WEIGHT")

            class TileCode(Generic[TC_D, TC_NUM, TC_WEIGHT]):  # type: ignore
                def __init__(
                    self,
                    dimensional_infos: Sequence[DimensionInfo[float]],
                    size_hint: Optional[int] = None,
                    *args: Any,
                    **kwargs: Any,
                ) -> None:
                    self._dimensionalInfos = dimensional_infos
                    self._sizeCache = size_hint if size_hint is not None else 0
                    self.num_tilings = kwargs.get("num_tilings", kwargs.get("numTilings", 0))

                def getDimension(self) -> int:
                    return len(self._dimensionalInfos) if hasattr(self, "_dimensionalInfos") else 0

                def getNumTilings(self) -> int:
                    return getattr(self, "num_tilings", getattr(self, "NUM_TILINGS", 0))

                def getSize(self) -> int:
                    return getattr(self, "_sizeCache", 0)

                def paramToGridValue(self, param: float, tiling_index: int, dimension_index: int) -> int:
                    raise NotImplementedError


D = TypeVar("D")
NUM_TILINGS = TypeVar("NUM_TILINGS")
WEIGHT_CONT = TypeVar("WEIGHT_CONT")

try:
    _TileCodeBase = TileCode[D, NUM_TILINGS, WEIGHT_CONT]  # type: ignore
except (TypeError, AttributeError):
    try:
        _TileCodeBase = TileCode[WEIGHT_CONT]  # type: ignore
    except (TypeError, AttributeError):
        _TileCodeBase = TileCode  # type: ignore


class TileCodeSuperFastHash(_TileCodeBase, Generic[D, NUM_TILINGS, WEIGHT_CONT]):
    """
    Tile Code using SuperFastHash.

    :tparam D: Number of dimensions.
    :tparam NUM_TILINGS: Number of tilings.
    :tparam WEIGHT_CONT: The container object to store the weights.
    """

    def __init__(
        self,
        dimensional_infos: Sequence[DimensionInfo[float]],
        size_hint: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        :param dimensional_infos: Array/sequence of DimensionInfo.
        :param size_hint: If size_hint is greater than calculated size, then size_hint
                          will be used instead. The bigger the size_hint, the less likely
                          is the collision during hashing.
        """
        dim_infos = dimensional_infos if dimensional_infos is not None else kwargs.pop("dimensionalInfos", None)
        s_hint = size_hint if size_hint is not None else kwargs.pop("sizeHint", None)

        if "numTilings" in kwargs:
            self._numTilings = kwargs["numTilings"]
        elif "num_tilings" in kwargs:
            self._numTilings = kwargs["num_tilings"]

        self._dimensionalInfos = dim_infos

        if s_hint is not None:
            self._sizeCache = s_hint
            try:
                super().__init__(dim_infos, s_hint, *args, **kwargs)
            except TypeError:
                super().__init__(dim_infos, s_hint)
        else:
            try:
                super().__init__(dim_infos, *args, **kwargs)
            except TypeError:
                super().__init__(dim_infos)
            if not hasattr(self, "_sizeCache"):
                self._sizeCache = getattr(self, "getSize", lambda: 0)()

    @classmethod
    def __class_getitem__(cls, params: Any) -> Any:
        try:
            return super().__class_getitem__(params)
        except TypeError:
            if not isinstance(params, tuple):
                params = (params,)
            d = params[0] if len(params) > 0 else None
            num_tilings = params[1] if len(params) > 1 else None
            weight_cont = params[2] if len(params) > 2 else None

            class _SpecializedTileCodeSuperFastHash(cls):  # type: ignore
                _D = d
                _NUM_TILINGS = num_tilings
                _WEIGHT_CONT = weight_cont

                def getDimension(self) -> int:
                    if self._D is not None and isinstance(self._D, int):
                        return self._D
                    return super(_SpecializedTileCodeSuperFastHash, self).getDimension()

                def getNumTilings(self) -> int:
                    if self._NUM_TILINGS is not None and isinstance(self._NUM_TILINGS, int):
                        return self._NUM_TILINGS
                    return super(_SpecializedTileCodeSuperFastHash, self).getNumTilings()

            _SpecializedTileCodeSuperFastHash.__name__ = f"{cls.__name__}[{', '.join(str(p) for p in params)}]"
            _SpecializedTileCodeSuperFastHash.__qualname__ = f"{cls.__qualname__}[{', '.join(str(p) for p in params)}]"
            return _SpecializedTileCodeSuperFastHash

    def getFeatureVector(self, parameters: Sequence[float]) -> FEATURE_VECTOR:
        """
        Hashes the parameters in Real space to a Natural space [0, infinity).

        :param parameters: Sequence of float parameters in continuous space.
        :return: List of discretized feature indices.
        """
        if hasattr(self, "getDimension") and callable(self.getDimension):
            dimension = self.getDimension()
        elif hasattr(self, "_dimensionalInfos") and self._dimensionalInfos is not None:
            dimension = len(self._dimensionalInfos)
        else:
            dimension = len(parameters)

        if hasattr(self, "getNumTilings") and callable(self.getNumTilings):
            num_tilings = self.getNumTilings()
        else:
            num_tilings = getattr(
                self,
                "_numTilings",
                getattr(
                    self,
                    "_num_tilings",
                    getattr(self, "num_tilings", getattr(self, "NUM_TILINGS", 0)),
                ),
            )

        size_cache = getattr(self, "_sizeCache", None)
        if size_cache is None:
            if hasattr(self, "getSize") and callable(self.getSize):
                size_cache = self.getSize()
            else:
                size_cache = 1

        tile_components: List[int] = [0] * (dimension + 1)
        fv: FEATURE_VECTOR = []

        for i in range(num_tilings):
            for j in range(dimension):
                tile_components[j] = int(self.paramToGridValue(parameters[j], i, j))

            # Add a unique number_tiling identifier.
            tile_components[dimension] = i

            hash_alg = SuperFastHash()
            byte_data = struct.pack(f"={len(tile_components)}q", *tile_components)

            try:
                hash_val = hash_alg.hash(byte_data, len(byte_data))
            except TypeError:
                hash_val = hash_alg.hash(byte_data)

            hash_val = int(hash_val) & 0xFFFFFFFF
            if size_cache > 0:
                fv.append(hash_val % size_cache)
            else:
                fv.append(hash_val)

        return fv

    get_feature_vector = getFeatureVector