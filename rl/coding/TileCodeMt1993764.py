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

from typing import (
    Any,
    Generic,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

from .TileCode import TileCode, DEFAULT_TILE_CONT
from .DimensionInfo import DimensionInfo

from ..declares import FLOAT, INT, FEATURE_VECTOR, floatArray


class seed_seq:
    """Emulates std::seed_seq from C++ <random>."""

    def __init__(self, seq: Sequence[int] | None = None) -> None:
        self.v: List[int] = []
        if seq is not None:
            self.v = [int(x) & 0xFFFFFFFF for x in seq]

    def generate(self, n_or_dest: int | List[int]) -> List[int]:
        if isinstance(n_or_dest, int):
            dest = [0x8B8B8B8B] * n_or_dest
            n = n_or_dest
        else:
            dest = n_or_dest
            n = len(dest)
            for i in range(n):
                dest[i] = 0x8B8B8B8B

        if n == 0:
            return dest

        s = len(self.v)
        t = (
            11
            if n >= 624
            else 9
            if n >= 398
            else 7
            if n >= 274
            else 5
            if n >= 37
            else 3
            if n >= 23
            else 1
        )
        p = (n - t) // 2
        q = p + t
        m = max(s + 1, n)

        for k in range(m):
            arg = (
                dest[k % n]
                ^ dest[(k + p) % n]
                ^ (dest[(k + n - 1) % n] >> 27)
            ) & 0xFFFFFFFF
            r1 = (1664525 * arg) & 0xFFFFFFFF
            if k == 0:
                r2 = (r1 + s) & 0xFFFFFFFF
            elif 1 <= k <= s:
                r2 = (r1 + (k % n) + self.v[k - 1]) & 0xFFFFFFFF
            else:
                r2 = (r1 + (k % n)) & 0xFFFFFFFF
            dest[(k + p) % n] = (dest[(k + p) % n] + r1) & 0xFFFFFFFF
            dest[(k + q) % n] = (dest[(k + q) % n] + r2) & 0xFFFFFFFF

        for k in range(m, m + n):
            arg = (
                dest[k % n]
                + dest[(k + p) % n]
                + (dest[(k + n - 1) % n] >> 27)
            ) & 0xFFFFFFFF
            r3 = (1566083941 * arg) & 0xFFFFFFFF
            r4 = (r3 - (k % n)) & 0xFFFFFFFF
            dest[(k + p) % n] = (dest[(k + p) % n] ^ r3) & 0xFFFFFFFF
            dest[(k + q) % n] = (dest[(k + q) % n] ^ r4) & 0xFFFFFFFF

        return dest


class mt19937_64:
    """Emulates std::mt19937_64 from C++ <random>."""

    NN = 312
    MM = 156
    MATRIX_A = 0xB5026F5AA96619E9
    UM = 0xFFFFFFFF80000000
    LM = 0x000000007FFFFFFF
    MASK64 = 0xFFFFFFFFFFFFFFFF

    def __init__(self, seed: Any = 5489) -> None:
        self.mt: List[int] = [0] * self.NN
        self.mti: int = self.NN + 1
        self.seed(seed)

    def seed(self, seed: Any = 5489) -> None:
        if isinstance(seed, seed_seq):
            words = seed.generate(self.NN * 2)
            self._seed_from_32bit_words(words)
        elif isinstance(seed, (list, tuple)):
            s = seed_seq(seed)
            words = s.generate(self.NN * 2)
            self._seed_from_32bit_words(words)
        elif hasattr(seed, "generate"):
            words = [0] * (self.NN * 2)
            seed.generate(words)
            self._seed_from_32bit_words(words)
        elif isinstance(seed, int):
            self.mt[0] = seed & self.MASK64
            for mti in range(1, self.NN):
                self.mt[mti] = (
                    6364136223846793005
                    * (self.mt[mti - 1] ^ (self.mt[mti - 1] >> 62))
                    + mti
                ) & self.MASK64
            self.mti = self.NN
        else:
            self.seed(hash(seed) & self.MASK64)

    def _seed_from_32bit_words(self, words: Sequence[int]) -> None:
        for i in range(self.NN):
            self.mt[i] = (
                (words[2 * i] & 0xFFFFFFFF)
                | ((words[2 * i + 1] & 0xFFFFFFFF) << 32)
            ) & self.MASK64
        if (self.mt[0] & self.UM) == 0 and all(x == 0 for x in self.mt[1:]):
            self.mt[0] = 1 << 63
        self.mti = self.NN

    def __call__(self) -> int:
        mag01 = [0, self.MATRIX_A]
        if self.mti >= self.NN:
            if self.mti == self.NN + 1:
                self.seed(5489)
            for i in range(self.NN - self.MM):
                x = (self.mt[i] & self.UM) | (self.mt[i + 1] & self.LM)
                self.mt[i] = self.mt[i + self.MM] ^ (x >> 1) ^ mag01[x & 1]
            for i in range(self.NN - self.MM, self.NN - 1):
                x = (self.mt[i] & self.UM) | (self.mt[i + 1] & self.LM)
                self.mt[i] = (
                    self.mt[i + (self.MM - self.NN)] ^ (x >> 1) ^ mag01[x & 1]
                )
            x = (self.mt[self.NN - 1] & self.UM) | (self.mt[0] & self.LM)
            self.mt[self.NN - 1] = (
                self.mt[self.MM - 1] ^ (x >> 1) ^ mag01[x & 1]
            )
            self.mti = 0

        x = self.mt[self.mti]
        self.mti += 1

        x ^= (x >> 29) & 0x5555555555555555
        x ^= (x << 17) & 0x71D67FFFEDA60000
        x ^= (x << 37) & 0xFFF7EEE000000000
        x ^= x >> 43

        return x & self.MASK64


D = TypeVar("D")
NUM_TILINGS = TypeVar("NUM_TILINGS")
try:
    WEIGHT_CONT = TypeVar("WEIGHT_CONT", default=DEFAULT_TILE_CONT)
except TypeError:
    WEIGHT_CONT = TypeVar("WEIGHT_CONT")


class TileCodeMt1993764(
    TileCode[D, NUM_TILINGS, WEIGHT_CONT],
    Generic[D, NUM_TILINGS, WEIGHT_CONT],
):
    """Tile Code using Mt1993764 hash.

    :tparam D: Number of dimensions.
    :tparam NUM_TILINGS: Number of tilings.
    :tparam WEIGHT_CONT: The container object to store the weights.
    """

    def __init__(
        self,
        dimensional_infos: Sequence[DimensionInfo[FLOAT]] | Sequence[Any] | None = None,
        size_hint: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        :param dimensional_infos: Information for each dimension.
        :param size_hint: If size_hint is greater than calculated size, size_hint
                          will be used instead. The bigger the size_hint, the
                          less likely collisions occur during hashing.
        """
        if dimensional_infos is None and "dimensionalInfos" in kwargs:
            dimensional_infos = kwargs.pop("dimensionalInfos")
        if size_hint is None and "sizeHint" in kwargs:
            size_hint = kwargs.pop("sizeHint")

        if size_hint is None:
            super().__init__(dimensional_infos, *args, **kwargs)
        else:
            super().__init__(dimensional_infos, size_hint, *args, **kwargs)

        self._prng: mt19937_64 = mt19937_64()

        if "numTilings" in kwargs:
            self._numTilings = kwargs["numTilings"]
        elif "num_tilings" in kwargs:
            self._numTilings = kwargs["num_tilings"]

    def getFeatureVector(
        self, parameters: floatArray | Sequence[FLOAT]
    ) -> FEATURE_VECTOR:
        dimension = (
            self.getDimension()
            if hasattr(self, "getDimension")
            else len(parameters)
        )
        tile_components: List[int] = [0] * (dimension + 1)
        fv: List[int] = []

        num_tilings = None
        if hasattr(self, "getNumTilings"):
            try:
                num_tilings = self.getNumTilings()
            except Exception:
                pass
        if num_tilings is None:
            num_tilings = getattr(self, "_numTilings", None)
        if num_tilings is None:
            orig = getattr(self, "__orig_class__", None)
            if orig and hasattr(orig, "__args__") and len(orig.__args__) >= 2:
                num_tilings = orig.__args__[1]
        if num_tilings is None:
            num_tilings = getattr(self, "NUM_TILINGS", 0)

        size_cache = getattr(self, "_sizeCache", None)
        if size_cache is None and hasattr(self, "getSize"):
            size_cache = self.getSize()

        for i in range(int(num_tilings)):
            for j in range(dimension):
                if hasattr(self, "paramToGridValue"):
                    tile_components[j] = int(
                        self.paramToGridValue(parameters[j], i, j)
                    )
                elif hasattr(self, "param_to_grid_value"):
                    tile_components[j] = int(
                        self.param_to_grid_value(parameters[j], i, j)
                    )

            # Add a unique number_tiling identifier.
            tile_components[dimension] = i

            seed1 = seed_seq(tile_components)
            self._prng.seed(seed1)

            if size_cache:
                fv.append(self._prng() % size_cache)
            else:
                fv.append(self._prng())

        return fv

    get_feature_vector = getFeatureVector


__all__ = [
    "TileCodeMt1993764",
    "seed_seq",
    "mt19937_64",
]