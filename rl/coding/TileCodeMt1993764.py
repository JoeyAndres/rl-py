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

from typing import Any, List, Sequence

from ..declares import FEATURE_VECTOR, FLOAT
from .TileCode import TileCode


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
            x = dest[k % n] ^ dest[(k + p) % n] ^ dest[(k + n - 1) % n]
            r1 = (1664525 * (x ^ (x >> 27))) & 0xFFFFFFFF
            if k == 0:
                r2 = (r1 + s) & 0xFFFFFFFF
            elif 1 <= k <= s:
                r2 = (r1 + (k % n) + self.v[k - 1]) & 0xFFFFFFFF
            else:
                r2 = (r1 + (k % n)) & 0xFFFFFFFF
            dest[(k + p) % n] = (dest[(k + p) % n] + r1) & 0xFFFFFFFF
            dest[(k + q) % n] = (dest[(k + q) % n] + r2) & 0xFFFFFFFF
            dest[k % n] = r2

        for k in range(m, m + n):
            x = (dest[k % n] + dest[(k + p) % n] + dest[(k + n - 1) % n]) & 0xFFFFFFFF
            r3 = (1566083941 * (x ^ (x >> 27))) & 0xFFFFFFFF
            r4 = (r3 - (k % n)) & 0xFFFFFFFF
            dest[(k + p) % n] = (dest[(k + p) % n] ^ r3) & 0xFFFFFFFF
            dest[(k + q) % n] = (dest[(k + q) % n] ^ r4) & 0xFFFFFFFF
            dest[k % n] = r4

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


class TileCodeMt1993764(TileCode):
    """Tile Code using Mt1993764 hash."""

    def __init__(self, dimensionalInfos, numTilings, sizeHint=None, weightCont=None) -> None:
        """
        :param sizeHint: If given, used as the weight vector size. The bigger
                         the size hint, the less likely collisions occur.
        """
        super().__init__(dimensionalInfos, numTilings, sizeHint, weightCont)
        self._prng = mt19937_64()

    def getFeatureVector(self, parameters: Sequence[FLOAT]) -> FEATURE_VECTOR:
        tileComponents: List[int] = [0] * (self.getDimension() + 1)
        fv: FEATURE_VECTOR = []
        for i in range(self.getNumTilings()):
            for j in range(self.getDimension()):
                tileComponents[j] = self.paramToGridValue(parameters[j], i, j)
            # Add a unique number_tiling identifier.
            tileComponents[self.getDimension()] = i
            self._prng.seed(seed_seq(tileComponents))
            fv.append(self._prng() % self._sizeCache)
        return fv
