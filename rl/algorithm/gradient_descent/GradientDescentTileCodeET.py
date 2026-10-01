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

from typing import List

from ...coding.TileCode import TileCode
from ...declares import FEATURE_VECTOR, FLOAT
from .GradientDescentAbstract import ActionParam, StateParam
from .GradientDescentTileCodeAbstract import GradientDescentTileCodeAbstract


class GradientDescentTileCodeET(GradientDescentTileCodeAbstract):
    """Gradient Descent with eligibility traces."""

    def __init__(
        self,
        tileCode: TileCode,
        stepSize: FLOAT,
        discountRate: FLOAT,
        lambda_: FLOAT,
    ) -> None:
        super().__init__(tileCode, stepSize, discountRate, lambda_)
        self._e: List[FLOAT] = [0.0] * self._courseCode.getSize()

    def incrementEligibilityTraces(self, fv: FEATURE_VECTOR) -> None:
        for f in fv:
            self._e[f] += 1

    def replaceEligibilityTraces(self, fv: FEATURE_VECTOR) -> None:
        for f in fv:
            self._e[f] = 1

    def decreaseEligibilityTraces(self) -> None:
        for i in range(self.getSize()):
            self._e[i] *= self._discountRateTimesLambda

    def resetEligibilityTraces(self) -> None:
        self._e = [0.0] * self.getSize()

    def backUpWeights(self, tdError: FLOAT) -> None:
        """
        :param tdError: Temporal difference error.
        """
        multiplier = (self._stepSize / self._tileCode.getNumTilings()) * tdError
        # The C++ loop stopped at n-1, skipping the last weight; update them all.
        for i in range(self.getSize()):
            self._tileCode[i] += multiplier * self._e[i]

    def updateWeights(
        self,
        currentStateVector: StateParam,
        currentActionVector: ActionParam,
        nextStateVector: StateParam,
        nextActionValue: FLOAT,
        reward: FLOAT,
    ) -> None:
        currentStateFv = self.getFeatureVector(
            tuple(currentStateVector) + tuple(currentActionVector)
        )
        self.incrementEligibilityTraces(currentStateFv)
        tdError = (
            reward
            + self._discountRate * nextActionValue
            - self.getValueFromFeatureVector(currentStateFv)
        )
        self.backUpWeights(tdError)
        self.decreaseEligibilityTraces()
