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

from abc import ABC, abstractmethod
from typing import Any, Generic, List, Sequence, TypeVar, Union

FLOAT = float
UINT = int
FEATURE_VECTOR = List[UINT]

D = TypeVar("D")
NUM_TILINGS = TypeVar("NUM_TILINGS")
WEIGHT_CONT = TypeVar("WEIGHT_CONT")
STATE_DIM = TypeVar("STATE_DIM")

floatArray = Sequence[FLOAT]

from .GradientDescentAbstract import GradientDescentAbstract

from ...coding.TileCode import TileCode

spTileCode = TileCode


class GradientDescentTileCodeAbstract(
    GradientDescentAbstract[D, STATE_DIM],
    Generic[D, NUM_TILINGS, WEIGHT_CONT, STATE_DIM],
):
    """
    Abstract class for Gradient Descent utilizing tile codes.

    :param D: Number of dimension.
    :param NUM_TILINGS: Number of tilings.
    :param WEIGHT_CONT: The container object to store the weights.
    :param STATE_DIM: Number of dimension in State.
                      This also implies ACTION_DIM = D - STATE_DIM.
    """

    _tileCode: TileCode[D, NUM_TILINGS, WEIGHT_CONT]

    def __init__(
        self,
        tileCode: TileCode[D, NUM_TILINGS, WEIGHT_CONT],
        stepSize: FLOAT,
        discountRate: FLOAT,
        lambda_: FLOAT,
    ) -> None:
        """
        :param tileCode: Type of course coding.
        :param stepSize: Step size for gradient descent.
        :param discountRate: Discount rate for gradient descent.
        :param lambda_: How influential is current state-action to other state-action.
        """
        super().__init__(tileCode, stepSize, discountRate, lambda_)
        # This is equivalent to _tileCode = _courseCode (downcast'd).
        self._tileCode = tileCode

    def getValueFromFeatureVector(self, fv: FEATURE_VECTOR) -> FLOAT:
        """
        Get the value of the parameters in the real space.

        :param fv: Feature vector.
        :return: Corresponding value.
        """
        return self._tileCode.getValueFromFeatureVector(fv)

    def getFeatureVector(self, parameters: Sequence[FLOAT]) -> FEATURE_VECTOR:
        """
        :param parameters: Parameters in the n-dimension space.
        :return: Feature vector output. Feature vector are samples taken around
                 the parameters in the n-dimension tilecode.
        """
        return self._tileCode.getFeatureVector(parameters)

    # Aliases for pythonic naming conventions
    get_value_from_feature_vector = getValueFromFeatureVector
    get_feature_vector = getFeatureVector