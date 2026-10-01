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

from typing import Any, Generic, List, TypeVar

from .ReinforcementLearningGDAbstract import ReinforcementLearningGDAbstract
from .GradientDescentTileCode import GradientDescentTileCode
from ...coding.TileCode import DEFAULT_TILE_CONT, TileCode
from ...policy.Policy import Policy

FLOAT = float

spTileCode = TileCode
spPolicy = Policy

D = TypeVar("D")
NUM_TILINGS = TypeVar("NUM_TILINGS")
WEIGHT_CONT = TypeVar("WEIGHT_CONT")
STATE_DIM = TypeVar("STATE_DIM")


class ReinforcementLearningGD(
    ReinforcementLearningGDAbstract[D, STATE_DIM],
    Generic[D, NUM_TILINGS, WEIGHT_CONT, STATE_DIM],
):
    """
    Gradient descent implementation of Reinforcement Learning.

    :tparam D: Number of dimension.
    :tparam NUM_TILINGS: Number of tilings.
    :tparam WEIGHT_CONT: The container object to store the weights.
    :tparam STATE_DIM: Number of dimension in State. This defaults to D-1.
                       This also implies ACTION_DIM = D - STATE_DIM.
    """

    def __init__(
        self,
        tile_code: spTileCode[D, NUM_TILINGS, WEIGHT_CONT],
        step_size: FLOAT,
        discount_rate: FLOAT,
        lambda_: FLOAT,
        policy: spPolicy,
    ) -> None:
        super().__init__(tile_code, step_size, discount_rate, lambda_, policy)
        self._gradientDescent = GradientDescentTileCode(
            tile_code, step_size, discount_rate, lambda_
        )
        self._gradient_descent = self._gradientDescent

    def __del__(self) -> None:
        pass