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

from typing import Any, Generic, TypeVar

from .ReinforcementLearningGDAbstract import ReinforcementLearningGDAbstract
from .GradientDescentTileCodeET import GradientDescentTileCodeET

D = TypeVar("D")
NUM_TILINGS = TypeVar("NUM_TILINGS")
WEIGHT_CONT = TypeVar("WEIGHT_CONT")
STATE_DIM = TypeVar("STATE_DIM")


class ReinforcementLearningGDET(
    ReinforcementLearningGDAbstract[D, STATE_DIM],
    Generic[D, NUM_TILINGS, WEIGHT_CONT, STATE_DIM],
):
    """Gradient descent implementation of Reinforcement Learning with Eligibility Traces.

    Template Parameters:
        D: Number of dimensions.
        NUM_TILINGS: Number of tilings.
        WEIGHT_CONT: The container object to store the weights.
        STATE_DIM: Number of dimensions in State. Defaults to D - 1.
                   This also implies ACTION_DIM = D - STATE_DIM.
    """

    def __init__(
        self,
        tile_code: Any,
        step_size: float,
        discount_rate: float,
        lambda_: float,
        policy: Any,
    ) -> None:
        """Initialize ReinforcementLearningGDET.

        Args:
            tile_code: TileCode implementation to aggregate.
            step_size: Step size for gradient descent updates.
            discount_rate: Discount rate for future state values.
            lambda_: Eligibility trace decay parameter.
            policy: Action selection control policy.
        """
        super().__init__(tile_code, step_size, discount_rate, lambda_, policy)
        self._gradient_descent = GradientDescentTileCodeET(
            tile_code, step_size, discount_rate, lambda_
        )
        self._gradientDescent = self._gradient_descent

    def __del__(self) -> None:
        """Destructor."""
        pass