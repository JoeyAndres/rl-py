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

from typing import Any, Generic, List, TypeVar

from .ReinforcementLearningGD import ReinforcementLearningGD

DEFAULT_TILE_CONT = List[float]

D = TypeVar("D")
NUM_TILINGS = TypeVar("NUM_TILINGS")
WEIGHT_CONT = TypeVar("WEIGHT_CONT")
STATE_DIM = TypeVar("STATE_DIM")


class QLearningGD(
    ReinforcementLearningGD[D, NUM_TILINGS, WEIGHT_CONT, STATE_DIM],
    Generic[D, NUM_TILINGS, WEIGHT_CONT, STATE_DIM],
):
    """Gradient Descent with QLearning implementation (Separate control
    and learning policy).

    :tparam D: Number of dimensions.
    :tparam NUM_TILINGS: Number of tilings.
    :tparam WEIGHT_CONT: The container object to store the weights.
    :tparam STATE_DIM: Number of dimension in State. This defaults to D-1.
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
        """Initialize QLearningGD.

        :param tile_code: Tile code representation.
        :param step_size: Learning step size.
        :param discount_rate: Discount rate.
        :param lambda_: Eligibility trace decay parameter.
        :param policy: Control policy.
        """
        super().__init__(
            tile_code,
            step_size,
            discount_rate,
            lambda_,
            policy,
        )


__all__ = ["DEFAULT_TILE_CONT", "QLearningGD"]