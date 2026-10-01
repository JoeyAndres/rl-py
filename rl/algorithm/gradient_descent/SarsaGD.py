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

from typing import Any, Generic, TypeVar, final

from .ReinforcementLearningGD import ReinforcementLearningGD

D = TypeVar("D")
NUM_TILINGS = TypeVar("NUM_TILINGS")
WEIGHT_CONT = TypeVar("WEIGHT_CONT")
STATE_DIM = TypeVar("STATE_DIM")


@final
class SarsaGD(
    ReinforcementLearningGD[D, NUM_TILINGS, WEIGHT_CONT, STATE_DIM],
    Generic[D, NUM_TILINGS, WEIGHT_CONT, STATE_DIM],
):
    """
    Gradient Descent with Sarsa implementation (the same policy for
    learning and action selection).

    :tparam D: Number of dimensions.
    :tparam NUM_TILINGS: Number of tilings.
    :tparam WEIGHT_CONT: The container object to store the weights.
                         Defaults to coding.DEFAULT_TILE_CONT.
    :tparam STATE_DIM: Number of dimensions in State. Defaults to D - 1.
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
        """
        Initialize SarsaGD.

        :param tile_code: TileCode implementation to be aggregated.
        :param step_size: Step size indicates how fast updates to estimates occur.
        :param discount_rate: How much current state influences future states.
        :param lambda_: Eligibility trace decay parameter (multiplied with discount rate).
        :param policy: Control and learning policy.
        """
        super().__init__(tile_code, step_size, discount_rate, lambda_, policy)
        if hasattr(self, "set_learning_policy"):
            self.set_learning_policy(policy)
        else:
            self.setLearningPolicy(policy)