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

from typing import TYPE_CHECKING, Any, Generic, TypeVar, final

from .reinforcement_learning_gdet import ReinforcementLearningGDET

if TYPE_CHECKING:
    from ...coding.tile_code import TileCode
    from ...policy.policy import Policy

D = TypeVar("D", bound=int)
NumTilings = TypeVar("NumTilings", bound=int)
WeightCont = TypeVar("WeightCont")
StateDim = TypeVar("StateDim", bound=int)


@final
class SarsaETGD(
    ReinforcementLearningGDET[D, NumTilings, WeightCont, StateDim],
    Generic[D, NumTilings, WeightCont, StateDim],
):
    """Gradient Descent with Sarsa implementation (the same policy for

    learning and action selection).
    """

    def __init__(
        self,
        tile_code: TileCode[D, NumTilings, WeightCont],
        step_size: float,
        discount_rate: float,
        lambda_: float,
        policy: Policy[Any, Any],
    ) -> None:
        super().__init__(tile_code, step_size, discount_rate, lambda_, policy)
        if hasattr(self, "set_learning_policy"):
            self.set_learning_policy(policy)
        elif hasattr(self, "setLearningPolicy"):
            self.setLearningPolicy(policy)
        else:
            self._learning_policy = policy