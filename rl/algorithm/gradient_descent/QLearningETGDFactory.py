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

try:
    from .ReinforcementLearningGDFactory import ReinforcementLearningGDFactory
    from .QLearningETGD import QLearningETGD
except (ImportError, ValueError):
    from ReinforcementLearningGDFactory import ReinforcementLearningGDFactory
    from QLearningETGD import QLearningETGD

try:
    from ...coding.TileCode import DEFAULT_TILE_CONT
except (ImportError, ValueError):
    try:
        from coding.TileCode import DEFAULT_TILE_CONT
    except (ImportError, ValueError):
        DEFAULT_TILE_CONT = List[float]

D = TypeVar("D")
NUM_TILINGS = TypeVar("NUM_TILINGS")
try:
    WEIGHT_CONT = TypeVar("WEIGHT_CONT", default=DEFAULT_TILE_CONT)
except TypeError:
    WEIGHT_CONT = TypeVar("WEIGHT_CONT")
STATE_DIM = TypeVar("STATE_DIM")


class QLearningETGDFactory(
    ReinforcementLearningGDFactory[D, NUM_TILINGS, WEIGHT_CONT, STATE_DIM, QLearningETGD],
    Generic[D, NUM_TILINGS, WEIGHT_CONT, STATE_DIM],
):
    """Factory method for QLearningETGD.

    Template parameters:
        D: Number of dimension.
        NUM_TILINGS: Number of tilings.
        WEIGHT_CONT: The container object to store the weights.
        STATE_DIM: Number of dimension in state.
                   Implies that action is D - STATE_DIM.
    """

    _algorithm_class = QLearningETGD

    def __init__(
        self,
        tile_code: Any,
        step_size: float,
        discount_rate: float,
        lambda_: float,
        policy: Any,
    ) -> None:
        super().__init__(tile_code, step_size, discount_rate, lambda_, policy)
        if getattr(self, "_instance", None) is None:
            self._instance = QLearningETGD(
                tile_code, step_size, discount_rate, lambda_, policy
            )


__all__ = ["QLearningETGDFactory"]