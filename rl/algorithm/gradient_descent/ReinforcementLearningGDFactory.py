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

from typing import Any, Generic, List, Optional, Type, TypeVar

D = TypeVar("D")
NUM_TILINGS = TypeVar("NUM_TILINGS")
WEIGHT_CONT = TypeVar("WEIGHT_CONT")
STATE_DIM = TypeVar("STATE_DIM")
ReinforcementLearningGDType = TypeVar("ReinforcementLearningGDType")

from ..LearningAlgorithmFactory import LearningAlgorithmFactory
from .ReinforcementLearningGD import ReinforcementLearningGD
from .ReinforcementLearningGDAbstract import ReinforcementLearningGDAbstract

from ...coding.TileCode import TileCode
from ...policy.Policy import Policy


class ReinforcementLearningGDFactory(
    LearningAlgorithmFactory[List[float], List[float]],
    Generic[D, NUM_TILINGS, WEIGHT_CONT, STATE_DIM, ReinforcementLearningGDType],
):
    """Factory method for ReinforcementLearningGDFactory.

    :tparam D: Number of dimension.
    :tparam NUM_TILINGS: Number of tilings.
    :tparam WEIGHT_CONT: The container object to store the weights.
    :tparam STATE_DIM: Number of dimension in state.
                       Implies that action is D - STATE_DIM.
    :tparam REINFORCEMENT_LEARNING_GD: ReinforcementLearningGDAbstract child class.
    """

    _algorithm_cls: Optional[Type[ReinforcementLearningGDType]] = None

    def __init__(
        self,
        tile_code: Any,
        step_size: float,
        discount_rate: float,
        lambda_: float,
        policy: Any,
        algorithm_cls: Optional[Type[ReinforcementLearningGDType]] = None,
    ) -> None:
        super().__init__()
        cls = algorithm_cls or self._algorithm_cls or ReinforcementLearningGD
        if cls is not None:
            self._instance = cls(
                tile_code, step_size, discount_rate, lambda_, policy
            )
        else:
            raise ValueError("Algorithm class must be specified.")